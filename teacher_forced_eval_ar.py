#!/usr/bin/env python3
import argparse
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ARData
from generate_brep import load_checkpoint, load_model_weights
from model import ARModel


@dataclass
class SampleEval:
    dataset_index: int
    seq_len: int
    valid_target_tokens: int
    token_correct: int
    loss_sum: float
    face_correct: int
    face_total: int
    edge_correct: int
    edge_total: int
    first_error_pos: Optional[int]
    errors: List[Dict[str, int]]

    @property
    def accuracy(self) -> float:
        if self.valid_target_tokens <= 0:
            return 0.0
        return self.token_correct / self.valid_target_tokens


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: ARData):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base_dataset[idx]
        item["dataset_index"] = idx
        item["seq_len"] = int(item["attention_mask"].sum().item())
        return item

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        padded = self.base_dataset.collate_fn(batch)
        padded["dataset_index"] = torch.tensor([x["dataset_index"] for x in batch], dtype=torch.long)
        padded["seq_len"] = torch.tensor([x["seq_len"] for x in batch], dtype=torch.long)
        return padded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict teacher-forced evaluation for AR model")
    parser.add_argument("--sequence_file", type=str, required=True, help="Grouped AR sequence pkl")
    parser.add_argument("--weight", type=str, required=True, help="AR checkpoint path")
    parser.add_argument("--image_feature_index_file", type=str, default="", help="Optional image feature index for prefix-conditioned AR")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means full split")
    parser.add_argument("--output_json", type=str, default="")
    parser.add_argument("--save_worst_k", type=int, default=10)
    parser.add_argument("--error_top_k", type=int, default=20)
    parser.add_argument("--dataset_type", type=str, choices=["abc", "deepcad", "furniture"], default="abc")
    parser.add_argument("--tb_log_dir", type=str, default="logs/teacher_forced_eval")
    parser.add_argument("--dir_name", type=str, default="checkpoints")
    parser.add_argument("--env", type=str, default="teacher_forced_eval")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--train_epoch", type=int, default=1)
    parser.add_argument("--test_epoch", type=int, default=1)
    parser.add_argument("--save_epoch", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    args = parser.parse_args()
    args.save_dir = f"{args.dir_name}/{args.env}"
    return args


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def build_model_from_checkpoint(checkpoint: Dict[str, Any], dataset: ARData, device: torch.device) -> ARModel:
    config_dict = checkpoint.get("config", {}) or {}
    state_dict = checkpoint.get("model_state_dict", {}) or {}
    wpe_len = None
    for key in ('model.transformer.wpe.weight', 'transformer.wpe.weight', 'module.model.transformer.wpe.weight', 'module.transformer.wpe.weight'):
        if key in state_dict:
            wpe_len = int(state_dict[key].shape[0])
            break
    max_seq_len = dataset.max_seq_len
    if wpe_len is None:
        wpe_len = config_dict.get("n_positions", max_seq_len)
    num_image_prefix_tokens = max(0, wpe_len - max_seq_len)
    model = ARModel(
        vocab_size=dataset.vocab_size,
        d_model=config_dict.get("n_embd", 256),
        nhead=config_dict.get("n_head", 8),
        num_layers=config_dict.get("n_layer", 8),
        dim_feedforward=config_dict.get("n_inner", 1024),
        dropout=0.0,
        max_seq_len=max_seq_len,
        pad_token_id=dataset.PAD_TOKEN,
        use_image_prefix=num_image_prefix_tokens > 0,
        image_feature_dim=1024,
        num_image_prefix_tokens=num_image_prefix_tokens,
    ).to(device)
    load_model_weights(model, checkpoint)
    model.eval()
    return model


def classify_positions(seq: torch.Tensor, dataset: ARData) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_list = seq.tolist()
    sep_token = dataset.SEP_TOKEN
    end_token = dataset.END_TOKEN
    valid_len = len(seq_list)
    if end_token in seq_list:
        valid_len = seq_list.index(end_token) + 1
    sep_pos = seq_list.index(sep_token) if sep_token in seq_list[:valid_len] else valid_len

    face_mask = torch.zeros(len(seq_list), dtype=torch.bool)
    edge_mask = torch.zeros(len(seq_list), dtype=torch.bool)

    for pos in range(1, min(sep_pos, valid_len)):
        face_mask[pos] = True
    for pos in range(sep_pos + 1, valid_len):
        if seq_list[pos] == end_token:
            break
        edge_mask[pos] = True
    return face_mask, edge_mask


def length_bucket(seq_len: int) -> str:
    if seq_len < 512:
        return "lt_512"
    if seq_len < 1024:
        return "512_1023"
    if seq_len < 1536:
        return "1024_1535"
    return "ge_1536"


def finalize_metrics(correct: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return correct / total


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"Using device: {device}")

    validate = args.split == "val"
    base_dataset = ARData(
        sequence_file=args.sequence_file,
        validate=validate,
        args=args,
        image_feature_index_file=args.image_feature_index_file or None,
    )
    dataset = EvalDataset(base_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = load_checkpoint(args.weight)
    model = build_model_from_checkpoint(checkpoint, base_dataset, device)

    total_loss_sum = 0.0
    total_valid_tokens = 0
    total_correct = 0
    total_top5 = 0
    face_correct = 0
    face_total = 0
    edge_correct = 0
    edge_total = 0
    bucket_stats: Dict[str, Dict[str, int]] = {}
    sample_evals: List[SampleEval] = []

    progress = tqdm(dataloader, desc=f"Teacher-forced eval ({args.split})")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            dataset_indices = batch["dataset_index"].tolist()
            seq_lens = batch["seq_len"].tolist()

            image_features = batch.get("image_features")
            if image_features is not None:
                image_features = image_features.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, image_features=image_features)
            prefix_len = outputs.logits.shape[1] - input_ids.shape[1]
            logits = outputs.logits[:, prefix_len:-1, :]
            targets = input_ids[:, 1:]
            target_mask = attention_mask[:, 1:].bool() & (targets != base_dataset.PAD_TOKEN)

            log_probs = F.log_softmax(logits, dim=-1)
            gather = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            nll = -gather

            preds = logits.argmax(dim=-1)
            correct = (preds == targets) & target_mask
            top5 = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
            top5_correct = top5.eq(targets.unsqueeze(-1)).any(dim=-1) & target_mask

            batch_loss_sum = nll.masked_select(target_mask).sum().item()
            batch_valid_tokens = int(target_mask.sum().item())
            batch_correct = int(correct.sum().item())
            batch_top5_correct = int(top5_correct.sum().item())

            total_loss_sum += batch_loss_sum
            total_valid_tokens += batch_valid_tokens
            total_correct += batch_correct
            total_top5 += batch_top5_correct

            for row in range(input_ids.shape[0]):
                seq = input_ids[row]
                valid_seq_len = seq_lens[row]
                face_mask_full, edge_mask_full = classify_positions(seq[:valid_seq_len].cpu(), base_dataset)

                row_target_mask = target_mask[row, :valid_seq_len - 1].cpu()
                row_correct = correct[row, :valid_seq_len - 1].cpu()
                row_targets = targets[row, :valid_seq_len - 1].cpu()
                row_preds = preds[row, :valid_seq_len - 1].cpu()
                row_nll = nll[row, :valid_seq_len - 1].cpu()

                face_target_mask = face_mask_full[1:valid_seq_len] & row_target_mask
                edge_target_mask = edge_mask_full[1:valid_seq_len] & row_target_mask

                row_face_total = int(face_target_mask.sum().item())
                row_edge_total = int(edge_target_mask.sum().item())
                row_face_correct = int((row_correct & face_target_mask).sum().item())
                row_edge_correct = int((row_correct & edge_target_mask).sum().item())

                face_total += row_face_total
                edge_total += row_edge_total
                face_correct += row_face_correct
                edge_correct += row_edge_correct

                first_error_pos = None
                errors: List[Dict[str, int]] = []
                valid_positions = torch.nonzero(row_target_mask, as_tuple=False).flatten().tolist()
                for pos in valid_positions:
                    if not bool(row_correct[pos].item()):
                        token_position = pos + 1
                        if first_error_pos is None:
                            first_error_pos = token_position
                        if len(errors) < args.error_top_k:
                            region = "face" if bool(face_mask_full[token_position].item()) else "edge" if bool(edge_mask_full[token_position].item()) else "other"
                            errors.append({
                                "position": token_position,
                                "target": int(row_targets[pos].item()),
                                "pred": int(row_preds[pos].item()),
                                "region": region,
                            })

                row_valid_targets = int(row_target_mask.sum().item())
                row_token_correct = int((row_correct & row_target_mask).sum().item())
                row_loss_sum = float(row_nll.masked_select(row_target_mask).sum().item())

                sample_evals.append(SampleEval(
                    dataset_index=int(dataset_indices[row]),
                    seq_len=int(valid_seq_len),
                    valid_target_tokens=row_valid_targets,
                    token_correct=row_token_correct,
                    loss_sum=row_loss_sum,
                    face_correct=row_face_correct,
                    face_total=row_face_total,
                    edge_correct=row_edge_correct,
                    edge_total=row_edge_total,
                    first_error_pos=first_error_pos,
                    errors=errors,
                ))

                bucket = length_bucket(int(valid_seq_len))
                if bucket not in bucket_stats:
                    bucket_stats[bucket] = {"samples": 0, "tokens": 0, "correct": 0}
                bucket_stats[bucket]["samples"] += 1
                bucket_stats[bucket]["tokens"] += row_valid_targets
                bucket_stats[bucket]["correct"] += row_token_correct

            running_acc = total_correct / total_valid_tokens if total_valid_tokens else 0.0
            running_ppl = math.exp(total_loss_sum / total_valid_tokens) if total_valid_tokens else float("inf")
            progress.set_postfix(acc=f"{running_acc:.4f}", ppl=f"{running_ppl:.2f}")

    overall_loss = total_loss_sum / total_valid_tokens if total_valid_tokens else float("inf")
    overall_accuracy = total_correct / total_valid_tokens if total_valid_tokens else 0.0
    overall_top5 = total_top5 / total_valid_tokens if total_valid_tokens else 0.0
    perplexity = math.exp(overall_loss) if total_valid_tokens else float("inf")

    worst_samples = sorted(sample_evals, key=lambda x: (x.accuracy, -x.seq_len))[: args.save_worst_k]
    best_samples = sorted(sample_evals, key=lambda x: (-x.accuracy, -x.seq_len))[: min(args.save_worst_k, len(sample_evals))]

    bucket_summary = {}
    for bucket, stats in bucket_stats.items():
        bucket_summary[bucket] = {
            "samples": stats["samples"],
            "tokens": stats["tokens"],
            "accuracy": stats["correct"] / stats["tokens"] if stats["tokens"] else None,
        }

    results = {
        "split": args.split,
        "sequence_file": args.sequence_file,
        "weight": args.weight,
        "device": str(device),
        "dataset_size": len(base_dataset),
        "evaluated_samples": len(sample_evals),
        "evaluated_tokens": total_valid_tokens,
        "overall": {
            "loss": overall_loss,
            "perplexity": perplexity,
            "accuracy": overall_accuracy,
            "top5_accuracy": overall_top5,
        },
        "regions": {
            "face_accuracy": finalize_metrics(face_correct, face_total),
            "face_tokens": face_total,
            "edge_accuracy": finalize_metrics(edge_correct, edge_total),
            "edge_tokens": edge_total,
        },
        "length_buckets": bucket_summary,
        "worst_samples": [
            {
                "dataset_index": s.dataset_index,
                "seq_len": s.seq_len,
                "accuracy": s.accuracy,
                "valid_target_tokens": s.valid_target_tokens,
                "first_error_pos": s.first_error_pos,
                "face_accuracy": finalize_metrics(s.face_correct, s.face_total),
                "edge_accuracy": finalize_metrics(s.edge_correct, s.edge_total),
                "errors": s.errors,
            }
            for s in worst_samples
        ],
        "best_samples": [
            {
                "dataset_index": s.dataset_index,
                "seq_len": s.seq_len,
                "accuracy": s.accuracy,
                "valid_target_tokens": s.valid_target_tokens,
                "first_error_pos": s.first_error_pos,
            }
            for s in best_samples
        ],
    }

    print("\n=== Teacher-forced Eval Summary ===")
    print(f"Split: {args.split}")
    print(f"Samples: {len(sample_evals)} / {len(base_dataset)}")
    print(f"Tokens: {total_valid_tokens}")
    print(f"Loss: {overall_loss:.6f}")
    print(f"Perplexity: {perplexity:.6f}")
    print(f"Accuracy: {overall_accuracy:.6f}")
    print(f"Top-5 Accuracy: {overall_top5:.6f}")
    print(f"Face Accuracy: {results['regions']['face_accuracy']}")
    print(f"Edge Accuracy: {results['regions']['edge_accuracy']}")
    print("Length Buckets:")
    for bucket in sorted(bucket_summary.keys()):
        stats = bucket_summary[bucket]
        print(f"  {bucket}: samples={stats['samples']}, tokens={stats['tokens']}, accuracy={stats['accuracy']}")

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Saved eval json to: {args.output_json}")


if __name__ == "__main__":
    main()
