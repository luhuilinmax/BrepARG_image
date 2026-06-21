#!/usr/bin/env python3
import argparse
import json
import os
from types import SimpleNamespace

import torch

from dataset import ARData
from generate_brep import load_checkpoint, load_model_weights
from model import ARModel


def parse_args():
    parser = argparse.ArgumentParser(description='Partial teacher forcing check for image-prefix AR')
    parser.add_argument('--sequence_file', type=str, required=True)
    parser.add_argument('--image_feature_index_file', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='val')
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_json', type=str, default='')
    parser.add_argument('--max_generate_len', type=int, default=2048)
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg.startswith('cuda') and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device('cpu')


def build_model_from_checkpoint(checkpoint, dataset, device):
    config_dict = checkpoint.get('config', {}) or {}
    state_dict = checkpoint.get('model_state_dict', {}) or {}
    wpe_len = None
    for key in ('model.transformer.wpe.weight', 'transformer.wpe.weight', 'module.model.transformer.wpe.weight', 'module.transformer.wpe.weight'):
        if key in state_dict:
            wpe_len = int(state_dict[key].shape[0])
            break
    max_seq_len = dataset.max_seq_len
    if wpe_len is None:
        wpe_len = config_dict.get('n_positions', max_seq_len)
    num_image_prefix_tokens = max(0, wpe_len - max_seq_len)
    model = ARModel(
        vocab_size=dataset.vocab_size,
        d_model=config_dict.get('n_embd', 256),
        nhead=config_dict.get('n_head', 8),
        num_layers=config_dict.get('n_layer', 8),
        dim_feedforward=config_dict.get('n_inner', 1024),
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


def longest_common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def infer_cut_points(tokens, sep_token):
    sep_pos = tokens.index(sep_token)
    candidates = [
        ('start_only', 1),
        ('first_face_block', min(sep_pos, 12)),
        ('through_sep', sep_pos + 1),
    ]
    unique = []
    seen = set()
    for name, k in candidates:
        k = max(1, min(k, len(tokens)))
        if k not in seen:
            unique.append((name, k))
            seen.add(k)
    return unique


def generate_from_prefix(model, prefix_ids, image_features, max_length, pad_token, end_token, start_token):
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=image_features.device).unsqueeze(0)
    prefix_mask = torch.ones_like(prefix)
    with torch.no_grad():
        generated = model.generate(
            input_ids=prefix,
            attention_mask=prefix_mask,
            image_features=image_features,
            max_length=max_length,
            temperature=0.6,
            top_k=0,
            top_p=0.6,
            repetition_penalty=1.0,
            do_sample=True,
            use_cache=True,
            pad_token_id=pad_token,
            eos_token_id=end_token,
            bos_token_id=start_token,
        )
    return generated[0].detach().cpu().tolist()


def main():
    args = parse_args()
    device = choose_device(args.device)
    ds_args = SimpleNamespace(max_seq_len=args.max_seq_len)
    validate = args.split == 'val'
    dataset = ARData(
        sequence_file=args.sequence_file,
        validate=validate,
        args=ds_args,
        image_feature_index_file=args.image_feature_index_file,
    )
    sample = dataset[args.sample_index]
    target_tokens = sample['input_ids'].tolist()
    image_features = sample['image_features'].unsqueeze(0).to(device)

    checkpoint = load_checkpoint(args.weight)
    model = build_model_from_checkpoint(checkpoint, dataset, device)

    cut_points = infer_cut_points(target_tokens, dataset.SEP_TOKEN)
    results = []
    for name, k in cut_points:
        prefix_ids = target_tokens[:k]
        max_length = min(model.config.n_positions, max(args.max_generate_len, len(prefix_ids) + 32))
        generated = generate_from_prefix(
            model,
            prefix_ids,
            image_features,
            max_length,
            dataset.PAD_TOKEN,
            dataset.END_TOKEN,
            dataset.START_TOKEN,
        )
        generated_continuation = generated[len(prefix_ids):] if generated[:len(prefix_ids)] == prefix_ids else generated
        compare_target = target_tokens[k:]
        compare_generated = generated_continuation
        suffix_match = 0
        for a, b in zip(compare_target, compare_generated):
            if a == b:
                suffix_match += 1
            else:
                break
        first_mismatch = None
        if suffix_match < min(len(compare_target), len(compare_generated)):
            first_mismatch = {
                'position_in_full_sequence': k + suffix_match,
                'position_in_continuation': suffix_match,
                'target': compare_target[suffix_match],
                'generated': compare_generated[suffix_match],
            }
        results.append({
            'mode': name,
            'prefix_len': k,
            'generated_total_len': len(generated),
            'generated_continuation_len': len(generated_continuation),
            'continuation_match_len': suffix_match,
            'target_continuation_len': len(compare_target),
            'first_mismatch': first_mismatch,
            'prefix_tokens_32': prefix_ids[:32],
            'generated_continuation_prefix_64': compare_generated[:64],
            'target_continuation_prefix_64': compare_target[:64],
        })

    output = {
        'split': args.split,
        'sample_index': args.sample_index,
        'cad_stem': sample.get('cad_stem'),
        'target_len': len(target_tokens),
        'sep_pos': target_tokens.index(dataset.SEP_TOKEN),
        'results': results,
    }
    print(json.dumps(output, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)


if __name__ == '__main__':
    main()
