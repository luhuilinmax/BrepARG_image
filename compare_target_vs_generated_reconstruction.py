#!/usr/bin/env python3
import argparse
import json
import os
from types import SimpleNamespace

import torch

from dataset import ARData
from generate_brep import load_checkpoint, load_model_weights
from model import ARModel
from reconstruct_vqvae_sample import write_brep_outputs
from utils import load_se_vqvae_model, reconstruct_cad_from_sequence


def parse_args():
    parser = argparse.ArgumentParser(description='Compare target-vs-generated CAD reconstruction for one sample')
    parser.add_argument('--sequence_file', type=str, required=True)
    parser.add_argument('--image_feature_index_file', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--se_vqvae', type=str, required=True)
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='val')
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset_type', type=str, default='abc')
    parser.add_argument('--output_dir', type=str, required=True)
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


def build_vocab_info(dataset):
    return {
        'vocab_size': dataset.vocab_size,
        'special_token_size': dataset.special_token_size,
        'face_index_size': dataset.face_index_size,
        'se_codebook_size': dataset.se_codebook_size,
        'bbox_index_size': dataset.bbox_index_size,
        'se_tokens_per_element': dataset.se_tokens_per_element,
        'bbox_tokens_per_element': dataset.bbox_tokens_per_element,
        'face_index_offset': dataset.face_index_offset,
        'se_token_offset': dataset.se_token_offset,
        'bbox_token_offset': dataset.bbox_token_offset,
        'START_TOKEN': dataset.START_TOKEN,
        'SEP_TOKEN': dataset.SEP_TOKEN,
        'END_TOKEN': dataset.END_TOKEN,
        'PAD_TOKEN': dataset.PAD_TOKEN,
    }


def generate_sequence(model, dataset, image_features, max_generate_len):
    prompt = torch.tensor([[dataset.START_TOKEN]], dtype=torch.long, device=image_features.device)
    prompt_mask = torch.ones_like(prompt)
    max_length = min(model.config.n_positions, max_generate_len)
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt,
            attention_mask=prompt_mask,
            image_features=image_features,
            max_length=max_length,
            temperature=0.6,
            top_k=0,
            top_p=0.6,
            repetition_penalty=1.0,
            do_sample=True,
            use_cache=True,
            pad_token_id=dataset.PAD_TOKEN,
            eos_token_id=dataset.END_TOKEN,
            bos_token_id=dataset.START_TOKEN,
        )
    return generated[0].detach().cpu().tolist()


def reconstruct_one(sequence, vocab_info, se_vqvae_model, device, output_dir, prefix):
    record = {
        'sequence_len': len(sequence),
        'sequence_path': os.path.join(output_dir, f'{prefix}_sequence.json'),
        'brep': None,
        'error': None,
    }
    with open(record['sequence_path'], 'w', encoding='utf-8') as f:
        json.dump(sequence, f)

    try:
        solid, debug_info = reconstruct_cad_from_sequence(
            sequence=sequence,
            vocab_info=vocab_info,
            se_vqvae_model=se_vqvae_model,
            device=device,
            scale_factor=1.0,
            verbose=False,
            return_debug=True,
        )
        debug_path = os.path.join(output_dir, f'{prefix}_brep_debug.pkl')
        import pickle
        with open(debug_path, 'wb') as f:
            pickle.dump(debug_info, f)
        record['debug_path'] = debug_path
        if solid is None:
            record['error'] = 'reconstruct_cad_from_sequence returned None'
        else:
            record['brep'] = write_brep_outputs(solid, output_dir, prefix)
    except Exception as exc:
        record['error'] = repr(exc)
    return record


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
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
    checkpoint = load_checkpoint(args.weight)
    model = build_model_from_checkpoint(checkpoint, dataset, device)
    se_vqvae_model = load_se_vqvae_model(args.se_vqvae, False, args.dataset_type, device)
    if se_vqvae_model is None:
        raise RuntimeError(f'Failed to load VQ-VAE checkpoint: {args.se_vqvae}')

    vocab_info = build_vocab_info(dataset)
    target_sequence = sample['input_ids'].tolist()
    image_features = sample['image_features'].unsqueeze(0).to(device)
    generated_sequence = generate_sequence(model, dataset, image_features, args.max_generate_len)

    result = {
        'cad_stem': sample.get('cad_stem'),
        'sample_index': args.sample_index,
        'split': args.split,
        'target_len': len(target_sequence),
        'generated_len': len(generated_sequence),
        'target': reconstruct_one(target_sequence, vocab_info, se_vqvae_model, device, args.output_dir, 'target'),
        'generated': reconstruct_one(generated_sequence, vocab_info, se_vqvae_model, device, args.output_dir, 'generated'),
    }

    metrics_path = os.path.join(args.output_dir, 'compare_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
