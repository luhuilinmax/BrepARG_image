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
    parser = argparse.ArgumentParser(description='Single-sample image-conditioned free-running check')
    parser.add_argument('--sequence_file', type=str, required=True)
    parser.add_argument('--image_feature_index_file', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='val')
    parser.add_argument('--max_seq_len', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_json', type=str, default='')
    return parser.parse_args()


def choose_device(device_arg: str) -> torch.device:
    if device_arg.startswith('cuda') and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device('cpu')


def build_model_from_checkpoint(checkpoint, dataset, device):
    config_dict = checkpoint.get('config', {}) or {}
    state_dict = checkpoint.get('model_state_dict', {}) or {}
    wpe_key = None
    for key in ('model.transformer.wpe.weight', 'transformer.wpe.weight', 'module.model.transformer.wpe.weight', 'module.transformer.wpe.weight'):
        if key in state_dict:
            wpe_key = key
            break
    if wpe_key is not None:
        n_positions = int(state_dict[wpe_key].shape[0])
    else:
        n_positions = config_dict.get('n_positions', dataset.max_seq_len)
    num_image_prefix_tokens = max(0, n_positions - dataset.max_seq_len)
    model = ARModel(
        vocab_size=dataset.vocab_size,
        d_model=config_dict.get('n_embd', 256),
        nhead=config_dict.get('n_head', 8),
        num_layers=config_dict.get('n_layer', 8),
        dim_feedforward=config_dict.get('n_inner', 1024),
        dropout=0.0,
        max_seq_len=dataset.max_seq_len,
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

    checkpoint = load_checkpoint(args.weight)
    model = build_model_from_checkpoint(checkpoint, dataset, device)

    input_ids = sample['input_ids'].to(device)
    target_tokens = input_ids.tolist()
    image_features = sample['image_features'].unsqueeze(0).to(device)
    prompt = torch.tensor([[dataset.START_TOKEN]], dtype=torch.long, device=device)
    prompt_mask = torch.ones_like(prompt)

    max_length = min(model.config.n_positions, prompt.shape[1] + args.max_new_tokens)

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
        )

    generated_tokens = generated[0].detach().cpu().tolist()
    prefix_len = longest_common_prefix(target_tokens, generated_tokens)
    first_mismatch = None
    if prefix_len < min(len(target_tokens), len(generated_tokens)):
        first_mismatch = {
            'position': prefix_len,
            'target': target_tokens[prefix_len],
            'generated': generated_tokens[prefix_len],
        }

    result = {
        'split': args.split,
        'sample_index': args.sample_index,
        'cad_stem': sample.get('cad_stem'),
        'target_len': len(target_tokens),
        'generated_len': len(generated_tokens),
        'common_prefix_len': prefix_len,
        'target_prefix_64': target_tokens[:64],
        'generated_prefix_64': generated_tokens[:64],
        'first_mismatch': first_mismatch,
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()
