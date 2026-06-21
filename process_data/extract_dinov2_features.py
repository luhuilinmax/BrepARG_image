import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_render_index(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'renders' in data:
        return data['renders']
    raise ValueError(f'Unsupported render index format: {path}')


def build_model(device):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.eval()
    model.to(device)
    return model


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def extract_patch_tokens(model, image_tensor):
    with torch.no_grad():
        features = model.forward_features(image_tensor)
    if 'x_norm_patchtokens' not in features:
        raise KeyError('DINOv2 forward_features output missing x_norm_patchtokens')
    return features['x_norm_patchtokens'][0].cpu()


def main():
    parser = argparse.ArgumentParser(description='Extract DINOv2 ViT-L patch-token features for rendered images.')
    parser.add_argument('--render_index', type=str, required=True, help='Merged render_index.pkl')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save per-sample .pt features')
    parser.add_argument('--index_output', type=str, required=True, help='Output pickle: cad_stem -> {image_path, feature_path}')
    parser.add_argument('--image_size', type=int, default=518, help='Resize/crop size for DINOv2')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    args = parser.parse_args()

    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = build_model(device)
    transform = build_transform(args.image_size)
    render_index = load_render_index(args.render_index)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_index = {}

    for stem, info in tqdm(render_index.items(), desc='Extracting DINOv2 features'):
        image_path = Path(info['image_path'])
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        patch_tokens = extract_patch_tokens(model, image_tensor)

        feature_path = output_dir / f'{stem}.pt'
        torch.save({'patch_tokens': patch_tokens}, feature_path)
        feature_index[stem] = {
            'image_path': str(image_path),
            'feature_path': str(feature_path),
        }

    index_output = Path(args.index_output)
    index_output.parent.mkdir(parents=True, exist_ok=True)
    with open(index_output, 'wb') as f:
        pickle.dump(feature_index, f)

    print(f'Saved features: {len(feature_index)}')
    print(f'Saved index: {index_output}')


if __name__ == '__main__':
    main()
