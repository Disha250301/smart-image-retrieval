import os
import time
import pathlib
import numpy as np
from PIL import Image
from argparse import ArgumentParser

import torch

from src.feature_extraction import MyResnet50, MyVGG16
from src.clip_feature_extractor import CLIPFeatureExtractor
from src.dataloader import get_transformation
from src.hypergraph_retrieval_engine import HypergraphRetrievalEngine

# Constants
ACCEPTED_IMAGE_EXTS = ['.jpg', '.png']
query_root = './dataset/groundtruth'
image_root = './dataset/paris'
feature_root = './dataset/features'
evaluate_root = './dataset/evaluation'

def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = [p.name for p in image_root.iterdir() if p.suffix in ACCEPTED_IMAGE_EXTS]
    return sorted(image_list)

def main():
    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=True, type=str, choices=["CLIP", "Resnet50", "VGG16"])
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--top_k", type=int, default=11)
    parser.add_argument("--crop", type=bool, default=False)

    print(f'📊 Ranking with Hypergraph-Based Manifold Ranking...')
    start = time.time()

    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"🧠 Using {args.feature_extractor} on {args.device}")

    # Load the appropriate extractor
    if args.feature_extractor == 'CLIP':
        extractor = CLIPFeatureExtractor(device=device)
        transform = extractor.clip_preprocess
    elif args.feature_extractor == 'Resnet50':
        extractor = MyResnet50(device)
        transform = extractor.transform
    elif args.feature_extractor == 'VGG16':
        extractor = MyVGG16(device)
        transform = extractor.transform
    else:
        raise ValueError("Invalid feature extractor")

    # Load dataset features
    feature_path = os.path.join(feature_root, f"{args.feature_extractor}_features.npy")
    filename_path = os.path.join(feature_root, f"{args.feature_extractor}_filenames.npy")
    if not os.path.exists(feature_path):
        print(f"❌ Feature file not found: {feature_path}")
        return

    features = np.load(feature_path, allow_pickle=True)
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)

    image_list = np.load(filename_path)
    image_paths = [str(pathlib.Path(image_root) / name) for name in image_list]

    # Build hypergraph
    engine = HypergraphRetrievalEngine(k_neighbors=10)
    engine.build_hypergraph(features, image_paths)

    # Process query images
    for path_file in os.listdir(query_root):
        if path_file.endswith("query.txt"):
            with open(os.path.join(query_root, path_file), "r") as file:
                img_query, left, top, right, bottom = file.read().split()

            test_image_path = pathlib.Path(image_root) / f"{img_query}.jpg"
            if not test_image_path.exists():
                print(f"⚠️ Skipping missing image: {test_image_path}")
                continue

            pil_image = Image.open(test_image_path).convert('RGB')
            path_crop = 'original'
            if args.crop:
                pil_image = pil_image.crop((float(left), float(top), float(right), float(bottom)))
                path_crop = 'crop'

            # Preprocess and extract query features
            image_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                if args.feature_extractor == 'CLIP':
                    query_feat = extractor.get_features(image_tensor)
                else:
                    query_feat = extractor(image_tensor).cpu().numpy()

            # Retrieve top-k
            ranked = engine.retrieve_top_k(query_feat, k=args.top_k)
            rank_list = [os.path.basename(path)[:-4] for path, _ in ranked]

            # Save rankings
            output_dir = os.path.join(evaluate_root, path_crop, args.feature_extractor)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{path_file[:-10]}.txt")

            with open(output_file, "w") as f:
                f.write("\n".join(rank_list))

    end = time.time()
    print(f'✅ Finished ranking in {round(end - start, 2)} seconds')

if __name__ == '__main__':
    main()
