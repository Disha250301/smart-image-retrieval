import os
import argparse
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from src.feature_extraction import MyResnet50, MyVGG16
from src.clip_feature_extractor import CLIPFeatureExtractor

# =========  Dataset Loader with Safety ==============
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, os.path.basename(path)
        except Exception as e:
            print(f" Skipping corrupted image: {path} ({str(e)})")
            return None

def collate_skip_none(batch):
    return [b for b in batch if b is not None]

# ========== Feature Extraction Logic ================
def extract_features(model, dataloader, extractor_name, device):
    if hasattr(model, 'eval'):
        model.eval()

    features = []
    filenames = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f" Extracting with {extractor_name}"):
            if not batch:
                continue
            batch_imgs, batch_names = zip(*batch)
            batch_imgs = torch.stack(batch_imgs).to(device)

            if extractor_name == "CLIP":
                output = model.get_features(batch_imgs)
            else:
                output = model.extract_features(batch_imgs)

            features.append(output)
            filenames.extend(batch_names)

    features = np.concatenate(features, axis=0)
    return features, filenames

# ==========  Main Routine =============================
def main(args):
    dataset_dir = os.path.join("dataset", "paris")
    output_dir = "dataset/features"
    os.makedirs(output_dir, exist_ok=True)

    features_path = os.path.join(output_dir, f"{args.feature_extractor}_features.npy")
    filenames_path = os.path.join(output_dir, f"{args.feature_extractor}_filenames.npy")

    if os.path.exists(features_path) and os.path.exists(filenames_path):
        print(" Features already extracted. Skipping..")
        return

    device = torch.device(args.device)

    # === Choose Model + Preprocessing ===
    if args.feature_extractor == "Resnet50":
        model = MyResnet50(device)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif args.feature_extractor == "VGG16":
        model = MyVGG16(device)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    elif args.feature_extractor == "CLIP":
        model = CLIPFeatureExtractor(device=device)
        transform = model.clip_preprocess
    else:
        raise ValueError(" Unknown feature extractor")

    # ===  Load Images Safely ===
    image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Optional pre-check for corrupted files
    def is_valid_image(path):
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except:
            return False

    image_paths = [p for p in image_paths if is_valid_image(p)]
    print(f"Found {len(image_paths)} valid images")

    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_skip_none)

    # === Extract Features ===
    features, filenames = extract_features(model, dataloader, args.feature_extractor, device)

    # === Save ===
    np.save(features_path, features)
    np.save(filenames_path, np.array(filenames))
    print(f"Saved features to: {features_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_extractor", type=str, required=True, choices=["Resnet50", "VGG16", "CLIP"])
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()
    main(args)
    
