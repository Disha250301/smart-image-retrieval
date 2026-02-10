import time
import os
import csv
import numpy as np
from argparse import ArgumentParser

from src.compute import compute_mAP


def save_csv(precisions, recalls, output_path="evaluation_results.csv"):
    import pandas as pd
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame({
        "Precision": precisions,
        "Recall": recalls
    })
    df.to_csv(output_path, index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument("--feature_extractor", required=False, type=str, default='CLIP', help="Feature extractor used (e.g., CLIP, Resnet50)")
    parser.add_argument("--crop", required=False, type=bool, default=False, help="Use cropped query images or not")

    print("Starting Evaluation...")
    start = time.time()

    # Compute metrics
    mAP, all_APs, all_precisions, all_recalls, all_filenames = compute_mAP(parser.parse_args().feature_extractor, parser.parse_args().crop)

    print(f"Mean Average Precision (mAP): {mAP:.4f}")

    avg_precision = np.mean([np.mean(p) for p in all_precisions])
    avg_recall = np.mean([np.mean(r) for r in all_recalls])
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    # Save detailed results
    save_csv(all_precisions, all_recalls)
    print("Evaluation results saved to: evaluation_metrics.csv")

    end = time.time()
    print(f" Finished evaluation in {end - start:.2f} seconds")


if __name__ == '__main__':
    main()
