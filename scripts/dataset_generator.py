import os
import argparse
from pathlib import Path

def setup_dataset_structure(base_path: str):
    """
    Creates the standardized dataset directory structure required for training 
    and evaluation of AdaptiveVision-LPR modules.
    """
    directories = [
        "raw",
        "videos",
        "extracted_frames",
        "annotations",
        "conditions",
        "occlusion_masks",
        "plate_crops",
        "train",
        "validation",
        "test"
    ]
    
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for d in directories:
        dir_path = base_dir / d
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create a basic .gitkeep to ensure empty directories are tracked if needed
        (dir_path / ".gitkeep").touch()
        print(f"Created: {dir_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize AdaptiveVision Dataset Pipeline")
    parser.add_argument("--path", type=str, default="./data", help="Base path for the dataset")
    args = parser.parse_args()
    
    print(f"Initializing Dataset Pipeline Structure at: {args.path}")
    setup_dataset_structure(args.path)
