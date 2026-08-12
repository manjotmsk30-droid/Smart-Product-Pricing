"""
Utility functions for downloading images and setting seeds.
NO EXTERNAL DATA SOURCES - only provided image URLs.
"""

import os
import time
import random
import hashlib
from pathlib import Path
from typing import Optional, List
from io import BytesIO  # CHANGED: use in-memory bytes for PIL open
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError  # CHANGED: import UnidentifiedImageError
from PIL import ImageFile  # CHANGED: tolerate truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True  # CHANGED: allow PIL to load slightly truncated files
import requests
from tqdm import tqdm

# Set seeds for reproducibility
def set_seed(seed: int = 2025):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def download_image(url: str, save_path: str, max_retries: int = 3, 
                   retry_delay: int = 2) -> bool:
    """
    Download a single image from URL with retry logic.

    Args:
        url: Image URL from the provided dataset only
        save_path: Where to save the image
        max_retries: Number of retry attempts if throttled
        retry_delay: Seconds to wait between retries

    Returns:
        Success status
    """
    # CHANGED: ensure target directory exists before writing
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        return True
    
    # CHANGED: single GET per image; broaden retryable statuses; open with BytesIO
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            status = resp.status_code

            # CHANGED: treat 429 and common 5xx as retryable
            if status in (429, 500, 502, 503, 504):
                time.sleep(retry_delay * (attempt + 1))
                continue

            if status != 200 or not resp.content:
                return False

            # CHANGED: use already-fetched bytes; no second GET; no leaked stream
            with Image.open(BytesIO(resp.content)) as img:
                img = img.convert('RGB')
                img.save(save_path, 'JPEG', quality=95, optimize=True)  # CHANGED: optimize flag
            return True

        except UnidentifiedImageError:
            # CHANGED: explicit handling for corrupt/unsupported images
            return False
        except Exception:
            if attempt == max_retries - 1:
                return False
            time.sleep(retry_delay * (attempt + 1))
    
    return False


def download_images_from_csv(csv_path: str, save_dir: str, 
                             url_column: str = 'image_link',
                             id_column: str = 'sample_id',
                             max_images: Optional[int] = None):
    """
    Download all images from a CSV file.
    
    Args:
        csv_path: Path to CSV with image URLs
        save_dir: Directory to save images
        url_column: Name of column containing image URLs
        id_column: Name of column containing sample IDs
        max_images: Limit number of images (for testing)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)

    # CHANGED: verify both required columns up front and exit early if missing
    if not verify_csv_schema(df, [id_column, url_column]):
        return
    
    if max_images:
        df = df.head(max_images)
    
    success_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        sample_id = row[id_column]
        image_url = row[url_column]
        
        if pd.isna(image_url):
            continue
        
        save_path = os.path.join(save_dir, f"{sample_id}.jpg")
        
        if download_image(image_url, save_path):
            success_count += 1
        
        # Rate limiting: small delay between requests
        time.sleep(0.1)
    
    print(f"\nDownloaded {success_count}/{len(df)} images successfully")
    print(f"Saved to: {save_dir}")


def get_file_hash(file_path: str) -> str:
    """Generate hash for a file to verify integrity."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_csv_schema(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Verify CSV has required columns."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        print(f"Error: Missing columns: {missing}")
        return False
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download images from CSV')
    parser.add_argument('--train', type=str, default='data/train.csv')
    parser.add_argument('--test', type=str, default='data/test.csv')
    parser.add_argument('--output', type=str, default='data/images/')
    parser.add_argument('--max-images', type=int, default=None)
    
    args = parser.parse_args()
    
    print("Downloading training images...")
    download_images_from_csv(args.train, args.output, max_images=args.max_images)
    
    print("\nDownloading test images...")
    download_images_from_csv(args.test, args.output, max_images=args.max_images)
