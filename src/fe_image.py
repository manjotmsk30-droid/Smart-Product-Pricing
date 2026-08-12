"""
Image feature engineering: CLIP/EfficientNet embeddings + hand-crafted features.
NO external labels - only pretrained encoders.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel


class ImageFeatureExtractor:
    """Extract features from product images."""

    def __init__(
        self,
        backbone: str = 'clip',
        image_size: Tuple[int, int] = (224, 224),
        device: str = None,
    ):
        """
        Args:
            backbone: 'clip' or 'efficientnet'
            image_size: Target image size (used only for certain backbones)
            device: Device to use (cuda/cpu)
        """
        self.backbone = backbone.lower()
        self.image_size = image_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.processor = None
        self.transform = None
        self.embedding_dim = 0

        self._load_model()

    def _load_model(self):
        """Load pretrained model with version-safe APIs."""
        if self.backbone == 'clip':
            model_name = "openai/clip-vit-base-patch32"
            # NOTE: This will use local cache if available; otherwise HF will download weights.
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            self.embedding_dim = 512
            print(f"✓ Loaded CLIP ViT-B/32 on {self.device}")

        elif self.backbone == 'efficientnet':
            # Handle both new and old TorchVision APIs.
            try:
                from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                try:
                    # New API (TorchVision >= 0.13)
                    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
                    self.model = efficientnet_b0(weights=weights).to(self.device)
                    self.transform = weights.transforms()
                except Exception:
                    # Fallback to old API
                    self.model = efficientnet_b0(pretrained=True).to(self.device)
                    self.transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ])
                # Remove classifier head -> global pooled features
                self.model.classifier = nn.Identity()
                self.model.eval()
                self.embedding_dim = 1280
                print(f"✓ Loaded EfficientNet-B0 on {self.device}")
            except ImportError as e:
                raise ImportError("torchvision is required for EfficientNet backbone") from e
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract embedding from a single image.

        Args:
            image_path: Path to image file

        Returns:
            Embedding vector (np.ndarray of shape (embedding_dim,))
        """
        # Default safe output
        zero = np.zeros(self.embedding_dim, dtype=np.float32)

        try:
            # Robust read via PIL for CLIP; OpenCV is fine for EfficientNet too,
            # but PIL keeps consistent RGB ordering.
            if not image_path or not os.path.exists(image_path):
                return zero

            image = Image.open(image_path).convert('RGB')

            if self.backbone == 'clip':
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    emb = image_features.cpu().numpy().astype(np.float32).flatten()
                # Return raw features (no normalization) to keep compatibility
                return emb if emb.size == self.embedding_dim else zero

            elif self.backbone == 'efficientnet':
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.model(image_tensor).cpu().numpy().astype(np.float32).flatten()
                return emb if emb.size == self.embedding_dim else zero

            return zero
        except Exception:
            # Never raise: return a valid zero vector so the pipeline keeps running
            return zero

    def extract_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings for multiple images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing (kept for API compatibility)

        Returns:
            Array of shape (n_images, embedding_dim)
        """
        # Keep simple per-image loop for robustness; preserves ordering.
        embeddings = [self.extract_embedding(p) for p in image_paths]
        return np.vstack(embeddings) if embeddings else np.zeros((0, self.embedding_dim), dtype=np.float32)


class HandCraftedImageFeatures:
    """Extract hand-crafted visual features from images."""

    @staticmethod
    def extract_from_path(image_path: str) -> dict:
        """
        Extract hand-crafted features from an image file.

        Args:
            image_path: Path to image

        Returns:
            Dictionary of features
        """
        features = {
            'img_brightness': 0.0,
            'img_contrast': 0.0,
            'img_entropy': 0.0,
            'edge_density': 0.0,
            'bg_white_ratio': 0.0,
            'dominant_hue_sin': 0.0,
            'dominant_hue_cos': 0.0,
            'aspect_ratio': 1.0,
        }

        try:
            if not image_path or not os.path.exists(image_path):
                return features

            img = cv2.imread(image_path)
            if img is None:
                return features

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Brightness (mean of grayscale) in [0,1]
            features['img_brightness'] = float(np.mean(gray) / 255.0)

            # Contrast (std of grayscale) in [0,1]
            features['img_contrast'] = float(np.std(gray) / 255.0)

            # Entropy with zero-sum guard
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).astype(np.float64)
            total = hist.sum()
            if total > 0:
                hist /= total
                nz = hist[hist > 0]
                features['img_entropy'] = float(-np.sum(nz * np.log2(nz)))
            else:
                features['img_entropy'] = 0.0

            # Edge density (Canny)
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = float(np.count_nonzero(edges) / edges.size)

            # White background ratio
            white_mask = (img >= 240).all(axis=2)
            features['bg_white_ratio'] = float(np.count_nonzero(white_mask) / white_mask.size)

            # Dominant hue (OpenCV hue range 0..179); use median as robust proxy
            hue_channel = hsv[:, :, 0].astype(np.float32)
            if hue_channel.size > 0:
                dominant_hue = float(np.median(hue_channel))
                angle = 2.0 * np.pi * (dominant_hue / 180.0)
                features['dominant_hue_sin'] = float(np.sin(angle))
                features['dominant_hue_cos'] = float(np.cos(angle))

            # Aspect ratio
            h, w = img.shape[:2]
            features['aspect_ratio'] = float(w / (h + 1e-6))

        except Exception:
            # Swallow and return the defaults so upstream never breaks
            return features

        return features

    @staticmethod
    def extract_batch(image_paths: List[str]) -> pd.DataFrame:
        """
        Extract features for multiple images.

        Args:
            image_paths: List of image paths

        Returns:
            DataFrame with features
        """
        features_list = [HandCraftedImageFeatures.extract_from_path(p) for p in image_paths]
        return pd.DataFrame(features_list)


def build_image_features(
    sample_ids: pd.Series,
    images_dir: str,
    extractor: Optional[ImageFeatureExtractor] = None,
    extract_embeddings: bool = True,
) -> Tuple[Optional[np.ndarray], pd.DataFrame]:
    """
    Build complete image feature set.

    Args:
        sample_ids: Series of sample IDs
        images_dir: Directory containing images
        extractor: Pre-initialized ImageFeatureExtractor
        extract_embeddings: Whether to extract deep embeddings

    Returns:
        Tuple of (embeddings, handcrafted_features)
    """
    image_paths = [os.path.join(images_dir, f"{sid}.jpg") for sid in sample_ids]
    valid_paths = [p if os.path.exists(p) else "" for p in image_paths]

    # Deep embeddings
    embeddings: Optional[np.ndarray]
    if extract_embeddings:
        if extractor is None:
            extractor = ImageFeatureExtractor()
        embeddings = extractor.extract_batch(valid_paths)
    else:
        embeddings = None  # keep API flexible if you want only handcrafted

    # Hand-crafted features
    handcrafted = HandCraftedImageFeatures.extract_batch(valid_paths)
    handcrafted.index = sample_ids.values

    return embeddings, handcrafted


if __name__ == "__main__":
    # Lightweight self-test (no external downloads required).
    print("Testing image feature extraction...")

    # Create a dummy image for testing
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    test_path = "/tmp/test_img.jpg"
    Image.fromarray(test_img).save(test_path)

    # Test hand-crafted features
    features = HandCraftedImageFeatures.extract_from_path(test_path)
    print(f"\nHand-crafted keys: {list(features.keys())}")
    print(f"Brightness: {features['img_brightness']:.3f}")

    # Test deep features (skip if CLIP/EfficientNet not available)
    try:
        extractor = ImageFeatureExtractor(backbone='clip')
        embedding = extractor.extract_embedding(test_path)
        print(f"\nCLIP embedding shape: {embedding.shape}")
    except Exception as e:
        print(f"\nSkipping deep feature test: {e}")

    if os.path.exists(test_path):
        os.remove(test_path)
