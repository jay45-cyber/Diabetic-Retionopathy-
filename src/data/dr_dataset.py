"""Reproducible PyTorch Dataset and Stratified Splitting for 5-Class DR Severity Grading."""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.preprocessing import crop_fundus_circle
from src.data.duplicate_detection import compute_dhash, hamming_distance

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def apply_clahe(image: Image.Image, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> Image.Image:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the luminance channel."""
    try:
        import cv2
        img_np = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_rgb)
    except ImportError:
        # Graceful fallback using PIL ImageOps autocontrast if cv2 is absent
        return ImageOps.autocontrast(image)


class DRGradingDataset(Dataset):
    """PyTorch Dataset for 5-Class Diabetic Retinopathy Severity Grading (APTOS 2019 / Clinical Fundus).
    
    Classes:
      0: No DR
      1: Mild DR
      2: Moderate DR
      3: Severe DR
      4: Proliferative DR
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = False,
        use_clahe: bool = False,
        crop_circle: bool = True,
        id_col: str = "id_code",
        label_col: str = "diagnosis"
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.is_training = is_training
        self.use_clahe = use_clahe
        self.crop_circle = crop_circle
        self.id_col = id_col
        self.label_col = label_col

        # Training-only data augmentations simulating clinical variation
        if self.is_training:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

    def __len__(self) -> int:
        return len(self.df)

    def _find_image_path(self, img_id: str) -> Path:
        """Find image supporting .png, .jpg, or .jpeg extensions."""
        for ext in [".png", ".jpg", ".jpeg", ""]:
            p = self.image_dir / f"{img_id}{ext}"
            if p.is_file():
                return p
        raise FileNotFoundError(f"Fundus image not found for id '{img_id}' in {self.image_dir}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_id = str(row[self.id_col])
        img_path = self._find_image_path(img_id)

        # 1. Load image in RGB
        image = Image.open(img_path).convert("RGB")

        # 2. Crop black borders
        if self.crop_circle:
            image = crop_fundus_circle(image)

        # 3. Optional CLAHE enhancement
        if self.use_clahe:
            image = apply_clahe(image)

        # 4. Standardized Transforms
        tensor_image = self.transform(image)

        # 5. Severity Label (0 to 4)
        label = int(row[self.label_col])
        assert 0 <= label <= 4, f"Invalid DR label {label}; expected 0 to 4."

        return tensor_image, torch.tensor(label, dtype=torch.long)


def compute_inverse_class_weights(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, List[int]],
    label_col: str = "diagnosis",
    num_classes: int = 5
) -> torch.Tensor:
    """Compute normalized inverse-frequency class weights for Weighted Cross-Entropy Loss."""
    if isinstance(data, pd.DataFrame):
        labels = data[label_col].values
    elif isinstance(data, pd.Series):
        labels = data.values
    else:
        labels = np.asarray(data)

    unique, counts = np.unique(labels, return_counts=True)
    count_dict = dict(zip(unique, counts))
    total_samples = len(labels)
    weights = []
    for c in range(num_classes):
        class_count = count_dict.get(c, 1)
        w = total_samples / (num_classes * class_count)
        weights.append(w)
    
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize weights so they sum to num_classes
    weights_tensor = weights_tensor * (num_classes / weights_tensor.sum())
    return weights_tensor


def create_stratified_dr_splits(
    df: Optional[pd.DataFrame] = None,
    csv_path: Optional[Union[str, Path]] = None,
    label_col: str = "diagnosis",
    id_col: str = "id_code",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    random_state: int = 42,
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, pd.DataFrame]:
    """Create a reproducible 70% / 15% / 15% stratified split for 5-class DR grading."""
    if df is None:
        if csv_path is None:
            raise ValueError("Either 'df' or 'csv_path' must be provided.")
        df = pd.read_csv(csv_path)

    t_size = train_ratio if train_ratio is not None else train_size
    v_size = val_ratio if val_ratio is not None else val_size
    te_size = test_ratio if test_ratio is not None else test_size

    assert abs(t_size + v_size + te_size - 1.0) < 1e-4, "Splits must sum to 1.0"

    # Step 1: Split into Train (70%) and Temp (30%)
    temp_size = v_size + te_size
    try:
        split1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_size, random_state=random_state)
        train_idx, temp_idx = next(split1.split(df, df[label_col]))
    except ValueError:
        from sklearn.model_selection import ShuffleSplit
        split1_fallback = ShuffleSplit(n_splits=1, test_size=temp_size, random_state=random_state)
        train_idx, temp_idx = next(split1_fallback.split(df))
    
    train_df = df.iloc[train_idx].copy().reset_index(drop=True)
    temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

    # Step 2: Split Temp into Val (15%) and Test (15%)
    val_fraction = v_size / temp_size
    try:
        split2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_fraction), random_state=random_state)
        val_idx, test_idx = next(split2.split(temp_df, temp_df[label_col]))
    except ValueError:
        from sklearn.model_selection import ShuffleSplit
        split2_fallback = ShuffleSplit(n_splits=1, test_size=(1.0 - val_fraction), random_state=random_state)
        val_idx, test_idx = next(split2_fallback.split(temp_df))

    val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)

    # Save to disk if output_dir specified
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_path / "dr_train.csv", index=False)
        val_df.to_csv(out_path / "dr_val.csv", index=False)
        test_df.to_csv(out_path / "dr_test.csv", index=False)
        # Also save aliases for backwards compatibility
        train_df.to_csv(out_path / "train_split.csv", index=False)
        val_df.to_csv(out_path / "val_split.csv", index=False)
        test_df.to_csv(out_path / "test_split.csv", index=False)

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }
