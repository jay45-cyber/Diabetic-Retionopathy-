"""PyTorch Dataset for Track B: Binary Systemic Diabetes Risk Screening.

Supports:
1. Strict clinical label provenance filtering (rejects DR-inferred labels).
2. Retinal image circular cropping and CLAHE enhancement.
3. Isolated DR=0 research cohort filtering.
4. Optional multimodal non-leaking clinical covariates (age, sex, BMI, systolic BP).
5. Mathematical target-leakage assertion on all tabular features.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.preprocessing import crop_fundus_circle
from src.data.dr_dataset import apply_clahe, IMAGENET_MEAN, IMAGENET_STD
from src.data.label_provenance import (
    validate_diabetes_label_provenance,
    assert_no_target_leakage,
    DiabetesProvenanceTier
)


class DiabetesScreeningDataset(Dataset):
    """PyTorch Dataset for Type 2 Diabetes Risk Screening from Retinal Images."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Optional[Union[str, Path]] = None,
        image_size: Tuple[int, int] = (224, 224),
        is_training: bool = False,
        use_clahe: bool = False,
        crop_circle: bool = True,
        allow_self_reported: bool = False,
        dr0_only: bool = False,
        clinical_feature_cols: Optional[List[str]] = None,
        clinical_scaler_stats: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_size = image_size
        self.is_training = is_training
        self.use_clahe = use_clahe
        self.crop_circle = crop_circle
        self.clinical_cols = clinical_feature_cols or []

        # Target leakage safeguard
        if self.clinical_cols:
            assert_no_target_leakage(self.clinical_cols)

        # Filter valid records strictly based on label provenance
        valid_records = []
        for idx, row in df.iterrows():
            lbl = row.get("diabetes_label")
            prov = str(row.get("diabetes_label_source", "Unknown"))
            dr_grade = row.get("dr_grade", None)

            is_valid, tier, msg = validate_diabetes_label_provenance(
                label=lbl,
                provenance_source=prov,
                dr_grade=dr_grade,
                allow_self_reported=allow_self_reported
            )

            if not is_valid:
                continue

            if dr0_only and dr_grade is not None:
                try:
                    if int(dr_grade) != 0:
                        continue
                except (ValueError, TypeError):
                    continue

            r_dict = row.to_dict()
            r_dict["_clean_label"] = float(lbl)
            r_dict["_provenance_tier"] = tier.value
            valid_records.append(r_dict)

        self.df = pd.DataFrame(valid_records).reset_index(drop=True)

        # Training vs Validation Transforms
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

        # Precompute clinical feature normalization stats (mean, std)
        self.scaler_stats = clinical_scaler_stats or {}
        if self.clinical_cols and not self.scaler_stats and len(self.df) > 0:
            for col in self.clinical_cols:
                vals = pd.to_numeric(self.df[col], errors="coerce")
                mean_v = float(vals.mean()) if not np.isnan(vals.mean()) else 0.0
                std_v = float(vals.std()) if not np.isnan(vals.std()) and vals.std() > 0 else 1.0
                self.scaler_stats[col] = (mean_v, std_v)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        """Find image by path column or by id_code in image_dir."""
        if "image_path" in row and pd.notnull(row["image_path"]):
            p = Path(row["image_path"])
            if p.is_file():
                return p

        img_id = str(row.get("image_id", row.get("participant_id", row.get("id_code", ""))))
        if self.image_dir:
            for ext in [".png", ".jpg", ".jpeg", ""]:
                candidate = self.image_dir / f"{img_id}{ext}"
                if candidate.is_file():
                    return candidate

        raise FileNotFoundError(f"Fundus image not found for id '{img_id}' (searched in {self.image_dir})")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row)

        # 1. Load image
        img = Image.open(img_path).convert("RGB")

        # 2. Circular ROI crop
        if self.crop_circle:
            try:
                img = crop_fundus_circle(img)
            except Exception:
                pass

        # 3. CLAHE enhancement
        if self.use_clahe:
            img = apply_clahe(img)

        # 4. Transform to tensor
        tensor_image = self.transform(img)

        # 5. Ground truth binary label
        target = torch.tensor(row["_clean_label"], dtype=torch.float32)

        # 6. Tabular clinical covariates if requested
        if self.clinical_cols:
            clin_vals = []
            for col in self.clinical_cols:
                raw_v = row.get(col, None)
                try:
                    num_v = float(raw_v) if pd.notnull(raw_v) else self.scaler_stats[col][0]
                except (ValueError, TypeError):
                    num_v = self.scaler_stats[col][0]
                mean_v, std_v = self.scaler_stats[col]
                norm_v = (num_v - mean_v) / (std_v + 1e-7)
                clin_vals.append(norm_v)

            clin_tensor = torch.tensor(clin_vals, dtype=torch.float32)
            return tensor_image, clin_tensor, target

        return tensor_image, target
