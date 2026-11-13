"""Dataset Readiness, Clinical Label Audit & Research Feasibility Tool.

Comprehensive CLI tool to audit retinal datasets for:
1. Dataset discovery & availability.
2. Metadata schema & patient identifier completeness.
3. Diabetes ground-truth label provenance & evidentiary tiering.
4. Prevention of target leakage (diagnostic lab tests in features).
5. Exact (MD5) and perceptual (dHash) duplicate detection.
6. Class and subgroup distributions (DR 0-4, Diabetes 0/1, DR=0 cohort).
7. Image quality & gradability assessment (Laplacian blur, illumination, contrast).
8. Patient-level independence & leakage detection.
9. Track A (DR Grading) & Track B (Systemic Diabetes) eligibility.
10. Machine-readable JSON/CSV output & human-readable Markdown summary.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_provenance import (
    DiabetesProvenanceTier,
    validate_diabetes_label_provenance,
    assert_no_target_leakage,
    LEAKAGE_BIOMARKER_COLUMNS
)
from src.data.duplicate_detection import compute_md5, compute_dhash, hamming_distance
from src.data.image_quality import assess_image_quality
from src.data.patient_manifest import validate_manifest, REQUIRED_COLUMNS
from src.utils.logging import get_logger

logger = get_logger("audit_datasets")

# Known dataset specifications
KNOWN_DATASETS = {
    "APTOS_2019": {
        "full_name": "APTOS 2019 Blindness Detection",
        "default_dir": "data/raw/aptos_2019",
        "default_csv": "data/raw/aptos_2019/train.csv",
        "source": "Aravind Eye Hospital, India / Kaggle",
        "access_requirement": "Kaggle Account & Competition Rules Acceptance",
        "track_a_eligible": True,
        "track_b_eligible": False,
        "notes": "Diabetic clinic recruitment. Grade 0 != Non-diabetic control. Auxiliary DR representation only."
    },
    "BRSET": {
        "full_name": "Brazilian Retinal Screening Dataset",
        "default_dir": "data/raw/brset",
        "default_csv": "data/raw/brset/labels.csv",
        "source": "PhysioNet (16,266 images, 8,524 patients)",
        "access_requirement": "PhysioNet Credentialed Access (CITI Training, Data Use Agreement)",
        "track_a_eligible": True,
        "track_b_eligible": True,
        "notes": "EHR-confirmed diabetes status. Bilateral OD/OS, ICDR grades. Primary Track B candidate."
    },
    "IDRiD": {
        "full_name": "Indian Diabetic Retinopathy Image Dataset",
        "default_dir": "data/raw/idrid",
        "default_csv": "data/raw/idrid/labels.csv",
        "source": "Eye Clinic, Nanded, Maharashtra, India / IEEE Dataport",
        "access_requirement": "IEEE Dataport Registration / Download",
        "track_a_eligible": True,
        "track_b_eligible": False,
        "notes": "Diabetic cohort. High-resolution Indian clinic images. Auxiliary DR representation only."
    },
    "Messidor_2": {
        "full_name": "Messidor-2 Retinal Image Dataset",
        "default_dir": "data/raw/messidor_2",
        "default_csv": "data/raw/messidor_2/messidor2_data.csv",
        "source": "French Ophthalmology Departments (1,748 images, 874 patients)",
        "access_requirement": "Messidor Research Permission Agreement",
        "track_a_eligible": True,
        "track_b_eligible": False,
        "notes": "External validation benchmark for DR and DME grading."
    }
}


class DatasetAuditor:
    """Core auditing engine evaluating dataset readiness, quality, and clinical provenance."""

    def __init__(self, root_dir: Path | str = PROJECT_ROOT):
        self.root_dir = Path(root_dir)

    def discover_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Discover presence and status of known and custom datasets in workspace."""
        inventory = {}
        for ds_id, spec in KNOWN_DATASETS.items():
            dir_path = self.root_dir / spec["default_dir"]
            csv_path = self.root_dir / spec["default_csv"]

            is_dir_present = dir_path.is_dir()
            is_csv_present = csv_path.is_file()
            img_count = 0
            if is_dir_present:
                img_count = len(list(dir_path.glob("*.jpg"))) + len(list(dir_path.glob("*.png"))) + len(list(dir_path.glob("*.jpeg")))

            status = "AVAILABLE_ON_DISK" if (is_dir_present and img_count > 0) else (
                "METADATA_ONLY" if is_csv_present else "NOT_DOWNLOADED"
            )

            inventory[ds_id] = {
                "full_name": spec["full_name"],
                "status": status,
                "image_dir": str(dir_path),
                "metadata_csv": str(csv_path),
                "images_found": img_count,
                "access_requirement": spec["access_requirement"],
                "track_a_eligible": spec["track_a_eligible"],
                "track_b_eligible": spec["track_b_eligible"],
                "notes": spec["notes"]
            }
        return inventory

    def audit_manifest(
        self,
        manifest_df: pd.DataFrame,
        check_images: bool = False,
        dhash_threshold: int = 5
    ) -> Dict[str, Any]:
        """Perform clinical, statistical, duplicate, and quality audit on a manifest DataFrame."""
        total_records = len(manifest_df)
        audit_results: Dict[str, Any] = {
            "total_records": total_records,
            "patient_metrics": {},
            "provenance_breakdown": {},
            "distributions": {},
            "duplicate_audit": {},
            "quality_audit": {},
            "leakage_risks": [],
            "eligibility": {}
        }

        if total_records == 0:
            audit_results["issues"] = ["Manifest contains zero records."]
            return audit_results

        # 1. Patient Identifiers Completeness
        has_pid = "participant_id" in manifest_df.columns
        if has_pid:
            missing_pid_count = manifest_df["participant_id"].isnull().sum()
            unique_patients = manifest_df["participant_id"].dropna().nunique()
            audit_results["patient_metrics"] = {
                "unique_patients": int(unique_patients),
                "missing_patient_ids": int(missing_pid_count),
                "patient_id_completeness_pct": round((1.0 - missing_pid_count / total_records) * 100, 2),
                "avg_images_per_patient": round(total_records / max(1, unique_patients), 2)
            }
            if missing_pid_count > 0:
                audit_results["leakage_risks"].append(
                    f"{missing_pid_count} records lack explicit patient IDs. Partitioning these without ID linkage risks leakage."
                )
        else:
            audit_results["patient_metrics"] = {
                "unique_patients": 0,
                "missing_patient_ids": total_records,
                "patient_id_completeness_pct": 0.0,
                "avg_images_per_patient": 1.0
            }
            audit_results["leakage_risks"].append(
                "No participant_id column found. True patient-level splitting cannot be guaranteed."
            )

        # 2. Conflicting Patient Labels
        if has_pid and "diabetes_label" in manifest_df.columns:
            valid_p_labels = manifest_df.dropna(subset=["participant_id", "diabetes_label"])
            conflicting_pts = (
                valid_p_labels.groupby("participant_id")["diabetes_label"]
                .nunique()
            )
            conflict_list = conflicting_pts[conflicting_pts > 1].index.tolist()
            audit_results["patient_metrics"]["conflicting_patient_labels_count"] = len(conflict_list)
            audit_results["patient_metrics"]["conflicting_patient_ids"] = conflict_list[:10]
            if conflict_list:
                audit_results["leakage_risks"].append(
                    f"Patient ID conflicts: {len(conflict_list)} patients have contradictory diabetes labels across visits/eyes."
                )

        # 3. Clinical Label Provenance Tiering
        provenance_counts = {tier.value: 0 for tier in DiabetesProvenanceTier}
        valid_track_b_count = 0
        dr0_target_count = 0

        for _, row in manifest_df.iterrows():
            lbl = row.get("diabetes_label")
            src = row.get("diabetes_label_source", "Unknown")
            dr = row.get("dr_grade", None)

            is_valid, tier, _ = validate_diabetes_label_provenance(
                label=lbl,
                provenance_source=src,
                dr_grade=dr,
                allow_self_reported=True
            )
            provenance_counts[tier.value] += 1
            if is_valid:
                valid_track_b_count += 1
                if dr is not None and int(dr) == 0 and int(lbl) == 1:
                    dr0_target_count += 1

        audit_results["provenance_breakdown"] = {
            "tier_counts": provenance_counts,
            "valid_track_b_candidates": valid_track_b_count,
            "dr0_diabetic_targets": dr0_target_count,
            "prohibited_dr_inferred_count": provenance_counts[DiabetesProvenanceTier.INVALID_DR_INFERRED.value],
            "unknown_label_count": provenance_counts[DiabetesProvenanceTier.UNKNOWN.value]
        }

        # 4. Target Leakage Assertion on Feature Columns
        candidate_features = [c for c in manifest_df.columns if c not in ["diabetes_label", "participant_id", "image_id", "image_path"]]
        leaking_features = []
        for feat in candidate_features:
            norm_f = feat.lower().strip().replace(" ", "_").replace("-", "_")
            if any(leak in norm_f for leak in LEAKAGE_BIOMARKER_COLUMNS):
                leaking_features.append(feat)

        audit_results["target_leakage_audit"] = {
            "detected_leaking_columns": leaking_features,
            "has_leakage": len(leaking_features) > 0
        }
        if leaking_features:
            audit_results["leakage_risks"].append(
                f"Target Leakage Detected: Metadata contains ground-truth biomarkers: {leaking_features}. "
                f"These must NOT be passed to any predictive model."
            )

        # 5. Class & Subgroup Distributions
        distributions = {}
        if "dr_grade" in manifest_df.columns:
            distributions["dr_grades"] = {
                str(k): int(v) for k, v in manifest_df["dr_grade"].value_counts(dropna=False).items()
            }
        if "diabetes_label" in manifest_df.columns:
            distributions["diabetes_status"] = {
                str(k): int(v) for k, v in manifest_df["diabetes_label"].value_counts(dropna=False).items()
            }
        if "eye_side" in manifest_df.columns:
            distributions["eye_side"] = {
                str(k): int(v) for k, v in manifest_df["eye_side"].value_counts(dropna=False).items()
            }
        audit_results["distributions"] = distributions

        # 6. Duplicates & Image Integrity Audit (if check_images is True and files exist)
        exact_dups = []
        perceptual_dups = []
        quality_fails = []
        files_checked = 0

        if check_images and "image_path" in manifest_df.columns:
            md5_cache = {}
            dhash_cache = {}

            for _, row in manifest_df.iterrows():
                p_str = row.get("image_path")
                if not p_str:
                    continue
                p = Path(p_str)
                if not p.is_file():
                    continue

                files_checked += 1
                try:
                    # MD5 check
                    md5_val = compute_md5(p)
                    if md5_val in md5_cache:
                        exact_dups.append((str(p), md5_cache[md5_val]))
                    else:
                        md5_cache[md5_val] = str(p)

                    # Perceptual dHash check
                    with Image.open(p) as img:
                        q_res = assess_image_quality(img)
                        if not q_res["gradable"]:
                            quality_fails.append({"path": str(p), "reasons": q_res["failure_reasons"]})

                        dh = compute_dhash(img)
                        for existing_path, existing_dh in dhash_cache.items():
                            dist = hamming_distance(dh, existing_dh)
                            if 0 < dist <= dhash_threshold:
                                perceptual_dups.append((str(p), existing_path, dist))
                        dhash_cache[str(p)] = dh
                except Exception as e:
                    quality_fails.append({"path": str(p), "reasons": [f"Unreadable/Corrupt: {str(e)}"]})

        audit_results["duplicate_audit"] = {
            "files_checked": files_checked,
            "exact_duplicate_pairs": len(exact_dups),
            "perceptual_duplicate_pairs": len(perceptual_dups),
            "sample_exact_dups": exact_dups[:5],
            "sample_perceptual_dups": perceptual_dups[:5]
        }
        audit_results["quality_audit"] = {
            "files_checked": files_checked,
            "ungradable_count": len(quality_fails),
            "ungradable_pct": round((len(quality_fails) / max(1, files_checked)) * 100, 2) if files_checked > 0 else 0.0,
            "sample_failures": quality_fails[:5]
        }

        # 7. Overall Task Eligibility
        has_dr = "dr_grade" in manifest_df.columns and manifest_df["dr_grade"].notnull().any()
        has_valid_diab = valid_track_b_count > 0

        audit_results["eligibility"] = {
            "track_a_dr_grading_eligible": bool(has_dr),
            "track_b_diabetes_screening_eligible": bool(has_valid_diab),
            "supports_dr0_preclinical_analysis": bool(dr0_target_count > 0 and (manifest_df["diabetes_label"] == 0).any())
        }

        return audit_results

    def generate_markdown_summary(
        self,
        inventory: Dict[str, Any],
        manifest_audit: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format discovery and audit results into human-readable Markdown."""
        lines = [
            "# Dataset Readiness, Clinical Label Audit & Research Feasibility Report",
            "",
            "**Project**: Explainable AI for Early Type 2 Diabetes Risk Screening Using Retinal Fundus Images in Rural India  ",
            "**Milestone**: Milestone 1 — Dataset Readiness, Clinical Label Audit & Research Feasibility  ",
            "**Verification Engine**: `scripts/audit_datasets.py`  ",
            "",
            "---",
            "",
            "## 1. Discoverable Datasets Inventory",
            "",
            "| Dataset ID | Full Name | Status on Disk | Images Found | Track A (DR) | Track B (Diabetes) | Access & Governance Protocol |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |"
        ]

        for ds_id, data in inventory.items():
            t_a = "Eligible" if data["track_a_eligible"] else "Ineligible"
            t_b = "Eligible" if data["track_b_eligible"] else "Ineligible"
            lines.append(
                f"| **{ds_id}** | {data['full_name']} | `{data['status']}` | {data['images_found']} | {t_a} | {t_b} | {data['access_requirement']} |"
            )

        lines.extend([
            "",
            "> [!NOTE]",
            "> **Clinical Provenance & Partitioning Notice**:",
            "> - **APTOS 2019 / IDRiD**: All recruited participants are diabetic. Grade 0 eyes are diabetic without retinopathy; they CANNOT serve as non-diabetic controls for Track B.",
            "> - **BRSET (PhysioNet)**: Contains true EHR-confirmed systemic diabetes labels across both diabetic cases and verified non-diabetic controls. Primary candidate for Track B.",
            "",
            "---",
            ""
        ])

        if manifest_audit:
            pm = manifest_audit.get("patient_metrics", {})
            pb = manifest_audit.get("provenance_breakdown", {})
            elig = manifest_audit.get("eligibility", {})
            leaks = manifest_audit.get("leakage_risks", [])

            lines.extend([
                "## 2. Active Manifest Audit Summary",
                "",
                f"- **Total Records Ingested**: {manifest_audit.get('total_records', 0)}",
                f"- **Unique Patient Count**: {pm.get('unique_patients', 'N/A')}",
                f"- **Patient Identifier Completeness**: {pm.get('patient_id_completeness_pct', 0.0)}%",
                f"- **Average Images per Patient**: {pm.get('avg_images_per_patient', 'N/A')}",
                "",
                "### Clinical Label Provenance Breakdown",
                "```json",
                json.dumps(pb.get("tier_counts", {}), indent=2),
                "```",
                "",
                f"- **Valid Track B Screening Candidates**: {pb.get('valid_track_b_candidates', 0)}",
                f"- **Early Research Target (DR = 0 & Diabetic)**: {pb.get('dr0_diabetic_targets', 0)}",
                f"- **Prohibited DR-Inferred Labels**: {pb.get('prohibited_dr_inferred_count', 0)}",
                "",
                "### Research Eligibility Assessment",
                f"- **Track A (5-Class DR Severity Grading)**: {'ELIGIBLE' if elig.get('track_a_dr_grading_eligible') else 'INELIGIBLE'}",
                f"- **Track B (Systemic Diabetes Screening)**: {'ELIGIBLE' if elig.get('track_b_diabetes_screening_eligible') else 'INELIGIBLE'}",
                f"- **Isolated DR = 0 Pre-Clinical Remodeling Analysis**: {'FEASIBLE' if elig.get('supports_dr0_preclinical_analysis') else 'INSUFFICIENT_DATA'}",
                "",
                "### Quality & Leakage Audit Alerts",
            ])

            if leaks:
                for lk in leaks:
                    lines.append(f"> [!WARNING]\n> {lk}\n")
            else:
                lines.append("No patient leakage or target leakage detected in audited manifest.\n")

        return "\n".join(lines)


def create_synthetic_audit_cohort(tmp_dir: Path) -> pd.DataFrame:
    """Create a verified synthetic cohort for audit demonstration and test suites."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    img_dir = tmp_dir / "audit_synthetic_images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # 1. Ten valid patients with both eyes (OD/OS)
    for i in range(10):
        pid = f"PATIENT_{i:03d}"
        is_diab = 1 if i < 5 else 0
        dr_grade = 0 if i in [0, 1, 5, 6, 7] else (1 if i in [2, 8] else 2)
        prov = "Laboratory_HbA1c" if i < 5 else "EHR_Diagnosis"

        for eye in ["OD", "OS"]:
            img_id = f"{pid}_{eye}"
            img_p = img_dir / f"{img_id}.png"
            # Create synthetic fundus image
            img_arr = np.random.randint(60, 200, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(img_arr).save(img_p)

            rows.append({
                "participant_id": pid,
                "image_id": img_id,
                "eye_side": eye,
                "image_path": str(img_p),
                "dataset_id": "SYNTHETIC_VERIFICATION",
                "diabetes_label": is_diab,
                "diabetes_label_source": prov,
                "dr_grade": dr_grade,
                "age": 45 + i,
                "sex": "F" if i % 2 == 0 else "M",
                "bmi": 26.5 + (i * 0.4),
                "systolic_bp": 120 + (i * 2)
            })

    # 2. Add an invalid self-reported and missing record to test provenance
    rows.append({
        "participant_id": "PATIENT_998",
        "image_id": "P998_OD",
        "eye_side": "OD",
        "image_path": str(img_dir / "audit_p998.png"),
        "dataset_id": "SYNTHETIC_VERIFICATION",
        "diabetes_label": 1,
        "diabetes_label_source": "Self_Reported_Survey",
        "dr_grade": 0
    })
    rows.append({
        "participant_id": "PATIENT_999",
        "image_id": "P999_OD",
        "eye_side": "OD",
        "image_path": str(img_dir / "audit_p999.png"),
        "dataset_id": "SYNTHETIC_VERIFICATION",
        "diabetes_label": None,
        "diabetes_label_source": "Unknown",
        "dr_grade": 1
    })
    # Save dummy images for 998 and 999
    Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128).save(img_dir / "audit_p998.png")
    Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128).save(img_dir / "audit_p999.png")

    df = pd.DataFrame(rows)
    return df


def main():
    parser = argparse.ArgumentParser(description="Audit dataset readiness, patient safety, and label provenance.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV to audit.")
    parser.add_argument("--check_images", action="store_true", help="Perform on-disk image integrity and blur audit.")
    parser.add_argument("--synthetic", action="store_true", help="Generate a self-contained synthetic cohort for verification.")
    parser.add_argument("--output_json", type=str, default=None, help="Path to save machine-readable JSON report.")
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save tabular dataset inventory CSV.")
    parser.add_argument("--output_md", type=str, default=None, help="Path to save Markdown audit summary.")
    args = parser.parse_args()

    auditor = DatasetAuditor(PROJECT_ROOT)
    logger.info("Executing Milestone 1 Dataset Readiness & Clinical Audit...")

    # 1. Discover all configured and local datasets
    inventory = auditor.discover_datasets()
    for ds_id, data in inventory.items():
        logger.info(f"Dataset: {ds_id:12s} | Status: {data['status']:20s} | Images: {data['images_found']:5d} | Track B: {data['track_b_eligible']}")

    # 2. Audit manifest if provided or if synthetic mode requested
    manifest_audit = None
    if args.synthetic:
        logger.info("Generating synthetic verification cohort for audit smoke test...")
        synth_dir = PROJECT_ROOT / "data" / "interim" / "audit_smoke_test"
        manifest_df = create_synthetic_audit_cohort(synth_dir)
        manifest_audit = auditor.audit_manifest(manifest_df, check_images=True)
    elif args.manifest:
        m_path = Path(args.manifest)
        if not m_path.is_file():
            logger.error(f"Provided manifest not found: {m_path}")
            sys.exit(1)
        logger.info(f"Auditing manifest at: {m_path}")
        manifest_df = pd.read_csv(m_path)
        manifest_audit = auditor.audit_manifest(manifest_df, check_images=args.check_images)

    # 3. Save Machine-Readable JSON
    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "inventory": inventory,
            "manifest_audit": manifest_audit
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Machine-readable JSON saved to: {out_json}")

    # 4. Save Tabular CSV
    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        inv_df = pd.DataFrame(list(inventory.values()))
        inv_df.to_csv(out_csv, index=False)
        logger.info(f"Dataset inventory CSV saved to: {out_csv}")

    # 5. Generate and Save Markdown Report
    md_report = auditor.generate_markdown_summary(inventory, manifest_audit)
    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(md_report)
        logger.info(f"Markdown audit summary saved to: {out_md}")

    print("\n" + md_report)
    logger.info("Milestone 1 Dataset Audit completed successfully!")


if __name__ == "__main__":
    main()
