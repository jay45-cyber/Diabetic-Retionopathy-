"""CLI Script for Batch Retinal Explainability Generation.

Generates and saves visual explanations:
- ResNet-50: Grad-CAM
- ViT-Base/16: ViT Token Grad-CAM & Attention Rollout
- Swin Transformer: Swin Grad-CAM
- All Models: Integrated Gradients

Output Path:
outputs/explanations/{model_name}/{method_name}/{image_id}.png
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from src.models.dr_models import build_dr_model
from src.explainability.dr_explainer import DRModelExplainer
from src.utils.logging import get_logger

logger = get_logger("generate_explanations")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main():
    parser = argparse.ArgumentParser(description="Generate Visual Explanations for DR Predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--model_name", type=str, default="vit_base_patch16_224", help="Model architecture")
    parser.add_argument("--image_dir", type=str, default="data/raw/aptos2019/train_images", help="Path to images directory")
    parser.add_argument("--csv_path", type=str, default="data/processed/splits/dr_test.csv", help="Test metadata CSV")
    parser.add_argument("--output_dir", type=str, default="outputs/explanations", help="Output directory")
    parser.add_argument("--methods", nargs="+", default=None, help="Explainability methods (e.g. gradcam, rollout, ig)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to explain")
    parser.add_argument("--device", type=str, default="auto", help="Device ('cuda', 'cpu', 'auto')")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    logger.info(f"Target Device: {device} | Architecture: {args.model_name}")

    # Load Model
    model = build_dr_model(model_name=args.model_name, num_classes=5, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    explainer = DRModelExplainer(model, model_architecture=args.model_name)

    # Determine methods to run
    if args.methods:
        methods = args.methods
    else:
        if "vit" in args.model_name.lower():
            methods = ["vit_token_gradcam", "rollout", "ig"]
        elif "resnet" in args.model_name.lower():
            methods = ["gradcam", "ig"]
        elif "swin" in args.model_name.lower():
            methods = ["swin_gradcam", "ig"]
        else:
            methods = ["ig"]

    # Locate images
    csv_file = Path(args.csv_path)
    img_dir = Path(args.image_dir)
    image_list = []

    if csv_file.is_file():
        df = pd.read_csv(csv_file)
        for _, row in df.head(args.num_samples).iterrows():
            id_code = str(row["id_code"])
            found = False
            for ext in [".png", ".jpg", ".jpeg", ""]:
                p = img_dir / f"{id_code}{ext}"
                if p.is_file():
                    image_list.append((id_code, p))
                    found = True
                    break
    elif img_dir.is_dir():
        files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        for p in files[:args.num_samples]:
            image_list.append((p.stem, p))

    if not image_list:
        logger.warning("No retinal images found to explain. Generating a synthetic retinal proxy image...")
        synth_dir = Path(args.output_dir) / "synthetic_inputs"
        synth_dir.mkdir(parents=True, exist_ok=True)
        synth_p = synth_dir / "demo_retina_001.png"
        dummy = np.random.randint(40, 220, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(dummy).save(synth_p)
        image_list.append(("demo_retina_001", synth_p))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    logger.info(f"Generating visual explanations for {len(image_list)} images with methods: {methods}")

    for img_id, img_path in image_list:
        orig_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(orig_img).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            logits = model(input_tensor)
            pred_class = int(torch.argmax(logits, dim=-1).item())

        for method in methods:
            out_file = explainer.save_explanation(
                original_img=orig_img,
                input_tensor=input_tensor,
                image_id=img_id,
                output_dir=args.output_dir,
                method=method,
                target_class=pred_class
            )
            logger.info(f"Saved [{method}] explanation -> {out_file}")

    logger.info("Batch explainability complete!")


if __name__ == "__main__":
    main()
