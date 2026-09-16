"""
FloodNet Dataset Sufficiency Analysis
======================================
Checks whether the FloodNet dataset is sufficient for training a DL segmentation model.
Analyzes: image counts, class distribution, flood prevalence, image resolution, 
          class imbalance ratios, and outputs a structured JSON + visualization.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
FLOODNET_ROOT = os.path.join(os.path.dirname(__file__), "..", "FloodNet")
OUTPUT_DIR = os.path.dirname(__file__)

SPLITS = {
    "train": {
        "images": os.path.join(FLOODNET_ROOT, "FloodNet-Supervised_v1.0", "train", "train-org-img"),
        "masks":  os.path.join(FLOODNET_ROOT, "ColorMasks-FloodNetv1", "ColorMasks-TrainSet"),
    },
    "val": {
        "images": os.path.join(FLOODNET_ROOT, "FloodNet-Supervised_v1.0", "val",   "val-org-img"),
        "masks":  os.path.join(FLOODNET_ROOT, "ColorMasks-FloodNetv1", "ColorMasks-ValSet"),
    },
    "test": {
        "images": os.path.join(FLOODNET_ROOT, "FloodNet-Supervised_v1.0", "test",  "test-org-img"),
        "masks":  os.path.join(FLOODNET_ROOT, "ColorMasks-FloodNetv1", "ColorMasks-TestSet"),
    },
}

# FloodNet 10-class colour palette
COLOR_PALETTE = {
    0: ("background",            (0,   0,   0)),
    1: ("building-flooded",      (255, 0,   0)),
    2: ("building-non-flooded",  (180, 120, 120)),
    3: ("road-flooded",          (160, 150, 20)),
    4: ("road-non-flooded",      (140, 140, 140)),
    5: ("water",                 (61,  230, 250)),
    6: ("tree",                  (0,   82,  255)),
    7: ("vehicle",               (255, 0,   245)),
    8: ("pool",                  (255, 235, 0)),
    9: ("grass",                 (4,   250, 7)),
}

FLOODED_CLASS_IDS  = {1, 3, 5}   # truly flooded classes
BUILDING_CLASS_IDS = {1, 2}
ROAD_CLASS_IDS     = {3, 4}

SOLID_COLORS = [
    "#E02424", "#9B1C1C", "#1A56DB", "#1E429F",
    "#057A55", "#03543F", "#C27803", "#8E4B10",
    "#5850EC", "#4B4ACF",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def rgb_mask_to_class_mask(arr: np.ndarray) -> np.ndarray:
    """Convert an H×W×3 RGB array to an H×W class-index array."""
    class_mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.int64)
    for class_id, (_, color) in COLOR_PALETTE.items():
        c = np.array(color, dtype=np.uint8)
        match = np.all(np.abs(arr.astype(int) - c.astype(int)) <= 15, axis=-1)
        class_mask[match] = class_id
    return class_mask


def list_images(directory: str):
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    return sorted([f for f in os.listdir(directory) if f.endswith(exts)])


def list_masks(directory: str):
    return sorted([f for f in os.listdir(directory) if f.endswith(".png")])


# ──────────────────────────────────────────────────────────────────────────────
# Per-split analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyse_split(split_name: str, paths: dict, max_masks: int = 200):
    print(f"\n{'='*60}")
    print(f"  Analysing split: {split_name.upper()}")
    print(f"{'='*60}")

    img_dir  = paths["images"]
    mask_dir = paths["masks"]

    images = list_images(img_dir)
    masks  = list_masks(mask_dir)

    print(f"  Images : {len(images)}")
    print(f"  Masks  : {len(masks)}")

    # ── Image resolution stats ──
    widths, heights = [], []
    for fname in images[:50]:                      # sample for speed
        try:
            img = Image.open(os.path.join(img_dir, fname))
            widths.append(img.width)
            heights.append(img.height)
        except Exception:
            pass

    res_stats = {}
    if widths:
        res_stats = {
            "min_w": int(min(widths)),  "max_w": int(max(widths)),
            "mean_w": float(np.mean(widths)),
            "min_h": int(min(heights)), "max_h": int(max(heights)),
            "mean_h": float(np.mean(heights)),
        }
        print(f"  Resolution: {res_stats['min_w']}×{res_stats['min_h']} – "
              f"{res_stats['max_w']}×{res_stats['max_h']}  "
              f"(mean {res_stats['mean_w']:.0f}×{res_stats['mean_h']:.0f})")

    # ── Class distribution ──
    class_pixel_counts = defaultdict(int)
    total_pixels       = 0
    flood_images       = 0
    building_images    = 0
    road_images        = 0
    analysed           = 0

    sample_masks = masks[:max_masks]
    print(f"  Analysing {len(sample_masks)} masks for class distribution …")

    for fname in sample_masks:
        fpath = os.path.join(mask_dir, fname)
        try:
            arr = np.array(Image.open(fpath).convert("RGB"))
            cm  = rgb_mask_to_class_mask(arr)

            for cid in range(10):
                class_pixel_counts[cid] += int(np.sum(cm == cid))

            total_pixels += cm.size
            if any(np.sum(cm == fid) > 0 for fid in FLOODED_CLASS_IDS):
                flood_images += 1
            if any(np.sum(cm == bid) > 0 for bid in BUILDING_CLASS_IDS):
                building_images += 1
            if any(np.sum(cm == rid) > 0 for rid in ROAD_CLASS_IDS):
                road_images += 1
            analysed += 1
        except Exception as e:
            print(f"    Warning: could not read {fname}: {e}")

    # Class pixel fractions
    class_fractions = {}
    for cid in range(10):
        label = COLOR_PALETTE[cid][0]
        pct   = class_pixel_counts[cid] / total_pixels * 100 if total_pixels > 0 else 0.0
        class_fractions[label] = round(pct, 4)
        print(f"    Class {cid:2d} ({label:24s}): {pct:6.2f}%")

    # Flood / damage prevalence
    flood_pct    = flood_images    / analysed * 100 if analysed > 0 else 0.0
    building_pct = building_images / analysed * 100 if analysed > 0 else 0.0
    road_pct     = road_images     / analysed * 100 if analysed > 0 else 0.0

    print(f"  Images with flooded pixels : {flood_images}/{analysed} ({flood_pct:.1f}%)")
    print(f"  Images with buildings      : {building_images}/{analysed} ({building_pct:.1f}%)")
    print(f"  Images with roads          : {road_images}/{analysed} ({road_pct:.1f}%)")

    # Imbalance ratio (max class / min non-zero class)
    nonzero = [v for v in class_pixel_counts.values() if v > 0]
    imbalance_ratio = max(nonzero) / min(nonzero) if len(nonzero) > 1 else 0.0
    print(f"  Class imbalance ratio      : {imbalance_ratio:.1f}:1")

    return {
        "split": split_name,
        "image_count": len(images),
        "mask_count":  len(masks),
        "masks_analysed": analysed,
        "resolution_stats": res_stats,
        "class_pixel_fractions_pct": class_fractions,
        "flood_image_prevalence_pct": round(flood_pct, 2),
        "building_image_prevalence_pct": round(building_pct, 2),
        "road_image_prevalence_pct": round(road_pct, 2),
        "class_imbalance_ratio": round(imbalance_ratio, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sufficiency verdict
# ──────────────────────────────────────────────────────────────────────────────
def compute_verdict(results: list[dict]) -> dict:
    total_images = sum(r["image_count"] for r in results)
    total_masks  = sum(r["mask_count"]  for r in results)

    # Heuristics for DL segmentation sufficiency:
    # - ≥500 labelled images per class is good; ≥200 is adequate; <50 is poor
    # - Imbalance ratio > 100:1 needs augmentation / class weighting
    # - Flood prevalence < 20% means the rare class needs oversampling

    val_result     = next((r for r in results if r["split"] == "val"),  None)
    train_result   = next((r for r in results if r["split"] == "train"), None)

    flood_pct      = val_result["flood_image_prevalence_pct"] if val_result else 0.0
    imbalance      = val_result["class_imbalance_ratio"]      if val_result else 0.0

    # Score factors
    size_score     = "good"     if total_masks >= 2000 else ("adequate" if total_masks >= 500 else "poor")
    balance_score  = "good"     if imbalance < 50      else ("moderate" if imbalance < 500  else "severe")
    flood_score    = "good"     if flood_pct  > 40     else ("moderate" if flood_pct  > 15  else "low")

    overall = "sufficient"
    if size_score == "poor" or balance_score == "severe":
        overall = "insufficient_without_augmentation"
    elif balance_score == "moderate" or flood_score in ("moderate", "low"):
        overall = "sufficient_with_augmentation_recommended"

    recommendations = []
    if balance_score in ("moderate", "severe"):
        recommendations.append("Apply class-weighted loss (e.g. DiceFocal or weighted CE) to handle imbalance.")
        recommendations.append("Use oversampling for flooded building / road images during training.")
    if flood_score in ("moderate", "low"):
        recommendations.append("Augment with horizontal/vertical flips, colour jitter, random crops to boost flood class diversity.")
    if train_result and train_result["image_count"] < 100:
        recommendations.append("Training split has very few original images — use val as primary training set or apply heavy augmentation.")
    recommendations.append("Pre-train on ADE20K/Cityscapes, then fine-tune on FloodNet to compensate for dataset size.")

    return {
        "total_images": total_images,
        "total_masks":  total_masks,
        "size_adequacy": size_score,
        "class_balance": balance_score,
        "flood_prevalence": flood_score,
        "overall_verdict": overall,
        "recommendations": recommendations,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────
def plot_class_distribution(results: list[dict], output_path: str):
    """Bar chart of class pixel % for each split."""
    labels = [COLOR_PALETTE[i][0] for i in range(10)]
    x      = np.arange(len(labels))
    width  = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="white")

    # ── Left: class distribution per split ──
    ax = axes[0]
    ax.set_facecolor("white")
    for s_idx, r in enumerate(results):
        vals = [r["class_pixel_fractions_pct"].get(COLOR_PALETTE[i][0], 0) for i in range(10)]
        offset = (s_idx - 1) * width
        bars = ax.bar(x + offset, vals, width, label=r["split"].capitalize(),
                      color=SOLID_COLORS[s_idx * 3], edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Pixel Coverage (%)", fontsize=10)
    ax.set_title("Class Pixel Distribution Across Splits", fontsize=12, fontweight="bold", color="#111827")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#D1D5DB")

    # ── Right: flood prevalence + imbalance summary ──
    ax2 = axes[1]
    ax2.set_facecolor("white")
    split_names  = [r["split"].capitalize() for r in results]
    flood_pcts   = [r["flood_image_prevalence_pct"]    for r in results]
    build_pcts   = [r["building_image_prevalence_pct"] for r in results]
    road_pcts    = [r["road_image_prevalence_pct"]     for r in results]

    x2    = np.arange(len(split_names))
    w2    = 0.25
    ax2.bar(x2 - w2, flood_pcts,  w2, label="Flood Prevalence %",    color="#E02424", edgecolor="white")
    ax2.bar(x2,      build_pcts,  w2, label="Building Presence %",   color="#1A56DB", edgecolor="white")
    ax2.bar(x2 + w2, road_pcts,   w2, label="Road Presence %",       color="#057A55", edgecolor="white")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(split_names, fontsize=10)
    ax2.set_ylabel("% of Images", fontsize=10)
    ax2.set_title("Damage Class Prevalence per Split", fontsize=12, fontweight="bold", color="#111827")
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, color="#D1D5DB")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Chart saved → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  FLOODNET DATASET SUFFICIENCY ANALYSIS")
    print("="*60)

    results = []
    for split_name, paths in SPLITS.items():
        r = analyse_split(split_name, paths, max_masks=200)
        results.append(r)

    verdict = compute_verdict(results)

    print("\n" + "="*60)
    print("  SUFFICIENCY VERDICT")
    print("="*60)
    print(f"  Total Images       : {verdict['total_images']}")
    print(f"  Total Masks        : {verdict['total_masks']}")
    print(f"  Size Adequacy      : {verdict['size_adequacy'].upper()}")
    print(f"  Class Balance      : {verdict['class_balance'].upper()}")
    print(f"  Flood Prevalence   : {verdict['flood_prevalence'].upper()}")
    print(f"  Overall Verdict    : {verdict['overall_verdict'].upper()}")
    print("\n  Recommendations:")
    for rec in verdict["recommendations"]:
        print(f"    • {rec}")

    # Save JSON report
    report = {
        "splits": results,
        "verdict": verdict,
    }
    json_path = os.path.join(OUTPUT_DIR, "dataset_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  JSON report saved → {json_path}")

    # Plot
    chart_path = os.path.join(OUTPUT_DIR, "class_distribution.png")
    plot_class_distribution(results, chart_path)

    return report


if __name__ == "__main__":
    main()
