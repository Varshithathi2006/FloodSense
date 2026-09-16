"""
FloodNet Damage Information Extractor
========================================
Uses FloodNet colour-mask segmentation to extract detailed damage
information from each image:

  • Flooded building count & area
  • Non-flooded building count & area
  • Flooded road coverage (%)
  • Non-flooded road coverage (%)
  • Water body area (%)
  • Vehicle count estimate
  • Pool / standing-water area (%)
  • Damage severity score (weighted formula)
  • Flood coverage % of total image
  • Per-class pixel distribution

Output per image: structured JSON  
Batch output  : damage_extraction_report.json + damage_summary.png
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
FLOODNET_ROOT = os.path.join(os.path.dirname(__file__), "..", "FloodNet")
OUTPUT_DIR    = os.path.dirname(__file__)

# ──────────────────────────────────────────────────────────────────────────────
# Class definitions
# ──────────────────────────────────────────────────────────────────────────────
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

# Solid display colours per class (for visualisation)
CLASS_DISPLAY_COLORS = {
    0: (55,  65,  81),    # dark gray
    1: (220, 38,  38),    # red     (building flooded)
    2: (180, 120, 120),   # pink    (building non-flooded)
    3: (202, 138, 4),     # amber   (road flooded)
    4: (107, 114, 128),   # gray    (road non-flooded)
    5: (6,   182, 212),   # cyan    (water)
    6: (5,   150, 105),   # green   (tree)
    7: (124, 58,  237),   # purple  (vehicle)
    8: (234, 179, 8),     # yellow  (pool)
    9: (74,  222, 128),   # light green (grass)
}

# Damage severity weights per class (higher = more severe)
SEVERITY_WEIGHTS = {
    1: 10.0,   # building-flooded  — critical
    3: 8.0,    # road-flooded      — severe
    5: 5.0,    # water             — moderate (extent indicator)
    8: 3.0,    # pool              — minor
    2: 1.0,    # building-non-flooded — asset exposure
    4: 0.5,    # road-non-flooded
    7: 2.0,    # vehicle
    6: 0.0,
    9: 0.0,
    0: 0.0,
}

# Minimum connected-component area (pixels) to count as a building/vehicle
MIN_BUILDING_PIXELS = 500
MIN_VEHICLE_PIXELS  = 50

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def rgb_mask_to_class_mask(arr: np.ndarray) -> np.ndarray:
    class_mask = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.int64)
    for class_id, (_, color) in COLOR_PALETTE.items():
        c     = np.array(color, dtype=np.uint8)
        match = np.all(np.abs(arr.astype(int) - c.astype(int)) <= 15, axis=-1)
        class_mask[match] = class_id
    return class_mask


def count_components(binary_mask: np.ndarray, min_pixels: int = 100) -> tuple[int, list[int]]:
    """Return (count, list_of_component_sizes) filtering by min_pixels."""
    labeled, num = ndimage.label(binary_mask)
    sizes = []
    for region_id in range(1, num + 1):
        sz = int(np.sum(labeled == region_id))
        if sz >= min_pixels:
            sizes.append(sz)
    return len(sizes), sorted(sizes, reverse=True)


def pixel_area_m2(pixel_count: int, image_shape: tuple,
                  uav_altitude_m: float = 30.0, fov_deg: float = 73.0) -> float:
    """
    Rough ground area estimate for pixel_count pixels.
    Assumes nadir-pointing UAV camera at uav_altitude_m with horizontal FoV fov_deg.
    GSD ≈ 2 * altitude * tan(fov/2) / image_width
    """
    h, w = image_shape[:2]
    fov_rad = np.deg2rad(fov_deg / 2)
    gsd_m   = 2 * uav_altitude_m * np.tan(fov_rad) / w   # meters per pixel
    return round(pixel_count * gsd_m ** 2, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Per-image damage extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_damage_info(mask_path: str, image_path: str = None) -> dict:
    """
    Extracts full damage information from a FloodNet colour mask.
    Optionally loads the corresponding original image for metadata.
    """
    mask_img = Image.open(mask_path).convert("RGB")
    mask_arr = np.array(mask_img)
    class_mask = rgb_mask_to_class_mask(mask_arr)

    total_pixels = class_mask.size
    h, w = class_mask.shape

    # ── Per-class pixel counts ──
    class_pixel_counts = {}
    class_pixel_pct    = {}
    for cid, (label, _) in COLOR_PALETTE.items():
        cnt = int(np.sum(class_mask == cid))
        class_pixel_counts[label] = cnt
        class_pixel_pct[label]    = round(cnt / total_pixels * 100, 3)

    # ── Buildings ──
    flooded_building_mask     = (class_mask == 1).astype(np.uint8)
    non_flooded_building_mask = (class_mask == 2).astype(np.uint8)

    flooded_bldg_count, flooded_bldg_sizes = count_components(
        flooded_building_mask, MIN_BUILDING_PIXELS)
    non_flooded_bldg_count, non_flooded_bldg_sizes = count_components(
        non_flooded_building_mask, MIN_BUILDING_PIXELS)

    flooded_bldg_pixels = int(np.sum(flooded_building_mask))
    non_flooded_bldg_pixels = int(np.sum(non_flooded_building_mask))

    # ── Roads ──
    flooded_road_pixels     = int(np.sum(class_mask == 3))
    non_flooded_road_pixels = int(np.sum(class_mask == 4))
    flooded_road_pct        = round(flooded_road_pixels     / total_pixels * 100, 3)
    non_flooded_road_pct    = round(non_flooded_road_pixels / total_pixels * 100, 3)

    # ── Water body ──
    water_pixels = int(np.sum(class_mask == 5))
    water_pct    = round(water_pixels / total_pixels * 100, 3)

    # ── Vehicles ──
    vehicle_mask                  = (class_mask == 7).astype(np.uint8)
    vehicle_count, vehicle_sizes  = count_components(vehicle_mask, MIN_VEHICLE_PIXELS)
    vehicle_pixels                = int(np.sum(vehicle_mask))

    # ── Pool / standing water ──
    pool_pixels = int(np.sum(class_mask == 8))
    pool_pct    = round(pool_pixels / total_pixels * 100, 3)

    # ── Overall flood coverage ──
    flood_pixels = flooded_bldg_pixels + flooded_road_pixels + water_pixels + pool_pixels
    flood_pct    = round(flood_pixels / total_pixels * 100, 3)

    # ── Damage severity score [0–100] ──
    severity_raw = 0.0
    max_possible = 0.0
    for cid, (label, _) in COLOR_PALETTE.items():
        w_class = SEVERITY_WEIGHTS.get(cid, 0.0)
        pct = class_pixel_pct[label]
        severity_raw  += w_class * pct
        max_possible  += w_class * 100.0

    damage_severity_score = round((severity_raw / max_possible) * 100, 2) if max_possible > 0 else 0.0

    # ── Severity category ──
    if damage_severity_score >= 15:
        severity_label = "CRITICAL"
    elif damage_severity_score >= 8:
        severity_label = "SEVERE"
    elif damage_severity_score >= 3:
        severity_label = "MODERATE"
    elif damage_severity_score >= 0.5:
        severity_label = "MINOR"
    else:
        severity_label = "NONE"

    # ── Approximate ground areas (rough UAV estimate) ──
    flooded_bldg_area_m2     = pixel_area_m2(flooded_bldg_pixels,     (h, w))
    non_flooded_bldg_area_m2 = pixel_area_m2(non_flooded_bldg_pixels, (h, w))
    flooded_road_area_m2     = pixel_area_m2(flooded_road_pixels,     (h, w))
    water_area_m2            = pixel_area_m2(water_pixels,            (h, w))
    total_flood_area_m2      = pixel_area_m2(flood_pixels,            (h, w))

    report = {
        "mask_file":               os.path.basename(mask_path),
        "image_file":              os.path.basename(image_path) if image_path else None,
        "image_resolution":        {"width": w, "height": h},
        "damage_summary": {
            "damage_severity_score":   damage_severity_score,
            "severity_label":          severity_label,
            "flood_coverage_pct":      flood_pct,
            "total_flood_area_m2":     total_flood_area_m2,
        },
        "buildings": {
            "flooded_building_count":         flooded_bldg_count,
            "flooded_building_area_pct":      round(flooded_bldg_pixels / total_pixels * 100, 3),
            "flooded_building_area_m2":       flooded_bldg_area_m2,
            "flooded_building_sizes_px":      flooded_bldg_sizes[:5],
            "non_flooded_building_count":     non_flooded_bldg_count,
            "non_flooded_building_area_pct":  round(non_flooded_bldg_pixels / total_pixels * 100, 3),
            "non_flooded_building_area_m2":   non_flooded_bldg_area_m2,
            "total_building_count":           flooded_bldg_count + non_flooded_bldg_count,
            "building_flood_ratio":           round(
                flooded_bldg_count / max(1, flooded_bldg_count + non_flooded_bldg_count), 3),
        },
        "roads": {
            "flooded_road_coverage_pct":      flooded_road_pct,
            "flooded_road_area_m2":           flooded_road_area_m2,
            "non_flooded_road_coverage_pct":  non_flooded_road_pct,
            "total_road_coverage_pct":        round(flooded_road_pct + non_flooded_road_pct, 3),
        },
        "water": {
            "water_body_coverage_pct":  water_pct,
            "water_body_area_m2":       water_area_m2,
            "pool_coverage_pct":        pool_pct,
        },
        "vehicles": {
            "vehicle_count":      vehicle_count,
            "vehicle_area_pct":   round(vehicle_pixels / total_pixels * 100, 3),
            "vehicle_sizes_px":   vehicle_sizes[:5],
        },
        "class_pixel_distribution_pct": class_pixel_pct,
    }

    return report


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation: overlay mask on original image
# ──────────────────────────────────────────────────────────────────────────────
def create_overlay_image(image_path: str, mask_path: str,
                         target_size=(512, 512)) -> np.ndarray:
    """Create a semi-transparent coloured overlay of the class mask on the image."""
    img  = np.array(Image.open(image_path).convert("RGB").resize(target_size, Image.BILINEAR))
    mask = np.array(Image.open(mask_path).convert("RGB").resize(target_size, Image.NEAREST))

    cm   = np.zeros((target_size[1], target_size[0]), dtype=np.int64)
    for cid, (_, color) in COLOR_PALETTE.items():
        c     = np.array(color, dtype=np.uint8)
        match = np.all(np.abs(mask.astype(int) - c.astype(int)) <= 15, axis=-1)
        cm[match] = cid

    # Build coloured mask
    colour_mask = np.zeros((*target_size[::-1], 3), dtype=np.uint8)
    for cid, rgb in CLASS_DISPLAY_COLORS.items():
        colour_mask[cm == cid] = rgb

    # Alpha blend: 60% image, 40% mask
    overlay = (0.6 * img + 0.4 * colour_mask).astype(np.uint8)
    return overlay, colour_mask, cm


# ──────────────────────────────────────────────────────────────────────────────
# Batch analysis + summary chart
# ──────────────────────────────────────────────────────────────────────────────
def run_batch_extraction(split="val", max_samples=50):
    mask_dir  = os.path.join(FLOODNET_ROOT, "ColorMasks-FloodNetv1", f"ColorMasks-{split.capitalize()}Set")
    image_dir = os.path.join(FLOODNET_ROOT, "FloodNet-Supervised_v1.0", split, f"{split}-org-img")

    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])[:max_samples]

    print(f"\n{'='*60}")
    print(f"  FLOODNET DAMAGE EXTRACTION — {split.upper()} split ({len(mask_files)} images)")
    print(f"{'='*60}")

    all_reports = []
    for i, mfname in enumerate(mask_files):
        mask_path  = os.path.join(mask_dir, mfname)
        # Match image: mask name is "XXXXX_lab.png", image is "XXXXX.jpg"
        img_stem   = mfname.replace("_lab.png", "")
        img_path   = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            candidate = os.path.join(image_dir, img_stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        try:
            report = extract_damage_info(mask_path, img_path)
            all_reports.append(report)

            if (i + 1) % 10 == 0 or i == 0:
                ds = report["damage_summary"]
                bld = report["buildings"]
                print(f"  [{i+1:>3}/{len(mask_files)}] {mfname[:20]:22s} | "
                      f"Severity: {ds['severity_label']:<8s} ({ds['damage_severity_score']:5.2f}) | "
                      f"Flood: {ds['flood_coverage_pct']:5.2f}% | "
                      f"Bldgs flooded/total: {bld['flooded_building_count']}/{bld['total_building_count']}")
        except Exception as e:
            print(f"  Error on {mfname}: {e}")

    # ── Aggregate statistics ──
    def avg(key_chain):
        vals = []
        for r in all_reports:
            obj = r
            for k in key_chain:
                obj = obj.get(k, {}) if isinstance(obj, dict) else {}
            if isinstance(obj, (int, float)):
                vals.append(obj)
        return round(float(np.mean(vals)), 3) if vals else 0.0

    agg = {
        "split": split,
        "images_analysed":              len(all_reports),
        "avg_damage_severity_score":    avg(["damage_summary", "damage_severity_score"]),
        "avg_flood_coverage_pct":       avg(["damage_summary", "flood_coverage_pct"]),
        "avg_flooded_buildings":        avg(["buildings", "flooded_building_count"]),
        "avg_total_buildings":          avg(["buildings", "total_building_count"]),
        "avg_building_flood_ratio":     avg(["buildings", "building_flood_ratio"]),
        "avg_flooded_road_pct":         avg(["roads", "flooded_road_coverage_pct"]),
        "avg_water_coverage_pct":       avg(["water", "water_body_coverage_pct"]),
        "avg_vehicle_count":            avg(["vehicles", "vehicle_count"]),
        "severity_distribution": {
            "CRITICAL": sum(1 for r in all_reports if r["damage_summary"]["severity_label"] == "CRITICAL"),
            "SEVERE":   sum(1 for r in all_reports if r["damage_summary"]["severity_label"] == "SEVERE"),
            "MODERATE": sum(1 for r in all_reports if r["damage_summary"]["severity_label"] == "MODERATE"),
            "MINOR":    sum(1 for r in all_reports if r["damage_summary"]["severity_label"] == "MINOR"),
            "NONE":     sum(1 for r in all_reports if r["damage_summary"]["severity_label"] == "NONE"),
        },
    }

    print("\n  ── Aggregate Summary ──")
    print(f"  Avg Damage Severity Score : {agg['avg_damage_severity_score']:.3f}")
    print(f"  Avg Flood Coverage        : {agg['avg_flood_coverage_pct']:.2f}%")
    print(f"  Avg Flooded Buildings     : {agg['avg_flooded_buildings']:.1f}")
    print(f"  Avg Total Buildings       : {agg['avg_total_buildings']:.1f}")
    print(f"  Avg Building Flood Ratio  : {agg['avg_building_flood_ratio']:.3f}")
    print(f"  Avg Flooded Road Coverage : {agg['avg_flooded_road_pct']:.2f}%")
    print(f"  Avg Water Coverage        : {agg['avg_water_coverage_pct']:.2f}%")
    print(f"  Avg Vehicles Detected     : {agg['avg_vehicle_count']:.1f}")
    print(f"\n  Severity Distribution: {agg['severity_distribution']}")

    # Save JSON
    full_report = {"aggregate": agg, "per_image": all_reports}
    json_path   = os.path.join(OUTPUT_DIR, "damage_extraction_report.json")
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n  Full report saved → {json_path}")

    # Plot summary
    plot_damage_summary(agg, all_reports)

    return full_report


def plot_damage_summary(agg: dict, reports: list[dict]):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")

    # ── Severity pie ──
    ax = axes[0]
    ax.set_facecolor("white")
    sev  = agg["severity_distribution"]
    keys = [k for k, v in sev.items() if v > 0]
    vals = [sev[k] for k in keys]
    SEV_COLORS = {
        "CRITICAL": "#E02424",
        "SEVERE":   "#C27803",
        "MODERATE": "#1A56DB",
        "MINOR":    "#057A55",
        "NONE":     "#9CA3AF",
    }
    ax.pie(vals, labels=keys, colors=[SEV_COLORS[k] for k in keys],
           autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5},
           textprops={"fontsize": 9})
    ax.set_title("Damage Severity Distribution", fontsize=11,
                 fontweight="bold", color="#111827")

    # ── Flood coverage histogram ──
    ax2 = axes[1]
    ax2.set_facecolor("white")
    flood_pcts = [r["damage_summary"]["flood_coverage_pct"] for r in reports]
    ax2.hist(flood_pcts, bins=20, color="#1A56DB", edgecolor="white", linewidth=0.5)
    ax2.axvline(np.mean(flood_pcts), color="#E02424", linewidth=2,
                linestyle="--", label=f"Mean: {np.mean(flood_pcts):.1f}%")
    ax2.set_xlabel("Flood Coverage (%)", fontsize=10)
    ax2.set_ylabel("Number of Images",   fontsize=10)
    ax2.set_title("Flood Coverage Distribution", fontsize=11,
                  fontweight="bold", color="#111827")
    ax2.legend(frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, color="#D1D5DB")

    # ── Building damage bar ──
    ax3 = axes[2]
    ax3.set_facecolor("white")
    metrics = {
        "Flooded\nBuildings":    agg["avg_flooded_buildings"],
        "Total\nBuildings":      agg["avg_total_buildings"],
        "Flood Road\nCoverage%": agg["avg_flooded_road_pct"],
        "Water\nCoverage%":      agg["avg_water_coverage_pct"],
        "Vehicles\nDetected":    agg["avg_vehicle_count"],
    }
    BAR_COLORS = ["#E02424", "#1A56DB", "#C27803", "#057A55", "#5850EC"]
    bars = ax3.bar(metrics.keys(), metrics.values(),
                   color=BAR_COLORS, edgecolor="white", linewidth=0.5)
    for bar in bars:
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.03,
                 f"{bar.get_height():.2f}",
                 ha="center", va="bottom", fontsize=8, color="#374151")
    ax3.set_ylabel("Average per Image", fontsize=10)
    ax3.set_title("Avg Damage Metrics per Image", fontsize=11,
                  fontweight="bold", color="#111827")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", linestyle="--", alpha=0.4, color="#D1D5DB")

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "damage_summary.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Chart saved → {chart_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FloodNet Damage Extractor")
    parser.add_argument("--split",   default="val",  help="Dataset split: train/val/test")
    parser.add_argument("--sample",  type=int, default=50, help="Max number of images to analyse")
    parser.add_argument("--mask",    default=None, help="Path to a single mask file for extraction")
    parser.add_argument("--image",   default=None, help="Path to corresponding image (optional)")
    args = parser.parse_args()

    if args.mask:
        # Single image mode
        report = extract_damage_info(args.mask, args.image)
        print(json.dumps(report, indent=2))
    else:
        # Batch mode
        run_batch_extraction(split=args.split, max_samples=args.sample)


if __name__ == "__main__":
    main()
