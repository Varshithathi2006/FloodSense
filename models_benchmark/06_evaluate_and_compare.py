import os
import json
import time
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from utils.dataset_loader import FloodNetDatasetLoader
from utils.metrics import CLASS_NAMES_FLOODNET

# FloodNet color mapping for visualization
COLOR_PALETTE_RGB = {
    0: (55, 65, 81),     # Background (dark slate)
    1: (220, 38, 38),    # Building-Flooded (bright red)
    2: (180, 120, 120),  # Building-Non-Flooded (tan/brown)
    3: (202, 138, 4),    # Road-Flooded (dark yellow)
    4: (107, 114, 128),  # Road-Non-Flooded (gray)
    5: (6, 182, 212),    # Water (cyan)
    6: (5, 150, 105),    # Tree (dark green)
    7: (124, 58, 237),   # Vehicle (purple)
    8: (234, 179, 8),    # Pool (yellow)
    9: (74, 222, 128),   # Grass (light green)
}

def mask_to_rgb(mask_2d):
    h, w = mask_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in COLOR_PALETTE_RGB.items():
        rgb[mask_2d == cid] = color
    return rgb

def run_all_benchmarks_and_evaluate(sample_count=20, results_dir="results", force_rerun=False):
    os.makedirs(results_dir, exist_ok=True)
    
    # Dynamically import model modules
    m1 = importlib.import_module("01_classical_thresholding")
    m2 = importlib.import_module("02_sam_segmentation")
    m3 = importlib.import_module("03_visual_llm")
    m4 = importlib.import_module("04_grounding_dino_sam2")
    m5 = importlib.import_module("05_segformer")
    
    models_info = [
        ("01_classical_thresholding.json", m1.run_classical_thresholding_benchmark),
        ("02_sam_segmentation.json", m2.run_sam_benchmark),
        ("03_visual_llm.json", m3.run_visual_llm_benchmark),
        ("04_grounding_dino_sam2.json", m4.run_grounding_dino_sam2_benchmark),
        ("05_segformer.json", m5.run_segformer_benchmark),
    ]
    
    results = []
    for json_file, runner in models_info:
        json_path = os.path.join(results_dir, json_file)
        if os.path.exists(json_path) and not force_rerun:
            print(f"Loading existing result from {json_path}...")
            with open(json_path, "r") as f:
                data = json.load(f)
            # Re-run if old format without multiclass
            if "multiclass" not in data:
                print(f"Re-running {json_file} for multi-class support...")
                data = runner(max_samples=sample_count, output_dir=results_dir)
            results.append(data)
        else:
            print(f"Running benchmark for {json_file}...")
            data = runner(max_samples=sample_count, output_dir=results_dir)
            results.append(data)
            
    # Build summary rows
    summary_rows = []
    per_class_rows = []
    
    for r in results:
        m_info = r.get("multiclass", {})
        model_name = r["model_name"]
        
        summary_rows.append({
            "model_name": model_name,
            "binary_iou": r.get("iou", 0.0),
            "binary_f1": r.get("f1_score", 0.0),
            "multiclass_mIoU": m_info.get("mIoU", 0.0),
            "multiclass_macro_f1": m_info.get("macro_f1", 0.0),
            "pixel_accuracy": m_info.get("overall_pixel_accuracy", r.get("pixel_accuracy", 0.0)),
            "avg_latency_ms": r.get("avg_latency_ms", 0.0),
            "fps": r.get("fps", 0.0)
        })
        
        per_class_dict = m_info.get("per_class", {})
        for c_name, c_metrics in per_class_dict.items():
            per_class_rows.append({
                "model_name": model_name,
                "class_name": c_name,
                "class_id": c_metrics.get("class_id", -1),
                "iou": c_metrics.get("iou", 0.0),
                "f1_score": c_metrics.get("f1_score", 0.0),
                "precision": c_metrics.get("precision", 0.0),
                "recall": c_metrics.get("recall", 0.0)
            })
            
    df_summary = pd.DataFrame(summary_rows)
    
    # Composite score formula: 0.35 * mIoU + 0.25 * Macro_F1 + 0.20 * Binary_IoU + 0.10 * Acc + 0.10 * FPS_norm
    max_fps = df_summary["fps"].max() if df_summary["fps"].max() > 0 else 1.0
    df_summary["fps_normalized"] = df_summary["fps"] / max_fps
    df_summary["composite_score"] = (
        0.35 * df_summary["multiclass_mIoU"] +
        0.25 * df_summary["multiclass_macro_f1"] +
        0.20 * df_summary["binary_iou"] +
        0.10 * df_summary["pixel_accuracy"] +
        0.10 * df_summary["fps_normalized"]
    )
    
    df_summary = df_summary.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
    best_model = df_summary.iloc[0]["model_name"]
    
    print("\n=========================================================================================")
    print("                 FLOODSENSE 5-MODEL MULTI-CLASS BENCHMARK SUMMARY TABLE                  ")
    print("=========================================================================================\n")
    print(df_summary[["model_name", "multiclass_mIoU", "multiclass_macro_f1", "binary_iou", "binary_f1", "pixel_accuracy", "avg_latency_ms", "fps", "composite_score"]].to_string(index=False))
    
    print("\n-----------------------------------------------------------------------------------------")
    print(f"TOP PERFORMING MODEL (Phase 1 Winner): {best_model}")
    print("-----------------------------------------------------------------------------------------\n")
    
    # Save CSVs
    csv_summary_path = os.path.join(results_dir, "benchmark_results_summary.csv")
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"Overall summary saved to {csv_summary_path}")
    
    df_per_class = pd.DataFrame(per_class_rows)
    csv_per_class_path = os.path.join(results_dir, "per_class_f1_iou_breakdown.csv")
    df_per_class.to_csv(csv_per_class_path, index=False)
    print(f"Per-class breakdown saved to {csv_per_class_path}")
    
    # Plot Overall Comparison Chart
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    x = np.arange(len(df_summary))
    width = 0.25
    plt.bar(x - width, df_summary["multiclass_mIoU"], width, label="Multi-Class mIoU", color="#1f77b4")
    plt.bar(x, df_summary["multiclass_macro_f1"], width, label="Macro F1-Score", color="#2ca02c")
    plt.bar(x + width, df_summary["pixel_accuracy"], width, label="Pixel Accuracy", color="#ff7f0e")
    plt.xticks(x, df_summary["model_name"], rotation=20, ha="right", fontsize=9)
    plt.ylabel("Score")
    plt.title("Multi-Class Segmentation Accuracy (FloodNet 10 Classes)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.subplot(1, 2, 2)
    plt.bar(df_summary["model_name"], df_summary["fps"], color="#9467bd", width=0.4)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.ylabel("FPS (Frames Per Second)")
    plt.title("Inference Speed (FPS)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    chart_path = os.path.join(results_dir, "benchmark_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Metric visualization chart saved to {chart_path}")
    
    # Plot Per-Class Breakdown Chart
    if not df_per_class.empty:
        plt.figure(figsize=(15, 7))
        classes = sorted(df_per_class["class_name"].unique())
        models = df_summary["model_name"].unique()
        
        bar_width = 0.15
        x = np.arange(len(classes))
        
        for idx, m_name in enumerate(models):
            sub = df_per_class[df_per_class["model_name"] == m_name].set_index("class_name")
            scores = [sub.loc[c]["f1_score"] if c in sub.index else 0.0 for c in classes]
            plt.bar(x + (idx - len(models)/2) * bar_width, scores, bar_width, label=m_name)
            
        plt.xticks(x, classes, rotation=25, ha="right", fontsize=9)
        plt.ylabel("F1 / Dice Score")
        plt.title("Per-Class F1-Score Across FloodNet 10 Classes")
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        per_class_chart = os.path.join(results_dir, "per_class_f1_breakdown.png")
        plt.savefig(per_class_chart, dpi=300)
        plt.close()
        print(f"Per-class breakdown chart saved to {per_class_chart}")
        
    # Generate Visual Overlays with 10-Class Colors
    generate_qualitative_multiclass_comparison(m1, m2, m3, m4, m5, results_dir)
    
    return df_summary, best_model

def generate_qualitative_multiclass_comparison(m1, m2, m3, m4, m5, results_dir="results"):
    print("Generating side-by-side 10-class multi-color prediction overlays...")
    loader = FloodNetDatasetLoader(root_dir="FloodNet", split="val", target_size=(512, 512))
    
    m1_fn = m1.segment_classical_multiclass
    m2_fn = m2.SAMSegmenter().segment_image
    m3_fn = m3.VisualLLMSegmenter().segment_image
    m4_fn = m4.GroundingDINOSAM2Pipeline().segment_image
    m5_fn = m5.SegFormerSegmenter().segment_image
    
    models = [
        ("Classical CV", m1_fn),
        ("SAM", m2_fn),
        ("Visual LLM", m3_fn),
        ("Grounding DINO + SAM 2", m4_fn),
        ("OneFormer / SegFormer", m5_fn)
    ]
    
    sample_indices = [0, 2]
    
    for idx in sample_indices:
        img, gt_binary, gt_class = loader.load_item(idx)
        
        plt.figure(figsize=(20, 5))
        
        # 1. Original
        plt.subplot(1, 7, 1)
        plt.imshow(img)
        plt.title("Original Image", fontsize=10)
        plt.axis("off")
        
        # 2. Ground Truth Multi-Class Mask
        plt.subplot(1, 7, 2)
        gt_rgb = mask_to_rgb(gt_class)
        plt.imshow(gt_rgb)
        plt.title("Ground Truth (10-Class)", fontsize=10)
        plt.axis("off")
        
        col_idx = 3
        for name, fn in models:
            plt.subplot(1, 7, col_idx)
            try:
                _, pred_class = fn(img)
                pred_rgb = mask_to_rgb(pred_class)
            except Exception as e:
                pred_rgb = np.zeros_like(img)
            plt.imshow(pred_rgb)
            plt.title(name, fontsize=9)
            plt.axis("off")
            col_idx += 1
            
        plt.tight_layout()
        qual_path = os.path.join(results_dir, f"visual_multiclass_comparison_sample_{idx}.png")
        plt.savefig(qual_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"10-class visual overlay saved to {qual_path}")

if __name__ == "__main__":
    run_all_benchmarks_and_evaluate(force_rerun=True)

