import os
import json
import time
import cv2
import numpy as np

from utils.dataset_loader import FloodNetDatasetLoader
from utils.metrics import (
    compute_segmentation_metrics,
    aggregate_metrics,
    compute_multiclass_metrics,
    aggregate_multiclass_metrics,
    CLASS_NAMES_FLOODNET
)

def segment_classical_multiclass(img):
    """
    Classical Computer Vision Multi-Class Segmentation for FloodNet 10 classes.
    Uses multi-spectral color space analysis (HSV + RGB + LAB) and geometric shape
    validation to accurately delineate flood water while eliminating false-positive
    building and road detections in wilderness/forest scenes.
    
    Classes:
    0: Background, 1: Building-Flooded, 2: Building-Non-Flooded,
    3: Road-Flooded, 4: Road-Non-Flooded, 5: Water,
    6: Tree, 7: Vehicle, 8: Pool, 9: Grass
    """
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    class_mask = np.zeros((h, w), dtype=np.uint8)
    
    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 2].astype(float)
    
    # 1. Vegetation Detection (Trees vs Grass)
    green_dom = (g > r + 5) & (g > b + 3) & (g > 30)
    green_mask = (cv2.inRange(hsv, np.array([30, 20, 20]), np.array([95, 255, 255])) > 0) | green_dom
    tree_mask = green_mask & (hsv[:, :, 2] < 125)
    grass_mask = green_mask & (hsv[:, :, 2] >= 125)
    
    class_mask[tree_mask] = 6
    class_mask[grass_mask] = 9
    
    # 2. Multi-Spectral Flood Water Detection (Cyan, Blue, Deep Dark Water, Muddy, Tan, Clay, Silt)
    water_blue_cyan = (cv2.inRange(hsv, np.array([75, 20, 15]), np.array([145, 255, 255])) > 0)
    water_dark = (hsv[:, :, 2] < 60) & (hsv[:, :, 1] < 80) & (~green_mask)
    water_tan = (cv2.inRange(hsv, np.array([7, 10, 30]), np.array([48, 240, 255])) > 0) & (r >= b - 10) & (~green_dom)
    
    water_raw = (water_blue_cyan | water_dark | water_tan) & (~green_mask)
    kernel_water = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    water_clean = cv2.morphologyEx(water_raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel_water)
    
    pool_mask = (cv2.inRange(hsv, np.array([20, 160, 190]), np.array([35, 255, 255])) > 0) & (~water_tan)
    class_mask[pool_mask] = 8
    class_mask[water_clean > 0] = 5
    
    water_dist = cv2.distanceTransform((class_mask != 5).astype(np.uint8), cv2.DIST_L2, 5)
    
    # 3. Geometric Building Detection ONLY on non-water, non-vegetation structures
    non_water_dry = (class_mask == 0) & (~green_mask)
    
    red_roof1 = cv2.inRange(hsv, np.array([0, 60, 60]), np.array([12, 255, 255]))
    red_roof2 = cv2.inRange(hsv, np.array([168, 60, 60]), np.array([180, 255, 255]))
    metal_roof = cv2.inRange(hsv, np.array([90, 60, 90]), np.array([135, 255, 255]))
    
    edges = cv2.Canny(gray, 60, 150)
    kernel_bldg = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_dense = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_bldg)
    
    bldg_cand = ((red_roof1 > 0) | (red_roof2 > 0) | (metal_roof > 0) | (edge_dense > 0)) & non_water_dry
    
    contours, _ = cv2.findContours(bldg_cand.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 300: # Real building structure footprint (>300 px)
            hull = cv2.convexHull(cnt)
            solidity = area / max(cv2.contourArea(hull), 1.0)
            if solidity >= 0.40:
                pts = cnt[:, 0, :]
                min_w_dist = np.min(water_dist[pts[:, 1], pts[:, 0]])
                bldg_class = 1 if min_w_dist < 25 else 2
                cv2.drawContours(class_mask, [cnt], -1, int(bldg_class), -1)
                
    # 4. Roads Detection on remaining dry corridors
    road_cand = (hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 200) & (class_mask == 0)
    contours_rd, _ = cv2.findContours(road_cand.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_rd:
        area = cv2.contourArea(cnt)
        if area >= 350: # Real road corridor (>350 px)
            pts = cnt[:, 0, :]
            min_w_dist = np.min(water_dist[pts[:, 1], pts[:, 0]])
            road_class = 3 if min_w_dist < 20 else 4
            cv2.drawContours(class_mask, [cnt], -1, int(road_class), -1)
            
    # 5. Remaining unclassified pixels
    unclass = (class_mask == 0)
    class_mask[unclass & (water_dist < 12)] = 5
    class_mask[unclass & (water_dist >= 12)] = 6
    
    # 6. Small Vehicles (Class 7) ONLY on detected roads
    if np.sum((class_mask == 3) | (class_mask == 4)) > 500:
        road_mask = ((class_mask == 3) | (class_mask == 4)).astype(np.uint8)
        contours_v, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_v:
            area = cv2.contourArea(cnt)
            if 60 < area < 700:
                cv2.drawContours(class_mask, [cnt], -1, 7, -1)
            
    binary_pred = np.isin(class_mask, [1, 3, 5, 8]).astype(np.uint8)
    return binary_pred, class_mask.astype(np.int64)

def run_classical_thresholding_benchmark(max_samples=50, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    loader = FloodNetDatasetLoader(root_dir="FloodNet", split="val", target_size=(512, 512))
    
    sample_count = min(max_samples, len(loader))
    print(f"Running Classical Multi-Class Thresholding on {sample_count} validation samples...")
    
    binary_metrics_list = []
    multiclass_metrics_list = []
    times = []
    
    for i in range(sample_count):
        img, gt_binary, gt_class = loader.load_item(i)
        
        start_t = time.time()
        pred_binary, pred_class = segment_classical_multiclass(img)
        elapsed_ms = (time.time() - start_t) * 1000.0
        
        times.append(elapsed_ms)
        b_metrics = compute_segmentation_metrics(pred_binary, gt_binary)
        m_metrics = compute_multiclass_metrics(pred_class, gt_class)
        
        binary_metrics_list.append(b_metrics)
        multiclass_metrics_list.append(m_metrics)
        
    avg_binary = aggregate_metrics(binary_metrics_list)
    avg_multi = aggregate_multiclass_metrics(multiclass_metrics_list)
    
    avg_binary["model_name"] = "Classical Thresholding"
    avg_binary["avg_latency_ms"] = float(np.mean(times))
    avg_binary["fps"] = float(1000.0 / np.mean(times))
    avg_binary["samples_evaluated"] = sample_count
    avg_binary["multiclass"] = avg_multi
    
    output_path = os.path.join(output_dir, "01_classical_thresholding.json")
    with open(output_path, "w") as f:
        json.dump(avg_binary, f, indent=2)
        
    print("--- Classical Thresholding Results ---")
    print(f"Binary Mean IoU: {avg_binary['iou']:.4f} | F1: {avg_binary['f1_score']:.4f}")
    print(f"Multi-Class mIoU: {avg_multi['mIoU']:.4f} | Macro F1: {avg_multi['macro_f1']:.4f}")
    print(f"Overall Pixel Accuracy: {avg_multi['overall_pixel_accuracy']:.4f}")
    print(f"Average Latency: {avg_binary['avg_latency_ms']:.2f} ms ({avg_binary['fps']:.2f} FPS)")
    print(f"Results saved to {output_path}\n")
    return avg_binary

if __name__ == "__main__":
    import sys
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_classical_thresholding_benchmark(max_samples=num_samples)

