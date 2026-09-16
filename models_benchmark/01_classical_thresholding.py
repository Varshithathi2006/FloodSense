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
    Uses multi-space color clustering (HSV + LAB), edge detection, and spatial morphological rules.
    
    Classes:
    0: Background, 1: Building-Flooded, 2: Building-Non-Flooded,
    3: Road-Flooded, 4: Road-Non-Flooded, 5: Water,
    6: Tree, 7: Vehicle, 8: Pool, 9: Grass
    """
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    class_mask = np.zeros((h, w), dtype=np.int64)
    
    # 1. Vegetation Detection (Grass vs Tree)
    # Green Hue in HSV: ~35-85
    green_mask = cv2.inRange(hsv, np.array([35, 30, 30]), np.array([85, 255, 255]))
    tree_mask = green_mask & (hsv[:, :, 2] < 120)  # Darker green = Trees (Class 6)
    grass_mask = green_mask & (hsv[:, :, 2] >= 120) # Brighter green = Grass (Class 9)
    
    class_mask[tree_mask > 0] = 6
    class_mask[grass_mask > 0] = 9
    
    # 2. Water / Pool Detection
    # Cyan / Blue water (Class 5)
    water_cyan = cv2.inRange(hsv, np.array([85, 40, 30]), np.array([135, 255, 255]))
    # Turbid / Muddy flood water
    water_muddy = cv2.inRange(hsv, np.array([10, 25, 40]), np.array([30, 160, 180])) & (class_mask == 0)
    # Bright rectangular / circular pools (Class 8)
    pool_mask = cv2.inRange(hsv, np.array([20, 150, 180]), np.array([35, 255, 255]))
    
    water_total = cv2.bitwise_or(water_cyan, water_muddy)
    kernel_sm = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    water_total = cv2.morphologyEx(water_total, cv2.MORPH_CLOSE, kernel_sm)
    
    class_mask[pool_mask > 0] = 8
    class_mask[water_total > 0] = 5
    
    # 3. Roads (Flooded vs Non-Flooded)
    # Gray / asphalt pavement (Class 4: Road Non-Flooded)
    road_candidate = (hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 200) & (class_mask == 0)
    
    # Check water proximity for flooded road (Class 3)
    water_dist = cv2.distanceTransform((class_mask != 5).astype(np.uint8), cv2.DIST_L2, 5)
    flooded_road = road_candidate & (water_dist < 15)
    non_flooded_road = road_candidate & (water_dist >= 15)
    
    class_mask[non_flooded_road] = 4
    class_mask[flooded_road] = 3
    
    # 4. Buildings (Flooded vs Non-Flooded)
    # High edge density / roof textures
    edges = cv2.Canny(gray, 50, 150)
    kernel_lg = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edge_dense = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_lg)
    building_candidate = (edge_dense > 0) & (class_mask == 0)
    
    flooded_building = building_candidate & (water_dist < 20)
    non_flooded_building = building_candidate & (water_dist >= 20)
    
    class_mask[non_flooded_building] = 2
    class_mask[flooded_building] = 1
    
    # 5. Small Vehicles (Class 7)
    # Compact bright/dark spots on roads
    contours, _ = cv2.findContours(road_candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    veh_overlay = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 800:
            cv2.drawContours(veh_overlay, [cnt], -1, 1, -1)
    class_mask[veh_overlay == 1] = 7
            
    # Flooded binary mask (Classes 1, 3, 5, 8)
    binary_pred = np.isin(class_mask, [1, 3, 5, 8]).astype(np.uint8)
    return binary_pred, class_mask

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

