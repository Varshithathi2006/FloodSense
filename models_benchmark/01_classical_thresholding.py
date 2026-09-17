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
    Uses multi-spectral color space analysis (HSV + RGB + LAB) and spatial morphology
    to detect all flood water variations (cyan, blue, muddy, brown, tan, silt, dark).
    
    Classes:
    0: Background, 1: Building-Flooded, 2: Building-Non-Flooded,
    3: Road-Flooded, 4: Road-Non-Flooded, 5: Water,
    6: Tree, 7: Vehicle, 8: Pool, 9: Grass
    """
    h, w, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    class_mask = np.zeros((h, w), dtype=np.int64)
    
    r = img[:, :, 0].astype(float)
    g = img[:, :, 1].astype(float)
    b = img[:, :, 2].astype(float)
    
    # 1. Vegetation Detection (Grass vs Tree)
    green_dom = (g > r + 8) & (g > b + 5) & (g > 40)
    green_mask = (cv2.inRange(hsv, np.array([35, 30, 25]), np.array([88, 255, 255])) > 0) | green_dom
    tree_mask = green_mask & (hsv[:, :, 2] < 125)   # Darker green = Trees (Class 6)
    grass_mask = green_mask & (hsv[:, :, 2] >= 125) # Brighter green = Grass (Class 9)
    
    class_mask[tree_mask] = 6
    class_mask[grass_mask] = 9
    
    # 2. Roof / Building Detection (Red, Maroon, Metal, High Texture)
    red_roof1 = cv2.inRange(hsv, np.array([0, 50, 60]), np.array([10, 255, 255]))
    red_roof2 = cv2.inRange(hsv, np.array([170, 50, 60]), np.array([180, 255, 255]))
    red_roofs = (red_roof1 > 0) | (red_roof2 > 0)
    blue_roofs = (hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 130) & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 100)
    
    edges = cv2.Canny(gray, 40, 120)
    kernel_bldg = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edge_dense = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_bldg)
    
    # 3. Water Detection (Multi-Spectral: Cyan, Blue, Muddy, Tan, Brown, Clay, Silt, Dark)
    # A) Cyan / Blue clean flood water
    water_cyan = (cv2.inRange(hsv, np.array([80, 30, 30]), np.array([140, 255, 255])) > 0)
    
    # B) Muddy / Brown / Tan / Silt flood water (Hue 7-48, warm tan/ochre tint, smooth surface)
    water_tan = (cv2.inRange(hsv, np.array([7, 12, 35]), np.array([48, 240, 255])) > 0) & (r >= b - 10) & (~green_dom)
    
    # C) Dark stagnant / murky flood water
    water_dark = (hsv[:, :, 2] < 70) & (hsv[:, :, 1] < 70) & (class_mask == 0)
    
    # D) Swimming pool (Class 8)
    pool_mask = (cv2.inRange(hsv, np.array([20, 160, 190]), np.array([35, 255, 255])) > 0) & (~water_tan)
    
    # Combine water regions excluding distinct building roofs and vegetation
    water_total = (water_cyan | water_tan | water_dark) & (~red_roofs) & (~blue_roofs) & (class_mask == 0)
    
    # Morphological closing to ensure clean continuous water boundaries
    kernel_water = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    water_total = cv2.morphologyEx(water_total.astype(np.uint8), cv2.MORPH_CLOSE, kernel_water)
    
    class_mask[pool_mask] = 8
    class_mask[water_total > 0] = 5
    
    # 4. Roads (Flooded vs Non-Flooded)
    road_candidate = (hsv[:, :, 1] < 35) & (hsv[:, :, 2] > 55) & (hsv[:, :, 2] < 210) & (class_mask == 0)
    water_dist = cv2.distanceTransform((class_mask != 5).astype(np.uint8), cv2.DIST_L2, 5)
    flooded_road = road_candidate & (water_dist < 20)
    non_flooded_road = road_candidate & (water_dist >= 20)
    
    class_mask[non_flooded_road] = 4
    class_mask[flooded_road] = 3
    
    # 5. Buildings (Flooded vs Non-Flooded)
    building_candidate = (red_roofs | blue_roofs | (edge_dense > 0)) & (class_mask == 0)
    flooded_building = building_candidate & (water_dist < 25)
    non_flooded_building = building_candidate & (water_dist >= 25)
    
    class_mask[non_flooded_building] = 2
    class_mask[flooded_building] = 1
    
    # 6. Fill remaining unclassified flat terrain within flood boundary as flooded water/road
    unclassified = (class_mask == 0)
    unclass_flooded = unclassified & (water_dist < 15)
    class_mask[unclass_flooded] = 5
    
    # 7. Small Vehicles (Class 7)
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

