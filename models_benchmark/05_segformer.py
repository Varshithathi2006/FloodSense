import os
import json
import time
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from utils.dataset_loader import FloodNetDatasetLoader
from utils.metrics import (
    compute_segmentation_metrics,
    aggregate_metrics,
    compute_multiclass_metrics,
    aggregate_multiclass_metrics,
    CLASS_NAMES_FLOODNET
)

class SegFormerSegmenter:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        print(f"Loading SegFormer Transformer model '{model_name}' on device {self.device}...")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def segment_image(self, img_np):
        """
        Segments image using SegFormer semantic segmentation model.
        Maps ADE20K categories to FloodNet 10 classes and detects flood inundation context.
        """
        h, w, _ = img_np.shape
        pil_img = Image.fromarray(img_np)
        
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits # (batch=1, num_classes, H_out, W_out)
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        
        pred_ade = upsampled_logits.argmax(dim=1).cpu().numpy()[0]
        
        # Initialize 10-class FloodNet mask
        class_mask = np.zeros((h, w), dtype=np.int64)
        
        # ADE20K Class Mappings:
        # Water: 21 (water), 26 (sea), 113 (lake), 128 (water body)
        water_mask = np.isin(pred_ade, [21, 26, 113, 128])
        # Pool: 60, 61
        pool_mask = np.isin(pred_ade, [60, 61])
        # Vegetation: Trees (4, 17, 66), Grass (9, 29, 46)
        tree_mask = np.isin(pred_ade, [4, 17, 66])
        grass_mask = np.isin(pred_ade, [9, 29, 46])
        # Vehicles: 20 (car), 80 (boat), 83 (bus), 102 (truck), 103 (van)
        veh_mask = np.isin(pred_ade, [20, 80, 83, 102, 103, 116])
        # Buildings: 1 (building), 25 (house), 48 (skyscraper), 84 (tower)
        bldg_mask = np.isin(pred_ade, [1, 25, 48, 84])
        # Roads / Pavement: 6 (road), 7 (bed), 11 (sidewalk), 13 (earth), 52 (path)
        road_mask = np.isin(pred_ade, [6, 11, 13, 52])
        
        # Apply base classes
        class_mask[tree_mask] = 6
        class_mask[grass_mask] = 9
        class_mask[veh_mask] = 7
        class_mask[pool_mask] = 8
        class_mask[water_mask] = 5
        
        # Spatial context for flood inundation on structures & roads
        if np.sum(water_mask) > 0:
            water_dist = cv2.distanceTransform((~water_mask).astype(np.uint8), cv2.DIST_L2, 5)
            flooded_bldg = bldg_mask & (water_dist < 25)
            non_flooded_bldg = bldg_mask & (water_dist >= 25)
            
            flooded_road = road_mask & (water_dist < 20)
            non_flooded_road = road_mask & (water_dist >= 20)
        else:
            flooded_bldg = np.zeros_like(bldg_mask, dtype=bool)
            non_flooded_bldg = bldg_mask
            flooded_road = np.zeros_like(road_mask, dtype=bool)
            non_flooded_road = road_mask
            
        class_mask[non_flooded_bldg] = 2
        class_mask[flooded_bldg] = 1
        class_mask[non_flooded_road] = 4
        class_mask[flooded_road] = 3
        
        # Binary mask (Classes 1, 3, 5, 8)
        binary_mask = np.isin(class_mask, [1, 3, 5, 8]).astype(np.uint8)
        
        # Fallback if empty water
        if np.sum(binary_mask) == 0 and np.sum(water_mask) > 0:
            binary_mask = water_mask.astype(np.uint8)
            
        return binary_mask, class_mask

def run_segformer_benchmark(max_samples=50, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    loader = FloodNetDatasetLoader(root_dir="FloodNet", split="val", target_size=(512, 512))
    
    sample_count = min(max_samples, len(loader))
    print(f"Running OneFormer / SegFormer Multi-Class Transformer on {sample_count} validation samples...")
    
    segmenter = SegFormerSegmenter()
    
    binary_metrics_list = []
    multiclass_metrics_list = []
    times = []
    
    for i in range(sample_count):
        img, gt_binary, gt_class = loader.load_item(i)
        
        start_t = time.time()
        pred_binary, pred_class = segmenter.segment_image(img)
        elapsed_ms = (time.time() - start_t) * 1000.0
        
        times.append(elapsed_ms)
        b_metrics = compute_segmentation_metrics(pred_binary, gt_binary)
        m_metrics = compute_multiclass_metrics(pred_class, gt_class)
        
        binary_metrics_list.append(b_metrics)
        multiclass_metrics_list.append(m_metrics)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{sample_count} samples...")
            
    avg_binary = aggregate_metrics(binary_metrics_list)
    avg_multi = aggregate_multiclass_metrics(multiclass_metrics_list)
    
    avg_binary["model_name"] = "OneFormer / SegFormer"
    avg_binary["avg_latency_ms"] = float(np.mean(times))
    avg_binary["fps"] = float(1000.0 / np.mean(times))
    avg_binary["samples_evaluated"] = sample_count
    avg_binary["multiclass"] = avg_multi
    
    output_path = os.path.join(output_dir, "05_segformer.json")
    with open(output_path, "w") as f:
        json.dump(avg_binary, f, indent=2)
        
    print("--- OneFormer / SegFormer Results ---")
    print(f"Binary Mean IoU: {avg_binary['iou']:.4f} | F1: {avg_binary['f1_score']:.4f}")
    print(f"Multi-Class mIoU: {avg_multi['mIoU']:.4f} | Macro F1: {avg_multi['macro_f1']:.4f}")
    print(f"Overall Pixel Accuracy: {avg_multi['overall_pixel_accuracy']:.4f}")
    print(f"Average Latency: {avg_binary['avg_latency_ms']:.2f} ms ({avg_binary['fps']:.2f} FPS)")
    print(f"Results saved to {output_path}\n")
    return avg_binary

if __name__ == "__main__":
    import sys
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_segformer_benchmark(max_samples=num_samples)

