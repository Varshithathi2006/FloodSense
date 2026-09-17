import os
import json
import time
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor

from utils.dataset_loader import FloodNetDatasetLoader
from utils.metrics import (
    compute_segmentation_metrics,
    aggregate_metrics,
    compute_multiclass_metrics,
    aggregate_multiclass_metrics,
    CLASS_NAMES_FLOODNET
)

class SAMSegmenter:
    def __init__(self, model_name="facebook/sam-vit-base", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        print(f"Loading SAM model '{model_name}' on device {self.device}...")
        self.processor = SamProcessor.from_pretrained(model_name)
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def classify_region(self, img_np, mask_bool):
        """Classifies a SAM segmented region into FloodNet 10 classes based on spectral & spatial features."""
        if np.sum(mask_bool) == 0:
            return 0
        pixels = img_np[mask_bool]
        mean_rgb = np.mean(pixels, axis=0) # [R, G, B]
        r, g, b = mean_rgb[0], mean_rgb[1], mean_rgb[2]
        
        # Vegetation: green dominant
        if g > r + 10 and g > b + 8 and g > 40:
            return 6 if np.mean(mean_rgb) < 115 else 9 # 6: Tree, 9: Grass
            
        # Water: cyan/blue dominant, muddy/tan/brown flood water, or dark stagnant water
        if (b > r + 10) or (b > 120 and g > 120 and r < 90):
            return 5 # Water (Cyan/Blue)
        if (r >= b - 5) and (g >= b - 15) and not (g > r + 15) and np.std(pixels) < 50:
            return 5 # Water (Muddy / Tan / Brown flood water)
        if np.mean(mean_rgb) < 65 and np.std([r, g, b]) < 20:
            return 5 # Water (Dark murky)
            
        # Bright high saturation pool
        if r > 160 and g > 160 and b < 60:
            return 8 # Pool
            
        # Road: low saturation gray
        color_std = np.std([r, g, b])
        if color_std < 18 and 50 < np.mean(mean_rgb) < 190:
            return 4 # Road non-flooded
            
        # Small compact region
        if np.sum(mask_bool) < 600:
            return 7 # Vehicle
            
        # Structure / building default
        return 2 # Building non-flooded

    def segment_image(self, img_np):
        """
        Segments image using SAM with point prompt grid and classifies regions.
        """
        h, w, _ = img_np.shape
        pil_img = Image.fromarray(img_np)
        
        # Define grid points across the image (4x4 point prompts for balance of coverage and speed)
        grid_x = np.linspace(w * 0.15, w * 0.85, 4)
        grid_y = np.linspace(h * 0.15, h * 0.85, 4)
        points = []
        for y in grid_y:
            for x in grid_x:
                points.append([int(x), int(y)])
                
        input_points = [[[p] for p in points]]
        
        inputs = self.processor(pil_img, input_points=input_points, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float64:
                    inputs[k] = v.to(torch.float32)
                inputs[k] = inputs[k].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0]
        
        iou_scores = outputs.iou_scores.cpu().numpy()[0]
        class_mask = np.zeros((h, w), dtype=np.int64)
        
        # Process masks
        for prompt_idx in range(len(points)):
            best_mask_idx = np.argmax(iou_scores[prompt_idx])
            mask = masks[prompt_idx, best_mask_idx].numpy()
            if np.sum(mask) > 50:
                cid = self.classify_region(img_np, mask)
                class_mask[mask] = cid
                
        # Contextual flood adjustment: if water is adjacent to roads/buildings, mark flooded
        water_mask = (class_mask == 5)
        if np.sum(water_mask) > 0:
            water_dist = cv2.distanceTransform((~water_mask).astype(np.uint8), cv2.DIST_L2, 5)
            class_mask[(class_mask == 2) & (water_dist < 20)] = 1 # Building Flooded
            class_mask[(class_mask == 4) & (water_dist < 15)] = 3 # Road Flooded
            
        binary_mask = np.isin(class_mask, [1, 3, 5, 8]).astype(np.uint8)
        return binary_mask, class_mask

def run_sam_benchmark(max_samples=50, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    loader = FloodNetDatasetLoader(root_dir="FloodNet", split="val", target_size=(512, 512))
    
    sample_count = min(max_samples, len(loader))
    print(f"Running SAM (Segment Anything Model) Multi-Class on {sample_count} validation samples...")
    
    segmenter = SAMSegmenter()
    
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
        
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{sample_count} samples...")
            
    avg_binary = aggregate_metrics(binary_metrics_list)
    avg_multi = aggregate_multiclass_metrics(multiclass_metrics_list)
    
    avg_binary["model_name"] = "SAM (Segment Anything)"
    avg_binary["avg_latency_ms"] = float(np.mean(times))
    avg_binary["fps"] = float(1000.0 / np.mean(times))
    avg_binary["samples_evaluated"] = sample_count
    avg_binary["multiclass"] = avg_multi
    
    output_path = os.path.join(output_dir, "02_sam_segmentation.json")
    with open(output_path, "w") as f:
        json.dump(avg_binary, f, indent=2)
        
    print("--- SAM (Segment Anything Model) Results ---")
    print(f"Binary Mean IoU: {avg_binary['iou']:.4f} | F1: {avg_binary['f1_score']:.4f}")
    print(f"Multi-Class mIoU: {avg_multi['mIoU']:.4f} | Macro F1: {avg_multi['macro_f1']:.4f}")
    print(f"Overall Pixel Accuracy: {avg_multi['overall_pixel_accuracy']:.4f}")
    print(f"Average Latency: {avg_binary['avg_latency_ms']:.2f} ms ({avg_binary['fps']:.2f} FPS)")
    print(f"Results saved to {output_path}\n")
    return avg_binary

if __name__ == "__main__":
    import sys
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_sam_benchmark(max_samples=num_samples)

