import os
import json
import time
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

from utils.dataset_loader import FloodNetDatasetLoader
from utils.metrics import (
    compute_segmentation_metrics,
    aggregate_metrics,
    compute_multiclass_metrics,
    aggregate_multiclass_metrics,
    CLASS_NAMES_FLOODNET
)

class GroundingDINOSAM2Pipeline:
    def __init__(self, dino_model="google/owlvit-base-patch32", sam_model="facebook/sam-vit-base", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        print(f"Loading Grounding DINO / Detector '{dino_model}' and SAM '{sam_model}' on {self.device}...")
        self.dino_processor = AutoProcessor.from_pretrained(dino_model)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model).to(self.device)
        self.dino_model.eval()
        
        self.sam_processor = SamProcessor.from_pretrained(sam_model)
        self.sam_model = SamModel.from_pretrained(sam_model).to(self.device)
        self.sam_model.eval()
        
        # Category prompts mapped to target FloodNet classes
        self.query_to_class = [
            ("flooded building", 1),
            ("building house", 2),
            ("flooded road", 3),
            ("road asphalt", 4),
            ("flood water puddle", 5),
            ("tree forest", 6),
            ("car vehicle", 7),
            ("swimming pool", 8),
            ("grass lawn", 9),
        ]
        self.text_prompt = [q[0] for q in self.query_to_class]

    def segment_image(self, img_np, score_threshold=0.10):
        h, w, _ = img_np.shape
        pil_img = Image.fromarray(img_np)
        
        # Step 1: Detect bounding boxes with Grounding DINO / zero-shot detector
        dino_inputs = self.dino_processor(text=self.text_prompt, images=pil_img, return_tensors="pt")
        for k, v in dino_inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float64:
                    dino_inputs[k] = v.to(torch.float32)
                dino_inputs[k] = dino_inputs[k].to(self.device)
                
        with torch.no_grad():
            dino_outputs = self.dino_model(**dino_inputs)
            
        target_sizes = torch.tensor([[h, w]], dtype=torch.float32).to(self.device)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs=dino_outputs,
            target_sizes=target_sizes,
            threshold=score_threshold
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        class_mask = np.zeros((h, w), dtype=np.int64)
        
        if len(boxes) == 0:
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            return binary_mask, class_mask
            
        # Filter top valid boxes
        valid_items = []
        for box, score, lbl in zip(boxes, scores, labels):
            if score > score_threshold:
                valid_items.append((box, score, lbl))
                
        if not valid_items:
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            return binary_mask, class_mask
            
        # Step 2: Pass bounding boxes to SAM for precise pixel-level mask refinement
        # Limit to top 8 boxes for speed & efficiency
        valid_items = sorted(valid_items, key=lambda x: x[1], reverse=True)[:8]
        input_boxes = [[[item[0][0], item[0][1], item[0][2], item[0][3]] for item in valid_items]]
        
        sam_inputs = self.sam_processor(pil_img, input_boxes=input_boxes, return_tensors="pt")
        for k, v in sam_inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float64:
                    sam_inputs[k] = v.to(torch.float32)
                sam_inputs[k] = sam_inputs[k].to(self.device)
                
        with torch.no_grad():
            sam_outputs = self.sam_model(**sam_inputs)
            
        masks = self.sam_processor.image_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
            sam_inputs["reshaped_input_sizes"].cpu()
        )[0]
        
        iou_scores = sam_outputs.iou_scores.cpu().numpy()[0]
        
        for box_idx, item in enumerate(valid_items):
            lbl_idx = item[2]
            cid = self.query_to_class[lbl_idx][1] if lbl_idx < len(self.query_to_class) else 5
            best_mask_idx = np.argmax(iou_scores[box_idx])
            refined_mask = masks[box_idx, best_mask_idx].numpy()
            class_mask[refined_mask] = cid
            
        binary_mask = np.isin(class_mask, [1, 3, 5, 8]).astype(np.uint8)
        return binary_mask, class_mask

def run_grounding_dino_sam2_benchmark(max_samples=20, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    loader = FloodNetDatasetLoader(root_dir="FloodNet", split="val", target_size=(512, 512))
    
    sample_count = min(max_samples, len(loader))
    print(f"Running Grounding DINO + SAM 2 pipeline on {sample_count} validation samples...")
    
    pipeline = GroundingDINOSAM2Pipeline()
    
    binary_metrics_list = []
    multiclass_metrics_list = []
    times = []
    
    for i in range(sample_count):
        img, gt_binary, gt_class = loader.load_item(i)
        
        start_t = time.time()
        pred_binary, pred_class = pipeline.segment_image(img)
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
    
    avg_binary["model_name"] = "Grounding DINO + SAM 2"
    avg_binary["avg_latency_ms"] = float(np.mean(times))
    avg_binary["fps"] = float(1000.0 / np.mean(times))
    avg_binary["samples_evaluated"] = sample_count
    avg_binary["multiclass"] = avg_multi
    
    output_path = os.path.join(output_dir, "04_grounding_dino_sam2.json")
    with open(output_path, "w") as f:
        json.dump(avg_binary, f, indent=2)
        
    print("--- Grounding DINO + SAM 2 Results ---")
    print(f"Binary Mean IoU: {avg_binary['iou']:.4f} | F1: {avg_binary['f1_score']:.4f}")
    print(f"Multi-Class mIoU: {avg_multi['mIoU']:.4f} | Macro F1: {avg_multi['macro_f1']:.4f}")
    print(f"Overall Pixel Accuracy: {avg_multi['overall_pixel_accuracy']:.4f}")
    print(f"Average Latency: {avg_binary['avg_latency_ms']:.2f} ms ({avg_binary['fps']:.2f} FPS)")
    print(f"Results saved to {output_path}\n")
    return avg_binary

if __name__ == "__main__":
    import sys
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_grounding_dino_sam2_benchmark(max_samples=num_samples)

