import os
import json
import time
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils.dataset_loader import FloodNetDatasetLoader
from utils.metrics import (
    compute_segmentation_metrics,
    aggregate_metrics,
    compute_multiclass_metrics,
    aggregate_multiclass_metrics,
    CLASS_NAMES_FLOODNET
)

class VisualLLMSegmenter:
    def __init__(self, model_name="google/owlvit-base-patch32", device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        print(f"Loading Visual LLM / VLM model '{model_name}' on device {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Category queries mapped to FloodNet Class IDs
        self.query_to_class = [
            ("flooded building", 1),
            ("building roof", 2),
            ("flooded road", 3),
            ("road pavement", 4),
            ("flood water", 5),
            ("green tree", 6),
            ("car vehicle", 7),
            ("swimming pool", 8),
            ("green grass lawn", 9),
        ]
        self.queries = [q[0] for q in self.query_to_class]

    def segment_image(self, img_np, score_threshold=0.08):
        """
        Queries Visual LLM / VLM for zero-shot open-vocabulary prompts across FloodNet categories.
        """
        h, w, _ = img_np.shape
        pil_img = Image.fromarray(img_np)
        
        inputs = self.processor(text=self.queries, images=pil_img, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float64:
                    inputs[k] = v.to(torch.float32)
                inputs[k] = inputs[k].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        target_sizes = torch.tensor([[h, w]], dtype=torch.float32).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=score_threshold
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        class_mask = np.zeros((h, w), dtype=np.int64)
        
        # Sort by confidence so higher confidence detections take precedence
        order = np.argsort(scores)
        for idx in order:
            box = boxes[idx]
            label_idx = labels[idx]
            if label_idx < len(self.query_to_class):
                cid = self.query_to_class[label_idx][1]
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                class_mask[y1:y2, x1:x2] = cid
                
        # Derive binary mask (Classes 1, 3, 5, 8)
        binary_mask = np.isin(class_mask, [1, 3, 5, 8]).astype(np.uint8)
        return binary_mask, class_mask

def run_visual_llm_benchmark(max_samples=20, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    loader = FloodNetDatasetLoader(root_dir="FloodNet", split="val", target_size=(512, 512))
    
    sample_count = min(max_samples, len(loader))
    print(f"Running Visual LLM (Zero-shot VLM) on {sample_count} validation samples...")
    
    segmenter = VisualLLMSegmenter()
    
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
    
    avg_binary["model_name"] = "Visual LLM"
    avg_binary["avg_latency_ms"] = float(np.mean(times))
    avg_binary["fps"] = float(1000.0 / np.mean(times))
    avg_binary["samples_evaluated"] = sample_count
    avg_binary["multiclass"] = avg_multi
    
    output_path = os.path.join(output_dir, "03_visual_llm.json")
    with open(output_path, "w") as f:
        json.dump(avg_binary, f, indent=2)
        
    print("--- Visual LLM Results ---")
    print(f"Binary Mean IoU: {avg_binary['iou']:.4f} | F1: {avg_binary['f1_score']:.4f}")
    print(f"Multi-Class mIoU: {avg_multi['mIoU']:.4f} | Macro F1: {avg_multi['macro_f1']:.4f}")
    print(f"Overall Pixel Accuracy: {avg_multi['overall_pixel_accuracy']:.4f}")
    print(f"Average Latency: {avg_binary['avg_latency_ms']:.2f} ms ({avg_binary['fps']:.2f} FPS)")
    print(f"Results saved to {output_path}\n")
    return avg_binary

if __name__ == "__main__":
    import sys
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_visual_llm_benchmark(max_samples=num_samples)

