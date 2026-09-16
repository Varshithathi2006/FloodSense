import numpy as np

CLASS_NAMES_FLOODNET = {
    0: "Background",
    1: "Building-Flooded",
    2: "Building-Non-Flooded",
    3: "Road-Flooded",
    4: "Road-Non-Flooded",
    5: "Water",
    6: "Tree",
    7: "Vehicle",
    8: "Pool",
    9: "Grass"
}

def compute_segmentation_metrics(pred_mask, gt_mask):
    """
    Computes binary segmentation metrics between predicted mask and ground truth mask.
    Both masks should be binary (0 or 1) numpy arrays of the same shape.
    """
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    
    total_pixels = pred.size
    
    # Intersection over Union (IoU)
    intersection = tp
    union = tp + fp + fn
    iou = float(intersection / union) if union > 0 else 1.0
    
    # Dice Coefficient / F1 Score
    f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 1.0
    
    # Precision & Recall
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # Pixel Accuracy
    accuracy = float((tp + tn) / total_pixels)
    
    return {
        "iou": iou,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "pixel_accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }

def compute_multiclass_metrics(pred_mask, gt_mask, num_classes=10, class_names=CLASS_NAMES_FLOODNET):
    """
    Computes multi-class semantic segmentation metrics across all classes.
    pred_mask and gt_mask are 2D integer arrays of same shape with values in range [0, num_classes-1].
    """
    pred = pred_mask.astype(np.int64)
    gt = gt_mask.astype(np.int64)
    
    total_pixels = gt.size
    correct_pixels = np.sum(pred == gt)
    pixel_accuracy = float(correct_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    per_class = {}
    ious = []
    f1_scores = []
    precisions = []
    recalls = []
    class_weights = []
    
    for c in range(num_classes):
        c_name = class_names.get(c, f"Class_{c}")
        
        pred_c = (pred == c)
        gt_c = (gt == c)
        
        tp = np.sum(pred_c & gt_c)
        fp = np.sum(pred_c & ~gt_c)
        fn = np.sum(~pred_c & gt_c)
        
        gt_count = int(np.sum(gt_c))
        class_weights.append(gt_count)
        
        union = tp + fp + fn
        if union > 0:
            c_iou = float(tp / union)
            c_f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
            c_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            c_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            ious.append(c_iou)
            f1_scores.append(c_f1)
            precisions.append(c_precision)
            recalls.append(c_recall)
        else:
            # Class not present in GT nor in Pred
            c_iou = 1.0
            c_f1 = 1.0
            c_precision = 1.0
            c_recall = 1.0
            
        per_class[c_name] = {
            "class_id": c,
            "iou": c_iou,
            "f1_score": c_f1,
            "precision": c_precision,
            "recall": c_recall,
            "gt_pixel_count": gt_count,
            "pred_pixel_count": int(np.sum(pred_c))
        }
        
    # Mean IoU and Macro F1
    mean_iou = float(np.mean(ious)) if ious else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    macro_precision = float(np.mean(precisions)) if precisions else 0.0
    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    
    # Frequency-weighted IoU
    total_gt = sum(class_weights)
    if total_gt > 0 and len(ious) == num_classes:
        fw_iou = float(sum(w * ious[i] for i, w in enumerate(class_weights)) / total_gt)
    else:
        fw_iou = mean_iou
        
    return {
        "mIoU": mean_iou,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "frequency_weighted_iou": fw_iou,
        "overall_pixel_accuracy": pixel_accuracy,
        "per_class": per_class
    }

def aggregate_metrics(metrics_list):
    """
    Aggregates list of metric dictionaries into mean values.
    """
    if not metrics_list:
        return {}
        
    keys = ["iou", "f1_score", "precision", "recall", "pixel_accuracy"]
    aggregated = {}
    for k in keys:
        values = [m[k] for m in metrics_list if k in m]
        aggregated[k] = float(np.mean(values)) if values else 0.0
        
    return aggregated

def aggregate_multiclass_metrics(multiclass_list):
    """
    Aggregates list of multi-class metric dictionaries across samples.
    """
    if not multiclass_list:
        return {}
        
    aggregated = {
        "mIoU": float(np.mean([m["mIoU"] for m in multiclass_list])),
        "macro_f1": float(np.mean([m["macro_f1"] for m in multiclass_list])),
        "macro_precision": float(np.mean([m["macro_precision"] for m in multiclass_list])),
        "macro_recall": float(np.mean([m["macro_recall"] for m in multiclass_list])),
        "frequency_weighted_iou": float(np.mean([m["frequency_weighted_iou"] for m in multiclass_list])),
        "overall_pixel_accuracy": float(np.mean([m["overall_pixel_accuracy"] for m in multiclass_list])),
        "per_class": {}
    }
    
    sample_classes = multiclass_list[0]["per_class"].keys()
    for c_name in sample_classes:
        aggregated["per_class"][c_name] = {
            "class_id": multiclass_list[0]["per_class"][c_name]["class_id"],
            "iou": float(np.mean([m["per_class"][c_name]["iou"] for m in multiclass_list])),
            "f1_score": float(np.mean([m["per_class"][c_name]["f1_score"] for m in multiclass_list])),
            "precision": float(np.mean([m["per_class"][c_name]["precision"] for m in multiclass_list])),
            "recall": float(np.mean([m["per_class"][c_name]["recall"] for m in multiclass_list]))
        }
        
    return aggregated

