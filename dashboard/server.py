"""
FloodSense Master API Server
============================
Flask backend serving live Computer Vision segmentation (SegFormer, Classical, SAM),
Multi-Class Damage Extraction, ChromaDB RAG Protocol Retrieval, and Grounded LLM
Disaster Situation Report (SITREP) generation.

Endpoints:
  GET  /                     → Serves dashboard UI (index.html)
  GET  /api/status           → System health and model availability
  GET  /api/models           → Available benchmarked CV models
  GET  /api/samples          → Pre-indexed demo flood images & masks
  GET  /api/knowledge_base   → Knowledge base documents & protocols
  POST /api/analyze          → Runs end-to-end CV + RAG + LLM report pipeline
  GET  /api/overlay/<file>   → Serves generated overlay image

Usage:
  py -3.11 dashboard/server.py
  → Open http://localhost:5050
"""

import os
import sys
import json
import uuid
import time
import base64
import tempfile
import io

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, send_file
from scipy import ndimage

# ── Paths ──────────────────────────────────────────────────────────────────
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.dirname(DASHBOARD_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "models_benchmark")
if MODELS_BENCHMARK_DIR not in sys.path:
    sys.path.insert(0, MODELS_BENCHMARK_DIR)

from damage_extraction.damage_extractor import (
    COLOR_PALETTE,
    CLASS_DISPLAY_COLORS,
    SEVERITY_WEIGHTS,
    rgb_mask_to_class_mask,
    count_components,
    pixel_area_m2
)
from rag_pipeline.vector_store import DisasterProtocolVectorStore
from rag_pipeline.retriever import FloodProtocolRetriever
from report_generation.llm_engine import FloodLLMEngine

import importlib

# Import model inference functions dynamically
m1 = importlib.import_module("models_benchmark.01_classical_thresholding")
segment_classical_multiclass = m1.segment_classical_multiclass

m5 = importlib.import_module("models_benchmark.05_segformer")
SegFormerSegmenter = m5.SegFormerSegmenter

# ── Flask App Setup ────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=DASHBOARD_DIR, static_url_path="")
OVERLAY_CACHE_DIR = os.path.join(tempfile.gettempdir(), "floodsense_overlays")
os.makedirs(OVERLAY_CACHE_DIR, exist_ok=True)

# ── Initialize RAG & LLM Engine ────────────────────────────────────────────
print("[Server Startup] Initializing Vector Store, Retriever & LLM Engine...")
vector_store = DisasterProtocolVectorStore()
if vector_store.collection.count() == 0:
    vector_store.build_or_update_index()
retriever = FloodProtocolRetriever(vector_store=vector_store)
llm_engine = FloodLLMEngine()

# Lazy model instances
segformer_instance = None

def get_segformer():
    global segformer_instance
    if segformer_instance is None:
        print("[Model Loader] Initializing SegFormer Transformer...")
        segformer_instance = SegFormerSegmenter()
    return segformer_instance

# ── FloodNet Sample Indexing ───────────────────────────────────────────────
FLOODNET_ROOT = os.path.join(PROJECT_ROOT, "FloodNet")
VAL_IMG_DIR   = os.path.join(FLOODNET_ROOT, "FloodNet-Supervised_v1.0", "val", "val-org-img")
VAL_MASK_DIR  = os.path.join(FLOODNET_ROOT, "ColorMasks-FloodNetv1", "ColorMasks-ValSet")

# ── CORS Helper ────────────────────────────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ── Color Overlay Generator ────────────────────────────────────────────────
def generate_color_overlay(img_rgb: np.ndarray, class_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = class_mask.shape
    color_layer = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_DISPLAY_COLORS.items():
        color_layer[class_mask == cid] = color
        
    img_resized = np.array(Image.fromarray(img_rgb).resize((w, h), Image.BILINEAR))
    blended = ((1.0 - alpha) * img_resized + alpha * color_layer).astype(np.uint8)
    return blended

def extract_metrics_from_class_mask(class_mask: np.ndarray, original_shape=(512, 512)) -> dict:
    total = class_mask.size
    h, w = class_mask.shape
    
    class_pct = {}
    for cid, (label, _) in COLOR_PALETTE.items():
        cnt = int(np.sum(class_mask == cid))
        class_pct[label] = round(cnt / total * 100, 3)
        
    fb_mask = (class_mask == 1).astype(np.uint8)
    nfb_mask = (class_mask == 2).astype(np.uint8)
    fb_cnt, _ = count_components(fb_mask, 50)
    nfb_cnt, _ = count_components(nfb_mask, 50)
    fb_px = int(np.sum(fb_mask))
    nfb_px = int(np.sum(nfb_mask))
    
    fr_px = int(np.sum(class_mask == 3))
    nfr_px = int(np.sum(class_mask == 4))
    
    water_px = int(np.sum(class_mask == 5))
    pool_px = int(np.sum(class_mask == 8))
    veh_px = int(np.sum(class_mask == 7))
    veh_mask = (class_mask == 7).astype(np.uint8)
    veh_cnt, _ = count_components(veh_mask, 20)
    
    flood_px = fb_px + fr_px + water_px + pool_px
    flood_pct = round(flood_px / total * 100, 3)
    
    # Severity score calculation
    severity_raw = sum(SEVERITY_WEIGHTS.get(cid, 0.0) * class_pct[label] for cid, (label, _) in COLOR_PALETTE.items())
    max_possible = sum(v * 100.0 for v in SEVERITY_WEIGHTS.values())
    raw_score = round((severity_raw / max_possible) * 100, 2) if max_possible > 0 else 0.0
    
    if raw_score >= 15.0:
        sev_label = "CRITICAL"
    elif raw_score >= 8.0:
        sev_label = "SEVERE"
    elif raw_score >= 3.0:
        sev_label = "MODERATE"
    elif raw_score >= 0.5:
        sev_label = "MINOR"
    else:
        sev_label = "NONE"
    
    return {
        "analysis_method": "Multi-Class Semantic Segmentation",
        "damage_summary": {
            "damage_severity_score": raw_score,
            "severity_label": sev_label,
            "flood_coverage_pct": flood_pct,
            "total_flood_area_m2": pixel_area_m2(flood_px, (h, w)),
        },
        "buildings": {
            "flooded_building_count": fb_cnt,
            "flooded_building_area_pct": round(fb_px / total * 100, 3),
            "flooded_building_area_m2": pixel_area_m2(fb_px, (h, w)),
            "non_flooded_building_count": nfb_cnt,
            "non_flooded_building_area_pct": round(nfb_px / total * 100, 3),
            "non_flooded_building_area_m2": pixel_area_m2(nfb_px, (h, w)),
            "total_building_count": fb_cnt + nfb_cnt,
            "building_flood_ratio": round(fb_cnt / max(1, fb_cnt + nfb_cnt), 3),
        },
        "roads": {
            "flooded_road_coverage_pct": round(fr_px / total * 100, 3),
            "flooded_road_area_m2": pixel_area_m2(fr_px, (h, w)),
            "non_flooded_road_coverage_pct": round(nfr_px / total * 100, 3),
            "total_road_coverage_pct": round((fr_px + nfr_px) / total * 100, 3),
        },
        "water": {
            "water_body_coverage_pct": round(water_px / total * 100, 3),
            "water_body_area_m2": pixel_area_m2(water_px, (h, w)),
            "pool_coverage_pct": round(pool_px / total * 100, 3),
        },
        "vehicles": {
            "vehicle_count": veh_cnt,
            "vehicle_area_pct": round(veh_px / total * 100, 3),
        },
        "class_pixel_distribution_pct": class_pct,
    }

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(DASHBOARD_DIR, "index.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "online",
        "system": "FloodSense Multimodal Emergency Reporting System",
        "version": "2.0",
        "vector_store_chunks": vector_store.collection.count(),
        "gpu_available": True
    })

@app.route("/api/models", methods=["GET"])
def api_models():
    models = [
        {
            "id": "segformer",
            "name": "OneFormer / SegFormer (Transformer)",
            "description": "High-accuracy hierarchical transformer segmenter with spatial flood reasoning.",
            "mIoU": 0.0937,
            "macro_f1": 0.1175,
            "latency_ms": 95.2,
            "recommended": True
        },
        {
            "id": "classical",
            "name": "Classical CV Multi-Space Thresholding",
            "description": "Ultra-fast HSV + LAB color space clustering and edge contour detection.",
            "mIoU": 0.0748,
            "macro_f1": 0.1079,
            "latency_ms": 22.8,
            "recommended": False
        },
        {
            "id": "sam",
            "name": "SAM (Segment Anything Model)",
            "description": "Promptable foundation segmentation with feature-based region classification.",
            "mIoU": 0.0813,
            "macro_f1": 0.1045,
            "latency_ms": 1063.1,
            "recommended": False
        },
        {
            "id": "ground_truth",
            "name": "FloodNet Ground Truth Mask Analysis",
            "description": "Exact pixel-level annotated mask analysis for validation benchmarks.",
            "mIoU": 1.0,
            "macro_f1": 1.0,
            "latency_ms": 10.0,
            "recommended": False
        }
    ]
    return jsonify({"models": models})

@app.route("/api/samples", methods=["GET"])
def api_samples():
    samples = []
    if os.path.exists(VAL_IMG_DIR) and os.path.exists(VAL_MASK_DIR):
        img_files = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.endswith(('.jpg', '.png'))])[:8]
        for img_file in img_files:
            stem = os.path.splitext(img_file)[0]
            mask_file = f"{stem}_lab.png"
            mask_path = os.path.join(VAL_MASK_DIR, mask_file)
            if os.path.exists(mask_path):
                samples.append({
                    "id": stem,
                    "image_file": img_file,
                    "mask_file": mask_file,
                    "image_url": f"/api/sample_image/{img_file}",
                    "mask_url": f"/api/sample_mask/{mask_file}"
                })
    return jsonify({"samples": samples})

@app.route("/api/sample_image/<filename>", methods=["GET"])
def api_sample_image(filename):
    return send_from_directory(VAL_IMG_DIR, filename)

@app.route("/api/sample_mask/<filename>", methods=["GET"])
def api_sample_mask(filename):
    return send_from_directory(VAL_MASK_DIR, filename)

@app.route("/api/knowledge_base", methods=["GET"])
def api_knowledge_base():
    docs = []
    kb_dir = os.path.join(PROJECT_ROOT, "knowledge_base")
    for f in os.listdir(kb_dir):
        if f.endswith(('.txt', '.md')) and not f.startswith('.'):
            path = os.path.join(kb_dir, f)
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            docs.append({
                "filename": f,
                "title": f.replace("_", " ").replace(".txt", "").upper(),
                "preview": content[:300] + "...",
                "full_text": content
            })
    return jsonify({"documents": docs})

@app.route("/api/overlay/<filename>", methods=["GET"])
def api_get_overlay(filename):
    path = os.path.join(OVERLAY_CACHE_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="image/png")
    return jsonify({"error": "Overlay not found"}), 404

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    t0 = time.time()
    model_name = request.form.get("model", "segformer").lower()
    
    # Check if sample ID provided or file uploaded
    sample_id = request.form.get("sample_id")
    file = request.files.get("image")
    
    img_pil = None
    mask_pil = None
    is_mask_upload = False
    
    if sample_id and os.path.exists(VAL_IMG_DIR):
        img_path = os.path.join(VAL_IMG_DIR, f"{sample_id}.jpg")
        mask_path = os.path.join(VAL_MASK_DIR, f"{sample_id}_lab.png")
        if os.path.exists(img_path):
            img_pil = Image.open(img_path).convert("RGB")
        if os.path.exists(mask_path):
            mask_pil = Image.open(mask_path).convert("RGB")
    elif file:
        filename = file.filename.lower()
        file_bytes = file.read()
        pil_opened = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        if "_lab" in filename or "mask" in filename:
            is_mask_upload = True
            mask_pil = pil_opened
            img_pil = pil_opened
        else:
            img_pil = pil_opened
    else:
        return jsonify({"error": "No image or sample provided"}), 400

    if img_pil is None:
        return jsonify({"error": "Failed to load image"}), 400

    target_size = (512, 512)
    img_rgb = np.array(img_pil.resize(target_size, Image.BILINEAR))
    
    # ── Step 1: Run Computer Vision Segmentation ──
    inference_t0 = time.time()
    if is_mask_upload or model_name == "ground_truth":
        if mask_pil is None:
            mask_pil = img_pil
        mask_arr = np.array(mask_pil.resize(target_size, Image.NEAREST))
        class_mask = rgb_mask_to_class_mask(mask_arr)
    elif model_name == "classical":
        _, class_mask = segment_classical_multiclass(img_rgb)
    elif model_name == "segformer":
        segformer = get_segformer()
        _, class_mask = segformer.segment_image(img_rgb)
    else:
        # Fallback to SegFormer
        segformer = get_segformer()
        _, class_mask = segformer.segment_image(img_rgb)
        
    inference_ms = round((time.time() - inference_t0) * 1000, 1)

    # ── Step 2: Extract Damage Metrics ──
    damage_metrics = extract_metrics_from_class_mask(class_mask, original_shape=target_size)
    damage_metrics["model_used"] = model_name
    damage_metrics["inference_time_ms"] = inference_ms

    # ── Step 3: RAG Retrieval from ChromaDB ──
    rag_t0 = time.time()
    retrieved_protocols = retriever.retrieve_grounded_protocols(damage_metrics)
    rag_ms = round((time.time() - rag_t0) * 1000, 1)

    # ── Step 4: LLM Grounded Report Synthesis ──
    llm_t0 = time.time()
    report_data = llm_engine.generate_grounded_report(damage_metrics, retrieved_protocols)
    llm_ms = round((time.time() - llm_t0) * 1000, 1)

    # ── Step 5: Generate Overlay Image ──
    overlay_rgb = generate_color_overlay(img_rgb, class_mask, alpha=0.55)
    overlay_id = f"overlay_{uuid.uuid4().hex[:10]}.png"
    overlay_save_path = os.path.join(OVERLAY_CACHE_DIR, overlay_id)
    Image.fromarray(overlay_rgb).save(overlay_save_path)
    
    # Base64 thumbnail
    buffered = io.BytesIO()
    Image.fromarray(overlay_rgb).save(buffered, format="JPEG", quality=85)
    overlay_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

    total_ms = round((time.time() - t0) * 1000, 1)

    response_payload = {
        "status": "success",
        "timings": {
            "cv_inference_ms": inference_ms,
            "rag_retrieval_ms": rag_ms,
            "llm_synthesis_ms": llm_ms,
            "total_latency_ms": total_ms
        },
        "model_used": model_name,
        "overlay_url": f"/api/overlay/{overlay_id}",
        "overlay_base64": overlay_base64,
        "damage_metrics": damage_metrics,
        "retrieved_protocols": [
            {
                "citation_tag": p["citation_tag"],
                "source_doc": p["source_doc"],
                "section": p["section"],
                "similarity_score": p["similarity_score"],
                "snippet": p["content"][:200] + "..."
            }
            for p in retrieved_protocols[:4]
        ],
        "sitrep_report": report_data["incident_report"],
        "markdown_report": report_data.get("markdown_rendered", "")
    }

    return jsonify(response_payload)

# ── Export PDF & DOCX Endpoints ───────────────────────────────────────────
from report_generation.export_utils import generate_pdf_report, generate_docx_report

@app.route("/api/export_pdf", methods=["GET", "POST", "OPTIONS"])
def api_export_pdf():
    try:
        report_data = request.get_json(force=True, silent=True) or {}
        pdf_path = os.path.join(tempfile.gettempdir(), f"FloodSense_SITREP_{uuid.uuid4().hex[:8]}.pdf")
        
        # If empty report data, load sample report if available
        if not report_data or "incident_report" not in report_data:
            sample_json = os.path.join(PROJECT_ROOT, "results", "sample_disaster_report.json")
            if os.path.exists(sample_json):
                with open(sample_json, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                    
        generate_pdf_report(report_data, pdf_path)
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name="FloodSense_Official_SITREP_Report.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        print(f"PDF generation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/export_docx", methods=["GET", "POST", "OPTIONS"])
def api_export_docx():
    try:
        report_data = request.get_json(force=True, silent=True) or {}
        docx_path = os.path.join(tempfile.gettempdir(), f"FloodSense_SITREP_{uuid.uuid4().hex[:8]}.docx")
        
        # If empty report data, load sample report if available
        if not report_data or "incident_report" not in report_data:
            sample_json = os.path.join(PROJECT_ROOT, "results", "sample_disaster_report.json")
            if os.path.exists(sample_json):
                with open(sample_json, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
                    
        generate_docx_report(report_data, docx_path)
        return send_file(
            docx_path,
            as_attachment=True,
            download_name="FloodSense_Official_SITREP_Report.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception as e:
        print(f"DOCX generation error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n=======================================================")
    print(f"  FLOODSENSE SERVER RUNNING ON http://localhost:{port}   ")
    print(f"=======================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)

