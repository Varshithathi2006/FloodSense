"""
FloodSense Master Report Generation Pipeline
=============================================
Orchestrates end-to-end flow:
1. Load image / mask and run multi-class damage extraction
2. Dynamically formulate RAG queries and retrieve NDMA/FEMA protocol clauses from ChromaDB
3. Synthesize CV features + RAG context into a grounded disaster situation report (JSON + Markdown)
"""

import os
import sys
import json
import time
from PIL import Image
import numpy as np

# Ensure root is in path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from damage_extraction.damage_extractor import extract_damage_info
from rag_pipeline.vector_store import DisasterProtocolVectorStore
from rag_pipeline.retriever import FloodProtocolRetriever
from report_generation.llm_engine import FloodLLMEngine

class FloodReportingPipeline:
    def __init__(self):
        print("Initializing FloodSense Pipeline components...")
        self.vector_store = DisasterProtocolVectorStore()
        # Build index if not present
        if self.vector_store.collection.count() == 0:
            print("Vector collection empty. Building knowledge base index...")
            self.vector_store.build_or_update_index()
            
        self.retriever = FloodProtocolRetriever(vector_store=self.vector_store)
        self.llm_engine = FloodLLMEngine()

    def generate_report_from_mask(self, mask_path: str, image_path: str = None) -> dict:
        """
        Runs complete damage extraction -> RAG -> LLM synthesis pipeline from a mask file.
        """
        t0 = time.time()
        print(f"\n[1/3] Extracting multi-class damage metrics from {os.path.basename(mask_path)}...")
        damage_metrics = extract_damage_info(mask_path, image_path)
        
        print("[2/3] Retrieving grounded NDMA/FEMA protocol clauses via ChromaDB RAG...")
        retrieved_protocols = self.retriever.retrieve_grounded_protocols(damage_metrics)
        print(f"      -> Retrieved {len(retrieved_protocols)} grounded protocol clauses.")
        
        print("[3/3] Generating grounded disaster situation report via LLM Synthesis Engine...")
        report = self.llm_engine.generate_grounded_report(damage_metrics, retrieved_protocols)
        
        total_time_ms = round((time.time() - t0) * 1000, 2)
        report["total_pipeline_time_ms"] = total_time_ms
        report["damage_metrics_extracted"] = damage_metrics
        report["retrieved_protocols"] = retrieved_protocols
        
        print(f"Pipeline completed successfully in {total_time_ms} ms!\n")
        return report

def run_test_pipeline():
    pipeline = FloodReportingPipeline()
    
    # Locate a sample mask from FloodNet validation set
    val_mask_dir = os.path.join(ROOT_DIR, "FloodNet", "ColorMasks-FloodNetv1", "ColorMasks-ValSet")
    mask_files = [f for f in os.listdir(val_mask_dir) if f.endswith(".png")]
    if not mask_files:
        print("No validation masks found for testing.")
        return
        
    sample_mask_path = os.path.join(val_mask_dir, mask_files[0])
    sample_img_stem = mask_files[0].replace("_lab.png", "")
    sample_img_path = os.path.join(ROOT_DIR, "FloodNet", "FloodNet-Supervised_v1.0", "val", "val-org-img", f"{sample_img_stem}.jpg")
    
    report = pipeline.generate_report_from_mask(sample_mask_path, sample_img_path if os.path.exists(sample_img_path) else None)
    
    # Save output sample
    os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)
    out_json = os.path.join(ROOT_DIR, "results", "sample_disaster_report.json")
    out_md = os.path.join(ROOT_DIR, "results", "sample_disaster_report.md")
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report.get("markdown_rendered", ""))
        
    print(f"Sample JSON report saved to: {out_json}")
    print(f"Sample Markdown SITREP saved to: {out_md}\n")
    print("--- SITREP Preview (First 500 chars) ---")
    print(report.get("markdown_rendered", "")[:500].encode("ascii", "replace").decode("ascii"))

if __name__ == "__main__":
    run_test_pipeline()
