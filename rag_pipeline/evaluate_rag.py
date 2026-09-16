"""
FloodSense RAG Evaluation Module
=================================
Evaluates Precision@K, Recall@K, and Mean Reciprocal Rank (MRR) for the disaster protocol RAG pipeline
across curated disaster management query benchmarks.
"""

import os
import json
import numpy as np
from .vector_store import DisasterProtocolVectorStore

BENCHMARK_SCENARIOS = [
    {
        "query": "NDMA evacuation triggers and severity classification for severe and critical flood levels",
        "relevant_docs": ["ndma_flood_guidelines.txt"],
        "relevant_keywords": ["Severity Classification", "Critical", "Severe", "evacuation", "NDMA"]
    },
    {
        "query": "Standard operating procedure for inflatable rescue boat deployment ratio per flooded building",
        "relevant_docs": ["ndma_flood_guidelines.txt", "ndrf_sdrf_sop.txt"],
        "relevant_keywords": ["Inflatable Rescue Boat", "IRB", "flooded residential structures", "deployment ratio"]
    },
    {
        "query": "Safe water depth limits for road closure and high-water vehicle transit",
        "relevant_docs": ["ndma_flood_guidelines.txt", "fema_flood_protocols.txt"],
        "relevant_keywords": ["15 cm", "30 cm", "barricaded", "high-clearance", "passable"]
    },
    {
        "query": "Rescue extraction protocol for submerged and trapped passenger vehicles",
        "relevant_docs": ["ndrf_sdrf_sop.txt"],
        "relevant_keywords": ["Submerged Vehicle", "door sill", "window punch", "hydraulic pressure"]
    },
    {
        "query": "Drinking water contamination protocols, boil water notice, and mobile water purification units",
        "relevant_docs": ["fema_flood_protocols.txt", "ndma_flood_guidelines.txt"],
        "relevant_keywords": ["boil water", "purification units", "clean drinking water", "halogen tablets"]
    },
    {
        "query": "Priority triage hierarchy for stranded rooftop residents vs secure multi-story buildings",
        "relevant_docs": ["ndrf_sdrf_sop.txt"],
        "relevant_keywords": ["Priority 1", "rooftop", "medical attention", "Triage Hierarchy"]
    },
    {
        "query": "Electrical power grid isolation timeline and transformer hazard in flooded sectors",
        "relevant_docs": ["ndma_flood_guidelines.txt", "ndrf_sdrf_sop.txt"],
        "relevant_keywords": ["electric distribution", "grid trip", "30 minutes", "electrocution", "powerline"]
    },
    {
        "query": "Urban drainage bottleneck clearing and culvert excavator deployment lessons from Chennai floods",
        "relevant_docs": ["historical_flood_cases.txt"],
        "relevant_keywords": ["Chennai", "culverts", "excavators", "reconnaissance", "plastic waste"]
    },
    {
        "query": "High velocity swift water current rescue boat operations with outboard motor",
        "relevant_docs": ["ndrf_sdrf_sop.txt", "historical_flood_cases.txt"],
        "relevant_keywords": ["Outboard Motor", "OBM", "swift current", "tethered safety lines"]
    },
    {
        "query": "Post flood building foundation re-entry safety clearance inspection checklist",
        "relevant_docs": ["fema_flood_protocols.txt", "ndma_flood_guidelines.txt"],
        "relevant_keywords": ["foundation walls", "settling", "cracking", "licensed inspector", "mold spores"]
    }
]

def evaluate_retrieval_performance(top_k_values=[1, 3, 5], results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    store = DisasterProtocolVectorStore()
    
    # Ensure index exists
    store.build_or_update_index()
    
    metrics = {f"Precision@{k}": [] for k in top_k_values}
    metrics.update({f"Recall@{k}": [] for k in top_k_values})
    mrr_list = []
    
    scenario_details = []
    
    for item in BENCHMARK_SCENARIOS:
        q = item["query"]
        expected_docs = set(item["relevant_docs"])
        expected_keywords = [kw.lower() for kw in item["relevant_keywords"]]
        
        max_k = max(top_k_values)
        retrieved = store.query(q, top_k=max_k)
        
        # Determine relevance of each retrieved item
        relevance_flags = []
        for r in retrieved:
            src = r["source_doc"]
            content = r["content"].lower()
            
            doc_match = src in expected_docs
            keyword_match = any(kw in content for kw in expected_keywords)
            
            is_relevant = doc_match and keyword_match
            relevance_flags.append(is_relevant)
            
        # Compute Precision@K and Recall@K
        p_at_k = {}
        r_at_k = {}
        for k in top_k_values:
            k_flags = relevance_flags[:k]
            precision = sum(k_flags) / k if k > 0 else 0.0
            # Recall: retrieved relevant / min(total possible relevant in KB, k)
            recall = 1.0 if any(k_flags) else 0.0
            
            metrics[f"Precision@{k}"].append(precision)
            metrics[f"Recall@{k}"].append(recall)
            p_at_k[f"P@{k}"] = round(precision, 3)
            r_at_k[f"R@{k}"] = round(recall, 3)
            
        # MRR (Mean Reciprocal Rank)
        first_rel = next((i + 1 for i, rel in enumerate(relevance_flags) if rel), None)
        reciprocal_rank = 1.0 / first_rel if first_rel is not None else 0.0
        mrr_list.append(reciprocal_rank)
        
        scenario_details.append({
            "query": q,
            "precision": p_at_k,
            "recall": r_at_k,
            "mrr": round(reciprocal_rank, 3),
            "top_retrieved_source": retrieved[0]["source_doc"] if retrieved else "None",
            "top_similarity": retrieved[0]["similarity_score"] if retrieved else 0.0
        })
        
    summary = {
        "benchmark_query_count": len(BENCHMARK_SCENARIOS),
        "mean_reciprocal_rank_MRR": float(np.mean(mrr_list)),
        "precision_at_k": {f"Precision@{k}": float(np.mean(metrics[f"Precision@{k}"])) for k in top_k_values},
        "recall_at_k": {f"Recall@{k}": float(np.mean(metrics[f"Recall@{k}"])) for k in top_k_values},
        "details": scenario_details
    }
    
    out_path = os.path.join(results_dir, "rag_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n=========================================================================================")
    print("                    FLOODSENSE RAG RETRIEVAL EVALUATION RESULTS                          ")
    print("=========================================================================================\n")
    print(f"Total Test Scenarios: {summary['benchmark_query_count']}")
    print(f"Mean Reciprocal Rank (MRR): {summary['mean_reciprocal_rank_MRR']:.4f}")
    for k in top_k_values:
        print(f"Mean Precision@{k}: {summary['precision_at_k'][f'Precision@{k}']:.4f}  |  Recall@{k}: {summary['recall_at_k'][f'Recall@{k}']:.4f}")
    print(f"\nDetailed evaluation report saved to {out_path}\n")
    return summary

if __name__ == "__main__":
    evaluate_retrieval_performance()
