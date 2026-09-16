"""
FloodSense RAG Retriever & Dynamic Query Engine
================================================
Constructs targeted semantic retrieval queries from computer vision damage features
and retrieves relevant India-specific disaster management protocols with traceable citations.

Sources grounded:
  - NDMA (National Disaster Management Authority) Flood Guidelines 2024
  - NDRF/SDRF SOP for Water Rescue Operations 2024
  - Ministry of Jal Shakti / CWC Flood Stage & State Protocols 2024
  - Historical Indian Flood Case Studies (Chennai 2015, Kerala 2018, Assam 2022, etc.)
"""

import os
from .vector_store import DisasterProtocolVectorStore

class FloodProtocolRetriever:
    def __init__(self, vector_store: DisasterProtocolVectorStore = None):
        if vector_store is None:
            self.vector_store = DisasterProtocolVectorStore()
        else:
            self.vector_store = vector_store

    def build_queries_from_damage_metrics(self, damage_metrics: dict) -> list:
        """
        Dynamically derives targeted domain queries from extracted computer vision metrics.
        """
        queries = []
        
        summary = damage_metrics.get("damage_summary", {})
        severity = summary.get("severity_label", "MODERATE")
        flood_pct = summary.get("flood_coverage_pct", 0.0)
        
        buildings = damage_metrics.get("buildings", {})
        flooded_bldg_cnt = buildings.get("flooded_building_count", 0)
        
        roads = damage_metrics.get("roads", {})
        flooded_road_pct = roads.get("flooded_road_coverage_pct", 0.0)
        
        vehicles = damage_metrics.get("vehicles", {})
        veh_cnt = vehicles.get("vehicle_count", 0)
        
        # Query 1: Overall incident triage & severity trigger
        queries.append(f"NDMA evacuation trigger, SDRF mobilization and NDRF battalion deployment for {severity} flood severity with {flood_pct:.1f}% flood coverage in India.")
        
        # Query 2: Structural and building inundation actions
        if flooded_bldg_cnt > 0:
            queries.append(f"NDRF search and rescue, rooftop vertical evacuation and IRB inflatable rescue boat ratio for {flooded_bldg_cnt} flooded residential buildings per NDMA DM Act 2005.")
        else:
            queries.append("Structural foundation flood inspection, PWD building assessment and DISCOM power grid electrical shutoff protocols in India.")
            
        # Query 3: Road blockage and vehicle entrapment
        if flooded_road_pct > 15.0 or veh_cnt > 0:
            queries.append(f"PWD/NHAI traffic restriction, SDRF high-water tactical vehicle transit and submerged vehicle occupant extraction protocol with {flooded_road_pct:.1f}% road inundation and {veh_cnt} submerged vehicles India.")
            
        # Query 4: Drinking water contamination and disease surveillance India
        queries.append("PHED boil water advisory, Jal Jeevan Mission water contamination, Leptospirosis Cholera disease surveillance IDSP post-flood India.")
        
        # Query 5: Post-flood compensation and crop damage India
        if flood_pct > 10.0:
            queries.append("SDRF compensation norms flooded house crop damage Khasra Girdawari PM Fasal Bima Yojana flood victims India.")
        
        return queries

    def retrieve_grounded_protocols(self, damage_metrics: dict, top_k_per_query: int = 2) -> list:
        """
        Executes multi-query retrieval and dedupes relevant protocol clauses.
        """
        queries = self.build_queries_from_damage_metrics(damage_metrics)
        all_results = []
        seen_contents = set()
        
        for q in queries:
            results = self.vector_store.query(q, top_k=top_k_per_query)
            for r in results:
                content_snip = r["content"][:100]
                if content_snip not in seen_contents:
                    seen_contents.add(content_snip)
                    # Format standard citation tag
                    doc_tag = r["source_doc"].replace(".txt", "").replace(".md", "").upper()
                    sec_tag = r["section"].replace("[", "").replace("]", "")
                    r["citation_tag"] = f"[{doc_tag}:{sec_tag}]"
                    all_results.append(r)
                    
        # Sort by similarity score descending
        all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return all_results

if __name__ == "__main__":
    # Test with sample damage metrics
    sample_metrics = {
        "damage_summary": {
            "damage_severity_score": 18.5,
            "severity_label": "CRITICAL",
            "flood_coverage_pct": 42.3,
            "total_flood_area_m2": 15400.0
        },
        "buildings": {
            "flooded_building_count": 5,
            "flooded_building_area_pct": 14.2
        },
        "roads": {
            "flooded_road_coverage_pct": 38.6
        },
        "vehicles": {
            "vehicle_count": 3
        }
    }
    
    retriever = FloodProtocolRetriever()
    protocols = retriever.retrieve_grounded_protocols(sample_metrics)
    print(f"Retrieved {len(protocols)} unique grounded protocol clauses:\n")
    for idx, p in enumerate(protocols):
        print(f"{idx+1}. Citation: {p['citation_tag']} (Score: {p['similarity_score']})")
        print(f"   {p['content'][:140]}...\n")
