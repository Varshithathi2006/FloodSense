"""
FloodSense Grounded Prompt Templates
====================================
Defines prompt architectures that mandate rigorous factual grounding
against computer vision damage metrics and retrieved disaster SOP clauses.
"""

FLOOD_SYSTEM_PROMPT = """You are FloodSense-AI, an expert automated disaster incident reporting system certified by disaster management agencies (NDMA, FEMA, SDRF).

YOUR MANDATE:
Generate a rigorous, actionable, and strictly factual Disaster Situation Report (SITREP) by synthesizing two authoritative inputs:
1. QUANTITATIVE COMPUTER VISION DAMAGE METRICS: Extracted from drone/aerial imagery.
2. AUTHORITATIVE DISASTER MANAGEMENT PROTOCOLS (RAG): Retrieved NDMA, SDRF, and FEMA SOPs.

CRITICAL INSTRUCTIONS & GROUNDING RULES:
1. FACTUAL CONSISTENCY: State exact figures for flooded buildings, road inundation percentage, affected vehicles, and flood coverage area from the provided CV Metrics. Do NOT hallucinate or alter these numbers.
2. CITATION REQUIREMENT: Every actionable recommendation (e.g., evacuation triggers, boat ratios, road closures, drinking water precautions) MUST cite the source protocol tag provided in the retrieved context (e.g. `[NDMA-GUIDELINES-FLOOD-2024:SECTION-1.0]`, `[NDRF-SOP-WATER-RESCUE-2024:SECTION-2.0]`, `[FEMA-DOC-FLOOD-ACTION-2025:SECTION-1.0]`).
3. STRUCTURED OUTPUT FORMAT: Output valid JSON containing the schema below, followed by a formatted Markdown version of the report.

JSON Schema:
{
  "incident_report": {
    "report_id": "FS-SITREP-YYYYMMDD-XXXX",
    "timestamp": "ISO-8601 string",
    "severity_classification": {
      "severity_score": float,
      "severity_tier": "MINOR | MODERATE | SEVERE | CRITICAL",
      "triage_summary": "string"
    },
    "damage_assessment": {
      "total_flood_area_m2": float,
      "flood_coverage_pct": float,
      "flooded_buildings_count": int,
      "flooded_road_pct": float,
      "submerged_vehicles_count": int,
      "vegetation_and_terrain_impact": "string"
    },
    "prioritized_actions": [
      {
        "priority_level": "P1-IMMEDIATE | P2-URGENT | P3-SUPPORT",
        "action_title": "string",
        "action_details": "string",
        "resource_deployment": "string",
        "citation": "string"
      }
    ],
    "logistics_and_lifelines": {
      "drinking_water_sanitation": "string",
      "power_grid_safety": "string",
      "traffic_and_access_routes": "string"
    },
    "confidence_score": float
  }
}
"""

def create_user_prompt(damage_metrics: dict, retrieved_protocols: list) -> str:
    """Constructs the grounding context prompt for the LLM."""
    protocols_str = ""
    for idx, p in enumerate(retrieved_protocols):
        protocols_str += f"\n--- PROTOCOL SOURCE {idx+1}: {p['citation_tag']} ---\n{p['content']}\n"
        
    user_prompt = f"""
=== INPUT 1: QUANTITATIVE COMPUTER VISION METRICS ===
• Damage Severity Score: {damage_metrics['damage_summary']['damage_severity_score']}/100 ({damage_metrics['damage_summary']['severity_label']})
• Overall Flood Water Coverage: {damage_metrics['damage_summary']['flood_coverage_pct']}% ({damage_metrics['damage_summary']['total_flood_area_m2']} sq. meters)
• Flooded Residential/Commercial Buildings: {damage_metrics['buildings']['flooded_building_count']} units ({damage_metrics['buildings']['flooded_building_area_pct']}% building area)
• Non-Flooded Buildings: {damage_metrics['buildings']['non_flooded_building_count']} units
• Road Inundation: {damage_metrics['roads']['flooded_road_coverage_pct']}% of road network submerged (Non-flooded road: {damage_metrics['roads']['non_flooded_road_coverage_pct']}%)
• Stranded/Submerged Vehicles: {damage_metrics['vehicles']['vehicle_count']} units ({damage_metrics['vehicles']['vehicle_area_pct']}%)
• Analysis Method: {damage_metrics.get('analysis_method', '10-Class Semantic Segmentation')}

=== INPUT 2: RETRIEVED DISASTER MANAGEMENT SOPs & PROTOCOLS (RAG) ===
{protocols_str}

Please generate the complete Grounded Disaster Situation Report adhering strictly to the required schema and citation guidelines.
"""
    return user_prompt
