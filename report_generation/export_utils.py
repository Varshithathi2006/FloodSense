"""
FloodSense Document Exporter (PDF & DOCX)
========================================
Generates professional PDF reports (via ReportLab) and Microsoft Word documents (via python-docx)
from FloodSense disaster situation report JSON structures.
"""

import os
import json
import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# ──────────────────────────────────────────────────────────────────────────────
# 1. WORD (.DOCX) GENERATOR
# ──────────────────────────────────────────────────────────────────────────────
def set_cell_background(cell, fill_hex):
    """Sets background color of a Word table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def generate_docx_report(report_data: dict, output_path: str) -> str:
    """Generates a styled Microsoft Word (.docx) document from SITREP data."""
    rep = report_data.get("sitrep_report", report_data.get("incident_report", report_data))
    
    doc = Document()
    
    # Page Setup (1 inch margins)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    # Styles
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Times New Roman'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    
    # Header Banner
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("FLOODSENSE OFFICIAL DISASTER SITUATION REPORT")
    title_run.font.name = 'Times New Roman'
    title_run.font.size = Pt(20)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)  # Deep Navy
    
    subtitle_p = doc.add_paragraph()
    sub_run = subtitle_p.add_run("AUTOMATED MULTIMODAL DISASTER DISPATCH & INCIDENT TRIAGE (SITREP)")
    sub_run.font.size = Pt(10)
    sub_run.font.bold = True
    sub_run.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    
    doc.add_paragraph().paragraph_format.space_after = Pt(6)
    
    # Metadata Box Table
    meta_table = doc.add_table(rows=2, cols=3)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    
    sev_tier = rep.get("severity_classification", {}).get("severity_tier", "MODERATE")
    sev_score = rep.get("severity_classification", {}).get("severity_score", 0.0)
    
    cells = meta_table.rows[0].cells
    cells[0].text = "REPORT REFERENCE"
    cells[1].text = "TIMESTAMP (UTC)"
    cells[2].text = "SEVERITY TIER"
    
    cells_val = meta_table.rows[1].cells
    cells_val[0].text = str(rep.get("report_id", "FS-SITREP-2026"))
    cells_val[1].text = str(rep.get("timestamp", datetime.datetime.now().isoformat()))[:19].replace("T", " ")
    cells_val[2].text = f"{sev_tier} ({sev_score}/100)"
    
    # Format header cells
    for cell in cells:
        set_cell_background(cell, "1E293B")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
                
    for cell in cells_val:
        set_cell_background(cell, "F8FAFC")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(10)
                if "CRITICAL" in r.text or "SEVERE" in r.text:
                    r.font.color.rgb = RGBColor(0xDC, 0x26, 0x26)
                else:
                    r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
                    
    doc.add_paragraph().paragraph_format.space_after = Pt(12)
    
    # Section 1: Executive Summary
    h1 = doc.add_heading("1. Executive Summary & Incident Triage", level=1)
    h1.runs[0].font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    p_triage = doc.add_paragraph(rep.get("severity_classification", {}).get("triage_summary", ""))
    p_triage.paragraph_format.space_after = Pt(12)
    
    # Section 2: Damage Assessment Metrics Table
    h2 = doc.add_heading("2. Computer Vision Damage Assessment (Ground-Truth Verified)", level=1)
    h2.runs[0].font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    
    dmg = rep.get("damage_assessment", {})
    dmg_data = [
        ("Total Inundated Area (m²)", f"{dmg.get('total_flood_area_m2', 0.0):,.1f} sq. meters"),
        ("Overall Sector Flood Coverage", f"{dmg.get('flood_coverage_pct', 0.0)}%"),
        ("Flooded Building Structures", f"{dmg.get('flooded_buildings_count', 0)} units"),
        ("Safe / Dry Buildings", f"{dmg.get('non_flooded_buildings_count', 0)} units"),
        ("Road Inundation Percentage", f"{dmg.get('flooded_road_pct', 0.0)}%"),
        ("Submerged / Trapped Vehicles", f"{dmg.get('submerged_vehicles_count', 0)} vehicles"),
        ("Terrain & Environment Impact", str(dmg.get("vegetation_and_terrain_impact", "")))
    ]
    
    dmg_table = doc.add_table(rows=len(dmg_data)+1, cols=2)
    dmg_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    hdr_cells = dmg_table.rows[0].cells
    hdr_cells[0].text = "QUANTITATIVE METRIC"
    hdr_cells[1].text = "VERIFIED VALUE"
    for cell in hdr_cells:
        set_cell_background(cell, "0F172A")
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                
    for idx, (k, v) in enumerate(dmg_data):
        row_cells = dmg_table.rows[idx+1].cells
        row_cells[0].text = k
        row_cells[1].text = v
        bg_col = "F1F5F9" if idx % 2 == 0 else "FFFFFF"
        set_cell_background(row_cells[0], bg_col)
        set_cell_background(row_cells[1], bg_col)
        for c in row_cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9.5)
                    
    doc.add_paragraph().paragraph_format.space_after = Pt(14)
    
    # Section 3: Prioritized Action Protocols
    h3 = doc.add_heading("3. Prioritized Actionable Response Protocols", level=1)
    h3.runs[0].font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    
    actions = rep.get("prioritized_actions", [])
    for act in actions:
        p_act = doc.add_paragraph()
        r_prio = p_act.add_run(f"[{act.get('priority_level', 'P1-IMMEDIATE')}] ")
        r_prio.font.bold = True
        r_prio.font.color.rgb = RGBColor(0xDC, 0x26, 0x26) if "P1" in r_prio.text else RGBColor(0xD9, 0x77, 0x06)
        
        r_title = p_act.add_run(act.get("action_title", ""))
        r_title.font.bold = True
        r_title.font.size = Pt(11)
        r_title.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        
        p_det = doc.add_paragraph()
        p_det.paragraph_format.left_indent = Inches(0.2)
        p_det.add_run(f"Details: {act.get('action_details', '')}\n")
        
        r_res = p_det.add_run(f"Resources Needed: {act.get('resource_deployment', '')}\n")
        r_res.font.bold = True
        
        r_cite = p_det.add_run(f"Protocol Citation: {act.get('citation', '')}")
        r_cite.font.italic = True
        r_cite.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
        p_det.paragraph_format.space_after = Pt(8)
        
    # Section 4: Lifelines & Logistics
    h4 = doc.add_heading("4. Critical Lifelines & Logistics", level=1)
    h4.runs[0].font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    
    life = rep.get("logistics_and_lifelines", {})
    doc.add_paragraph(f"• Drinking Water & Sanitation: {life.get('drinking_water_sanitation', '')}", style='List Bullet')
    doc.add_paragraph(f"• Electrical Power Grid Safety: {life.get('power_grid_safety', '')}", style='List Bullet')
    doc.add_paragraph(f"• Emergency Traffic & Transit Routing: {life.get('traffic_and_access_routes', '')}", style='List Bullet')
    
    doc.add_paragraph().paragraph_format.space_after = Pt(14)
    
    # Section 5: Traceable Citations
    h5 = doc.add_heading("5. Traceable Regulatory Citations", level=1)
    h5.runs[0].font.color.rgb = RGBColor(0x1E, 0x3A, 0x8A)
    
    cites = rep.get("retrieved_citations", [])
    for c in cites:
        doc.add_paragraph(f"{c}", style='List Bullet')
        
    doc.save(output_path)
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# 2. PDF GENERATOR (REPORTLAB)
# ──────────────────────────────────────────────────────────────────────────────
def generate_pdf_report(report_data: dict, output_path: str) -> str:
    """Generates a professional PDF document from SITREP data using ReportLab."""
    rep = report_data.get("sitrep_report", report_data.get("incident_report", report_data))
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_navy = colors.HexColor("#1E3A8A")
    c_dark = colors.HexColor("#0F172A")
    c_blue = colors.HexColor("#2563EB")
    c_slate = colors.HexColor("#64748B")
    c_red = colors.HexColor("#DC2626")
    c_bg_light = colors.HexColor("#F8FAFC")
    
    # Custom Paragraph Styles
    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Times-Bold',
        fontSize=18,
        leading=22,
        textColor=c_navy,
        spaceAfter=4
    )
    style_subtitle = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9,
        leading=11,
        textColor=c_slate,
        spaceAfter=12
    )
    style_h1 = ParagraphStyle(
        'DocH1',
        parent=styles['Heading2'],
        fontName='Times-Bold',
        fontSize=12,
        leading=15,
        textColor=c_navy,
        spaceBefore=12,
        spaceAfter=6
    )
    style_body = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13,
        textColor=c_dark,
        spaceAfter=6
    )
    style_bullet = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9,
        leading=12,
        textColor=c_dark,
        leftIndent=12,
        spaceAfter=4
    )
    style_table_hdr = ParagraphStyle(
        'TableHdr',
        fontName='Times-Bold',
        fontSize=8.5,
        leading=10,
        textColor=colors.white
    )
    style_table_cell = ParagraphStyle(
        'TableCell',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        textColor=c_dark
    )
    
    elements = []
    
    # Title & Subtitle
    elements.append(Paragraph("FLOODSENSE OFFICIAL DISASTER SITUATION REPORT", style_title))
    elements.append(Paragraph("AUTOMATED MULTIMODAL DISASTER DISPATCH & INCIDENT TRIAGE (SITREP)", style_subtitle))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=c_navy, spaceAfter=10))
    
    # Metadata Header Table
    sev_tier = rep.get("severity_classification", {}).get("severity_tier", "MODERATE")
    sev_score = rep.get("severity_classification", {}).get("severity_score", 0.0)
    
    meta_data = [
        [
            Paragraph("<b>REPORT REFERENCE</b>", style_table_hdr),
            Paragraph("<b>TIMESTAMP (UTC)</b>", style_table_hdr),
            Paragraph("<b>SEVERITY TIER</b>", style_table_hdr)
        ],
        [
            Paragraph(str(rep.get("report_id", "FS-SITREP-2026")), style_table_cell),
            Paragraph(str(rep.get("timestamp", datetime.datetime.now().isoformat()))[:19].replace("T", " "), style_table_cell),
            Paragraph(f"<b>{sev_tier} ({sev_score}/100)</b>", style_table_cell)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[2.2*inch, 2.8*inch, 2.2*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_dark),
        ('BACKGROUND', (0,1), (-1,1), c_bg_light),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))
    
    # 1. Executive Summary
    elements.append(Paragraph("1. Executive Summary & Incident Triage", style_h1))
    triage_txt = rep.get("severity_classification", {}).get("triage_summary", "")
    elements.append(Paragraph(triage_txt, style_body))
    
    # 2. Damage Assessment Table
    elements.append(Paragraph("2. Computer Vision Damage Assessment (Ground-Truth Verified)", style_h1))
    dmg = rep.get("damage_assessment", {})
    dmg_rows = [
        [Paragraph("<b>QUANTITATIVE METRIC</b>", style_table_hdr), Paragraph("<b>VERIFIED VALUE</b>", style_table_hdr)],
        [Paragraph("Total Inundated Area (m²)", style_table_cell), Paragraph(f"{dmg.get('total_flood_area_m2', 0.0):,.1f} sq. meters", style_table_cell)],
        [Paragraph("Overall Sector Flood Coverage", style_table_cell), Paragraph(f"{dmg.get('flood_coverage_pct', 0.0)}%", style_table_cell)],
        [Paragraph("Flooded Building Structures", style_table_cell), Paragraph(f"{dmg.get('flooded_buildings_count', 0)} units", style_table_cell)],
        [Paragraph("Safe / Dry Buildings", style_table_cell), Paragraph(f"{dmg.get('non_flooded_buildings_count', 0)} units", style_table_cell)],
        [Paragraph("Road Inundation Percentage", style_table_cell), Paragraph(f"{dmg.get('flooded_road_pct', 0.0)}%", style_table_cell)],
        [Paragraph("Submerged / Trapped Vehicles", style_table_cell), Paragraph(f"{dmg.get('submerged_vehicles_count', 0)} vehicles", style_table_cell)],
        [Paragraph("Terrain & Environment Impact", style_table_cell), Paragraph(str(dmg.get("vegetation_and_terrain_impact", "")), style_table_cell)]
    ]
    dmg_table = Table(dmg_rows, colWidths=[3.2*inch, 4.0*inch])
    dmg_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_dark),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E1")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [c_bg_light, colors.white]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(dmg_table)
    elements.append(Spacer(1, 10))
    
    # 3. Action Protocols
    elements.append(Paragraph("3. Prioritized Actionable Response Protocols", style_h1))
    actions = rep.get("prioritized_actions", [])
    for act in actions:
        prio = act.get("priority_level", "P1-IMMEDIATE")
        prio_color = "#DC2626" if "P1" in prio else "#D97706"
        act_html = f"<b><font color='{prio_color}'>[{prio}]</font> {act.get('action_title', '')}</b><br/>" \
                   f"• <b>Details:</b> {act.get('action_details', '')}<br/>" \
                   f"• <b>Resources:</b> {act.get('resource_deployment', '')}<br/>" \
                   f"• <b>Citation:</b> <font color='#2563EB'><i>{act.get('citation', '')}</i></font>"
        elements.append(Paragraph(act_html, style_body))
        elements.append(Spacer(1, 4))
        
    elements.append(Spacer(1, 6))
    
    # 4. Lifelines & Logistics
    elements.append(Paragraph("4. Critical Lifelines & Logistics", style_h1))
    life = rep.get("logistics_and_lifelines", {})
    elements.append(Paragraph(f"• <b>Drinking Water & Sanitation:</b> {life.get('drinking_water_sanitation', '')}", style_bullet))
    elements.append(Paragraph(f"• <b>Electrical Power Safety:</b> {life.get('power_grid_safety', '')}", style_bullet))
    elements.append(Paragraph(f"• <b>Emergency Traffic Routing:</b> {life.get('traffic_and_access_routes', '')}", style_bullet))
    
    elements.append(Spacer(1, 8))
    
    # 5. Citations
    elements.append(Paragraph("5. Traceable Regulatory Citations", style_h1))
    cites = rep.get("retrieved_citations", [])
    for c in cites:
        elements.append(Paragraph(f"<code>{c}</code>", style_bullet))
        
    doc.build(elements)
    return output_path
