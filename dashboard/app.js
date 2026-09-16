/**
 * FloodSense Dashboard Interactive Logic
 * Handles model selection, drone photo analysis, live overlay rendering,
 * ChromaDB RAG citations, and Grounded SITREP exports.
 */

// ── State ──────────────────────────────────────────────────────────────────
let activeReportData = null;
let currentRetrievedProtocols = [];

// ── Sidebar ───────────────────────────────────────────────────────────────
function setSidebarOpen(isOpen) {
  const shell = document.querySelector(".app-shell");
  if (!shell) return;

  shell.classList.toggle("sidebar-open", isOpen);
  shell.classList.toggle("sidebar-collapsed", !isOpen);
  document.getElementById("sidebar-open")?.setAttribute("aria-expanded", String(isOpen));
}

// ── Navigation ─────────────────────────────────────────────────────────────
function navigate(sectionId) {
  document.querySelectorAll('.page-section').forEach(sec => sec.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(btn => btn.classList.remove('active'));

  const targetSection = document.getElementById(`section-${sectionId}`);
  const targetNav = document.getElementById(`nav-${sectionId}`);

  if (targetSection) targetSection.classList.add('active');
  if (targetNav) targetNav.classList.add('active');

  const titleMap = {
    live: "Live Incident Reporting & RAG-LLM Synthesis",
    rag: "Disaster Knowledge Base & Vector Index",
    models: "5-Model Computer Vision Multi-Class Benchmark",
    damage: "Damage Information Extractor Engine",
    dataset: "FloodNet Dataset Sufficiency Analysis"
  };

  const topbarTitle = document.getElementById("topbar-title");
  if (topbarTitle && titleMap[sectionId]) {
    topbarTitle.textContent = titleMap[sectionId];
  }
}

// ── App Init ───────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadSampleScenarios();
  loadKnowledgeBaseDocs();
  checkServerHealth();
});

// ── Event Listeners ────────────────────────────────────────────────────────
function setupEventListeners() {
  const sidebarOpen = document.getElementById("sidebar-open");
  const sidebarClose = document.getElementById("sidebar-close");

  sidebarOpen?.addEventListener("click", () => setSidebarOpen(true));
  sidebarClose?.addEventListener("click", () => setSidebarOpen(false));

  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const opacitySlider = document.getElementById("opacity-slider");
  const opacityVal = document.getElementById("opacity-val");

  if (dropZone && fileInput) {
    dropZone.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropZone.classList.add("drag-over");
    });
    
    dropZone.addEventListener("dragleave", () => {
      dropZone.classList.remove("drag-over");
    });
    
    dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropZone.classList.remove("drag-over");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  if (opacitySlider && opacityVal) {
    opacitySlider.addEventListener("input", (e) => {
      const val = e.target.value;
      opacityVal.textContent = `${val}%`;
      const overlayImg = document.getElementById("overlay-preview");
      if (overlayImg) {
        overlayImg.style.opacity = (val / 100).toString();
      }
    });
  }
}

// ── Server Health Check ────────────────────────────────────────────────────
async function checkServerHealth() {
  const badge = document.getElementById("server-badge");
  try {
    const res = await fetch("/api/status");
    if (res.ok) {
      const data = await res.json();
      if (badge) {
        badge.className = "status-badge green";
        badge.innerHTML = `<span class="status-dot"></span>Server: Online (${data.vector_store_chunks} chunks)`;
      }
    }
  } catch (err) {
    if (badge) {
      badge.className = "status-badge amber";
      badge.innerHTML = `<span class="status-dot"></span>Server: Offline (Demo Mode)`;
    }
  }
}

// ── Load Sample Scenarios ──────────────────────────────────────────────────
async function loadSampleScenarios() {
  const container = document.getElementById("sample-buttons");
  if (!container) return;

  try {
    const res = await fetch("/api/samples");
    if (res.ok) {
      const data = await res.json();
      container.innerHTML = "";
      if (data.samples && data.samples.length > 0) {
        data.samples.slice(0, 5).forEach((s, idx) => {
          const btn = document.createElement("button");
          btn.className = "sample-pill-btn";
          btn.textContent = `Scene ${idx + 1} (${s.id})`;
          btn.onclick = () => loadAndAnalyzeSample(s.id);
          container.appendChild(btn);
        });
      } else {
        container.innerHTML = `<span class="empty-cite-text">No pre-loaded validation samples found.</span>`;
      }
    }
  } catch (err) {
    container.innerHTML = `<span class="empty-cite-text">Sample loader offline. Upload an image manually.</span>`;
  }
}

// ── Load Knowledge Base Documents ──────────────────────────────────────────
async function loadKnowledgeBaseDocs() {
  const grid = document.getElementById("kb-doc-grid");
  if (!grid) return;

  try {
    const res = await fetch("/api/knowledge_base");
    if (res.ok) {
      const data = await res.json();
      grid.innerHTML = "";
      data.documents.forEach(doc => {
        const card = document.createElement("div");
        card.className = "doc-card";
        card.innerHTML = `
          <div class="doc-card-title">📜 ${doc.title}</div>
          <div class="doc-card-preview">${doc.preview}</div>
        `;
        grid.appendChild(card);
      });
    }
  } catch (err) {
    grid.innerHTML = `<div class="doc-card"><p>Knowledge base explorer offline.</p></div>`;
  }
}

// ── Analyze Uploaded File ──────────────────────────────────────────────────
async function handleFileUpload(file) {
  showLoading("Reading image and uploading to FloodSense Ops Server...");

  // Show raw preview
  const reader = new FileReader();
  reader.onload = (e) => {
    const rawImg = document.getElementById("raw-preview");
    const dropPrompt = document.getElementById("drop-prompt");
    const previewWrapper = document.getElementById("preview-wrapper");
    if (rawImg && dropPrompt && previewWrapper) {
      rawImg.src = e.target.result;
      dropPrompt.style.display = "none";
      previewWrapper.style.display = "block";
    }
  };
  reader.readAsDataURL(file);

  const model = document.getElementById("model-select").value || "segformer";
  const formData = new FormData();
  formData.append("image", file);
  formData.append("model", model);

  await executeAnalysisRequest(formData);
}

// ── Analyze Sample Scene ───────────────────────────────────────────────────
async function loadAndAnalyzeSample(sampleId) {
  showLoading(`Loading Flight Scene ${sampleId} & executing multi-class inference...`);
  
  const rawImg = document.getElementById("raw-preview");
  const dropPrompt = document.getElementById("drop-prompt");
  const previewWrapper = document.getElementById("preview-wrapper");
  if (rawImg && dropPrompt && previewWrapper) {
    rawImg.src = `/api/sample_image/${sampleId}.jpg`;
    dropPrompt.style.display = "none";
    previewWrapper.style.display = "block";
  }

  const model = document.getElementById("model-select").value || "segformer";
  const formData = new FormData();
  formData.append("sample_id", sampleId);
  formData.append("model", model);

  await executeAnalysisRequest(formData);
}

// ── Send Analysis Request to Backend ───────────────────────────────────────
async function executeAnalysisRequest(formData) {
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error(`Server returned error ${res.status}`);
    }

    const data = await res.json();
    renderAnalysisResults(data);
  } catch (err) {
    console.error("Analysis failed:", err);
    alert(`Analysis request failed: ${err.message}. Please check that the server is running on http://localhost:5050`);
  } finally {
    hideLoading();
  }
}

// ── Render Analysis Results ────────────────────────────────────────────────
function renderAnalysisResults(data) {
  activeReportData = data;
  currentRetrievedProtocols = data.retrieved_protocols || [];

  // 1. Overlay image
  const overlayImg = document.getElementById("overlay-preview");
  if (overlayImg && (data.overlay_url || data.overlay_base64)) {
    overlayImg.src = data.overlay_url || data.overlay_base64;
    const opacitySlider = document.getElementById("opacity-slider");
    overlayImg.style.opacity = opacitySlider ? (opacitySlider.value / 100).toString() : "0.55";
  }

  // 2. Latency badge
  const latencyBadge = document.getElementById("pipeline-latency-badge");
  if (latencyBadge && data.timings) {
    latencyBadge.textContent = `Latency: ${data.timings.total_latency_ms} ms (CV: ${data.timings.cv_inference_ms}ms · RAG: ${data.timings.rag_retrieval_ms}ms · LLM: ${data.timings.llm_synthesis_ms}ms)`;
  }

  // 3. Quick stats pills
  const metrics = data.damage_metrics || {};
  const sum = metrics.damage_summary || {};
  const bldgs = metrics.buildings || {};
  const roads = metrics.roads || {};
  const vehs = metrics.vehicles || {};

  const statSeverity = document.getElementById("stat-severity");
  const statScore = document.getElementById("stat-score");
  if (statSeverity && statScore) {
    statSeverity.textContent = sum.severity_label || "MODERATE";
    statSeverity.className = `stat-pill-value ${getSeverityColorClass(sum.severity_label)}`;
    statScore.textContent = `Score: ${sum.damage_severity_score || 0}/100`;
  }

  const statFloodPct = document.getElementById("stat-flood-pct");
  const statFloodArea = document.getElementById("stat-flood-area");
  if (statFloodPct && statFloodArea) {
    statFloodPct.textContent = `${sum.flood_coverage_pct || 0}%`;
    statFloodArea.textContent = `${(sum.total_flood_area_m2 || 0).toLocaleString()} m²`;
  }

  const statBldgs = document.getElementById("stat-bldgs");
  const statBldgSub = document.getElementById("stat-bldg-sub");
  if (statBldgs && statBldgSub) {
    statBldgs.textContent = `${bldgs.flooded_building_count || 0} Flooded`;
    statBldgSub.textContent = `${bldgs.non_flooded_building_count || 0} Safe units`;
  }

  const statRoads = document.getElementById("stat-roads");
  const statVeh = document.getElementById("stat-veh");
  if (statRoads && statVeh) {
    statRoads.textContent = `${roads.flooded_road_coverage_pct || 0}%`;
    statVeh.textContent = `${vehs.vehicle_count || 0} Vehicles trapped`;
  }

  // 4. Render Grounded SITREP Body
  const sitrepContent = document.getElementById("sitrep-content");
  if (sitrepContent) {
    sitrepContent.innerHTML = formatMarkdownToHtml(data.markdown_report || "No report generated.");
  }

  // 5. Render Citations Chips
  const citationChips = document.getElementById("citation-chips");
  if (citationChips) {
    if (currentRetrievedProtocols.length > 0) {
      citationChips.innerHTML = "";
      currentRetrievedProtocols.forEach((p, idx) => {
        const chip = document.createElement("button");
        chip.className = "cite-chip";
        chip.innerHTML = `📜 ${p.citation_tag} <span style="opacity:0.7">(${(p.similarity_score * 100).toFixed(0)}%)</span>`;
        chip.onclick = () => openCitationModal(p);
        citationChips.appendChild(chip);
      });
    } else {
      citationChips.innerHTML = `<span class="empty-cite-text">No active report citations.</span>`;
    }
  }
}

// ── Citation Inspection Modal ──────────────────────────────────────────────
function openCitationModal(protocol) {
  const modal = document.getElementById("citation-modal");
  const title = document.getElementById("modal-cite-title");
  const docBadge = document.getElementById("modal-cite-doc");
  const scoreBadge = document.getElementById("modal-cite-score");
  const bodyText = document.getElementById("modal-cite-text");

  if (modal && title && docBadge && scoreBadge && bodyText) {
    title.textContent = `Protocol: ${protocol.citation_tag}`;
    docBadge.textContent = `Document: ${protocol.source_doc} (${protocol.section})`;
    scoreBadge.textContent = `Cosine Relevance: ${(protocol.similarity_score * 100).toFixed(1)}%`;
    bodyText.textContent = protocol.snippet || protocol.content || "Protocol snippet text unavailable.";
    modal.style.display = "flex";
  }
}

function closeCitationModal() {
  const modal = document.getElementById("citation-modal");
  if (modal) modal.style.display = "none";
}

// ── Export PDF & DOCX Reports ──────────────────────────────────────────────
async function exportReportPDF() {
  if (!activeReportData) {
    alert("Please analyze an aerial image first before exporting.");
    return;
  }

  showLoading("Generating official PDF SITREP Document...");
  try {
    const res = await fetch("/api/export_pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(activeReportData)
    });

    if (!res.ok) throw new Error("Failed to generate PDF");

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `FloodSense_Official_SITREP_${Date.now()}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error("PDF Export Error:", err);
    alert("PDF generation error: " + err.message);
  } finally {
    hideLoading();
  }
}

async function exportReportDOCX() {
  if (!activeReportData) {
    alert("Please analyze an aerial image first before exporting.");
    return;
  }

  showLoading("Generating Word (.docx) Document...");
  try {
    const res = await fetch("/api/export_docx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(activeReportData)
    });

    if (!res.ok) throw new Error("Failed to generate Word document");

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `FloodSense_Official_SITREP_${Date.now()}.docx`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error("DOCX Export Error:", err);
    alert("Word document generation error: " + err.message);
  } finally {
    hideLoading();
  }
}

// ── Helpers ────────────────────────────────────────────────────────────────
function showLoading(msg) {
  const loading = document.getElementById("loading-spinner");
  const text = document.getElementById("loading-status-text");
  if (loading) loading.style.display = "flex";
  if (text) text.textContent = msg || "Processing...";
}

function hideLoading() {
  const loading = document.getElementById("loading-spinner");
  if (loading) loading.style.display = "none";
}

function getSeverityColorClass(tier) {
  switch ((tier || "").toUpperCase()) {
    case "CRITICAL": return "red";
    case "SEVERE": return "red";
    case "MODERATE": return "amber";
    case "MINOR": return "blue";
    default: return "green";
  }
}

function formatMarkdownToHtml(md) {
  let html = md
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    .replace(/`(.*?)`/gim, '<code>$1</code>')
    .replace(/^\- (.*$)/gim, '<li>$1</li>')
    .replace(/\n\n/gim, '<br/><br/>');
  return html;
}
