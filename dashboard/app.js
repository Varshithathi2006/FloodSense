/**
 * FloodSense Dashboard Interactive Logic
 * Handles model selection, drone photo analysis, live overlay rendering,
 * ChromaDB RAG citations, and Grounded SITREP exports.
 */

// ── State ──────────────────────────────────────────────────────────────────
let activeReportData = null;
let currentRetrievedProtocols = [];

function isStaticDeployment() {
  const h = window.location.hostname;
  return h !== "localhost" && h !== "127.0.0.1" && h !== "0.0.0.0" && h !== "";
}

const API_BASE_URL = isStaticDeployment() ? "https://floodsense-ru60.onrender.com" : "";

function apiUrl(path) {
  return `${API_BASE_URL}${path}`;
}

// ── Sidebar ───────────────────────────────────────────────────────────────
function setSidebarOpen(isOpen) {
  const shell = document.querySelector(".app-shell");
  if (!shell) return;

  shell.classList.toggle("sidebar-open", isOpen);
  shell.classList.toggle("sidebar-collapsed", !isOpen);
  
  const openBtn = document.getElementById("sidebar-open");
  if (openBtn) {
    openBtn.setAttribute("aria-expanded", String(isOpen));
  }

  // Prevent background scrolling when off-canvas drawer is active on mobile/tablet
  if (window.innerWidth <= 1024) {
    document.body.classList.toggle("drawer-open", isOpen);
  } else {
    document.body.classList.remove("drawer-open");
  }
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

  // Auto-close sidebar on mobile/tablet after navigating
  if (window.innerWidth <= 1024) {
    setSidebarOpen(false);
  }

  // Smoothly scroll back to top of page
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ── App Init ───────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadSampleScenarios();
  loadKnowledgeBaseDocs();
  wakeUpServer(); // pre-warm Render on load; falls back to health check locally
});

// ── Event Listeners ────────────────────────────────────────────────────────
function setupEventListeners() {
  const sidebarOpen = document.getElementById("sidebar-open");
  const sidebarClose = document.getElementById("sidebar-close");
  const sidebarBackdrop = document.getElementById("sidebar-backdrop");

  sidebarOpen?.addEventListener("click", () => setSidebarOpen(true));
  sidebarClose?.addEventListener("click", () => setSidebarOpen(false));
  sidebarBackdrop?.addEventListener("click", () => setSidebarOpen(false));

  // Escape key to dismiss drawer or modal
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      setSidebarOpen(false);
      closeCitationModal();
    }
  });

  // Handle window resizing to clean up mobile drawer state
  let resizeTimeout;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
      if (window.innerWidth > 1024) {
        document.body.classList.remove("drawer-open");
      }
    }, 150);
  });

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
      const val = parseInt(e.target.value, 10);
      opacityVal.textContent = `${val}%`;
      const overlayImg = document.getElementById("overlay-preview");
      if (overlayImg) {
        // 0% slider = fully opaque overlay; 100% slider = fully transparent (hidden)
        overlayImg.style.opacity = ((100 - val) / 100).toString();
      }
    });
  }
}

// ── Server Health Check & Wake-Up ─────────────────────────────────────────
let serverOnline = false;

async function checkServerHealth() {
  const badge = document.getElementById("server-badge");
  try {
    const res = await fetch(apiUrl("/api/status"), { signal: AbortSignal.timeout(8000) });
    if (res.ok) {
      const data = await res.json();
      serverOnline = true;
      if (badge) {
        badge.className = "status-badge green";
        badge.innerHTML = `<span class="status-dot"></span>Server: Online (${data.vector_store_chunks} chunks)`;
      }
      hideWakeUpBanner();
      return true;
    }
  } catch (err) {
    serverOnline = false;
    if (badge) {
      badge.className = "status-badge amber";
      badge.innerHTML = `<span class="status-dot"></span>Server: Waking up...`;
    }
  }
  return false;
}

// Pre-warm the Render server on page load with live countdown banner
async function wakeUpServer() {
  if (!isStaticDeployment()) return; // only needed for cloud deployment
  const online = await checkServerHealth();
  if (online) return;

  showWakeUpBanner();
  const MAX_WAIT = 45; // seconds
  let elapsed = 0;
  const interval = setInterval(async () => {
    elapsed += 5;
    updateWakeUpBanner(elapsed, MAX_WAIT);
    const ok = await checkServerHealth();
    if (ok || elapsed >= MAX_WAIT) {
      clearInterval(interval);
      if (!ok) {
        hideWakeUpBanner();
        showToast("Server did not respond. Analysis may be slow on first request.", "warning", 6000);
      }
    }
  }, 5000);
}

function showWakeUpBanner() {
  let banner = document.getElementById("wakeup-banner");
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "wakeup-banner";
    banner.className = "wakeup-banner";
    document.querySelector(".main-content")?.prepend(banner);
  }
  banner.innerHTML = `
    <span class="wakeup-spinner"></span>
    <span id="wakeup-text">Waking up Render server (free tier cold start)... <strong>0s</strong></span>
    <span class="wakeup-sub">Analysis will be available in ~15–30 seconds</span>
  `;
  banner.style.display = "flex";
}

function updateWakeUpBanner(elapsed, max) {
  const el = document.getElementById("wakeup-text");
  if (el) el.innerHTML = `Waking up server... <strong>${elapsed}s</strong> / ${max}s — please wait`;
}

function hideWakeUpBanner() {
  const banner = document.getElementById("wakeup-banner");
  if (banner) banner.style.display = "none";
}

// ── Load Sample Scenarios ──────────────────────────────────────────────────
async function loadSampleScenarios() {
  const container = document.getElementById("sample-buttons");
  if (!container) return;

  try {
    const res = await fetch(apiUrl("/api/samples"));
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
    const res = await fetch(apiUrl("/api/knowledge_base"));
    if (res.ok) {
      const data = await res.json();
      grid.innerHTML = "";
      data.documents.forEach(doc => {
        const card = document.createElement("div");
        card.className = "doc-card";
        card.innerHTML = `
          <div class="doc-card-title">${doc.title}</div>
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
  showLoading("Image is getting uploaded...");

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
    rawImg.src = apiUrl(`/api/sample_image/${sampleId}.jpg`);
    dropPrompt.style.display = "none";
    previewWrapper.style.display = "block";
  }

  const model = document.getElementById("model-select").value || "segformer";
  const formData = new FormData();
  formData.append("sample_id", sampleId);
  formData.append("model", model);

  await executeAnalysisRequest(formData);
}

// ── Send Analysis Request to Backend (with auto-retry) ────────────────────
async function executeAnalysisRequest(formData, attempt = 1) {
  const MAX_ATTEMPTS = 3;
  const RETRY_DELAY_MS = 8000;

  try {
    if (attempt > 1) {
      showLoading(`Server waking up... retry ${attempt}/${MAX_ATTEMPTS}`);
    }

    const res = await fetch(apiUrl("/api/analyze"), {
      method: "POST",
      body: formData,
      signal: AbortSignal.timeout(90000) // 90s timeout for cold start
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => "");
      throw new Error(`Server error ${res.status}${errText ? ": " + errText : ""}`);
    }

    const data = await res.json();
    serverOnline = true;
    hideWakeUpBanner();
    renderAnalysisResults(data);

  } catch (err) {
    console.error(`Analysis attempt ${attempt} failed:`, err);

    const isNetworkError = err.name === "TypeError" || err.name === "AbortError" || err.message.includes("fetch");

    if (isNetworkError && attempt < MAX_ATTEMPTS) {
      // Auto-retry: server is likely in cold start
      const wait = RETRY_DELAY_MS / 1000;
      showLoading(`Server is waking up (cold start)... retrying in ${wait}s (${attempt}/${MAX_ATTEMPTS})`);
      await new Promise(r => setTimeout(r, RETRY_DELAY_MS));
      return executeAnalysisRequest(formData, attempt + 1);
    }

    // All retries exhausted — show inline toast, not alert()
    hideLoading();
    const isOffline = isNetworkError;
    showToast(
      isOffline
        ? `Could not reach the server after ${MAX_ATTEMPTS} attempts. The Render server may be sleeping — please try again in 30 seconds.`
        : `Analysis failed: ${err.message}`,
      isOffline ? "warning" : "error",
      8000
    );
    return;
  }

  hideLoading();
}

// ── Toast Notification System ──────────────────────────────────────────────
function showToast(message, type = "error", durationMs = 5000) {
  let container = document.getElementById("toast-container");
  if (!container) {
    container = document.createElement("div");
    container.id = "toast-container";
    container.className = "toast-container";
    document.body.appendChild(container);
  }

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  const icon = type === "warning" ? "⚠️" : type === "success" ? "✅" : "❌";
  toast.innerHTML = `
    <span class="toast-icon">${icon}</span>
    <span class="toast-msg">${message}</span>
    <button class="toast-close" onclick="this.parentElement.remove()">×</button>
  `;
  container.appendChild(toast);

  // Animate in
  requestAnimationFrame(() => toast.classList.add("toast-visible"));

  // Auto-remove
  setTimeout(() => {
    toast.classList.remove("toast-visible");
    setTimeout(() => toast.remove(), 350);
  }, durationMs);
}

// ── Render Analysis Results ────────────────────────────────────────────────
function renderAnalysisResults(data) {
  activeReportData = data;
  currentRetrievedProtocols = data.retrieved_protocols || [];

  // 1. Overlay image
  const overlayImg = document.getElementById("overlay-preview");
  if (overlayImg && (data.overlay_url || data.overlay_base64)) {
    overlayImg.src = data.overlay_base64 || new URL(data.overlay_url, API_BASE_URL || window.location.origin).href;
    const opacitySlider = document.getElementById("opacity-slider");
    overlayImg.style.opacity = opacitySlider ? (opacitySlider.value / 100).toString() : "0.55";
  }

  // 2. Latency badge + model used
  const latencyBadge = document.getElementById("pipeline-latency-badge");
  if (latencyBadge && data.timings) {
    const autoTag = data.auto_selected
      ? ` <span style="background:#7c3aed;color:#fff;padding:1px 7px;border-radius:999px;font-size:0.72rem;margin-left:6px;">⚡ Auto-Selected</span>`
      : "";
    const modelLabel = data.model_display_name || data.model_used || "";
    latencyBadge.innerHTML = `Model: <strong>${modelLabel}</strong>${autoTag} &nbsp;|&nbsp; Latency: ${data.timings.total_latency_ms} ms (CV: ${data.timings.cv_inference_ms}ms · RAG: ${data.timings.rag_retrieval_ms}ms · LLM: ${data.timings.llm_synthesis_ms}ms)`;
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
    const res = await fetch(apiUrl("/api/export_pdf"), {
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
    const res = await fetch(apiUrl("/api/export_docx"), {
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
