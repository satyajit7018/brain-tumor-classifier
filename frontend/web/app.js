let currentFile = null;
let currentPrediction = null;
let wipePos = 50;
let isDraggingWipe = false;
let samples = {};

document.addEventListener('DOMContentLoaded', async () => {
    initTabs();
    initControls();
    initDropzone();
    await loadSamples();
    await checkHealth();
});

// Tab Switcher
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.style.display = 'none');

            btn.classList.add('active');
            const target = document.getElementById(btn.dataset.tab);
            if (target) target.style.display = 'block';
        });
    });
}

// Check Backend Health
async function checkHealth() {
    try {
        const res = await fetch('/health');
        if (res.ok) {
            const data = await res.json();
            const badge = document.getElementById('health-badge');
            if (data.model_loaded) {
                badge.className = 'badge badge-online';
                badge.innerHTML = '🟢 API Online (ResNet50 Active)';
            }
        }
    } catch (e) {
        console.warn('API health check failed:', e);
    }
}

// Load Pre-configured Samples
async function loadSamples() {
    try {
        const res = await fetch('/samples');
        if (res.ok) {
            samples = await res.json();
            const gallery = document.getElementById('sample-gallery');
            gallery.innerHTML = '';

            for (const [cls, info] of Object.entries(samples)) {
                const chip = document.createElement('div');
                chip.className = 'sample-chip';
                chip.innerHTML = `
                    <span>🧪 ${cls.replace('_', ' ').toUpperCase()}</span>
                    <small>${info.filename.slice(0, 16)}...</small>
                `;
                chip.onclick = () => selectSample(cls, info);
                gallery.appendChild(chip);
            }
        }
    } catch (e) {
        console.warn('Could not load samples:', e);
    }
}

// Select a sample scan
async function selectSample(cls, info) {
    const byteCharacters = atob(info.image_b64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: 'image/jpeg' });
    currentFile = new File([blob], info.filename, { type: 'image/jpeg' });

    document.getElementById('file-info').innerText = `Loaded Sample: ${info.filename} (${(blob.size / 1024).toFixed(1)} KB)`;
    await runInference();
}

// File Drag & Drop
function initDropzone() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');

    dropzone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            currentFile = e.target.files[0];
            document.getElementById('file-info').innerText = `Loaded: ${currentFile.name} (${(currentFile.size / 1024).toFixed(1)} KB)`;
            runInference();
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            currentFile = e.dataTransfer.files[0];
            document.getElementById('file-info').innerText = `Loaded: ${currentFile.name} (${(currentFile.size / 1024).toFixed(1)} KB)`;
            runInference();
        }
    });
}

// Execute Inference & Explainability
async function runInference() {
    if (!currentFile) return;

    const btn = document.getElementById('run-btn');
    btn.innerText = '⚡ Processing Convolutional Attention...';
    btn.disabled = true;

    try {
        const colormap = document.getElementById('colormap-select').value;
        const formData = new FormData();
        formData.append('file', currentFile);

        const res = await fetch(`/predict?colormap=${colormap}&alpha=0.4`, {
            method: 'POST',
            body: formData,
        });

        if (res.ok) {
            currentPrediction = await res.json();
            renderPredictionResults(currentPrediction);
            updateViewportImages();
        } else {
            alert(`Inference failed: ${res.statusText}`);
        }
    } catch (e) {
        alert(`Error: ${e.message}`);
    } finally {
        btn.innerText = '🚀 Re-Run Diagnostic & Explainability Inference';
        btn.disabled = false;
    }
}

// Render Results & Probability Bars
function renderPredictionResults(data) {
    document.getElementById('diag-panel').style.display = 'block';

    const statusPill = document.getElementById('status-pill');
    if (data.clinical_status.includes('HIGH')) {
        statusPill.className = 'status-pill review';
        statusPill.innerHTML = `🚨 <b>CLINICAL STATUS: ${data.clinical_status}</b><br>${data.status_description}`;
    } else if (data.clinical_status.includes('MODERATE')) {
        statusPill.className = 'status-pill moderate';
        statusPill.innerHTML = `⚠️ <b>CLINICAL STATUS: ${data.clinical_status}</b><br>${data.status_description}`;
    } else {
        statusPill.className = 'status-pill confident';
        statusPill.innerHTML = `✅ <b>CLINICAL STATUS: ${data.clinical_status}</b><br>${data.status_description}`;
    }

    document.getElementById('val-pathology').innerText = data.predicted_class.replace('_', ' ').toUpperCase();
    document.getElementById('val-confidence').innerText = `${(data.confidence * 100).toFixed(2)}%`;
    document.getElementById('val-epistemic').innerText = `±${data.epistemic_uncertainty.toFixed(4)}`;
    document.getElementById('val-entropy').innerText = `${data.predictive_entropy.toFixed(4)} / 1.0`;

    // Render Probabilities
    const probList = document.getElementById('prob-list');
    probList.innerHTML = '';

    for (const [cname, p] of Object.entries(data.probabilities)) {
        const std = data.std_probabilities[cname] || 0;
        const row = document.createElement('div');
        row.className = 'prob-row';
        row.innerHTML = `
            <div class="prob-labels">
                <span>${cname.replace('_', ' ').toUpperCase()}</span>
                <span>${(p * 100).toFixed(2)}% <small style="color:var(--text-muted)">(±${(std * 100).toFixed(2)}%)</small></span>
            </div>
            <div class="prob-track">
                <div class="prob-fill fill-${cname}" style="width: ${(p * 100).toFixed(1)}%"></div>
            </div>
        `;
        probList.appendChild(row);
    }
}

// Update PACS Viewport & Split Screen
function updateViewportImages() {
    if (!currentPrediction) return;

    const wipeBefore = document.getElementById('wipe-before');
    const wipeAfter = document.getElementById('wipe-after');

    wipeBefore.style.backgroundImage = `url(data:image/png;base64,${currentPrediction.original_image})`;
    wipeAfter.style.backgroundImage = `url(data:image/png;base64,${currentPrediction.gradcam_overlay})`;

    applyWindowLevel();
    updateWipePosition(wipePos);
}

// Controls: Window Level Brightness/Contrast + Wipe Slider
function initControls() {
    const brightnessSlider = document.getElementById('brightness-slider');
    const contrastSlider = document.getElementById('contrast-slider');
    const colormapSelect = document.getElementById('colormap-select');

    brightnessSlider.addEventListener('input', applyWindowLevel);
    contrastSlider.addEventListener('input', applyWindowLevel);
    colormapSelect.addEventListener('change', () => {
        if (currentFile) runInference();
    });

    const wipeContainer = document.getElementById('wipe-container');
    const wipeHandle = document.getElementById('wipe-handle');

    const onMove = (e) => {
        if (!isDraggingWipe) return;
        const rect = wipeContainer.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const x = clientX - rect.left;
        let pos = (x / rect.width) * 100;
        pos = Math.max(0, Math.min(100, pos));
        updateWipePosition(pos);
    };

    wipeContainer.addEventListener('mousedown', () => isDraggingWipe = true);
    window.addEventListener('mouseup', () => isDraggingWipe = false);
    window.addEventListener('mousemove', onMove);

    wipeContainer.addEventListener('touchstart', () => isDraggingWipe = true);
    window.addEventListener('touchend', () => isDraggingWipe = false);
    window.addEventListener('touchmove', onMove);

    document.getElementById('run-btn').addEventListener('click', runInference);
    document.getElementById('pdf-btn').addEventListener('click', previewPdfReport);
    document.getElementById('triage-btn').addEventListener('click', runTriageSimulation);
}

function updateWipePosition(pos) {
    wipePos = pos;
    const wipeHandle = document.getElementById('wipe-handle');
    const wipeAfter = document.getElementById('wipe-after');
    const label = document.getElementById('wipe-val');

    wipeHandle.style.left = `${pos}%`;
    wipeAfter.style.clipPath = `polygon(${pos}% 0%, 100% 0%, 100% 100%, ${pos}% 100%)`;
    if (label) label.innerText = `${Math.round(pos)}%`;
}

function applyWindowLevel() {
    const b = document.getElementById('brightness-slider').value;
    const c = document.getElementById('contrast-slider').value;
    document.getElementById('brightness-val').innerText = `${b}%`;
    document.getElementById('contrast-val').innerText = `${c}%`;

    const viewport = document.getElementById('wipe-container');
    viewport.style.filter = `brightness(${b}%) contrast(${c}%)`;
}

// In-Browser PDF Report Preview
async function previewPdfReport() {
    if (!currentFile) return;

    const modal = document.getElementById('pdf-modal');
    const frame = document.getElementById('pdf-frame');
    modal.style.display = 'flex';
    frame.src = 'about:blank';

    try {
        const formData = new FormData();
        formData.append('file', currentFile);
        const res = await fetch('/report', { method: 'POST', body: formData });
        if (res.ok) {
            const blob = await res.blob();
            const blobUrl = URL.createObjectURL(blob);
            frame.src = blobUrl;

            document.getElementById('download-pdf-btn').onclick = () => {
                const a = document.createElement('a');
                a.href = blobUrl;
                a.download = `clinical_diagnostic_report_${currentFile.name}.pdf`;
                a.click();
            };
        }
    } catch (e) {
        alert('Could not compile PDF: ' + e.message);
    }
}

function closeModal() {
    document.getElementById('pdf-modal').style.display = 'none';
}

// Emergency Department Triage Queue Simulation
async function runTriageSimulation() {
    const btn = document.getElementById('triage-btn');
    btn.innerText = '⏳ Analyzing Patient Queue with ResNet50...';
    btn.disabled = true;

    try {
        const res = await fetch('/triage', { method: 'POST' });
        if (res.ok) {
            const list = await res.json();
            const tbody = document.getElementById('triage-tbody');
            tbody.innerHTML = '';

            list.forEach((item, idx) => {
                const tr = document.createElement('tr');
                let priorityClass = 'priority-clear';
                if (item.priority_level === 1) priorityClass = 'priority-stat';
                else if (item.priority_level === 2) priorityClass = 'priority-oncology';

                tr.innerHTML = `
                    <td><b>#${idx + 1}</b></td>
                    <td><code>${item.filename}</code></td>
                    <td><b>${item.predicted_class.toUpperCase()}</b></td>
                    <td>${(item.confidence * 100).toFixed(1)}%</td>
                    <td>${item.predictive_entropy.toFixed(3)}</td>
                    <td><span class="${priorityClass}">${item.triage_priority}</span></td>
                `;
                tbody.appendChild(tr);
            });
            document.getElementById('triage-results').style.display = 'block';
        }
    } catch (e) {
        alert('Triage queue error: ' + e.message);
    } finally {
        btn.innerText = '⚡ Run Emergency Cohort Triage Simulation (10 Patients)';
        btn.disabled = false;
    }
}
