/* ==========================================================================
   NEVER-AI v5.0 — PILLAR 3 & 4: FULL-DUPLEX VOICE MODE, NATIVE VISION,
   SCENE ILLUSTRATION, LIVE HTML/PYTHON WASM ATELIER, 120HZ MOBILE GESTURES,
   ENCRYPTED VAULT SYNC & ATOMIC SHA-256 OTA ROLLBACK
   ========================================================================== */
let _pendingVisionDataUrl = null;
let _pendingVisionName = '';
let _voiceActive = false;
let _voiceRecog = null;
let _voiceAudioCtx = null;
let _voiceAnalyser = null;
let _voiceMicStream = null;
let _voiceAnimId = null;
let _pyodideInstance = null;

/* ---------------- 1. NATIVE VISION (CAMERA / IMAGE / PASTE) ---------------- */
function _compressImageFileToDataUrl(file, maxDim = 1024, quality = 0.84) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (ev) => {
      const img = new Image();
      img.onload = () => {
        let w = img.width;
        let h = img.height;
        if (w > maxDim || h > maxDim) {
          if (w >= h) { h = Math.round((h * maxDim) / w); w = maxDim; }
          else { w = Math.round((w * maxDim) / h); h = maxDim; }
        }
        const cv = document.createElement('canvas');
        cv.width = w;
        cv.height = h;
        const ctx = cv.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);
        resolve(cv.toDataURL('image/jpeg', quality));
      };
      img.onerror = reject;
      img.src = ev.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

async function onVisionFileSelected(inp) {
  const f = inp && inp.files && inp.files[0];
  if (!f) return;
  try {
    _pendingVisionDataUrl = await _compressImageFileToDataUrl(f);
    _pendingVisionName = f.name || 'image.jpg';
    renderPendingVisionBar();
    toast('Image jointe : ' + _pendingVisionName);
  } catch (_) {
    toast('Impossible de lire cette image');
  } finally {
    inp.value = '';
  }
}

function clearPendingVision() {
  _pendingVisionDataUrl = null;
  _pendingVisionName = '';
  renderPendingVisionBar();
}

function renderPendingVisionBar() {
  const bar = document.getElementById('composerAttachments');
  if (!bar) return;
  if (!_pendingVisionDataUrl) {
    bar.innerHTML = '';
    bar.classList.add('hidden');
    return;
  }
  bar.classList.remove('hidden');
  bar.innerHTML = `
    <div class="composer-att-chip">
      <img src="${_pendingVisionDataUrl}" alt="Aperçu" />
      <span>${esc(_pendingVisionName || 'Photo')}</span>
      <button class="icon-btn" style="width:20px;height:20px;" onclick="clearPendingVision()" title="Retirer">×</button>
    </div>
  `;
}

document.addEventListener('paste', async (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const it of items) {
    if (it.type && it.type.startsWith('image/')) {
      const f = it.getAsFile();
      if (f) {
        _pendingVisionDataUrl = await _compressImageFileToDataUrl(f);
        _pendingVisionName = 'capture-presse-papiers.jpg';
        renderPendingVisionBar();
        toast('Image collée dans le message');
      }
      break;
    }
  }
});

function openImageLightbox(src) {
  const modal = document.getElementById('svgModal');
  const body = document.getElementById('svgModalBody');
  if (!modal || !body) return;
  body.innerHTML = `<img src="${esc(src)}" style="max-width:92vw;max-height:82vh;border-radius:12px;border:1px solid rgba(255,255,255,0.2);" />`;
  modal.classList.remove('hidden');
}

/* ---------------- 2. SCENE & PROMPT ILLUSTRATION ---------------- */
async function illustrateTurn(msgIdx) {
  const m = currentMessages[msgIdx];
  if (!m) return;
  const sg = extractSceneGraphJS(currentMessages.slice(0, msgIdx + 1));
  const rawBasis = ((sg.summary ? sg.summary + '. ' : '') + String(m.content || '')).replace(/[*«»"]/g, ' ').slice(0, 320);
  const seed = Math.floor(Math.random() * 999999);
  const imgUrl =
    'https://image.pollinations.ai/prompt/' +
    encodeURIComponent('cinematic atmospheric monochrome photography, dramatic lighting, high detail: ' + rawBasis) +
    `?width=896&height=512&nologo=true&seed=${seed}`;
  m.illustration_url = imgUrl;
  renderChat(false);
  syncCurrentThreadToBackend();
  toast('Illustration de la scène générée');
}

/* ---------------- 3. FULL-DUPLEX HANDS-FREE VOICE MODE ---------------- */
function _drawVoiceScope() {
  const cv = document.getElementById('voiceScopeCanvas');
  if (!cv || !_voiceActive) return;
  const ctx = cv.getContext('2d');
  const w = (cv.width = cv.clientWidth * (window.devicePixelRatio || 1));
  const h = (cv.height = cv.clientHeight * (window.devicePixelRatio || 1));
  ctx.fillStyle = '#030304';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = '#FFFFFF';
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.beginPath();

  if (_voiceAnalyser) {
    const data = new Uint8Array(_voiceAnalyser.fftSize);
    _voiceAnalyser.getByteTimeDomainData(data);
    const slice = w / data.length;
    let x = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i] / 128.0;
      const y = (v * h) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += slice;
    }
  } else {
    const t = Date.now() / 220;
    const speaking = window.speechSynthesis && window.speechSynthesis.speaking;
    const amp = speaking ? h * 0.26 : h * 0.05;
    for (let x = 0; x <= w; x += 4) {
      const y = h / 2 + Math.sin(x * 0.025 + t) * Math.cos(x * 0.01 - t * 0.7) * amp;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
  _voiceAnimId = requestAnimationFrame(_drawVoiceScope);
}

function speakTextFrench(text, onDone) {
  if (!('speechSynthesis' in window)) { if (onDone) onDone(); return; }
  window.speechSynthesis.cancel();
  const clean = String(text || '')
    .replace(/```[\s\S]*?```/g, ' bloc de code. ')
    .replace(/[*#>`_~]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (!clean) { if (onDone) onDone(); return; }
  const utt = new SpeechSynthesisUtterance(clean);
  utt.lang = 'fr-FR';
  utt.rate = 1.04;
  const voices = window.speechSynthesis.getVoices() || [];
  const frVoice = voices.find((v) => /fr[-_]/i.test(v.lang) && /natural|neural|google|thomas|am[eé]lie|denise/i.test(v.name))
    || voices.find((v) => /fr/i.test(v.lang));
  if (frVoice) utt.voice = frVoice;
  utt.onend = () => { if (onDone) onDone(); };
  utt.onerror = () => { if (onDone) onDone(); };
  window.speechSynthesis.speak(utt);
}

async function openVoiceMode() {
  const modal = document.getElementById('voiceModal');
  if (!modal) return;
  modal.classList.remove('hidden');
  _voiceActive = true;

  try {
    _voiceMicStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    _voiceAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const src = _voiceAudioCtx.createMediaStreamSource(_voiceMicStream);
    _voiceAnalyser = _voiceAudioCtx.createAnalyser();
    _voiceAnalyser.fftSize = 512;
    src.connect(_voiceAnalyser);
  } catch (_) {
    _voiceAnalyser = null;
  }
  _drawVoiceScope();
  startVoiceListeningCycle();
}

function startVoiceListeningCycle() {
  if (!_voiceActive) return;
  const stEl = document.getElementById('voiceStatusLbl');
  const trEl = document.getElementById('voiceLiveTranscript');
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    if (stEl) stEl.textContent = 'Reconnaissance vocale non supportée sur ce navigateur (utilisez Chrome / Edge / Safari)';
    return;
  }
  if (_voiceRecog) { try { _voiceRecog.abort(); } catch (_) {} }
  _voiceRecog = new SR();
  _voiceRecog.lang = 'fr-FR';
  _voiceRecog.interimResults = true;
  _voiceRecog.continuous = false;

  if (stEl) stEl.textContent = 'ÉCOUTE EN DIRECT — PARLEZ NATURELLEMENT…';
  let finalTranscript = '';

  _voiceRecog.onresult = (ev) => {
    let interim = '';
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const seg = ev.results[i][0].transcript;
      if (ev.results[i].isFinal) finalTranscript += seg + ' ';
      else interim += seg;
    }
    if (trEl) trEl.textContent = (finalTranscript + interim).trim() || '…';
  };

  _voiceRecog.onend = async () => {
    if (!_voiceActive) return;
    const spoken = finalTranscript.trim();
    if (!spoken) {
      if (_voiceActive && !(window.speechSynthesis && window.speechSynthesis.speaking)) {
        setTimeout(startVoiceListeningCycle, 350);
      }
      return;
    }
    if (stEl) stEl.textContent = 'GÉNÉRATION SYNAPTIQUE…';
    document.getElementById('composerInput').value = spoken;
    await sendMessage();
    if (!_voiceActive) return;
    const lastMsg = currentMessages[currentMessages.length - 1];
    const replyText = (lastMsg && lastMsg.role !== 'user' && lastMsg.content) ? lastMsg.content : '';
    if (stEl) stEl.textContent = 'NEVER-AI PARLE…';
    speakTextFrench(replyText, () => {
      if (_voiceActive) startVoiceListeningCycle();
    });
  };

  try { _voiceRecog.start(); } catch (_) {}
}

function closeVoiceMode() {
  _voiceActive = false;
  if (_voiceAnimId) cancelAnimationFrame(_voiceAnimId);
  if (_voiceRecog) { try { _voiceRecog.abort(); } catch (_) {} _voiceRecog = null; }
  if (window.speechSynthesis) window.speechSynthesis.cancel();
  if (_voiceMicStream) { _voiceMicStream.getTracks().forEach((t) => t.stop()); _voiceMicStream = null; }
  if (_voiceAudioCtx) { _voiceAudioCtx.close().catch(() => {}); _voiceAudioCtx = null; _voiceAnalyser = null; }
  const modal = document.getElementById('voiceModal');
  if (modal) modal.classList.add('hidden');
}

/* ---------------- 4. LIVE SPLIT-PANE ATELIER (HTML/JS & PYTHON WASM) ---------------- */
function openHtmlAtelier(codeIdx) {
  const code = codeSnippets[codeIdx] || '';
  const drawer = document.getElementById('atelierDrawer');
  const title = document.getElementById('atelierTitle');
  const body = document.getElementById('atelierBody');
  if (!drawer || !body) return;
  title.textContent = 'Aperçu Interactif Live (Sandbox HTML / JS)';
  body.innerHTML = `<iframe class="atelier-iframe" sandbox="allow-scripts" srcdoc="${esc(code)}"></iframe>`;
  drawer.classList.remove('hidden');
}

async function runPythonAtelier(codeIdx) {
  const code = codeSnippets[codeIdx] || '';
  if (!code) return;
  const drawer = document.getElementById('atelierDrawer');
  const title = document.getElementById('atelierTitle');
  const body = document.getElementById('atelierBody');
  if (!drawer || !body) return;
  title.textContent = 'Exécution Python Souveraine (WASM / Sandbox)';
  body.innerHTML = `<div id="atelierConsole" class="atelier-console">Initialisation de l'environnement Python…</div>`;
  drawer.classList.remove('hidden');
  const outEl = document.getElementById('atelierConsole');

  /* Try fast backend sandbox if running locally, else in-browser Pyodide WASM */
  if (!window._standaloneEdgeMode) {
    try {
      const r = await fetch('/api/plugins/invoke', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plugin: 'python_sandbox', params: { code } }),
      });
      if (r.ok) {
        const d = await r.json();
        if (!d.standalone) {
          outEl.textContent = (d.stdout || '') + (d.stderr ? '\n' + d.stderr : '') + (d.error ? '\nErreur: ' + d.error : '') || '(Exécuté sans sortie stdout)';
          return;
        }
      }
    } catch (_) {}
  }

  try {
    if (!_pyodideInstance) {
      outEl.textContent = 'Chargement de Python 3.12 WebAssembly (Pyodide)…\n';
      if (!window.loadPyodide) {
        await new Promise((resolve, reject) => {
          const s = document.createElement('script');
          s.src = 'https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js';
          s.onload = resolve;
          s.onerror = reject;
          document.head.appendChild(s);
        });
      }
      _pyodideInstance = await window.loadPyodide();
    }
    let captured = '';
    _pyodideInstance.setStdout({ batched: (line) => { captured += line + '\n'; } });
    _pyodideInstance.setStderr({ batched: (line) => { captured += '[stderr] ' + line + '\n'; } });
    const result = await _pyodideInstance.runPythonAsync(code);
    if (result !== undefined && result !== null && String(result) !== 'None') {
      captured += '=> ' + String(result) + '\n';
    }
    outEl.textContent = captured.trim() || '(Exécution terminée avec succès)';
  } catch (err) {
    outEl.textContent = 'Erreur Python :\n' + (err.message || String(err));
  }
}

function closeAtelier() {
  const drawer = document.getElementById('atelierDrawer');
  if (drawer) drawer.classList.add('hidden');
}

/* ---------------- 5. ENCRYPTED VAULT SYNC & ATOMIC OTA ROLLBACK ---------------- */
async function exportEncryptedVaultFile() {
  const pass = (document.getElementById('inpVaultPassphrase') || {}).value || '';
  if (pass.length < 4) { toast('Choisissez une clé secrète (min. 4 caractères)'); return; }
  try {
    const bundle = await NeverAIVault.encryptVaultPayload(pass);
    const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `NeverAI-Vault-${new Date().toISOString().slice(0, 10)}.nva.json`;
    a.click();
    toast('Coffre chiffré AES-256-GCM exporté');
  } catch (e) {
    toast('Erreur de chiffrement : ' + (e.message || e));
  }
}

async function importEncryptedVaultFile(inp) {
  const f = inp && inp.files && inp.files[0];
  if (!f) return;
  const pass = (document.getElementById('inpVaultPassphrase') || {}).value || '';
  if (pass.length < 4) { toast('Saisissez d\'abord la clé secrète du coffre'); inp.value = ''; return; }
  try {
    const bundle = JSON.parse(await f.text());
    const n = await NeverAIVault.decryptAndMergeVault(bundle, pass);
    await loadThreads();
    toast(`${n} discussion(s) déchiffrée(s) et fusionnée(s)`);
  } catch (_) {
    toast('Clé secrète incorrecte ou fichier de coffre invalide');
  } finally {
    inp.value = '';
  }
}

async function pushEncryptedCloudVault() {
  const vid = ((document.getElementById('inpVaultSyncId') || {}).value || '').trim();
  const pass = (document.getElementById('inpVaultPassphrase') || {}).value || '';
  if (vid.length < 4 || pass.length < 4) {
    toast('Renseignez un identifiant de coffre et une clé secrète (min. 4 caractères)');
    return;
  }
  localStorage.setItem('never_ai_vault_id', vid);
  try {
    const bundle = await NeverAIVault.encryptVaultPayload(pass);
    const r = await fetch('/api/vault/push', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vault_id: vid, ...bundle }),
    });
    const d = await r.json();
    if (d && d.ok) toast('Coffre chiffré synchronisé (AES-256-GCM)');
  } catch (e) {
    toast('Erreur de synchronisation : ' + (e.message || e));
  }
}

async function pullEncryptedCloudVault() {
  const vid = ((document.getElementById('inpVaultSyncId') || {}).value || '').trim();
  const pass = (document.getElementById('inpVaultPassphrase') || {}).value || '';
  if (vid.length < 4 || pass.length < 4) {
    toast('Renseignez votre identifiant de coffre et votre clé secrète');
    return;
  }
  try {
    const r = await fetch('/api/vault/pull/' + encodeURIComponent(vid));
    if (!r.ok) { toast('Aucun coffre distant trouvé pour cet identifiant'); return; }
    const bundle = await r.json();
    const n = await NeverAIVault.decryptAndMergeVault(bundle, pass);
    await loadThreads();
    toast(`${n} discussion(s) synchronisée(s) et déchiffrée(s)`);
  } catch (_) {
    toast('Échec du déchiffrement (vérifiez votre clé secrète)');
  }
}

async function rollbackOtaNow() {
  const fb = document.getElementById('otaFeedback');
  setCardBusy('cardOta', true);
  if (fb) fb.textContent = 'Restauration atomique de la version précédente…';
  try {
    const r = await fetch('/api/ota/rollback', { method: 'POST' });
    const d = await r.json();
    if (d && d.rolled_back) {
      if (fb) fb.textContent = d.message || 'Version précédente restaurée.';
      scheduleSafeReload('Rollback effectué — rechargement…');
    } else {
      if (fb) fb.textContent = (d && (d.error || d.message)) || 'Aucune sauvegarde antérieure enregistrée.';
    }
  } finally {
    setCardBusy('cardOta', false);
  }
}

/* ---------------- 6. NATIVE 120HZ MOBILE GESTURES & VIEWPORT ---------------- */
(function initMobileGesturesAndViewport() {
  let startX = 0;
  let startY = 0;
  let tracking = false;

  document.addEventListener('touchstart', (e) => {
    if (!e.touches || e.touches.length !== 1) return;
    startX = e.touches[0].clientX;
    startY = e.touches[0].clientY;
    const sb = document.getElementById('sidebar');
    const isOpen = sb && sb.classList.contains('mobile-open');
    tracking = startX < 30 || isOpen;
  }, { passive: true });

  document.addEventListener('touchend', (e) => {
    if (!tracking || !e.changedTouches || !e.changedTouches.length) return;
    tracking = false;
    const dx = e.changedTouches[0].clientX - startX;
    const dy = Math.abs(e.changedTouches[0].clientY - startY);
    if (dy > 70) return;
    const sb = document.getElementById('sidebar');
    const bd = document.getElementById('drawerBackdrop');
    if (!sb) return;
    if (startX < 30 && dx > 64) {
      sb.classList.add('mobile-open');
      if (bd) bd.classList.add('mobile-open');
      if (navigator.vibrate) navigator.vibrate(8);
    } else if (sb.classList.contains('mobile-open') && dx < -64) {
      closeMobileSidebar();
    }
  }, { passive: true });

  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', () => {
      const shell = document.querySelector('.app-shell');
      if (shell && window.innerWidth <= 820) {
        shell.style.height = window.visualViewport.height + 'px';
      }
    });
  }
})();
