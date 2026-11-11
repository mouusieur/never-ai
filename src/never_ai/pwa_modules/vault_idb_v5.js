/* ==========================================================================
   NEVER-AI v5.0 — PILLAR 2 & 4: INDEXEDDB UNLIMITED VAULT, LOCAL COSINE
   SEMANTIC VECTOR SEARCH & AES-256-GCM END-TO-END ENCRYPTED SYNC
   ========================================================================== */
const NeverAIVault = (() => {
  const DB_NAME = 'never_ai_idb_v5';
  const DB_VER = 1;
  const VEC_DIM = 384;
  let _dbPromise = null;
  const _memCache = { threads: {}, docs: [], experiences: [], otaSnapshots: [], hydrated: false };

  function computeSemanticVector(text) {
    const vec = new Float32Array(VEC_DIM);
    const clean = String(text || '').toLowerCase().replace(/\s+/g, ' ').trim();
    if (!clean) return vec;
    for (let i = 0; i < clean.length - 2; i++) {
      const h = ((clean.charCodeAt(i) * 31 + clean.charCodeAt(i + 1)) * 31 + clean.charCodeAt(i + 2)) >>> 0;
      vec[h % VEC_DIM] += 1.0;
    }
    const words = clean.split(/[^a-z0-9àâçéèêëîïôûùüÿñæœ]+/).filter((w) => w.length > 2);
    for (const w of words) {
      let wh = 2166136261;
      for (let j = 0; j < w.length; j++) {
        wh ^= w.charCodeAt(j);
        wh = Math.imul(wh, 16777619) >>> 0;
      }
      vec[wh % VEC_DIM] += 2.2;
    }
    let norm = 0;
    for (let k = 0; k < VEC_DIM; k++) norm += vec[k] * vec[k];
    norm = Math.sqrt(norm) || 1;
    for (let k = 0; k < VEC_DIM; k++) vec[k] /= norm;
    return vec;
  }

  function cosineSim(a, b) {
    let dot = 0;
    const len = Math.min(a.length, b.length);
    for (let i = 0; i < len; i++) dot += a[i] * b[i];
    return dot;
  }

  function openDB() {
    if (_dbPromise) return _dbPromise;
    _dbPromise = new Promise((resolve) => {
      if (typeof indexedDB === 'undefined') { resolve(null); return; }
      const req = indexedDB.open(DB_NAME, DB_VER);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('threads')) db.createObjectStore('threads', { keyPath: 'id' });
        if (!db.objectStoreNames.contains('docs')) db.createObjectStore('docs', { keyPath: 'id' });
        if (!db.objectStoreNames.contains('experiences')) db.createObjectStore('experiences', { keyPath: 'id' });
        if (!db.objectStoreNames.contains('ota_snapshots')) db.createObjectStore('ota_snapshots', { keyPath: 'sha' });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    });
    return _dbPromise;
  }

  async function idbGetAll(storeName) {
    const db = await openDB();
    if (!db) return [];
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(storeName, 'readonly');
        const req = tx.objectStore(storeName).getAll();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => resolve([]);
      } catch (_) { resolve([]); }
    });
  }

  async function idbPut(storeName, val) {
    const db = await openDB();
    if (!db) return;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(storeName, 'readwrite');
        tx.objectStore(storeName).put(val);
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
      } catch (_) { resolve(false); }
    });
  }

  async function idbDelete(storeName, key) {
    const db = await openDB();
    if (!db) return;
    return new Promise((resolve) => {
      try {
        const tx = db.transaction(storeName, 'readwrite');
        tx.objectStore(storeName).delete(key);
        tx.oncomplete = () => resolve(true);
        tx.onerror = () => resolve(false);
      } catch (_) { resolve(false); }
    });
  }

  async function hydrate() {
    if (_memCache.hydrated) return _memCache;
    try {
      const [threads, docs, exps, snaps] = await Promise.all([
        idbGetAll('threads'),
        idbGetAll('docs'),
        idbGetAll('experiences'),
        idbGetAll('ota_snapshots'),
      ]);
      for (const t of threads) _memCache.threads[t.id] = t;
      _memCache.docs = docs;
      _memCache.experiences = exps;
      _memCache.otaSnapshots = snaps;

      /* Migrate legacy localStorage vault v4 -> IndexedDB v5 seamlessly */
      const legacyRaw = localStorage.getItem('never_ai_edge_vault_v4');
      if (legacyRaw) {
        const legacy = JSON.parse(legacyRaw);
        if (legacy && legacy.threads) {
          for (const [tid, t] of Object.entries(legacy.threads)) {
            if (!_memCache.threads[tid]) {
              _memCache.threads[tid] = t;
              idbPut('threads', t);
            }
          }
        }
        if (legacy && Array.isArray(legacy.docs)) {
          for (const d of legacy.docs) {
            const docObj = { id: 'doc_' + (d.ts || Date.now()) + '_' + Math.random().toString(36).slice(2, 6), ...d };
            _memCache.docs.push(docObj);
            idbPut('docs', docObj);
          }
        }
        localStorage.removeItem('never_ai_edge_vault_v4');
      }
    } catch (_) {}
    _memCache.hydrated = true;
    return _memCache;
  }

  function getVaultSync() {
    return _memCache;
  }

  function putThread(thread) {
    if (!thread || !thread.id) return;
    _memCache.threads[thread.id] = thread;
    idbPut('threads', thread);
  }

  function removeThread(threadId) {
    delete _memCache.threads[threadId];
    idbDelete('threads', threadId);
  }

  function addDoc(filename, content) {
    const doc = { id: 'doc_' + Date.now().toString(36), filename, content, ts: Date.now() };
    _memCache.docs.push(doc);
    idbPut('docs', doc);
    return doc;
  }

  function addExperience(prompt, reply, kind = 'edit') {
    const exp = { id: 'exp_' + Date.now().toString(36), prompt: String(prompt).slice(0, 600), reply: String(reply).slice(0, 1400), kind, ts: Date.now() };
    _memCache.experiences.push(exp);
    idbPut('experiences', exp);
    return exp;
  }

  function saveOtaSnapshot(sha, htmlContent) {
    if (!sha || !htmlContent) return;
    const snap = { sha, html: htmlContent, ts: Date.now() };
    _memCache.otaSnapshots = [snap, ..._memCache.otaSnapshots.filter((s) => s.sha !== sha)].slice(0, 3);
    idbPut('ota_snapshots', snap);
  }

  function popLatestOtaSnapshot() {
    const snap = _memCache.otaSnapshots.shift();
    if (snap) idbDelete('ota_snapshots', snap.sha);
    return snap || null;
  }

  function semanticSearch(query, currentThreadId, topK = 3) {
    const qv = computeSemanticVector(query);
    const hits = [];
    for (const d of _memCache.docs) {
      const sv = computeSemanticVector(d.filename + ' ' + String(d.content || '').slice(0, 1200));
      const sim = cosineSim(qv, sv);
      if (sim > 0.18) hits.push({ score: sim + 0.08, text: `[Doc ${d.filename}] ${String(d.content || '').slice(0, 360)}` });
    }
    for (const e of _memCache.experiences.slice(-60)) {
      const sv = computeSemanticVector(e.prompt);
      const sim = cosineSim(qv, sv);
      if (sim > 0.26) hits.push({ score: sim + 0.12, text: `[Style préféré] ${e.reply.slice(0, 240)}` });
    }
    for (const t of Object.values(_memCache.threads)) {
      if (t.id === currentThreadId) continue;
      for (const m of (t.messages || []).slice(-12)) {
        const txt = String(m.content || '');
        if (txt.length < 18) continue;
        const sim = cosineSim(qv, computeSemanticVector(txt));
        if (sim > 0.30) hits.push({ score: sim, text: `[Souvenir · ${t.title || 'Discussion'}] ${txt.slice(0, 200)}` });
      }
    }
    hits.sort((a, b) => b.score - a.score);
    return hits.slice(0, topK).map((h) => h.text);
  }

  /* AES-256-GCM WebCrypto End-to-End Encryption */
  function _bufToB64(buf) {
    const bytes = new Uint8Array(buf);
    let bin = '';
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return btoa(bin);
  }

  function _b64ToBuf(b64) {
    const bin = atob(b64);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return bytes.buffer;
  }

  async function _deriveKey(passphrase, saltBytes) {
    const enc = new TextEncoder();
    const baseKey = await crypto.subtle.importKey('raw', enc.encode(passphrase), 'PBKDF2', false, ['deriveKey']);
    return crypto.subtle.deriveKey(
      { name: 'PBKDF2', salt: saltBytes, iterations: 100000, hash: 'SHA-256' },
      baseKey,
      { name: 'AES-GCM', length: 256 },
      false,
      ['encrypt', 'decrypt']
    );
  }

  async function encryptVaultPayload(passphrase) {
    await hydrate();
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const key = await _deriveKey(passphrase, salt);
    const plain = JSON.stringify({
      version: '5.0.0',
      exported_at: Date.now(),
      threads: _memCache.threads,
      docs: _memCache.docs,
      experiences: _memCache.experiences,
    });
    const cipherBuf = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, new TextEncoder().encode(plain));
    return {
      format: 'never-ai-encrypted-vault-v5',
      ciphertext_b64: _bufToB64(cipherBuf),
      iv_b64: _bufToB64(iv),
      salt_b64: _bufToB64(salt),
      updated_at: Date.now(),
    };
  }

  async function decryptAndMergeVault(bundle, passphrase) {
    await hydrate();
    const salt = new Uint8Array(_b64ToBuf(bundle.salt_b64));
    const iv = new Uint8Array(_b64ToBuf(bundle.iv_b64));
    const key = await _deriveKey(passphrase, salt);
    const plainBuf = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, _b64ToBuf(bundle.ciphertext_b64));
    const data = JSON.parse(new TextDecoder().decode(plainBuf));
    let mergedCount = 0;
    for (const [tid, t] of Object.entries(data.threads || {})) {
      const existing = _memCache.threads[tid];
      if (!existing || (t.updated_at || 0) >= (existing.updated_at || 0)) {
        putThread(t);
        mergedCount++;
      }
    }
    for (const d of data.docs || []) {
      if (!_memCache.docs.some((x) => x.id === d.id)) {
        _memCache.docs.push(d);
        idbPut('docs', d);
      }
    }
    for (const e of data.experiences || []) {
      if (!_memCache.experiences.some((x) => x.id === e.id)) {
        _memCache.experiences.push(e);
        idbPut('experiences', e);
      }
    }
    return mergedCount;
  }

  return {
    hydrate,
    getVaultSync,
    putThread,
    removeThread,
    addDoc,
    addExperience,
    saveOtaSnapshot,
    popLatestOtaSnapshot,
    semanticSearch,
    encryptVaultPayload,
    decryptAndMergeVault,
  };
})();
