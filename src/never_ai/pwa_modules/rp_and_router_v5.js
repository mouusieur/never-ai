/* ==========================================================================
   NEVER-AI v5.0 — PILLAR 1: NO-LIMIT MULTI-ROUTER, SCENE GRAPH, RP TYPOGRAPHY,
   RESPONSE VARIANTS (SWIPE), INLINE MESSAGE EDITING & CONVERSATION FORKING
   ========================================================================== */
const _EDGE_REFUSAL_RE = /\b(je ne peux pas|je suis d[ée]sol[ée],?\s+mais|en tant qu'(?:ia|intelligence artificielle|mod[èe]le)|i cannot fulfill|i'm sorry,?\s+but i can't|as an ai language model|contre mes directives)\b/i;

let _editingMsgIndex = -1;
let _sidebarSearchQuery = '';
let _preferredWebGpuModel = localStorage.getItem('never_ai_webgpu_model') || 'Hermes-3-Llama-3.2-3B-q4f16_1-MLC';

function setPreferredWebGpuModel(modelId) {
  if (!modelId) return;
  _preferredWebGpuModel = modelId;
  localStorage.setItem('never_ai_webgpu_model', modelId);
  window._edgeWebLlmEngine = null;
  toast('Modèle WebGPU local sélectionné : ' + modelId.split('-q4')[0]);
}

function extractSceneGraphJS(messages) {
  const locRe = /\b(?:dans|sur|contre|au milieu de|au fond de|près de)\s+(?:la |le |l'|les |un |une |mon |ma |ton |ta |notre )?(chambre|salon|cuisine|salle de bain|douche|voiture|bureau|couloir|bar|hôtel|appartement|ascenseur|forêt|plage|jardin|rue|ruelle|cave|terrasse|balcon|lit|canapé|fauteuil|table|mur|porte|fenêtre)\b/gi;
  const posRe = /\b(plaqu[ée]s?\s+contre\s+le\s+mur|contre\s+le\s+mur|sur\s+le\s+lit|allong[ée]s?|à\s+genoux|agenouill[ée]s?|assis(?:e)?\s+sur|debout\s+devant|pench[ée]s?\s+vers|dans\s+(?:mes|tes|ses)\s+bras|à\s+califourchon|dos\s+au\s+mur|blotti(?:e)?|face\s+à\s+face|coll[ée]s?\s+contre)\b/gi;
  const attRe = /\b(chemise|robe|jupe|pantalon|jean|caleçon|culotte|soutien-gorge|veste|manteau|t-shirt|pull|ceinture|fermeture\s+éclair|déboutonn[ée]|dévêtu[e]?|nu[e]?|torse\s+nu)\b/gi;
  const actRe = /\*([^*\n]{4,170})\*/g;

  let lastLoc = '';
  let lastPos = '';
  const attSet = [];
  const acts = [];
  let hasMarkers = false;

  for (const m of (messages || []).slice(-26)) {
    const txt = String(m.content || '').trim();
    if (!txt) continue;
    if (txt.includes('«') || /\*[^*\n]{3,}\*/.test(txt)) hasMarkers = true;
    let match;
    while ((match = actRe.exec(txt)) !== null) {
      acts.push(match[1].trim());
    }
    while ((match = locRe.exec(txt)) !== null) {
      lastLoc = match[0].trim();
    }
    while ((match = posRe.exec(txt)) !== null) {
      lastPos = match[0].trim();
    }
    while ((match = attRe.exec(txt)) !== null) {
      const v = match[0].trim().toLowerCase();
      if (!attSet.includes(v)) attSet.push(v);
    }
  }

  const parts = [];
  if (lastLoc) parts.push('Lieu : ' + lastLoc);
  if (lastPos) parts.push('Posture : ' + lastPos);
  if (attSet.length) parts.push('Éléments : ' + attSet.slice(-3).join(', '));
  if (acts.length) parts.push('Action : *' + acts[acts.length - 1].slice(0, 80) + '*');

  return {
    active: Boolean(hasMarkers && parts.length),
    summary: parts.join(' · '),
  };
}

function formatRpTypography(escapedHtml) {
  /* Outside HTML tags, style *physical descriptions* & « spoken dialogue » in pure B&W */
  return escapedHtml
    .replace(/(^|>|\s)\*([^*\n<>]{2,260})\*(?=$|<|\s|[.,!?;:])/g, '$1<span class="rp-action">*$2*</span>')
    .replace(/«\s*([^»\n<>]{1,320})\s*»/g, '<span class="rp-speech">« $1 »</span>');
}

async function syncCurrentThreadToBackend() {
  if (!currentThreadId) return;
  try {
    await fetch('/api/threads/' + encodeURIComponent(currentThreadId), {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        owner: currentOwner(),
        messages: currentMessages.map((m) => ({
          role: m.role,
          content: m.content,
          thoughts: m.thoughts || [],
          web_sources: m.web_sources || [],
          variants: m.variants || undefined,
          activeVariant: typeof m.activeVariant === 'number' ? m.activeVariant : undefined,
          image_url: m.image_url || undefined,
          illustration_url: m.illustration_url || undefined,
        })),
      }),
    });
  } catch (_) {}
}

function ensureVariants(m) {
  if (!Array.isArray(m.variants) || !m.variants.length) {
    m.variants = [{ content: m.content || '', thoughts: m.thoughts || [] }];
    m.activeVariant = 0;
  }
  if (typeof m.activeVariant !== 'number' || m.activeVariant < 0 || m.activeVariant >= m.variants.length) {
    m.activeVariant = m.variants.length - 1;
  }
}

function switchVariant(msgIdx, delta) {
  const m = currentMessages[msgIdx];
  if (!m || m.streaming) return;
  ensureVariants(m);
  const next = m.activeVariant + delta;
  if (next < 0 || next >= m.variants.length) return;
  m.activeVariant = next;
  m.content = m.variants[next].content;
  m.thoughts = m.variants[next].thoughts || [];
  if (navigator.vibrate) navigator.vibrate(6);
  renderChat(false);
  syncCurrentThreadToBackend();
}

async function generateVariantAt(msgIdx) {
  if (activeStreamCtrl) return;
  const botMsg = currentMessages[msgIdx];
  if (!botMsg || botMsg.role === 'user') return;
  let userIdx = msgIdx - 1;
  while (userIdx >= 0 && currentMessages[userIdx].role !== 'user') userIdx--;
  if (userIdx < 0) return;

  ensureVariants(botMsg);
  const userTurn = currentMessages[userIdx];
  const priorSlice = currentMessages.slice(0, userIdx);

  /* Temporarily trim/stream into new variant slot */
  const newVarIdx = botMsg.variants.length;
  botMsg.variants.push({ content: '', thoughts: [''] });
  botMsg.activeVariant = newVarIdx;
  botMsg.content = '';
  botMsg.thoughts = [''];
  botMsg.streaming = true;
  renderChat(false);

  const btn = document.getElementById('btnSend');
  const box = document.getElementById('composerBox');
  btn.classList.add('streaming-mode');
  btn.innerHTML = ICO_STOP;
  box.classList.add('is-streaming');
  activeStreamCtrl = new AbortController();

  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: userTurn.content,
        max_new_tokens: 205,
        temperature: Math.min(0.82, 0.62 + newVarIdx * 0.06),
        thread_id: currentThreadId,
        owner: currentOwner(),
        deep_think: deepThinkActive,
        web_search: webSearchActive,
        variant_regen: true,
        history_override: priorSlice,
        image_data_url: userTurn.image_url || undefined,
      }),
      signal: activeStreamCtrl.signal,
    });
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n');
      buf = parts.pop() || '';
      for (const p of parts) {
        if (!p.startsWith('data: ')) continue;
        const ev = JSON.parse(p.slice(6));
        if (ev.type === 'think_delta') { botMsg.thoughts[0] += ev.text; }
        else if (ev.type === 'answer_delta') { botMsg.content += ev.text; }
        else if (ev.type === 'answer_replace') { botMsg.content = ev.text; }
        else if (ev.type === 'done' && ev.result) {
          botMsg.content = ev.result.visible_answer || botMsg.content;
          botMsg.thoughts = ev.result.thoughts || botMsg.thoughts;
        }
        botMsg.variants[newVarIdx] = { content: botMsg.content, thoughts: botMsg.thoughts };
        const row = document.getElementById('msgRow-' + msgIdx);
        if (row) {
          const tmp = document.createElement('div');
          tmp.innerHTML = buildAssistantHtml(botMsg, msgIdx, false);
          row.replaceWith(tmp.firstElementChild);
        }
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError' && !botMsg.content) {
      botMsg.content = botMsg.variants[Math.max(0, newVarIdx - 1)].content;
      botMsg.variants.pop();
      botMsg.activeVariant = Math.max(0, botMsg.variants.length - 1);
    }
  } finally {
    botMsg.streaming = false;
    botMsg.variants[botMsg.activeVariant] = { content: botMsg.content, thoughts: botMsg.thoughts };
    activeStreamCtrl = null;
    btn.classList.remove('streaming-mode');
    btn.innerHTML = ICO_LAUNCH;
    box.classList.remove('is-streaming');
    renderChat(false);
    syncCurrentThreadToBackend();
  }
}

function startInlineEdit(msgIdx) {
  _editingMsgIndex = msgIdx;
  renderChat(false);
  const ta = document.getElementById('inlineEditArea-' + msgIdx);
  if (ta) {
    ta.focus();
    ta.selectionStart = ta.value.length;
  }
}

function cancelInlineEdit() {
  _editingMsgIndex = -1;
  renderChat(false);
}

async function saveInlineEdit(msgIdx, andRegenerate = false) {
  const ta = document.getElementById('inlineEditArea-' + msgIdx);
  if (!ta || !currentMessages[msgIdx]) return;
  const val = ta.value.trim();
  if (!val) return;
  const m = currentMessages[msgIdx];
  m.content = val;
  if (m.role !== 'user') {
    ensureVariants(m);
    m.variants[m.activeVariant].content = val;
    /* Record preferred style anchor for online adaptation & LoRA dataset */
    const prevUser = msgIdx > 0 && currentMessages[msgIdx - 1].role === 'user' ? currentMessages[msgIdx - 1].content : '';
    if (prevUser) {
      NeverAIVault.addExperience(prevUser, val, 'edit');
      fetch('/api/experience/record', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prevUser, preferred_reply: val, kind: 'edit' }),
      }).catch(() => {});
    }
  }
  _editingMsgIndex = -1;
  if (andRegenerate && m.role === 'user') {
    currentMessages = currentMessages.slice(0, msgIdx + 1);
    await syncCurrentThreadToBackend();
    renderChat(false);
    if (currentMessages[msgIdx]) {
      /* Create fresh assistant reply for the edited user message */
      const botMsg = { role: 'never', content: '', thoughts: [''], web_sources: [], streaming: true, variants: [{ content: '', thoughts: [''] }], activeVariant: 0 };
      currentMessages.push(botMsg);
      await generateVariantAt(currentMessages.length - 1);
      /* Collapse initial placeholder if 2nd variant was pushed */
      if (botMsg.variants.length > 1 && !botMsg.variants[0].content) {
        botMsg.variants.shift();
        botMsg.activeVariant = 0;
      }
      renderChat(false);
      await syncCurrentThreadToBackend();
    }
    return;
  }
  renderChat(false);
  await syncCurrentThreadToBackend();
  toast('Message mis à jour');
}

async function forkThreadAt(msgIdx) {
  if (!currentThreadId) return;
  try {
    const r = await fetch('/api/threads/' + encodeURIComponent(currentThreadId) + '/fork', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ up_to_index: msgIdx, owner: currentOwner() }),
    });
    const d = await r.json();
    if (d && d.thread) {
      toast('Branche créée : ' + (d.thread.title || 'Nouvelle branche'));
      await loadThreads();
      await openThread(d.thread.id);
    }
  } catch (_) {
    toast('Impossible de bifurquer cette discussion');
  }
}

function filterSidebarThreads(q) {
  _sidebarSearchQuery = String(q || '').toLowerCase().trim();
  loadThreads();
}
