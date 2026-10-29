// static/js/raw_requests.js
//
// Fork: viewer for the exact model request bodies captured for a chat
// (Settings → Appearance → "Raw requests"; src/request_capture.py). The
// server keeps the last few in memory only, so this is a debugging window,
// not history.

const _h = (tag, cls, text) => {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  return el;
};

const _approxTokens = (s) => Math.ceil(String(s || '').length / 4);
const _fmt = (n) => Number(n || 0).toLocaleString();

function _section(title, meta, body, open = false) {
  const det = _h('details', 'rawreq-section');
  det.open = open;
  const sum = _h('summary');
  sum.appendChild(_h('span', 'rawreq-section-title', title));
  if (meta) sum.appendChild(_h('span', 'rawreq-meta', meta));
  det.appendChild(sum);
  // Render the (possibly large) body only when first opened.
  let built = false;
  const build = () => {
    if (built) return;
    built = true;
    det.appendChild(typeof body === 'function' ? body() : body);
  };
  if (open) build();
  det.addEventListener('toggle', () => { if (det.open) build(); });
  return det;
}

const _pre = (text) => _h('pre', 'rawreq-pre', text);

function _contentText(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content.map((part) => {
      if (typeof part === 'string') return part;
      if (part?.type === 'text') return part.text || '';
      if (part?.type === 'image_url' || part?.type === 'image') return '[image]';
      return JSON.stringify(part, null, 2);
    }).join('\n');
  }
  return content == null ? '' : JSON.stringify(content, null, 2);
}

function _renderRequest(rec) {
  const card = _h('div', 'rawreq-card');
  let payload = null;
  try { payload = JSON.parse(rec.body); } catch { /* shown raw below */ }

  const head = _h('div', 'rawreq-card-head');
  const when = new Date(rec.time * 1000).toLocaleTimeString();
  let path = rec.url;
  try { path = new URL(rec.url).pathname; } catch { /* keep url */ }
  head.appendChild(_h('span', 'rawreq-source', rec.source || 'request'));
  head.appendChild(_h('span', 'rawreq-meta',
    `${when} · ${payload?.model || '?'} · ${path} · ${_fmt(rec.bytes)} bytes ≈ ${_fmt(_approxTokens(rec.body))} tokens`
    + (rec.truncated ? ' · TRUNCATED' : '')));
  const copy = _h('button', 'rawreq-btn', 'Copy JSON');
  copy.type = 'button';
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(rec.body);
      copy.textContent = 'Copied';
    } catch {
      copy.textContent = 'Copy failed';
    }
    setTimeout(() => { copy.textContent = 'Copy JSON'; }, 1500);
  });
  head.appendChild(copy);
  card.appendChild(head);

  if (!payload || typeof payload !== 'object') {
    card.appendChild(_section('Body', null, _pre(rec.body), true));
    return card;
  }

  const { messages, tools, prompt, system, ...rest } = payload;
  if (system != null) {
    const text = _contentText(system);
    card.appendChild(_section('system', `≈ ${_fmt(_approxTokens(text))} tokens`, () => _pre(text)));
  }
  if (Array.isArray(messages)) {
    const list = _h('div', 'rawreq-messages');
    messages.forEach((m, i) => {
      const text = _contentText(m?.content);
      const extra = m?.tool_calls ? `\n\n[tool_calls]\n${JSON.stringify(m.tool_calls, null, 2)}` : '';
      const label = `${i + 1}. ${m?.role || '?'}${m?.name ? ` (${m.name})` : ''}`;
      list.appendChild(_section(label, `≈ ${_fmt(_approxTokens(text + extra))} tokens`,
        () => _pre(text + extra)));
    });
    card.appendChild(_section(`Messages (${messages.length})`, null, list, true));
  }
  if (typeof prompt === 'string') {
    card.appendChild(_section('Prompt (raw)', `≈ ${_fmt(_approxTokens(prompt))} tokens`, () => _pre(prompt), true));
  }
  if (Array.isArray(tools) && tools.length) {
    const json = JSON.stringify(tools, null, 2);
    const names = tools.map((t) => t?.function?.name || t?.name).filter(Boolean).join(', ');
    card.appendChild(_section(`Tools (${tools.length})`, `≈ ${_fmt(_approxTokens(json))} tokens`, () => {
      const wrap = _h('div');
      wrap.appendChild(_h('div', 'rawreq-meta rawreq-toolnames', names));
      wrap.appendChild(_pre(json));
      return wrap;
    }));
  }
  if (Object.keys(rest).length) {
    card.appendChild(_section('Parameters', null, () => _pre(JSON.stringify(rest, null, 2))));
  }
  card.appendChild(_section('Full JSON', null, () => _pre(JSON.stringify(payload, null, 2))));
  return card;
}

function _renderList(container, data) {
  container.replaceChildren();
  const rows = data?.requests || [];
  if (!rows.length) {
    container.appendChild(_h('div', 'rawreq-empty',
      'Nothing captured for this chat yet. With the toggle on, send a message and refresh. '
      + 'Captures are kept in memory only and clear when Odysseus restarts.'));
    return;
  }
  // Group by turn, newest turn first (rows arrive newest first).
  const turns = new Map();
  rows.forEach((r) => {
    if (!turns.has(r.turn_id)) turns.set(r.turn_id, []);
    turns.get(r.turn_id).push(r);
  });
  let n = turns.size;
  turns.forEach((reqs) => {
    reqs.reverse(); // oldest first within a turn: router, round 1, round 2...
    const turn = _h('div', 'rawreq-turn');
    turn.appendChild(_h('div', 'rawreq-turn-title',
      `Turn ${n--} · ${reqs.length} request${reqs.length === 1 ? '' : 's'}`));
    reqs.forEach((r) => turn.appendChild(_renderRequest(r)));
    container.appendChild(turn);
  });
}

export async function openRawRequestsViewer(sessionId, apiBase = '') {
  if (!sessionId) return;
  document.querySelector('.rawreq-overlay')?.remove();
  const overlay = _h('div', 'rawreq-overlay');
  const modal = _h('div', 'rawreq-modal');
  modal.setAttribute('role', 'dialog');
  modal.setAttribute('aria-label', 'Raw model requests');
  const bar = _h('div', 'rawreq-bar');
  bar.appendChild(_h('span', 'rawreq-title', 'Raw requests'));
  const spacer = _h('span', 'rawreq-spacer');
  bar.appendChild(spacer);
  const refresh = _h('button', 'rawreq-btn', 'Refresh');
  const clear = _h('button', 'rawreq-btn', 'Clear');
  const close = _h('button', 'rawreq-btn', 'Close');
  [refresh, clear, close].forEach((b) => { b.type = 'button'; bar.appendChild(b); });
  modal.appendChild(bar);
  modal.appendChild(_h('div', 'rawreq-meta rawreq-note',
    'Exact JSON bodies Odysseus sent to the model for this chat (memory-only, last few requests). '
    + 'Ollama then applies the model\'s chat template to this.'));
  const list = _h('div', 'rawreq-list');
  modal.appendChild(list);
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  const url = `${apiBase}/api/session/${encodeURIComponent(sessionId)}/raw-requests`;
  const load = async () => {
    list.replaceChildren(_h('div', 'rawreq-empty', 'Loading…'));
    try {
      const res = await fetch(url, { credentials: 'same-origin' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      _renderList(list, await res.json());
    } catch (err) {
      list.replaceChildren(_h('div', 'rawreq-empty', `Could not load: ${err.message || err}`));
    }
  };
  const dismiss = () => {
    overlay.remove();
    document.removeEventListener('keydown', onKey, true);
  };
  const onKey = (ev) => {
    if (ev.key === 'Escape') { ev.preventDefault(); ev.stopPropagation(); dismiss(); }
  };
  document.addEventListener('keydown', onKey, true);
  overlay.addEventListener('pointerdown', (ev) => { if (ev.target === overlay) dismiss(); });
  close.addEventListener('click', dismiss);
  refresh.addEventListener('click', load);
  clear.addEventListener('click', async () => {
    try { await fetch(url, { method: 'DELETE', credentials: 'same-origin' }); } catch { /* reload shows state */ }
    load();
  });
  await load();
}
