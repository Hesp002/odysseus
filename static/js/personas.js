// ============================================
// Personas — sidebar section (between Email and Tools)
// ============================================
// A persona is a character a NEW chat talks as ("Respond to me as Marcus
// Aurelius") with its own memory; tools, email, web and documents stay with
// the default Odysseus. Clicking a persona starts a new chat with it. Backend:
// routes/persona_routes.py, src/personas.py.

import uiModule from './ui.js';
import sessionModule from './sessions.js';

const API_BASE = window.location.origin;

let _personas = [];      // [{id, name, personality, default?}], Odysseus first (id null)
let _activeId = null;    // persona of the current/pending chat (null = Odysseus)
let _syncSeq = 0;

function _el(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}

const _ICON_PERSONA = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;opacity:0.5;"><circle cx="12" cy="8" r="4"/><path d="M4 21v-1a6 6 0 0 1 6-6h4a6 6 0 0 1 6 6v1"/></svg>';
const _ICON_EDIT = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>';
const _ICON_DELETE = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>';

export async function loadPersonas() {
  try {
    const res = await fetch(`${API_BASE}/api/personas`, { credentials: 'same-origin' });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    _personas = Array.isArray(data.personas) ? data.personas : [];
  } catch (e) {
    console.warn('[personas] failed to load', e);
    _personas = [{ id: null, name: 'Odysseus', personality: '', default: true }];
  }
  renderPersonaList();
}

export function renderPersonaList() {
  const list = document.getElementById('persona-list');
  if (!list) return;
  list.innerHTML = '';
  _personas.forEach(p => {
    const row = _el('div', 'list-item persona-item');
    row.dataset.personaId = p.id || '';
    row.title = p.id
      ? `New chat with ${p.name} (own memory, no tools)`
      : 'New chat with Odysseus (memory, tools, email, web)';
    if ((p.id || null) === _activeId) row.classList.add('active-session');
    row.innerHTML = _ICON_PERSONA;
    row.appendChild(_el('span', 'grow', p.name));
    if (p.id) {
      const actions = _el('span', 'persona-actions');
      const edit = _el('button', 'persona-action-btn');
      edit.type = 'button';
      edit.title = `Edit ${p.name}`;
      edit.innerHTML = _ICON_EDIT;
      edit.addEventListener('click', (e) => { e.stopPropagation(); openPersonaEditor(p); });
      const del = _el('button', 'persona-action-btn');
      del.type = 'button';
      del.title = `Delete ${p.name}`;
      del.innerHTML = _ICON_DELETE;
      del.addEventListener('click', (e) => { e.stopPropagation(); deletePersona(p); });
      actions.append(edit, del);
      row.appendChild(actions);
    }
    row.addEventListener('click', () => startChat(p));
    list.appendChild(row);
  });
}

export async function startChat(persona) {
  await sessionModule.startPersonaChat(persona && persona.id ? persona : null);
}

/** Highlight the persona of the current/pending chat. */
export function syncActive(personaId) {
  _activeId = personaId || null;
  document.querySelectorAll('#persona-list .persona-item').forEach(row => {
    row.classList.toggle('active-session', (row.dataset.personaId || null) === _activeId);
  });
}

/** Look up and highlight the persona of an existing chat. */
export async function syncForSession(sessionId) {
  const seq = ++_syncSeq;
  if (!sessionId) { syncActive(null); return; }
  try {
    const res = await fetch(`${API_BASE}/api/personas/session/${encodeURIComponent(sessionId)}`, { credentials: 'same-origin' });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    if (seq !== _syncSeq) return;
    syncActive(data.persona ? data.persona.id : null);
  } catch (_) {
    if (seq === _syncSeq) syncActive(null);
  }
}

async function deletePersona(p) {
  const ok = await uiModule.styledConfirm(
    `Delete "${p.name}"?\n\nThis also deletes everything ${p.name} remembers about you. Existing chats with ${p.name} stay, but continue as Odysseus.`,
    { confirmText: 'Delete', danger: true, title: 'Delete persona' },
  );
  if (!ok) return;
  try {
    const res = await fetch(`${API_BASE}/api/personas/${encodeURIComponent(p.id)}`, { method: 'DELETE', credentials: 'same-origin' });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const data = await res.json();
    uiModule.showToast(`Deleted ${p.name}` + (data.memories_removed ? ` and ${data.memories_removed} memories` : ''));
    if (_activeId === p.id) _activeId = null;
    await loadPersonas();
  } catch (e) {
    uiModule.showError('Failed to delete persona: ' + e.message);
  }
}

// ── Editor dialog ───────────────────────────────────────────────────

function _ensureEditor() {
  let overlay = document.getElementById('persona-editor-overlay');
  if (overlay) return overlay;
  overlay = _el('div', 'modal hidden');
  overlay.id = 'persona-editor-overlay';
  overlay.innerHTML =
    '<div class="modal-content styled-confirm-box persona-editor-box" role="dialog" aria-modal="true" aria-labelledby="persona-editor-title">' +
      '<div class="modal-header"><h4 id="persona-editor-title"></h4></div>' +
      '<div class="modal-body">' +
        '<label class="persona-editor-label" for="persona-editor-name">Name</label>' +
        '<input type="text" id="persona-editor-name" class="styled-prompt-input" maxlength="80" placeholder="Marcus Aurelius" />' +
        '<label class="persona-editor-label" for="persona-editor-prompt">How should they respond?</label>' +
        '<textarea id="persona-editor-prompt" class="styled-prompt-input persona-editor-prompt" rows="7" maxlength="8000" placeholder="Respond to me as Marcus Aurelius: a Stoic emperor writing calm, practical reflections. Draw on the Meditations."></textarea>' +
        '<p class="persona-editor-hint">Personas only talk and remember. Tools, email, web search and documents stay with Odysseus.</p>' +
      '</div>' +
      '<div class="modal-footer">' +
        '<button type="button" id="persona-editor-expand" class="confirm-btn confirm-btn-secondary" title="Have the AI write a fuller character description from the name and your notes">Expand with AI</button>' +
        '<span style="flex:1"></span>' +
        '<button type="button" id="persona-editor-cancel" class="confirm-btn confirm-btn-secondary">Cancel</button>' +
        '<button type="button" id="persona-editor-save" class="confirm-btn confirm-btn-primary">Save</button>' +
      '</div>' +
    '</div>';
  document.body.appendChild(overlay);
  return overlay;
}

export function openPersonaEditor(persona = null) {
  const overlay = _ensureEditor();
  const title = document.getElementById('persona-editor-title');
  const nameInput = document.getElementById('persona-editor-name');
  const promptInput = document.getElementById('persona-editor-prompt');
  const saveBtn = document.getElementById('persona-editor-save');
  const cancelBtn = document.getElementById('persona-editor-cancel');
  const expandBtn = document.getElementById('persona-editor-expand');

  title.textContent = persona ? `Edit ${persona.name}` : 'New persona';
  nameInput.value = persona ? persona.name : '';
  promptInput.value = persona ? (persona.personality || '') : '';
  saveBtn.disabled = false;
  expandBtn.disabled = false;
  expandBtn.textContent = 'Expand with AI';

  const prevFocus = document.activeElement;
  overlay.classList.remove('hidden');
  overlay.style.display = '';
  setTimeout(() => nameInput.focus(), 0);

  function close() {
    overlay.classList.add('hidden');
    overlay.style.display = 'none';
    saveBtn.removeEventListener('click', onSave);
    cancelBtn.removeEventListener('click', close);
    expandBtn.removeEventListener('click', onExpand);
    overlay.removeEventListener('click', onBackdrop);
    document.removeEventListener('keydown', onKey, true);
    try { prevFocus && prevFocus.focus && prevFocus.focus(); } catch (_) {}
  }
  function onBackdrop(e) { if (e.target === overlay) close(); }
  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); }
  }

  async function onExpand() {
    const name = nameInput.value.trim();
    const draft = promptInput.value.trim();
    if (!name && !draft) { uiModule.showError('Enter a name or a few notes first.'); return; }
    expandBtn.disabled = true;
    expandBtn.textContent = 'Expanding…';
    try {
      const res = await fetch(`${API_BASE}/api/presets/expand`, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, prompt: draft }),
      });
      const data = await res.json();
      if (!data.success || !data.prompt) throw new Error(data.message || 'No result');
      promptInput.value = data.prompt;
    } catch (e) {
      uiModule.showError('Expand failed: ' + e.message);
    } finally {
      expandBtn.disabled = false;
      expandBtn.textContent = 'Expand with AI';
    }
  }

  async function onSave() {
    const name = nameInput.value.trim();
    const personality = promptInput.value.trim();
    if (!name) { uiModule.showError('A persona needs a name.'); nameInput.focus(); return; }
    saveBtn.disabled = true;
    try {
      const url = persona
        ? `${API_BASE}/api/personas/${encodeURIComponent(persona.id)}`
        : `${API_BASE}/api/personas`;
      const res = await fetch(url, {
        method: persona ? 'PUT' : 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, personality }),
      });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
      close();
      uiModule.showToast(persona ? `Saved ${name}` : `Created ${name}`);
      await loadPersonas();
    } catch (e) {
      saveBtn.disabled = false;
      uiModule.showError('Failed to save persona: ' + e.message);
    }
  }

  saveBtn.addEventListener('click', onSave);
  cancelBtn.addEventListener('click', close);
  expandBtn.addEventListener('click', onExpand);
  overlay.addEventListener('click', onBackdrop);
  document.addEventListener('keydown', onKey, true);
}

function init() {
  const newBtn = document.getElementById('persona-new-btn');
  if (newBtn) newBtn.addEventListener('click', (e) => { e.stopPropagation(); openPersonaEditor(null); });
  loadPersonas();
}

const personaModule = { loadPersonas, renderPersonaList, startChat, syncActive, syncForSession, openPersonaEditor };
try { window.personaModule = personaModule; } catch (_) {}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}

export default personaModule;
