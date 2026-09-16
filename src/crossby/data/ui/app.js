/**
 * crossby UI — drives any number of AI tool sessions, each on a server-side PTY.
 *
 * Transport notes:
 * - **One** EventSource carries every session's output, tagged by session id. A
 *   browser allows ~6 HTTP/1.1 connections per origin; a stream per session
 *   spends one apiece, and the sixth terminal stalls every other request,
 *   keystrokes included.
 * - Output is base64 inside each frame, because a read boundary routinely
 *   splits a UTF-8 sequence or an escape and SSE is newline-framed text.
 * - Input and resize go back as small POSTs, serialized per session so two
 *   in-flight writes can never be applied out of order.
 */
"use strict";

const TOKEN = new URLSearchParams(location.search).get("token") || "";

const el = (id) => document.getElementById(id);
const ui = {
  form: el("launch-form"), tool: el("tool"), model: el("model"), effort: el("effort"),
  message: el("initial-message"), yolo: el("yolo"), launch: el("launch"), stop: el("stop"),
  error: el("error"), modelField: el("model-field"), effortField: el("effort-field"),
  messageField: el("message-field"), yoloField: el("yolo-field"), placeholder: el("placeholder"),
  stopHint: el("stop-hint"),
  tabs: el("tabs"), panes: el("panes"), meta: el("session-meta"), metaStatus: el("meta-status"),
  metaPid: el("meta-pid"), metaSize: el("meta-size"), metaCommand: el("meta-command"),
  projectRoot: el("project-root"),
};

let tools = [];
let stream = null;
let activeId = null;
/** @type {Map<string, object>} session id → terminal, addon, pane, tab, state */
const sessions = new Map();

/* ---------------- transport ---------------- */

async function api(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: {
      "X-Crossby-Token": TOKEN,
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `${method} ${path} failed (${response.status})`);
  return payload;
}

/* ---------------- one session ---------------- */

function createSession(info) {
  if (sessions.has(info.id)) return sessions.get(info.id);

  const pane = document.createElement("div");
  pane.className = "pane";
  pane.dataset.session = info.id;
  ui.panes.appendChild(pane);

  const term = new Terminal({
    // Construct at the PTY's *current* geometry rather than xterm's 80x24
    // default. Replayed scrollback is full of absolute cursor-positioning
    // escapes computed for the size the tool was drawing at; interpreting them
    // at any other width puts the content off-screen entirely. The pane is
    // refitted (and the new size pushed to the server) when it is activated.
    cols: info.cols || 80,
    rows: info.rows || 24,
    fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
    fontSize: 13,
    lineHeight: 1.2,
    cursorBlink: true,
    scrollback: 10000,
    allowProposedApi: true,
    theme: {
      background: "#0f1115", foreground: "#e4e7ee", cursor: "#5b9dff",
      selectionBackground: "#2b3450",
    },
  });
  const fit = new FitAddon.FitAddon();
  term.loadAddon(fit);
  term.open(pane);

  const entry = {
    id: info.id,
    info,
    term,
    fit,
    pane,
    running: info.running !== false,
    // Outgoing bytes are held until the tool produces its first output. xterm
    // answers device-attribute queries and focus events on its own; sent before
    // the tool switches the tty to raw mode, those replies get echoed as
    // literal `^[[I` / `^[[?1;2c` junk in the first frame.
    started: false,
    outbox: "",
    pending: "",
    inFlight: false,
  };

  term.onData((data) => {
    entry.pending += data;
    void flushInput(entry);
  });
  term.onResize(({ cols, rows }) => {
    if (entry.id === activeId) ui.metaSize.textContent = `${cols}×${rows}`;
    if (!entry.running) return;
    api("POST", `/api/sessions/${entry.id}/resize`, { cols, rows }).catch((err) =>
      showError(err.message),
    );
  });

  entry.tab = createTab(entry);
  sessions.set(info.id, entry);
  ui.placeholder.hidden = true;
  document.body.dataset.session = "live";
  return entry;
}

async function flushInput(entry) {
  if (entry.inFlight || !entry.pending || !entry.running) return;
  if (!entry.started) return;      // held until the tool has drawn something
  entry.inFlight = true;
  const data = entry.pending;
  entry.pending = "";
  try {
    await api("POST", `/api/sessions/${entry.id}/input`, { data });
  } catch (err) {
    showError(err.message);
  } finally {
    entry.inFlight = false;
    if (entry.pending) void flushInput(entry);
  }
}

function createTab(entry) {
  const tab = document.createElement("button");
  tab.type = "button";
  tab.className = "tab";
  tab.setAttribute("role", "tab");
  tab.dataset.session = entry.id;

  const dot = document.createElement("span");
  dot.className = "dot";
  dot.dataset.state = entry.running ? "running" : "exited";

  const label = document.createElement("span");
  label.textContent = labelFor(entry.info.tool) || "session";

  const close = document.createElement("span");
  close.className = "tab-close";
  close.textContent = "×";
  close.title = "Close session";
  close.addEventListener("click", (event) => {
    event.stopPropagation();
    void closeSession(entry.id);
  });

  tab.append(dot, label, close);
  tab.addEventListener("click", () => activate(entry.id));
  ui.tabs.appendChild(tab);
  entry.dot = dot;
  return tab;
}

function activate(id) {
  const entry = sessions.get(id);
  if (!entry) return;
  activeId = id;
  for (const other of sessions.values()) {
    const isActive = other.id === id;
    other.pane.dataset.active = String(isActive);
    other.tab.setAttribute("aria-selected", String(isActive));
  }
  refit(entry);
  // xterm parses writes into its buffer regardless of visibility, but does not
  // paint into a hidden container and does not repaint merely on becoming
  // visible. Without this, a background tab shows blank until something else
  // forces a redraw — most visibly after a reload, when every restored tab but
  // the active one looked empty.
  entry.term.refresh(0, entry.term.rows - 1);
  if (entry.running) entry.term.focus();
  renderMeta(entry);
  ui.stop.hidden = !entry.running;
  ui.stopHint.hidden = !entry.running;
}

function refit(entry) {
  try {
    entry.fit.fit();
  } catch {
    /* pane not measurable yet */
  }
}

function renderMeta(entry) {
  ui.meta.hidden = false;
  ui.metaCommand.hidden = false;
  ui.metaCommand.textContent = (entry.info.command || []).join(" ");
  ui.metaPid.textContent = entry.info.pid ?? "—";
  ui.metaSize.textContent = `${entry.term.cols}×${entry.term.rows}`;
  ui.metaStatus.textContent = entry.running ? "running" : entry.endedLabel || "ended";
}

/**
 * A negative exit code is a signal death, which surfaces as a baffling
 * "exit -1" unless translated — and a session the user stopped is not a crash.
 */
function endedLabel({ stopped, exit_code: code, exit_signal: signal }) {
  if (stopped) return "stopped";
  if (signal) return `killed (${signal})`;
  if (code === 0 || code === null || code === undefined) return "ended";
  return `exit ${code}`;
}

function markExited(id, detail) {
  const entry = sessions.get(id);
  if (!entry || !entry.running) return;
  entry.running = false;
  entry.endedLabel = endedLabel(detail);
  entry.dot.dataset.state = detail.stopped ? "stopped" : "exited";
  entry.term.write(`\r\n\x1b[2m── session ${entry.endedLabel} ──\x1b[0m\r\n`);
  if (entry.id === activeId) {
    ui.stop.hidden = true;
    ui.stopHint.hidden = true;
    renderMeta(entry);
  }
}

async function stopSession(id) {
  const entry = sessions.get(id);
  if (!entry || !entry.running) return;
  ui.stop.disabled = true;
  try {
    await api("DELETE", `/api/sessions/${id}`);
    // The stream's exit frame normally lands first; mark it here too so the tab
    // settles immediately even if that frame is delayed.
    markExited(id, { stopped: true, exit_code: null, exit_signal: null });
  } catch (err) {
    showError(err.message);
  } finally {
    ui.stop.disabled = false;
  }
}

async function closeSession(id) {
  const entry = sessions.get(id);
  if (!entry) return;
  try {
    if (entry.running) await api("DELETE", `/api/sessions/${id}`);
  } catch (err) {
    showError(err.message);
  }
  entry.term.dispose();
  entry.pane.remove();
  entry.tab.remove();
  sessions.delete(id);
  if (activeId === id) {
    activeId = null;
    const next = sessions.keys().next();
    if (next.done) {
      ui.placeholder.hidden = false;
      ui.meta.hidden = true;
      ui.metaCommand.hidden = true;
      ui.stop.hidden = true;
      ui.stopHint.hidden = true;
      document.body.dataset.session = "ended";
    } else {
      activate(next.value);
    }
  }
}

/* ---------------- multiplexed stream ---------------- */

/** Frames for a session whose geometry has not been fetched yet. */
const pendingAdoption = new Map();

function adopt(id, chunk) {
  const held = pendingAdoption.get(id);
  if (held) {
    held.push(chunk);
    return;
  }
  pendingAdoption.set(id, [chunk]);
  api("GET", `/api/sessions/${id}`)
    .then((info) => {
      const entry = createSession(info);
      for (const pending of pendingAdoption.get(id) || []) writeChunk(entry, pending);
      pendingAdoption.delete(id);
      if (activeId === null) activate(id);
    })
    .catch((err) => {
      pendingAdoption.delete(id);
      showError(err.message);
    });
}

function writeChunk(entry, chunk) {
  const binary = atob(chunk);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  entry.term.write(bytes);
  if (!entry.started) {
    entry.started = true;                 // tool is drawing; safe to send now
    if (entry.pending) void flushInput(entry);
    if (entry.id === activeId) entry.term.focus();
  }
}

function openStream() {
  if (stream) stream.close();
  stream = new EventSource(`/api/stream?token=${encodeURIComponent(TOKEN)}`);

  stream.addEventListener("output", (event) => {
    const { session: id, chunk } = JSON.parse(event.data);
    const entry = sessions.get(id);
    if (!entry) {
      // A session this page has not seen. Its geometry is unknown and guessing
      // would corrupt the replay, so hold the frames until the server tells us
      // the real size.
      adopt(id, chunk);
      return;
    }
    writeChunk(entry, chunk);
  });

  stream.addEventListener("exit", (event) => {
    const detail = JSON.parse(event.data);
    markExited(detail.session, detail);
  });

  // EventSource reconnects on its own. On reconnect the server replays every
  // session's scrollback, so a reset keeps the screen from doubling up.
  stream.onerror = () => {
    for (const entry of sessions.values()) {
      if (entry.running) entry.term.reset();
    }
  };
}

/* ---------------- launching ---------------- */

async function startSession(event) {
  event.preventDefault();
  hideError();
  ui.launch.disabled = true;
  ui.launch.textContent = "Starting…";

  // Size the new session from the active terminal, or a sane default.
  const reference = sessions.get(activeId);
  const cols = reference ? reference.term.cols : 80;
  const rows = reference ? reference.term.rows : 24;

  try {
    const info = await api("POST", "/api/sessions", {
      tool: ui.tool.value,
      model: ui.model.value || null,
      effort: ui.effort.value || null,
      yolo: ui.yolo.checked,
      initial_message: ui.message.value.trim() || null,
      cols,
      rows,
    });
    const entry = createSession(info);
    activate(info.id);
    refit(entry);
    // Tell the server the size this pane actually resolved to.
    if (entry.term.cols !== cols || entry.term.rows !== rows) {
      api("POST", `/api/sessions/${info.id}/resize`, {
        cols: entry.term.cols, rows: entry.term.rows,
      }).catch(() => {});
    }
    ui.panes.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    showError(err.message);
  } finally {
    ui.launch.disabled = false;
    ui.launch.textContent = "Launch session";
  }
}

/* ---------------- form ---------------- */

function labelFor(id) {
  const match = tools.find((tool) => tool.id === id);
  return match ? match.display_name : id;
}

function syncFormToTool() {
  const selected = tools.find((tool) => tool.id === ui.tool.value);
  if (!selected) return;
  ui.modelField.hidden = !selected.supports_model_flag || selected.models.length === 0;
  ui.model.replaceChildren(new Option("Tool default", ""));
  selected.models.forEach((model) => ui.model.add(new Option(model, model)));
  ui.effortField.hidden = !selected.supports_effort;
  ui.effort.replaceChildren(new Option("Tool default", ""));
  selected.supported_efforts.forEach((level) => ui.effort.add(new Option(level, level)));
  ui.yoloField.hidden = !selected.supports_yolo;
  if (!selected.supports_yolo) ui.yolo.checked = false;
  ui.messageField.hidden = !selected.supports_initial_message;
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function hideError() {
  ui.error.hidden = true;
}

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const entry = sessions.get(activeId);
    if (entry) refit(entry);
  }, 80);
});

/* ---------------- boot ---------------- */

async function boot() {
  if (!TOKEN) {
    showError("Missing access token. Open the URL printed by `crossby ui`.");
    ui.launch.disabled = true;
    return;
  }
  try {
    const payload = await api("GET", "/api/tools");
    tools = payload.tools;
    ui.projectRoot.textContent = payload.project_root;
    ui.projectRoot.title = payload.project_root;
  } catch (err) {
    showError(err.message);
    ui.launch.disabled = true;
    return;
  }

  if (tools.length === 0) {
    showError("No terminal AI tools detected on PATH.");
    ui.launch.disabled = true;
  } else {
    ui.tool.replaceChildren();
    tools.forEach((tool) => ui.tool.add(new Option(tool.display_name, tool.id)));
    syncFormToTool();
    ui.tool.addEventListener("change", syncFormToTool);
    ui.form.addEventListener("submit", startSession);
  }
  ui.stop.addEventListener("click", () => stopSession(activeId));

  // Reattach to sessions that outlived the page: a reload must not orphan a
  // running tool that only the server can still see.
  try {
    const { sessions: existing } = await api("GET", "/api/sessions");
    for (const info of existing) {
      const entry = createSession(info);
      if (!info.running) markExited(info.id, info);
      if (activeId === null) activate(info.id);
    }
  } catch (err) {
    showError(err.message);
  }

  openStream();
}

window.addEventListener("beforeunload", () => stream && stream.close());
void boot();
