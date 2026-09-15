/**
 * crossby UI — drives one AI tool session running on a server-side PTY.
 *
 * Output arrives base64-encoded over SSE (a read boundary routinely splits a
 * UTF-8 sequence or an escape, so bytes travel as bytes and xterm.js does the
 * decoding). Input and resize events go back as small POSTs; on loopback the
 * round trip is negligible.
 */
"use strict";

const TOKEN = new URLSearchParams(location.search).get("token") || "";

const el = (id) => document.getElementById(id);
const ui = {
  form: el("launch-form"), tool: el("tool"), model: el("model"), effort: el("effort"),
  message: el("initial-message"), yolo: el("yolo"), launch: el("launch"), stop: el("stop"),
  error: el("error"), modelField: el("model-field"), effortField: el("effort-field"),
  messageField: el("message-field"), yoloField: el("yolo-field"), placeholder: el("placeholder"),
  statusDot: el("status-dot"), statusText: el("status-text"), meta: el("session-meta"),
  metaStatus: el("meta-status"), metaPid: el("meta-pid"), metaSize: el("meta-size"),
  metaCommand: el("meta-command"), projectRoot: el("project-root"),
};

let tools = [];
let session = null;
let stream = null;

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

/* ---------------- terminal ---------------- */

const term = new Terminal({
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
term.open(el("terminal"));

/**
 * Keystrokes are coalesced and sent strictly one request at a time: two
 * concurrent POSTs could otherwise be applied out of order, which corrupts
 * input in ways that are miserable to debug.
 */
let pending = "";
let inFlight = false;

function queueInput(data) {
  if (!session) return;
  pending += data;
  void flushInput();
}

async function flushInput() {
  if (inFlight || !pending || !session) return;
  inFlight = true;
  const data = pending;
  pending = "";
  try {
    await api("POST", `/api/sessions/${session.id}/input`, { data });
  } catch (err) {
    showError(err.message);
  } finally {
    inFlight = false;
    if (pending) void flushInput();
  }
}

term.onData(queueInput);

let resizeTimer = null;
function scheduleFit() {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    try { fit.fit(); } catch { /* terminal not visible yet */ }
  }, 80);
}
window.addEventListener("resize", scheduleFit);

term.onResize(({ cols, rows }) => {
  ui.metaSize.textContent = `${cols}×${rows}`;
  if (!session) return;
  api("POST", `/api/sessions/${session.id}/resize`, { cols, rows }).catch((err) =>
    showError(err.message),
  );
});

/* ---------------- session lifecycle ---------------- */

function openStream(id) {
  closeStream();
  stream = new EventSource(`/api/sessions/${id}/stream?token=${encodeURIComponent(TOKEN)}`);

  stream.addEventListener("output", (event) => {
    const binary = atob(event.data);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
    term.write(bytes);
  });

  stream.addEventListener("exit", (event) => {
    let code = null;
    try { code = JSON.parse(event.data).exit_code; } catch { /* keep null */ }
    markExited(code);
  });

  stream.onerror = () => {
    // EventSource retries on its own; only a finished session is terminal.
    if (session && !session.running) closeStream();
  };
}

function closeStream() {
  if (stream) { stream.close(); stream = null; }
}

async function startSession(event) {
  event.preventDefault();
  hideError();
  ui.launch.disabled = true;
  ui.launch.textContent = "Starting…";

  try {
    fit.fit();
  } catch { /* fall through to defaults */ }

  const body = {
    tool: ui.tool.value,
    model: ui.model.value || null,
    effort: ui.effort.value || null,
    yolo: ui.yolo.checked,
    initial_message: ui.message.value.trim() || null,
    cols: term.cols,
    rows: term.rows,
  };

  try {
    session = await api("POST", "/api/sessions", body);
    term.reset();
    term.focus();
    ui.placeholder.hidden = true;
    document.body.dataset.session = "live";
    ui.stop.hidden = false;
    ui.meta.hidden = false;
    ui.metaCommand.hidden = false;
    ui.metaCommand.textContent = session.command.join(" ");
    ui.metaPid.textContent = session.pid;
    ui.metaSize.textContent = `${session.cols}×${session.rows}`;
    setStatus("running", `${labelFor(session.tool)} · session ${session.id.slice(0, 8)}`);
    ui.metaStatus.textContent = "running";
    openStream(session.id);
    scheduleFit();
    // On a single-column layout the terminal may start below the fold.
    el("terminal").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    showError(err.message);
    setStatus("idle", "No session");
  } finally {
    ui.launch.disabled = false;
    ui.launch.textContent = "Launch session";
  }
}

async function stopSession() {
  if (!session) return;
  ui.stop.disabled = true;
  try {
    await api("DELETE", `/api/sessions/${session.id}`);
  } catch (err) {
    showError(err.message);
  } finally {
    ui.stop.disabled = false;
  }
}

function markExited(code) {
  closeStream();
  if (session) session.running = false;
  const suffix = code === null || code === undefined ? "" : ` (exit ${code})`;
  setStatus("exited", `Session ended${suffix}`);
  ui.metaStatus.textContent = `exited${suffix}`;
  ui.stop.hidden = true;
  document.body.dataset.session = "ended";
  term.write(`\r\n\x1b[2m── session ended${suffix} ──\x1b[0m\r\n`);
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

function setStatus(state, text) {
  ui.statusDot.dataset.state = state;
  ui.statusText.textContent = text;
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = false;
}

function hideError() {
  ui.error.hidden = true;
}

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
    return;
  }

  ui.tool.replaceChildren();
  tools.forEach((tool) => ui.tool.add(new Option(tool.display_name, tool.id)));
  syncFormToTool();
  ui.tool.addEventListener("change", syncFormToTool);
  ui.form.addEventListener("submit", startSession);
  ui.stop.addEventListener("click", stopSession);
  scheduleFit();
}

window.addEventListener("beforeunload", closeStream);
void boot();
