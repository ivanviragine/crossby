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

/** How long outgoing bytes wait for the tool's first output before being sent anyway. */
const INPUT_HOLD_MS = 1500;

const el = (id) => document.getElementById(id);
const ui = {
  form: el("launch-form"), tool: el("tool"), model: el("model"), effort: el("effort"),
  message: el("initial-message"), launch: el("launch"), stop: el("stop"),
  autonomy: el("autonomy"), autonomyField: el("autonomy-field"), autonomyNote: el("autonomy-note"),
  sandbox: el("sandbox"), sandboxField: el("sandbox-field"),
  network: el("network"), networkField: el("network-field"),
  error: el("error"), modelField: el("model-field"), effortField: el("effort-field"),
  messageField: el("message-field"), placeholder: el("placeholder"),
  stopHint: el("stop-hint"),
  tabs: el("tabs"), panes: el("panes"), meta: el("session-meta"), metaStatus: el("meta-status"),
  metaPid: el("meta-pid"), metaSize: el("meta-size"), metaCommand: el("meta-command"),
  projectRoot: el("project-root"),
  folder: el("folder"), folderPath: el("folder-path"), browser: el("browser"),
  browserUp: el("browser-up"), browserHere: el("browser-here"),
  browserList: el("browser-list"), browserPick: el("browser-pick"),
};

let tools = [];
let stream = null;
let activeId = null;
/** Directory new sessions launch in; null means the server's default. */
let workdir = null;
/** Directory the picker is currently showing, which may differ from `workdir`. */
let browsing = null;
/** @type {Map<string, object>} session id → terminal, addon, pane, tab, state */
const sessions = new Map();


/** How each autonomy rung reads in the form, and what it means. */
const AUTONOMY = {
  "default": ["Ask before acting", "The tool prompts for edits and commands."],
  "plan": ["Plan only (read-only)", "Native plan mode: the tool proposes, changes nothing."],
  "accept-edits": ["Auto-accept edits", "File edits apply without asking; commands still prompt."],
  "auto": ["Auto", "The tool's own classifier decides what needs asking."],
  "yolo": ["YOLO — skip all prompts", "No permission prompts at all. Use deliberately."],
};

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
  if (response.status === 404) {
    const missing = new Error(payload.error || "not found");
    missing.status = 404;
    throw missing;
  }
  if (response.status === 401) {
    // Every `crossby ui` run mints a new token, so a stale URL — a bookmark, a
    // reopened tab, a reload after a restart — is the usual cause. Say so,
    // rather than leaving someone to decode "invalid token".
    throw new Error(
      "This page's access token is no longer valid. The server was most likely " +
        "restarted — open the URL it printed most recently.",
    );
  }
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
    pending: "",
    inFlight: false,
  };

  // Releasing on first output alone deadlocks a tool that prints nothing until
  // it is written to — the input is then held forever. The hold is only a
  // mitigation for the echo race at startup, so it expires on its own.
  entry.releaseTimer = setTimeout(() => {
    if (!entry.started) {
      entry.started = true;
      if (entry.pending) void flushInput(entry);
    }
  }, INPUT_HOLD_MS);

  // Returning false keeps xterm from handling the key *and* from forwarding it
  // to the tool, so a tab switch never leaks a stray keystroke into the session.
  term.attachCustomKeyEventHandler((event) => {
    if (event.type !== "keydown") return true;
    return !isShortcut(event);
  });

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

  sessions.set(info.id, entry);
  entry.tab = createTab(entry);
  relabelTabs();
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

/**
 * Distinguish same-tool tabs. Two sessions both reading "Claude Code" gave no
 * way to tell which was which; an ordinal per tool does.
 */
function tabLabel(entry) {
  const name = labelFor(entry.info.tool) || "session";
  // With sessions across several folders the tool name alone is ambiguous, so
  // the folder becomes the distinguishing part.
  const folders = new Set([...sessions.values()].map((other) => other.info.cwd));
  if (folders.size > 1 && entry.info.cwd) {
    const leaf = entry.info.cwd.replace(/\/+$/, "").split("/").pop();
    if (leaf) return `${name} · ${leaf}`;
  }
  const sameTool = [...sessions.values()].filter(
    (other) => labelFor(other.info.tool) === name,
  );
  if (sameTool.length <= 1) return name;
  const ordinal = sameTool.findIndex((other) => other.id === entry.id) + 1;
  return `${name} ${ordinal || sameTool.length}`;
}

function relabelTabs() {
  const entries = [...sessions.values()];
  entries.forEach((entry, offset) => {
    const label = entry.tab && entry.tab.querySelector(".tab-label");
    if (label) label.textContent = tabLabel(entry);
    const index = entry.tab && entry.tab.querySelector(".tab-index");
    if (!index) return;
    // Mirror handleShortcut exactly: 1-8 are positional and 9 is always the
    // last tab. Numbering straight through would print a "9" on the ninth tab
    // while the key took you to the twelfth.
    const position = offset + 1;
    const isLast = offset === entries.length - 1;
    if (position <= 8) index.textContent = String(position);
    else if (isLast) index.textContent = "9";
    else index.textContent = "";
  });
}

function createTab(entry) {
  const tab = document.createElement("button");
  tab.type = "button";
  tab.className = "tab";
  tab.setAttribute("role", "tab");
  tab.dataset.session = entry.id;
  tab.title = "Click to switch; Delete or Backspace to close";

  const dot = document.createElement("span");
  dot.className = "dot";
  dot.dataset.state = entry.running ? "running" : "exited";

  const index = document.createElement("span");
  index.className = "tab-index";

  const label = document.createElement("span");
  label.className = "tab-label";
  label.textContent = tabLabel(entry);

  const close = document.createElement("span");
  close.className = "tab-close";
  close.textContent = "×";
  close.title = "Close session";
  close.setAttribute("aria-hidden", "true");
  close.addEventListener("click", (event) => {
    event.stopPropagation();
    void requestClose(entry.id);
  });

  tab.append(dot, index, label, close);
  tab.addEventListener("click", () => activate(entry.id));
  // The "x" is decorative, not a nested button — a button inside a button is
  // invalid and unreachable by keyboard. Closing is bound to the tab itself so
  // it works without a mouse.
  tab.addEventListener("keydown", (event) => {
    if (event.key === "Delete" || event.key === "Backspace") {
      event.preventDefault();
      void requestClose(entry.id);
    }
  });
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
  if (entry.restored) nudgeRedraw(entry);
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

/**
 * Make a reattached tool repaint from its own state.
 *
 * Replayed scrollback is a cushion, not a transcript: a tool with an idle
 * animation (Codex emits ~10.8 KB/s doing nothing) pushes real output out of the
 * buffer within seconds, so a reattaching viewer can replay nothing but
 * animation. A one-column resize makes the tool redraw what it is actually
 * showing, which no replay can reconstruct.
 */
function nudgeRedraw(entry) {
  if (!entry.running || entry.nudged) return;
  entry.nudged = true;
  const { cols, rows } = entry.term;
  if (cols <= 1) return;
  const resize = (c, r) =>
    api("POST", `/api/sessions/${entry.id}/resize`, { cols: c, rows: r }).catch(() => {});
  void resize(cols - 1, rows).then(() => setTimeout(() => void resize(cols, rows), 120));
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

/**
 * Record how a session ended.
 *
 * `announce` controls the closing line in the terminal, and is off while
 * restoring: the scrollback replay has not run yet at that point, so writing it
 * there would leave "session ended" sitting above the output it followed. The
 * stream delivers an exit frame after that session's backlog, which is when the
 * line belongs.
 *
 * Not gated on `entry.running`: a restored session is constructed
 * already-exited, and that guard discarded its stopped/exit_code/exit_signal
 * detail so every restored tab read a generic "ended".
 */
function markExited(id, detail, { announce = true } = {}) {
  const entry = sessions.get(id);
  if (!entry) return;

  if (!entry.endedLabel) {
    entry.running = false;
    entry.endedLabel = endedLabel(detail);
    entry.dot.dataset.state = detail.stopped ? "stopped" : "exited";
    if (entry.id === activeId) {
      ui.stop.hidden = true;
      ui.stopHint.hidden = true;
      renderMeta(entry);
    }
  }

  if (announce && !entry.announced) {
    entry.announced = true;
    entry.term.write(`\r\n\x1b[2m── session ${entry.endedLabel} ──\x1b[0m\r\n`);
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

/**
 * Closing a *running* session kills the tool, and the x sits a few pixels from
 * the tab label — easy to hit while aiming to switch tabs. Ask first. An
 * already-ended tab holds nothing but text, so it just closes.
 */
async function requestClose(id) {
  const entry = sessions.get(id);
  if (!entry) return;
  if (entry.running) {
    const label = entry.tab.querySelector(".tab-label");
    const name = (label && label.textContent) || "this session";
    if (!window.confirm(`End ${name}? The tool is still running.`)) return;
  }
  await closeSession(id);
}

async function closeSession(id) {
  const entry = sessions.get(id);
  if (!entry) return;

  // Always ask the server to drop it, running or not: an exited session stays
  // in the registry until a later launch reaps it, so skipping the DELETE meant
  // a reload re-adopted a tab the user had closed.
  try {
    await api("DELETE", `/api/sessions/${id}`);
  } catch (err) {
    // Already reaped is the outcome we wanted.
    if (err.status !== 404) {
      // Removing the tab now would strand a running tool with no controls.
      showError(`${err.message} — the session is still running.`);
      return;
    }
  }

  clearTimeout(entry.releaseTimer);
  entry.term.dispose();
  entry.pane.remove();
  entry.tab.remove();
  sessions.delete(id);
  relabelTabs();
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

/**
 * Exit frames that arrived before the page had a tab for the session.
 *
 * A short-lived tool can exit while the launch POST or the adoption fetch is
 * still in flight, and `markExited` drops a frame for a session it cannot
 * find. The snapshot those requests return may predate the exit, so the tab
 * they create would stay marked running forever. Holding the detail here lets
 * whichever path creates the tab settle it.
 *
 * Keying this off a live adoption was not enough: a tool that exits without
 * writing a byte produces nothing to adopt, so its frame was still lost.
 * Holding every tab-less id instead means ids that never get a tab — another
 * browser tab's silent session — would accumulate, so each entry expires.
 */
const pendingExit = new Map();
const PENDING_EXIT_TTL_MS = 60_000;

function holdExit(detail) {
  dropExit(detail.session);
  pendingExit.set(detail.session, {
    detail,
    timer: setTimeout(() => pendingExit.delete(detail.session), PENDING_EXIT_TTL_MS),
  });
}

/** Settle a freshly created tab from its held exit frame, if one is waiting. */
function settleExit(id) {
  const held = pendingExit.get(id);
  if (!held) return;
  dropExit(id);
  markExited(id, held.detail);
}

function dropExit(id) {
  const held = pendingExit.get(id);
  if (!held) return;
  clearTimeout(held.timer);
  pendingExit.delete(id);
}

/** Write a session's held frames into its tab, in arrival order, and clear them. */
function flushAdoption(entry) {
  for (const chunk of pendingAdoption.get(entry.id) || []) writeChunk(entry, chunk);
  pendingAdoption.delete(entry.id);
}

function adopt(id, chunk) {
  const held = pendingAdoption.get(id);
  if (held) {
    held.push(chunk);
    return;
  }
  pendingAdoption.set(id, [chunk]);
  api("GET", `/api/sessions/${id}`)
    .then((info) => {
      // The launch POST may have created the tab while this fetch was in
      // flight; a second createSession() would leave a duplicate pane behind.
      const existing = sessions.get(id);
      const entry = existing || createSession(info);
      if (!existing) entry.restored = true;   // a reattachment: needs a redraw nudge
      flushAdoption(entry);
      settleExit(id);             // after the backlog, so the closing line lands last
      if (activeId === null) activate(id);
    })
    .catch((err) => {
      pendingAdoption.delete(id);
      dropExit(id);
      showError(err.message);
    });
}

function writeChunk(entry, chunk) {
  // A reconnect arms this instead of clearing the terminal outright, because
  // only the replay can say whether clearing is safe: an exited session the
  // server has already reaped gets none, and its final output then exists
  // nowhere but here. Consuming the flag on the first chunk means the screen is
  // cleared exactly when there is something to rebuild it from.
  if (entry.pendingReset) {
    entry.pendingReset = false;
    entry.term.reset();
  }
  const binary = atob(chunk);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  entry.term.write(bytes);
  if (!entry.started) {
    entry.started = true;                 // tool is drawing; safe to send now
    clearTimeout(entry.releaseTimer);
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
    // `pendingAdoption` is checked even when a tab exists: the launch POST can
    // create it while adoption still holds earlier frames, and writing this one
    // straight through would put it *before* them. Terminal output is a stateful
    // escape stream, so reordering corrupts the screen, not just the scrollback.
    if (!entry || pendingAdoption.has(id)) {
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
    if (!sessions.has(detail.session)) {
      // No tab yet — a launch or an adoption is still in flight, or the
      // session belongs to another page. Hold the detail so the tab, if one
      // appears, shows how the session ended rather than staying marked
      // running.
      holdExit(detail);
      return;
    }
    markExited(detail.session, detail);
  });

  // A reconnect replays every session's scrollback, so each terminal has to be
  // cleared to stop the screen doubling up — but only those the server will
  // actually replay. `create()` reaps exited sessions from the registry, so a
  // tab whose tool finished before the last launch gets no replay at all, and
  // clearing it would wipe a final transcript that survives only in this
  // browser. Arming the reset and letting the first replayed chunk spend it
  // covers both without asking the server which sessions it still holds.
  //
  // This belongs on `open`, not `error`: error fires on every failed attempt
  // too, which wiped the terminals with no replay coming.
  let opened = false;
  stream.onopen = () => {
    if (opened) {
      for (const entry of sessions.values()) {
        entry.pendingReset = true;
        // The replay is a cushion, not a transcript. A tool with an idle
        // animation pushes its real screen out of that buffer within seconds,
        // so replaying alone can leave a live session blank or stale. Every
        // running tool is asked to redraw from its own state — which previously
        // happened only for tabs restored on load, so a session launched here
        // and then briefly disconnected had no way back. Clearing the one-shot
        // flag also lets a second reconnect nudge again.
        //
        // An exited session needs none of this: it has nothing left to redraw,
        // and nudgeRedraw ignores it.
        entry.nudged = false;
        nudgeRedraw(entry);
      }
    }
    opened = true;
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
      cwd: workdir,
      autonomy: ui.autonomy.value || "default",
      initial_message: ui.message.value.trim() || null,
      sandbox: ui.sandboxField.hidden ? true : ui.sandbox.checked,
      network_access: ui.networkField.hidden ? false : ui.network.checked,
      cols,
      rows,
    });
    // Adoption may already have built the tab from a first output frame.
    const entry = sessions.get(info.id) || createSession(info);
    // Output can beat this response. Draining the held frames here — rather
    // than waiting for adoption's metadata GET to resolve — is what keeps them
    // ahead of everything that arrives next, and it costs no round trip since
    // the POST already told us the geometry.
    flushAdoption(entry);
    // A tool that exited before this response landed has its frame held; apply
    // it now, or the tab reads as running for the rest of the page's life.
    settleExit(info.id);
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

  // Only the rungs this adapter implements — the tools genuinely differ
  // (OpenCode has no YOLO, Codex no plan mode, only Claude has auto).
  ui.autonomy.replaceChildren();
  selected.autonomy.forEach((mode) => {
    const [label] = AUTONOMY[mode] || [mode];
    ui.autonomy.add(new Option(label, mode));
  });
  ui.autonomyField.hidden = selected.autonomy.length <= 1;
  syncAutonomyNote();

  ui.sandboxField.hidden = !selected.supports_sandbox_toggle;
  if (!selected.supports_sandbox_toggle) ui.sandbox.checked = true;
  ui.networkField.hidden = !selected.supports_network_access;
  if (!selected.supports_network_access) ui.network.checked = false;

  ui.messageField.hidden = !selected.supports_initial_message;
}

function syncAutonomyNote() {
  const entry = AUTONOMY[ui.autonomy.value];
  ui.autonomyNote.textContent = entry ? entry[1] : "";
  // Plan mode is exclusive: the tool changes nothing, so an initial message
  // would be a planning brief rather than a task.
  const planning = ui.autonomy.value === "plan";
  ui.message.placeholder = planning
    ? "What should it plan?"
    : "Ask for something to start with…";
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

/* ---------------- folder picker ---------------- */

const LAST_FOLDER_KEY = "crossby.lastFolder";

/**
 * Shorten a path for display, keeping the tail — the informative part.
 *
 * Truncation happens here rather than via CSS: `direction: rtl` does ellipsize
 * at the start, but it also reorders the path's leading "/" to the end, which
 * reads as a stray trailing slash.
 */
function shortenPath(path, limit = 38) {
  const home = path.replace(/^\/(Users|home)\/[^/]+/, "~");
  return home.length <= limit ? home : `…${home.slice(-(limit - 1))}`;
}

function setWorkdir(path) {
  workdir = path;
  ui.folderPath.textContent = shortenPath(path);
  ui.folder.title = path;
  try {
    localStorage.setItem(LAST_FOLDER_KEY, path);
  } catch {
    /* private window, or storage disabled — the choice just will not persist */
  }
}

async function showBrowser(path) {
  let listing;
  try {
    listing = await api("GET", `/api/directories?path=${encodeURIComponent(path)}`);
  } catch (err) {
    showError(err.message);
    return;
  }
  browsing = listing.path;
  ui.browserHere.textContent = shortenPath(listing.path);
  ui.browserHere.title = listing.path;
  ui.browserUp.disabled = !listing.parent;
  ui.browserUp.dataset.parent = listing.parent || "";

  ui.browserList.replaceChildren();
  if (listing.children.length === 0) {
    const empty = document.createElement("li");
    empty.className = "browser-empty";
    empty.textContent = "No sub-folders here";
    ui.browserList.appendChild(empty);
  }
  for (const name of listing.children) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    button.addEventListener("click", () => showBrowser(`${listing.path}/${name}`));
    item.appendChild(button);
    ui.browserList.appendChild(item);
  }
  ui.browser.hidden = false;
  ui.folder.setAttribute("aria-expanded", "true");
}

function hideBrowser() {
  ui.browser.hidden = true;
  ui.folder.setAttribute("aria-expanded", "false");
}

function setupFolderPicker(defaultPath) {
  // Reuse the last folder when it is still permitted; the server is the judge,
  // since the operator may have restarted with different roots.
  let remembered = null;
  try {
    remembered = localStorage.getItem(LAST_FOLDER_KEY);
  } catch {
    /* storage unavailable */
  }
  setWorkdir(defaultPath);
  if (remembered && remembered !== defaultPath) {
    api("GET", `/api/directories?path=${encodeURIComponent(remembered)}`)
      .then(() => setWorkdir(remembered))
      .catch(() => {
        try {
          localStorage.removeItem(LAST_FOLDER_KEY);
        } catch {
          /* nothing to clean up */
        }
      });
  }

  ui.folder.addEventListener("click", () => {
    if (ui.browser.hidden) void showBrowser(browsing || workdir);
    else hideBrowser();
  });
  ui.browserUp.addEventListener("click", () => {
    const parent = ui.browserUp.dataset.parent;
    if (parent) void showBrowser(parent);
  });
  ui.browserPick.addEventListener("click", () => {
    if (browsing) setWorkdir(browsing);
    hideBrowser();
  });
}

/* ---------------- keyboard shortcuts ---------------- */

/**
 * Switch tabs with Cmd/Ctrl + 1-9 (9 = last tab, as browsers and editors do).
 *
 * Cmd is the right modifier here: macOS never delivers it to a terminal
 * application, so nothing collides with the tool's own bindings, whereas Ctrl-
 * and Option- combinations are the tool's to use.
 *
 * The catch is that Chrome reserves Cmd+1-9 for its *own* tab strip and handles
 * it in the browser process, where a page cannot intercept it —
 * `preventDefault()` does not help. It is free when there is no browser tab
 * strip: an installed PWA / "Open as window", or another browser that does not
 * reserve it. So a second binding, Cmd/Ctrl+Alt+1-9, is registered alongside and
 * is not reserved anywhere, giving a combination that always works in a normal
 * tab while the plain one works wherever the browser allows it.
 */
function isShortcut(event) {
  return (
    (event.metaKey || event.ctrlKey) && !event.shiftKey && /^Digit[1-9]$/.test(event.code)
  );
}

function handleShortcut(event) {
  if (!(event.metaKey || event.ctrlKey) || event.shiftKey) return false;
  // `code`, not `key`: with Option held, macOS reports Option+1 as "¡".
  const match = /^Digit([1-9])$/.exec(event.code);
  if (!match) return false;

  const ids = [...sessions.keys()];
  if (ids.length === 0) return false;
  const requested = Number(match[1]);
  // 9 means "last tab" however many there are; otherwise the nth, if it exists.
  const id = requested === 9 ? ids[ids.length - 1] : ids[requested - 1];
  if (!id) return false;

  activate(id);
  event.preventDefault();
  event.stopPropagation();
  return true;
}

function setupShortcuts() {
  // Capture phase so the handler runs before xterm sees the key.
  window.addEventListener("keydown", handleShortcut, { capture: true });
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
    ui.projectRoot.textContent = shortenPath(payload.project_root);
    ui.projectRoot.title = payload.project_root;
    setupFolderPicker(payload.project_root);
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
    ui.autonomy.addEventListener("change", syncAutonomyNote);
    ui.form.addEventListener("submit", startSession);
  }
  ui.stop.addEventListener("click", () => stopSession(activeId));

  // Reattach to sessions that outlived the page: a reload must not orphan a
  // running tool that only the server can still see.
  try {
    const { sessions: existing } = await api("GET", "/api/sessions");
    for (const info of existing) {
      const entry = createSession(info);
      entry.restored = true;      // needs a redraw nudge, not just a replay
      // State only: the closing line waits for the stream's exit frame, which
      // arrives after this session's scrollback.
      if (!info.running) markExited(info.id, info, { announce: false });
      if (activeId === null) activate(info.id);
    }
  } catch (err) {
    showError(err.message);
  }

  setupShortcuts();
  openStream();
}

window.addEventListener("beforeunload", () => stream && stream.close());
void boot();
