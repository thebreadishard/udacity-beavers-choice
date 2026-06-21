/* =====================================================================================
   viewer/app.js — Beaver's Choice pixel office controller
   -------------------------------------------------------------------------------------
   Pulls agent events from the server (`/events?since=N`) and animates them. A single
   event QUEUE drives both modes:
     - LIVE   : poll the server and enqueue new events as they appear.
     - REPLAY : load the whole transcript once and play it back at a chosen speed.
   Each event maps an agent to its office station and triggers a state animation.
   ===================================================================================== */

(() => {
  "use strict";

  // --- Agent id -> display name. Unknown agents fall back to the raw id. ---
  const DISPLAY_NAMES = {
    orchestrator_agent: "Orchestrator",
    inventory_agent: "Inventory",
    quoting_agent: "Quoting",
    sales_agent: "Sales",
    customer: "Customer",
  };

  // Transient station classes cleared whenever a new state arrives.
  const STATE_CLASSES = ["is-thinking", "is-working", "is-done"];

  // --- DOM references ---
  const stationEls = {};
  document.querySelectorAll(".station").forEach((el) => {
    stationEls[el.dataset.agent] = el;
    // Inject the sprite's eyes + legs (kept out of the HTML for tidiness).
    const sprite = el.querySelector('[data-role="sprite"]');
    if (sprite) sprite.innerHTML = '<i class="eyes"></i><i class="legs"></i>';
  });

  const feedEl = document.getElementById("feed");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const replayControls = document.getElementById("replay-controls");
  const playBtn = document.getElementById("replay-play");
  const restartBtn = document.getElementById("replay-restart");
  const speedSel = document.getElementById("replay-speed");
  const modeButtons = document.querySelectorAll(".mode-btn");

  // --- Runtime state ---
  let mode = "live"; // "live" | "replay"
  let playing = true; // replay play/pause
  let seen = 0; // events already pulled from the server (live)
  let allEvents = []; // full transcript (replay)
  const queue = []; // pending events to animate
  let pollTimer = null;
  let drainTimer = null;
  let idleTicks = 0;

  const LIVE_DRAIN_MS = 300; // how fast queued events animate in live mode
  const POLL_MS = 700; // how often we ask the server for new events

  // ----------------------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------------------

  function truncate(text, n) {
    if (!text) return "";
    text = String(text).replace(/\s+/g, " ").trim();
    return text.length > n ? text.slice(0, n - 1) + "…" : text;
  }

  // Recursively collect only the *values* from a parsed JSON structure (skips the keys),
  // so tool arguments like {"task": "Check A4 stock"} become "Check A4 stock".
  function collectValues(value, out) {
    if (value === null || value === undefined) return;
    if (typeof value === "string") {
      if (value.trim()) out.push(value.trim());
    } else if (typeof value === "number" || typeof value === "boolean") {
      out.push(String(value));
    } else if (Array.isArray(value)) {
      value.forEach((v) => collectValues(v, out));
    } else if (typeof value === "object") {
      Object.values(value).forEach((v) => collectValues(v, out));
    }
  }

  // Turn machine-y content (JSON args / results) into something that reads like words.
  function humanize(text) {
    if (!text) return "";
    let s = String(text).trim();
    if ((s.startsWith("{") && s.endsWith("}")) ||
        (s.startsWith("[") && s.endsWith("]"))) {
      try {
        const values = [];
        collectValues(JSON.parse(s), values);
        if (values.length) return cleanupWords(values.join(" — "));
      } catch (_) {
        // Not valid JSON — fall through to a light cleanup below.
      }
    }
    return cleanupWords(s);
  }

  // Remove leftover markup and smolagents managed-agent boilerplate so snippets read
  // like a conversation rather than a template.
  function cleanupWords(s) {
    return String(s)
      .replace(/^\s*date of request:\s*\d{4}-\d{2}-\d{2}[.\-–,\s]*/i, "")       // drop date preamble
      .replace(/#{1,6}\s*\d+\.\s*/g, "")                                  // "### 1. " numbering
      .replace(/#{1,6}\s*/g, "")                                          // stray hashes
      .replace(/here is the final answer from your managed agent[^:]*:\s*/gi, "")
      .replace(/task outcome \([^)]*\)\s*:?\s*/gi, "")                     // "(short version):"
      .replace(/additional context[^:]*:\s*/gi, "")
      .replace(/[{}\[\]"]/g, " ")                                          // residual JSON punctuation
      .replace(/\s+/g, " ")
      .trim();
  }



  function setStatus(kind, text) {
    statusDot.classList.remove("is-live", "is-idle", "is-error");
    if (kind) statusDot.classList.add(kind);
    statusText.textContent = text;
  }

  function clearTransient(station) {
    STATE_CLASSES.forEach((c) => station.classList.remove(c));
  }

  // ----------------------------------------------------------------------------------
  // Station animations (one per agent state)
  // ----------------------------------------------------------------------------------

  function activate(station) {
    if (station.classList.contains("is-active")) return;
    station.classList.add("is-active", "entering", "is-walking");
    setTimeout(() => station.classList.remove("entering", "is-walking"), 720);
  }

  function applyState(station, ev) {
    const bubble = station.querySelector('[data-role="bubble"]');
    const tool = station.querySelector('[data-role="tool"]');

    switch (ev.state) {
      case "start":
        activate(station);
        break;

      case "thinking":
        clearTransient(station);
        station.classList.add("is-thinking");
        if (bubble) bubble.textContent = truncate(humanize(ev.content), 16) || "…";
        break;

      case "tool_use":
        clearTransient(station);
        station.classList.add("is-working");
        if (tool) tool.textContent = truncate(ev.tool_name || "tool", 18);
        break;

      case "tool_result":
        station.classList.add("flash-result");
        setTimeout(() => station.classList.remove("flash-result"), 500);
        break;

      case "done":
        clearTransient(station);
        station.classList.add("is-done");
        if (bubble) bubble.textContent = "✓";
        break;

      default:
        break;
    }
  }

  // ----------------------------------------------------------------------------------
  // Activity feed
  // ----------------------------------------------------------------------------------

  function addFeedRow(ev) {
    const who = DISPLAY_NAMES[ev.agent] || ev.agent || "system";
    let what = ev.state;
    if (ev.state === "tool_use" && ev.tool_name) what = `tool · ${ev.tool_name}`;

    const li = document.createElement("li");
    li.className = "feed-row";
    li.dataset.agent = ev.agent || "";
    li.innerHTML =
      '<span class="tick"></span>' +
      '<span class="body"><span class="who"></span> · <span class="what"></span><br><span class="snippet"></span></span>';
    li.querySelector(".who").textContent = who;
    li.querySelector(".what").textContent = what;
    li.querySelector(".snippet").textContent = truncate(humanize(ev.content), 90);

    feedEl.appendChild(li);
    // Cap the DOM size during long runs.
    while (feedEl.children.length > 250) feedEl.removeChild(feedEl.firstChild);
    feedEl.scrollTop = feedEl.scrollHeight;
  }

  // ----------------------------------------------------------------------------------
  // Render one event
  // ----------------------------------------------------------------------------------

  function render(ev) {
    const station = stationEls[ev.agent];
    if (station) {
      activate(station);
      applyState(station, ev);
    }
    addFeedRow(ev);
  }

  // ----------------------------------------------------------------------------------
  // Queue draining (shared by both modes)
  // ----------------------------------------------------------------------------------

  function drainTick() {
    if (!playing) return;
    if (queue.length === 0) {
      if (mode === "replay") setStatus("is-idle", "replay complete");
      return;
    }
    render(queue.shift());
    if (mode === "replay") {
      setStatus("is-live", `replay (${queue.length} left)`);
    }
  }

  function startDrain(intervalMs) {
    if (drainTimer) clearInterval(drainTimer);
    drainTimer = setInterval(drainTick, intervalMs);
  }

  // ----------------------------------------------------------------------------------
  // Live mode
  // ----------------------------------------------------------------------------------

  async function poll() {
    try {
      const res = await fetch(`/events?since=${seen}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const incoming = data.events || [];
      if (incoming.length) {
        incoming.forEach((ev) => queue.push(ev));
        seen += incoming.length;
        idleTicks = 0;
        setStatus("is-live", `live · ${seen} events`);
      } else {
        idleTicks += 1;
        if (queue.length === 0) {
          setStatus(idleTicks > 3 ? "is-idle" : "is-live",
            idleTicks > 3 ? "live · idle" : `live · ${seen} events`);
        }
      }
    } catch (err) {
      setStatus("is-error", "server offline");
    }
  }

  function startLive() {
    mode = "live";
    playing = true;
    seen = 0;
    queue.length = 0;
    resetAll();
    replayControls.hidden = true;
    setStatus("is-live", "connecting…");
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(poll, POLL_MS);
    poll();
    startDrain(LIVE_DRAIN_MS);
  }

  // ----------------------------------------------------------------------------------
  // Replay mode
  // ----------------------------------------------------------------------------------

  async function loadAll() {
    const res = await fetch(`/events?since=0`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return data.events || [];
  }

  function queueReplay() {
    queue.length = 0;
    resetAll();
    allEvents.forEach((ev) => queue.push(ev));
    playing = true;
    syncPlayBtn();
  }

  async function startReplay() {
    mode = "replay";
    if (pollTimer) clearInterval(pollTimer);
    replayControls.hidden = false;
    setStatus("is-idle", "loading transcript…");
    try {
      allEvents = await loadAll();
    } catch (err) {
      setStatus("is-error", "server offline");
      return;
    }
    if (allEvents.length === 0) {
      setStatus("is-idle", "transcript empty");
      return;
    }
    queueReplay();
    startDrain(Number(speedSel.value));
    setStatus("is-live", `replay (${queue.length} left)`);
  }

  // ----------------------------------------------------------------------------------
  // Reset
  // ----------------------------------------------------------------------------------

  function resetAll() {
    Object.values(stationEls).forEach((station) => {
      station.classList.remove("is-active", "entering", "is-walking", "flash-result",
        ...STATE_CLASSES);
      const bubble = station.querySelector('[data-role="bubble"]');
      const tool = station.querySelector('[data-role="tool"]');
      if (bubble) bubble.textContent = "";
      if (tool) tool.textContent = "";
    });
    feedEl.innerHTML = "";
  }

  // ----------------------------------------------------------------------------------
  // Controls wiring
  // ----------------------------------------------------------------------------------

  function syncPlayBtn() {
    playBtn.innerHTML = playing ? "&#10073;&#10073;" : "&#9654;"; // pause / play glyphs
  }

  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.classList.contains("is-active")) return;
      modeButtons.forEach((b) => b.classList.toggle("is-active", b === btn));
      if (btn.dataset.mode === "live") startLive();
      else startReplay();
    });
  });

  playBtn.addEventListener("click", () => {
    playing = !playing;
    syncPlayBtn();
    if (playing && mode === "replay") setStatus("is-live", `replay (${queue.length} left)`);
    else if (mode === "replay") setStatus("is-idle", "paused");
  });

  restartBtn.addEventListener("click", () => {
    if (mode === "replay") {
      queueReplay();
      startDrain(Number(speedSel.value));
    }
  });

  speedSel.addEventListener("change", () => {
    if (mode === "replay") startDrain(Number(speedSel.value));
  });

  // ----------------------------------------------------------------------------------
  // Boot
  // ----------------------------------------------------------------------------------

  syncPlayBtn();
  startLive();
})();
