/* Standalone AMBI trace viewer. Pure helpers also run under Node for tests. */
(function (root) {
  "use strict";
  const axes = {event_index: "Chronological event", critic_updates: "Critic updates", actor_updates: "Actor updates", temperature_updates: "Temperature updates", round_index: "Collection round", decision_index: "Real decision"};
  const colors = ["#146c58", "#ba632c", "#526cc4", "#a34c80", "#6d7e31", "#26879a", "#7847a8", "#8b6547"];
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const episodeKey = episode => JSON.stringify([episode.episode_id, episode.seed]);
  function series(trace, metric, axis) {
    if (!trace || !Object.hasOwn(trace.metrics, metric)) return [];
    const values = trace.metrics[metric], statuses = trace.nonfinite[metric] || [];
    return values.map((value, index) => ({x: axis === "decision_index" ? trace.decision_index : (trace[axis] || [])[index], value, nonfinite: statuses[index] || null,
      decision: trace.decision_index, event: trace.event_index[index], phase: trace.phase[index], round: trace.round_index[index],
      critic: trace.critic_updates[index], actor: trace.actor_updates[index], temperature: trace.temperature_updates[index]}))
      .filter(point => finite(point.x) && (finite(point.value) || point.nonfinite));
  }
  function selectTraces(run, mode, selection) {
    return run.traces.filter(trace => trace.mode === mode && trace.selection_key === selection)
      .sort((a, b) => a.decision_index - b.decision_index);
  }
  function selections(runs, mode) {
    const found = new Map();
    runs.forEach(run => {
      if (mode === "episodes") run.episodes.forEach(episode => found.set(episodeKey(episode), {key: episodeKey(episode), label: `Episode ${episode.episode_id} · seed ${episode.seed}`}));
      run.traces.filter(trace => trace.mode === mode).forEach(trace => {
        const label = mode === "bank" ? `Root ${trace.root_id} · repeat ${trace.repeat} · solver ${trace.solver_seed} · bank ${trace.root_bank_id}` : `Episode ${trace.episode_id} · seed ${trace.seed}`;
        found.set(trace.selection_key, {key: trace.selection_key, label});
      });
    });
    return Array.from(found.values()).sort((a, b) => a.label.localeCompare(b.label, undefined, {numeric: true}));
  }
  function extent(values) {
    let low = Infinity, high = -Infinity;
    for (const value of values) if (finite(value)) {low = Math.min(low, value); high = Math.max(high, value);}
    return low === Infinity ? null : [low, high];
  }
  function heatmapCells(traces, metric, axis) {
    const cells = new Map();
    traces.forEach(trace => series(trace, metric, axis).forEach(point => {
      const key = `${trace.decision_index}:${point.x}`, previous = cells.get(key);
      cells.set(key, {decision: trace.decision_index, ...point, count: previous ? previous.count + 1 : 1});
    }));
    return Array.from(cells.values());
  }
  function overlay(runs, mode, selection, decision, metric, axis) {
    return runs.map(run => {
      const traces = selectTraces(run, mode, selection);
      return {run, points: mode === "episodes" && axis === "decision_index" ? traces.flatMap(trace => series(trace, metric, axis))
        : series(mode === "bank" ? traces[0] : traces.find(trace => trace.decision_index === decision), metric, axis)};
    });
  }
  function format(value) {
    if (!finite(value)) return "—";
    if (value === 0) return "0";
    return Math.abs(value) >= 10000 || Math.abs(value) < 0.001 ? value.toExponential(3) : Number(value.toPrecision(5)).toString();
  }
  const api = {series, selectTraces, selections, extent, heatmapCells, overlay, episodeKey};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.AMBIReport = api;
  if (typeof document === "undefined") return;

  function boot(data) {
    const $ = id => document.getElementById(id);
    const state = {selected: new Set(data.runs.map(run => run.key)), heat: null};
    const color = run => colors[data.runs.indexOf(run) % colors.length];
    const chosen = () => data.runs.filter(run => state.selected.has(run.key));
    function option(select, value, label) {const item = document.createElement("option"); item.value = value; item.textContent = label; select.appendChild(item);}
    function replaceOptions(select, entries, requested) {
      select.replaceChildren(); entries.forEach(entry => option(select, entry.key, entry.label));
      if (entries.some(entry => entry.key === requested)) select.value = requested;
    }
    function svg(tag, attributes, parent, text) {
      const item = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attributes || {}).forEach(([key, value]) => item.setAttribute(key, value));
      if (text !== undefined) item.textContent = text;
      (parent || $("curve")).appendChild(item); return item;
    }
    function text(id, value) {$(id).textContent = value;}
    function refreshOptions() {
      const selected = chosen();
      replaceOptions($("primary-run"), selected.map(run => ({key: run.key, label: `${run.label} (${run.id})`})), $("primary-run").value);
      replaceOptions($("selection"), selections(selected, $("mode").value), $("selection").value);
      text("selection-label", $("mode").value === "bank" ? "Shared root" : "Episode");
    }
    function outcomeRows() {
      const target = $("outcomes"); target.replaceChildren();
      chosen().forEach(run => {
        const episodes = run.episodes.length ? run.episodes : [null];
        episodes.forEach(episode => {
          const row = document.createElement("tr");
          const ending = episode ? [episode.terminated && "terminated", episode.truncated && "truncated", episode.capped && "capped"].filter(Boolean).join(", ") || "incomplete" : "root-bank only";
          const status = episode && episode.status || run.status || "unknown";
          const complete = episode && (episode.terminated || episode.truncated || episode.capped) && !["failed", "partial", "running"].includes(status);
          [run.label, episode ? `${episode.episode_id} / ${episode.seed}` : "—", format(episode && episode.return), format(complete ? episode.paired_return_delta : null),
            episode ? episode.length : "—", format(episode && episode.control_seconds), ending, status].forEach(value => {
            const cell = document.createElement("td"); cell.textContent = value; row.appendChild(cell);
          }); target.appendChild(row);
        });
      });
    }
    function heatmap() {
      const primary = chosen().find(run => run.key === $("primary-run").value), metric = $("metric").value, axis = $("axis").value;
      const selection = $("selection").value, mode = $("mode").value;
      const outerAxis = mode === "episodes" && axis === "decision_index";
      const traces = primary ? selectTraces(primary, mode, selection) : [];
      const cells = heatmapCells(traces, metric, axis);
      const allTraces = chosen().flatMap(run => selectTraces(run, mode, selection));
      const allCells = chosen().flatMap(run => heatmapCells(selectTraces(run, mode, selection), metric, axis));
      const limits = extent(allCells.map(cell => cell.value));
      const xextent = extent(allCells.map(cell => cell.x));
      const selectedEpisodes = chosen().flatMap(run => run.episodes.filter(episode => episodeKey(episode) === selection));
      const maxDecision = Math.max(0, ...allTraces.map(trace => trace.decision_index), ...selectedEpisodes.map(episode => episode.length - 1));
      $("decision").max = String(maxDecision);
      const selectedDecision = mode === "bank" && allTraces.length ? (traces[0] || allTraces[0]).decision_index : Math.max(0, Math.min(maxDecision, Math.trunc(Number($("decision").value) || 0)));
      $("decision").value = String(selectedDecision);
      $("decision").disabled = mode === "bank";
      $("previous").disabled = mode === "bank"; $("next").disabled = mode === "bank";
      text("heat-title", `${outerAxis ? "Values across real decisions" : mode === "bank" ? "Learning at a shared root" : "Learning across an episode"}${primary ? ` · ${primary.label}` : ""}`);
      text("heat-selection", `Selected decision ${selectedDecision} · ${cells.length} logged cells`);
      text("heat-axis", axes[axis]);
      text("heat-ylabel", outerAxis ? "Episode summary" : mode === "bank" ? "Root solve decision" : "Real decision");
      $("heat-empty").hidden = cells.length !== 0;
      $("heat-content").hidden = cells.length === 0;
      if (!cells.length) {
        text("heat-empty", !traces.length ? "No inner trace for this run and selection. A prior-only baseline has no optimizer updates." : "This metric was not logged for this run and selection.");
        state.heat = null; return;
      }
      const canvas = $("heatmap"), ctx = canvas.getContext("2d");
      const width = 1000, height = mode === "bank" || outerAxis ? 72 : Math.max(120, Math.min(500, maxDecision + 1));
      canvas.width = width; canvas.height = height;
      const xlow = xextent[0], xhigh = xextent[1], columns = xhigh - xlow + 1, rows = mode === "bank" || outerAxis ? 1 : maxDecision + 1;
      ctx.fillStyle = "#eef0ed"; ctx.fillRect(0, 0, width, height);
      cells.forEach(cell => {
        const t = limits && limits[1] !== limits[0] ? (cell.value - limits[0]) / (limits[1] - limits[0]) : 0.5;
        ctx.fillStyle = cell.nonfinite ? "#c53635" : `rgb(${Math.round(241 - 221*t)},${Math.round(244 - 136*t)},${Math.round(238 - 150*t)})`;
        ctx.fillRect((cell.x - xlow) * width / columns, (mode === "bank" || outerAxis ? 0 : cell.decision) * height / rows, width / columns + 0.05, height / rows + 0.05);
      });
      ctx.strokeStyle = "#d38019"; ctx.lineWidth = 2;
      if (outerAxis) ctx.strokeRect((selectedDecision - xlow) * width / columns, 1, Math.max(2, width / columns), height - 2);
      else ctx.strokeRect(1, (mode === "bank" ? 0 : selectedDecision) * height / rows, width - 2, Math.max(2, height / rows));
      state.heat = {cells: new Map(cells.map(cell => [`${cell.decision}:${cell.x}`, cell])), xlow, columns, rows, bankDecision: mode === "bank" ? selectedDecision : null, outerAxis};
      text("heat-xmin", xlow); text("heat-xmax", xhigh);
      text("color-min", limits ? format(limits[0]) : "nonfinite"); text("color-max", limits ? format(limits[1]) : "nonfinite");
    }
    function curves() {
      const metric = $("metric").value, axis = $("axis").value, mode = $("mode").value, selection = $("selection").value;
      const outerAxis = mode === "episodes" && axis === "decision_index";
      const decision = Number($("decision").value), entries = overlay(chosen(), mode, selection, decision, metric, axis);
      const points = entries.flatMap(entry => entry.points), xextent = extent(points.map(point => point.x));
      const yextent = extent(points.map(point => point.value)), chart = $("curve"); chart.replaceChildren();
      $("curve-legend").replaceChildren();
      entries.forEach(entry => {
        const item = document.createElement("span"), swatch = document.createElement("span");
        swatch.className = "swatch"; swatch.style.background = color(entry.run); item.appendChild(swatch);
        item.appendChild(document.createTextNode(`${entry.run.label}${entry.points.length ? ` · ${entry.points.length} events` : " · no trace"}`));
        $("curve-legend").appendChild(item);
      });
      text("curve-title", `${metric || "Metric"} · ${outerAxis ? `episode trace, highlighting decision ${decision}` : `decision ${decision}`}`);
      text("point-readout", "Hover a plotted point for its exact event and update counters.");
      $("curve-empty").hidden = points.length !== 0; chart.style.display = points.length ? "block" : "none";
      if (!points.length) {text("curve-empty", "No logged values for this selection."); return;}
      const bounds = {left: 78, right: 970, top: 20, bottom: 274};
      const xmin = xextent[0], xmax = xextent[1] === xmin ? xmin + 1 : xextent[1];
      let ymin = yextent ? yextent[0] : 0, ymax = yextent ? yextent[1] : 1;
      const padding = ymax === ymin ? Math.max(0.1, Math.abs(ymax) * 0.1) : (ymax - ymin) * 0.07;
      ymin -= padding; ymax += padding;
      const sx = value => bounds.left + (value - xmin) / (xmax - xmin) * (bounds.right - bounds.left);
      const sy = value => bounds.bottom - (value - ymin) / (ymax - ymin) * (bounds.bottom - bounds.top);
      if (outerAxis && decision >= xmin && decision <= xmax) svg("line", {x1: sx(decision), x2: sx(decision), y1: bounds.top, y2: bounds.bottom, stroke: "#d38019", "stroke-width": 1.5});
      for (let index = 0; index <= 4; index++) {
        const y = ymin + (ymax - ymin) * index / 4;
        svg("line", {x1: bounds.left, x2: bounds.right, y1: sy(y), y2: sy(y), stroke: "#e5ebe6"});
        svg("text", {x: bounds.left - 9, y: sy(y) + 4, "text-anchor": "end", fill: "#5d6d67", "font-size": 11}, null, format(y));
        const x = xmin + (xmax - xmin) * index / 4;
        svg("text", {x: sx(x), y: bounds.bottom + 20, "text-anchor": "middle", fill: "#5d6d67", "font-size": 11}, null, format(x));
      }
      svg("text", {x: 525, y: 322, "text-anchor": "middle", fill: "#5d6d67", "font-size": 12}, null, axes[axis]);
      const unit = (data.metric_catalog[metric] || {}).unit;
      if (unit) svg("text", {x: 8, y: 12, fill: "#5d6d67", "font-size": 11}, null, unit);
      entries.forEach(entry => {
        let segment = [], previousRound;
        const flush = () => {if (segment.length) svg("polyline", {points: segment.join(" "), fill: "none", stroke: color(entry.run), "stroke-width": 1.7, "stroke-opacity": 0.8}); segment = [];};
        entry.points.forEach(point => {
          if (!outerAxis && previousRound !== undefined && point.round !== previousRound) svg("line", {x1: sx(point.x), x2: sx(point.x), y1: bounds.top, y2: bounds.bottom, stroke: color(entry.run), "stroke-dasharray": "3 5", "stroke-opacity": 0.22});
          previousRound = point.round;
          if (point.nonfinite) {flush(); svg("text", {x: sx(point.x), y: bounds.top + 12, fill: "#c53635", "font-size": 15}, null, "×");}
          else segment.push(`${sx(point.x)},${sy(point.value)}`);
        }); flush();
        entry.points.forEach(point => {
          const marker = svg("circle", {cx: sx(point.x), cy: point.nonfinite ? bounds.top + 6 : sy(point.value), r: 4,
            fill: point.nonfinite ? "#c53635" : color(entry.run), "fill-opacity": 0.85, tabindex: 0});
          const description = `${entry.run.label} · ${metric}=${point.nonfinite || format(point.value)} · decision ${point.decision} · event ${point.event} · ${point.phase} · round ${point.round} · critic ${point.critic} / actor ${point.actor} / temperature ${point.temperature}`;
          svg("title", {}, marker, description);
          const inspect = () => text("point-readout", description);
          marker.addEventListener("mouseenter", inspect); marker.addEventListener("focus", inspect);
        });
      });
      text("curve-note", outerAxis ? "Each point is a logged value at a real decision; per-decision metrics summarize that completed solve. The orange line highlights the selected decision. Episodes keep their actual lengths."
        : "Dashed lines mark each run’s collection-round changes. Different update budgets keep their actual endpoints; red × marks a nonfinite value.");
    }
    function render() {
      const mode = $("mode").value, semantic = data.metric_catalog[$("metric").value] || {};
      text("comparison-note", mode === "bank" ? "Shared-root comparison: only identical bank/root/repeat/solver-seed identities are overlaid. Missing matches remain absent." : "Trajectory comparison: episode IDs and seeds match, but states may differ after controllers take different actions.");
      const fallback = semantic.preferred_axis && !Object.hasOwn(axes, semantic.preferred_axis) ? `This per-solve view uses event order; ${semantic.preferred_axis} identifies the selected outer decision.` : "";
      text("metric-note", [semantic.definition, semantic.sampling_phase && `Sampling: ${semantic.sampling_phase}`, semantic.unit && `Unit: ${semantic.unit}`, fallback].filter(Boolean).join(" · "));
      outcomeRows(); heatmap(); curves();
    }
    Object.keys(data.metric_catalog).sort().forEach(name => option($("metric"), name, name));
    if (Object.hasOwn(data.metric_catalog, "critic_loss")) $("metric").value = "critic_loss";
    const preferred = () => {const value = (data.metric_catalog[$("metric").value] || {}).preferred_axis; $("axis").value = Object.hasOwn(axes, value) ? value : "event_index";};
    data.runs.forEach(run => {
      const label = document.createElement("label"), input = document.createElement("input"), swatch = document.createElement("span");
      label.className = "run-choice"; input.type = "checkbox"; input.checked = true; input.value = run.key; input.setAttribute("aria-label", `Compare ${run.label}`);
      swatch.className = "swatch"; swatch.style.background = color(run);
      label.append(input, swatch, document.createTextNode(`${run.label} (${run.id})`)); $("run-list").appendChild(label);
      input.addEventListener("change", () => {if (input.checked) state.selected.add(run.key); else state.selected.delete(run.key); refreshOptions(); render();});
    });
    $("mode").addEventListener("change", () => {refreshOptions(); $("decision").value = "0"; render();});
    $("metric").addEventListener("change", () => {preferred(); render();});
    ["primary-run", "axis", "selection", "decision"].forEach(id => $(id).addEventListener("change", render));
    $("previous").addEventListener("click", () => {$("decision").value = String(Math.max(0, Number($("decision").value) - 1)); render();});
    $("next").addEventListener("click", () => {$("decision").value = String(Number($("decision").value) + 1); render();});
    function heatPoint(event) {
      if (!state.heat) return null;
      const rectangle = $("heatmap").getBoundingClientRect();
      const x = state.heat.xlow + Math.min(state.heat.columns - 1, Math.max(0, Math.floor((event.clientX - rectangle.left) / rectangle.width * state.heat.columns)));
      const decision = state.heat.outerAxis ? x : state.heat.bankDecision === null ? Math.min(state.heat.rows - 1, Math.max(0, Math.floor((event.clientY - rectangle.top) / rectangle.height * state.heat.rows))) : state.heat.bankDecision;
      return {decision, x, cell: state.heat.cells.get(`${decision}:${x}`)};
    }
    $("heatmap").addEventListener("mousemove", event => {
      const point = heatPoint(event); if (!point) return;
      text("heat-readout", `Decision ${point.decision} · ${axes[$("axis").value]} ${point.x} · ${point.cell ? `${point.cell.nonfinite || format(point.cell.value)} · event ${point.cell.event} · ${point.cell.phase}${point.cell.count > 1 ? ` · final of ${point.cell.count} events at this coordinate` : ""}` : "no logged value"}`);
    });
    $("heatmap").addEventListener("click", event => {const point = heatPoint(event); if (point) {$("decision").value = String(point.decision); render();}});
    text("provenance", `Checkpoint ${data.checkpoint.sha256} · ${data.runs.length} runs · ${data.protocol.action_rule} · ${data.protocol.max_steps} decision cap`);
    text("source-details", data.sources.map(source => `${source.evaluation_id} · ${source.status || "unknown status"} · ${JSON.stringify(source.code)} · ${source.directory}`).join("\n"));
    preferred(); refreshOptions();
    const initialHeatRun = chosen().find(run => run.traces.some(trace => series(trace, $("metric").value, $("axis").value).length));
    if (initialHeatRun) $("primary-run").value = initialHeatRun.key;
    render();
  }
  try {boot(JSON.parse(document.getElementById("report-data").textContent));}
  catch (error) {document.getElementById("error").textContent = `Could not render report: ${error.message}`;}
})(typeof globalThis !== "undefined" ? globalThis : this);
