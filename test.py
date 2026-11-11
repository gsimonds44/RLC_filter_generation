"""
Test a trained inverse model (response -> topology) on BRAND NEW random
circuits it has never seen.

Generates --count fresh topologies with topology_gen.py (the
same standard-filter generator used for training data), simulates each
one's real AC frequency response, feeds that response into the trained
model, and compares the model's generated netlist against the actual
ground-truth netlist that produced the response.

It also closes the loop on the response side: the model's PREDICTED
netlist is itself re-simulated (same MNA/SPICE-equivalent solver, same
frequency grid) and its resulting magnitude/phase curves are compared
directly against the requested response -- not just component-level
structural matching. This is the real test of whether the predicted
circuit actually behaves like what was asked for, since two structurally
different netlists can still be near-equivalent electrically (and vice
versa -- a "close" structural match can still behave very differently).

The report also closes the loop VISUALLY: draw_schematic.py (a
reference schematic renderer, standard IEC/US-style component symbols
rather than a node/edge graph dump) draws both the input circuit -- the
ground-truth netlist a response was simulated from -- and the model's
predicted output circuit, side by side, right above each circuit's
magnitude/phase comparison. Seeing the actual predicted topology drawn
out makes it immediate whether the model reinvented an equivalent
circuit, got the right filter shape with different values, or produced
something structurally nonsensical -- much faster to read than a raw
netlist dump. A predicted netlist doesn't always come out as a clean
left-to-right ladder the way generator output always does (see
draw_schematic.validate_ladder_topology), so a predicted schematic is
only drawn when it validates as one; when it doesn't, the report says
why instead of drawing something misleading.

A side-by-side HTML report (schematics + magnitude + phase, requested
vs. predicted) is written for every circuit -- see --report.

This is the same free-running generation check train.py runs on its own
held-out split at the end of training, but as a standalone script you
can re-run any time against fresh topologies -- not the frozen
validation set from a particular training run.

Usage:
    python test.py --model inverse_model.pt --count 10
    python test.py --model inverse_model.pt --count 25 --seed 7
    python test.py --model inverse_model.pt --report my_report.html
"""

import argparse
import json
import math
import random

import torch

from model import TenLayerTransformer
from topology_gen import random_filter_topology, simulate_ac_response
from draw_schematic import render_schematic_svg, validate_ladder_topology


# ----------------------------------------------------------------------
# Response tokenization -- must match train.py exactly (same binning),
# since the model was trained on tokens produced by this exact scheme.
# ----------------------------------------------------------------------
MAG_MIN, MAG_MAX, MAG_STEP = -160, 20, 1
PH_MIN, PH_MAX, PH_STEP = -180, 180, 2


def _bin_value(x, lo, hi, step):
    x = max(lo, min(hi, x))
    return int(round(x / step) * step)


def mag_token(mag_db: float) -> str:
    return f"MAG_{_bin_value(mag_db, MAG_MIN, MAG_MAX, MAG_STEP)}"


def phase_token(phase_deg: float) -> str:
    return f"PH_{_bin_value(phase_deg, PH_MIN, PH_MAX, PH_STEP)}"


def response_tokens(response: dict) -> list:
    tokens = []
    for i, (mag, ph) in enumerate(zip(response["mag_db"], response["phase_deg"])):
        tokens += [f"F{i}", mag_token(mag), phase_token(ph)]
    return tokens


# ----------------------------------------------------------------------
# Value-token decoding (mantissa + decade token -> value in base SI
# units), same as train.py -- needed to report the predicted component
# values in human-readable form and to score value error.
# ----------------------------------------------------------------------
_PREFIX_EXPONENT = {"": 0, "p": -12, "n": -9, "u": -6, "m": -3, "k": 3, "M": 6, "G": 9}


def _decode_component_value(mantissa_token, decade_token):
    if not decade_token.startswith("x1"):
        return None
    prefix = decade_token[2:]
    if prefix not in _PREFIX_EXPONENT:
        return None
    try:
        mantissa = float(mantissa_token)
    except ValueError:
        return None
    return mantissa * (10 ** _PREFIX_EXPONENT[prefix])


class Vocab:
    """Rebuilt from the checkpoint's saved vocab_itos -- guaranteed to
    match exactly what the model was trained with, rather than
    re-deriving the deterministic vocab and hoping it agrees."""
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    def encode(self, tokens):
        return [self.stoi[t] for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)


def _parse_netlist_tokens(tokens):
    """Parse a flat, generated netlist-description token list of the form
    IN Na GND Nb OUT Nc (TYPE Na Nb VAL mantissa decade)* into a
    structured dict, or return None if it doesn't match that shape."""
    try:
        i = 0
        assert tokens[i] == "IN"
        in_node = tokens[i + 1]
        i += 2
        assert tokens[i] == "GND"
        gnd_node = tokens[i + 1]
        i += 2
        assert tokens[i] == "OUT"
        out_node = tokens[i + 1]
        i += 2

        components = []
        while i < len(tokens):
            ctype = tokens[i]
            if ctype not in ("R", "L", "C"):
                break
            node_a, node_b = tokens[i + 1], tokens[i + 2]
            assert tokens[i + 3] == "VAL"
            mantissa, decade = tokens[i + 4], tokens[i + 5]
            components.append({
                "type": ctype, "node_a": node_a, "node_b": node_b,
                "mantissa": mantissa, "decade_token": decade,
            })
            i += 6

        return {"in_node": in_node, "gnd_node": gnd_node, "out_node": out_node,
                "components": components}
    except (IndexError, AssertionError):
        return None


def _match_components(true_components, gen_components):
    """Greedy match of generated components to true components by
    (type, unordered node pair). Returns (n_matched, value_rel_errors)."""
    used = [False] * len(gen_components)
    n_matched = 0
    value_rel_errors = []

    for tc in true_components:
        t_pair = frozenset((f"N{tc['node_a']}", f"N{tc['node_b']}"))
        for gi, gc in enumerate(gen_components):
            if used[gi] or gc["type"] != tc["type"]:
                continue
            if frozenset((gc["node_a"], gc["node_b"])) != t_pair:
                continue
            used[gi] = True
            n_matched += 1
            gen_val = _decode_component_value(gc["mantissa"], gc["decade_token"])
            if gen_val is not None and gen_val > 0 and tc["value_si"] > 0:
                value_rel_errors.append(abs(gen_val - tc["value_si"]) / tc["value_si"])
            break

    return n_matched, value_rel_errors


def _parsed_netlist_to_topology(parsed):
    """Convert a parsed (possibly model-generated) netlist dict into the
    topology-dict shape simulate_ac_response() (and draw_schematic's
    render_schematic_svg()) expect, so a PREDICTED circuit can be
    simulated -- and drawn -- exactly like a real one and compared
    directly against the requested response.

    Node tokens are relabeled so whatever node the model called GND
    becomes node 0 -- the MNA solver hardcodes ground as node 0, and a
    circuit's node NAMES are arbitrary (relabeling isn't a change to the
    circuit, just to how it's written down). Returns None if the
    netlist doesn't parse into a well-formed, simulatable circuit (bad
    node tokens, a component with no components at all, or a value
    token that doesn't decode to a positive number) -- a real, useful
    outcome to report, not just a structural mismatch.
    """
    def node_num(tok):
        if not (isinstance(tok, str) and tok.startswith("N") and tok[1:].isdigit()):
            return None
        return int(tok[1:])

    raw_gnd = node_num(parsed["gnd_node"])
    raw_in = node_num(parsed["in_node"])
    raw_out = node_num(parsed["out_node"])
    if raw_gnd is None or raw_in is None or raw_out is None or not parsed["components"]:
        return None

    all_raw_nodes = {raw_in, raw_out, raw_gnd}
    parsed_components = []
    for c in parsed["components"]:
        a, b = node_num(c["node_a"]), node_num(c["node_b"])
        if a is None or b is None:
            return None
        value_si = _decode_component_value(c["mantissa"], c["decade_token"])
        if value_si is None or value_si <= 0:
            return None
        all_raw_nodes.add(a)
        all_raw_nodes.add(b)
        parsed_components.append({"type": c["type"], "raw_a": a, "raw_b": b, "value_si": value_si})

    # remap: raw_gnd -> 0 (the solver's hardcoded ground), everything
    # else gets 1..N in a stable order
    remap = {raw_gnd: 0}
    next_id = 1
    for n in sorted(all_raw_nodes):
        if n not in remap:
            remap[n] = next_id
            next_id += 1

    return {
        "num_nodes": next_id,
        "in_node": remap[raw_in],
        "gnd_node": 0,
        "out_node": remap[raw_out],
        "components": [
            {"type": c["type"], "node_a": remap[c["raw_a"]], "node_b": remap[c["raw_b"]],
             "value_si": c["value_si"]}
            for c in parsed_components
        ],
    }


def _phase_error_deg(true_deg, pred_deg):
    """Wrap-aware phase difference in degrees (e.g. 179 vs -179 is a 2
    degree error, not 358) -- a plain subtraction would badly overstate
    error right at the +/-180 wraparound boundary."""
    d = abs(true_deg - pred_deg) % 360.0
    return min(d, 360.0 - d)


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    vocab = Vocab(checkpoint["vocab_itos"])
    cfg = checkpoint["config"]
    model = TenLayerTransformer(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab, checkpoint


def predict_topology(model, vocab, response, device, max_new_tokens):
    """Autoregressively generate a netlist from a response alone (no
    teacher forcing) -- this is the real design-generation use case, not
    a training-time metric."""
    prompt_tokens = ["[BOS]"] + response_tokens(response) + ["[SEP]"]
    prompt_ids = torch.tensor(
        vocab.encode(prompt_tokens), dtype=torch.long, device=device
    ).unsqueeze(0)

    generated = prompt_ids
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(generated, use_causal_mask=True)
            next_id = int(logits[0, -1].argmax())
            generated = torch.cat(
                [generated, torch.tensor([[next_id]], device=device)], dim=1
            )
            if vocab.itos[next_id] == "[EOS]":
                break

    gen_tokens = vocab.decode(generated[0].tolist())
    gen_target = gen_tokens[len(prompt_tokens):]
    if gen_target and gen_target[0] == "[TOPO]":
        gen_target = gen_target[1:]
    if gen_target and gen_target[-1] == "[EOS]":
        gen_target = gen_target[:-1]

    return _parse_netlist_tokens(gen_target)


def format_true_netlist(topo: dict) -> str:
    lines = [f"  IN=N{topo['in_node']}  GND=N{topo['gnd_node']}  OUT=N{topo['out_node']}"]
    for c in topo["components"]:
        lines.append(f"    {c['type']}  N{c['node_a']} - N{c['node_b']}   ({c['human']})")
    return "\n".join(lines)


def format_predicted_netlist(parsed) -> str:
    if parsed is None:
        return "  (model did not generate a well-formed netlist)"
    lines = [f"  IN={parsed['in_node']}  GND={parsed['gnd_node']}  OUT={parsed['out_node']}"]
    for c in parsed["components"]:
        val = _decode_component_value(c["mantissa"], c["decade_token"])
        val_str = f"~{val:.4g}" if val is not None else "?"
        lines.append(f"    {c['type']}  {c['node_a']} - {c['node_b']}   "
                      f"VAL {c['mantissa']} {c['decade_token']}  ({val_str})")
    return "\n".join(lines)


def _render_schematic(topo, title):
    """Best-effort schematic render for the HTML report: returns
    (svg_markup_or_None, note_or_None). Only random_filter_topology's
    own output is GUARANTEED to be the simple left-to-right ladder
    draw_schematic.py's layout algorithm assumes -- a model-predicted
    netlist certainly isn't. validate_ladder_topology catches that up
    front so we never silently drop a component the renderer can't
    place rather than draw something misleading; when a topology
    doesn't qualify, the report explains why instead of showing a
    schematic."""
    ok, reason = validate_ladder_topology(topo)
    if not ok:
        return None, f"Schematic not rendered -- {reason}."
    return render_schematic_svg(topo, title=title), None


def _build_report_html(circuits, meta) -> str:
    """Self-contained HTML report: one row per circuit, each starting
    with a side-by-side pair of drawn schematics (input circuit vs.
    predicted circuit, via draw_schematic.py) followed by side-by-side
    magnitude/phase panels comparing the requested response to the
    response of the model's PREDICTED circuit, re-simulated on the same
    frequency grid. Two different-scale measures (dB vs degrees) get
    two separate panels, never one dual-axis chart. Colors: blue =
    requested, orange = predicted -- same categorical pair already
    validated (CVD-safe, both light and dark) for the training-curves
    report earlier in this project. The schematics themselves are drawn
    with fixed, explicit colors (they're technical diagrams, not
    theme-following chart series) and sit on their own white card so
    they stay readable in dark mode too."""
    data_json = json.dumps(circuits)
    meta_json = json.dumps(meta)

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Design eval: requested vs. predicted response</title>
<style>
  .viz-root {
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page-plane:     #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --series-1:       #2a78d6; /* requested */
    --series-2:       #eb6834; /* predicted */
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) .viz-root {
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page-plane:     #0d0d0d;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --border:         rgba(255,255,255,0.10);
      --series-1:       #3987e5;
      --series-2:       #d95926;
    }
  }
  :root[data-theme="dark"] .viz-root {
    color-scheme: dark;
    --surface-1:      #1a1a19;
    --page-plane:     #0d0d0d;
    --text-primary:   #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted:     #898781;
    --gridline:       #2c2c2a;
    --baseline:       #383835;
    --border:         rgba(255,255,255,0.10);
    --series-1:       #3987e5;
    --series-2:       #d95926;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  /* background/color live on .viz-root, not html/body: custom properties
     defined on .viz-root are invisible to its ANCESTORS (html/body can't
     see a descendant's scoped vars), so putting var(--page-plane) on
     html/body would silently resolve to nothing and leave the page white
     in dark mode. min-height covers short pages so no gutter shows. */
  .viz-root { min-height: 100vh; background: var(--page-plane); color: var(--text-primary); padding: 28px 16px; }
  .wrap { max-width: 1080px; margin: 0 auto; }
  .header { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 18px; }
  .header h1 { font-size: 17px; font-weight: 600; margin: 0; }
  .header .sub { font-size: 12.5px; color: var(--text-secondary); }
  .theme-toggle {
    font-size: 12px; color: var(--text-secondary); background: none;
    border: 1px solid var(--border); border-radius: 6px; padding: 4px 9px; cursor: pointer;
  }
  .circuit-panel {
    background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
    padding: 16px 20px 12px; margin-bottom: 16px;
  }
  .circuit-title { font-size: 13px; font-weight: 600; color: var(--text-primary); display: flex; align-items: center; gap: 8px; }
  .circuit-sub { font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; }
  .stat-row { display: flex; gap: 18px; font-size: 11.5px; color: var(--text-secondary); margin-bottom: 8px; }
  .stat-row b { color: var(--text-primary); font-weight: 600; }
  .schem-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 10px 0 14px; }
  .schem-col .chart-label { font-size: 11.5px; font-weight: 600; color: var(--text-primary); margin-bottom: 4px; }
  .schem-frame {
    background: #ffffff; border: 1px solid var(--border); border-radius: 8px;
    padding: 10px; overflow-x: auto;
  }
  .schem-frame svg { display: block; width: 100%; height: auto; min-width: 360px; }
  .schem-frame .schem-note { color: #6b6b6b; font-size: 12px; font-style: italic; padding: 22px 4px; }
  .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .chart-col .legend { display: flex; gap: 16px; margin-bottom: 6px; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--text-secondary); }
  .swatch { width: 14px; height: 2px; border-radius: 1px; }
  .chart-label { font-size: 11.5px; font-weight: 600; color: var(--text-primary); margin-bottom: 2px; }
  svg { display: block; width: 100%; height: auto; overflow: visible; }
  .gridline { stroke: var(--gridline); stroke-width: 1; }
  .baseline { stroke: var(--baseline); stroke-width: 1; }
  .axis-label { fill: var(--text-muted); font-size: 10px; }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .crosshair-line { stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 2,2; opacity: 0; pointer-events: none; }
  .hover-dot { r: 3.5; stroke: var(--surface-1); stroke-width: 1.5; opacity: 0; pointer-events: none; }
  .hover-target { fill: transparent; }
  .malformed-note { font-size: 12px; color: var(--text-muted); font-style: italic; padding: 8px 0; }
  .tooltip {
    position: absolute; pointer-events: none; background: var(--surface-1);
    border: 1px solid var(--border); border-radius: 8px; padding: 7px 9px; font-size: 11px;
    color: var(--text-primary); box-shadow: 0 4px 14px rgba(0,0,0,0.12); opacity: 0;
    transition: opacity 0.08s; white-space: nowrap; z-index: 10;
  }
  .tooltip .t-freq { color: var(--text-muted); margin-bottom: 3px; }
  .tooltip .t-row { display: flex; align-items: center; gap: 6px; }
  .tooltip .t-dot { width: 7px; height: 7px; border-radius: 50%; }
</style>
</head>
<body>
<div class="viz-root">
  <div class="wrap">
    <div class="header">
      <div>
        <h1>Design eval: requested vs. predicted response</h1>
        <div class="sub" id="headerSub"></div>
      </div>
      <button class="theme-toggle" id="themeToggle">Dark mode</button>
    </div>
    <div id="circuits"></div>
  </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
const circuits = """ + data_json + """;
const meta = """ + meta_json + """;

document.getElementById('headerSub').textContent =
  circuits.length + ' circuits · ' + meta.n_simulatable + ' predicted netlists simulatable · ' +
  meta.n_malformed + ' malformed generations';

const root = document.querySelector('.viz-root');
function seriesColor(varName) { return getComputedStyle(root).getPropertyValue(varName).trim(); }
const tooltip = document.getElementById('tooltip');
const ns = 'http://www.w3.org/2000/svg';
function el(tag, attrs) {
  const e = document.createElementNS(ns, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

function buildSchemCol(label, svgMarkup, note) {
  const col = document.createElement('div');
  col.className = 'schem-col';
  const lbl = document.createElement('div');
  lbl.className = 'chart-label';
  lbl.textContent = label;
  col.appendChild(lbl);
  const frame = document.createElement('div');
  frame.className = 'schem-frame';
  if (svgMarkup) {
    frame.innerHTML = svgMarkup;
  } else {
    const n = document.createElement('div');
    n.className = 'schem-note';
    n.textContent = note || 'Not available.';
    frame.appendChild(n);
  }
  col.appendChild(frame);
  return col;
}

function buildLogXChart(svg, freqs, seriesA, seriesB, labelA, labelB, opts) {
  const W = 480, H = 190;
  const M = { top: 10, right: 12, bottom: 24, left: 40 };
  const plotW = W - M.left - M.right;
  const plotH = H - M.top - M.bottom;
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);

  const logF = freqs.map(f => Math.log10(f));
  const fMin = logF[0], fMax = logF[logF.length - 1];
  const allVals = seriesA.concat(seriesB);
  let yMin = Math.min(...allVals), yMax = Math.max(...allVals);
  const pad = (yMax - yMin) * 0.12 || 1;
  yMin -= pad; yMax += pad;

  const x = lf => M.left + (lf - fMin) / (fMax - fMin) * plotW;
  const y = v => M.top + (1 - (v - yMin) / (yMax - yMin)) * plotH;

  const steps = 3;
  for (let i = 0; i <= steps; i++) {
    const v = yMin + (yMax - yMin) * (i / steps);
    const yy = y(v);
    svg.appendChild(el('line', { x1: M.left, x2: M.left + plotW, y1: yy, y2: yy, class: 'gridline' }));
    const label = el('text', { x: M.left - 6, y: yy + 3, class: 'axis-label', 'text-anchor': 'end' });
    label.textContent = opts.fmt(v);
    svg.appendChild(label);
  }
  svg.appendChild(el('line', { x1: M.left, x2: M.left + plotW, y1: M.top + plotH, y2: M.top + plotH, class: 'baseline' }));

  // log-decade x ticks
  const decadeStart = Math.ceil(fMin), decadeEnd = Math.floor(fMax);
  for (let d = decadeStart; d <= decadeEnd; d += 2) {
    const xx = x(d);
    const label = el('text', { x: xx, y: M.top + plotH + 16, class: 'axis-label', 'text-anchor': 'middle' });
    const hz = Math.pow(10, d);
    label.textContent = hz >= 1e9 ? (hz/1e9)+'G' : hz >= 1e6 ? (hz/1e6)+'M' : hz >= 1e3 ? (hz/1e3)+'k' : hz+'';
    svg.appendChild(label);
  }

  function path(series, color) {
    let d = '';
    logF.forEach((lf, i) => {
      const cmd = i === 0 ? 'M' : 'L';
      d += cmd + x(lf).toFixed(2) + ',' + y(series[i]).toFixed(2) + ' ';
    });
    svg.appendChild(el('path', { d, class: 'line', stroke: color }));
  }
  const colorA = seriesColor('--series-1'), colorB = seriesColor('--series-2');
  path(seriesA, colorA);
  path(seriesB, colorB);

  const crosshair = el('line', { x1: 0, x2: 0, y1: M.top, y2: M.top + plotH, class: 'crosshair-line' });
  svg.appendChild(crosshair);
  const dotA = el('circle', { class: 'hover-dot', fill: colorA });
  const dotB = el('circle', { class: 'hover-dot', fill: colorB });
  svg.appendChild(dotA); svg.appendChild(dotB);
  const target = el('rect', { x: M.left, y: M.top, width: plotW, height: plotH, class: 'hover-target' });
  svg.appendChild(target);

  target.addEventListener('mousemove', (ev) => {
    const rect = svg.getBoundingClientRect();
    const scale = W / rect.width;
    const mx = (ev.clientX - rect.left) * scale;
    const relX = (mx - M.left) / plotW;
    let idx = Math.round(relX * (logF.length - 1));
    idx = Math.max(0, Math.min(logF.length - 1, idx));
    const xx = x(logF[idx]);
    crosshair.setAttribute('x1', xx); crosshair.setAttribute('x2', xx); crosshair.style.opacity = 1;
    dotA.setAttribute('cx', xx); dotA.setAttribute('cy', y(seriesA[idx])); dotA.style.opacity = 1;
    dotB.setAttribute('cx', xx); dotB.setAttribute('cy', y(seriesB[idx])); dotB.style.opacity = 1;
    tooltip.innerHTML =
      '<div class="t-freq">' + freqs[idx].toPrecision(4) + ' Hz</div>' +
      '<div class="t-row"><span class="t-dot" style="background:' + colorA + '"></span>' + labelA + ': <strong>' + opts.fmt(seriesA[idx]) + '</strong></div>' +
      '<div class="t-row"><span class="t-dot" style="background:' + colorB + '"></span>' + labelB + ': <strong>' + opts.fmt(seriesB[idx]) + '</strong></div>';
    tooltip.style.opacity = 1;
    tooltip.style.left = (ev.clientX + 14) + 'px';
    tooltip.style.top = (ev.clientY - 10) + 'px';
  });
  target.addEventListener('mouseleave', () => {
    crosshair.style.opacity = 0; dotA.style.opacity = 0; dotB.style.opacity = 0; tooltip.style.opacity = 0;
  });
}

const container = document.getElementById('circuits');
circuits.forEach(c => {
  const panel = document.createElement('div');
  panel.className = 'circuit-panel';

  const title = document.createElement('div');
  title.className = 'circuit-title';
  const titleText = document.createElement('span');
  titleText.textContent = 'Circuit ' + c.index + '  (' + c.n_components + ' components' +
    (c.stage_str ? ', ' + c.stage_str : '') + ')';
  title.appendChild(titleText);
  panel.appendChild(title);

  // Schematics always come first: the input circuit is always drawable
  // (or explains why not), and the predicted circuit is drawn whenever
  // the model's generation parsed, simulated, AND validated as a simple
  // ladder.
  const schemRow = document.createElement('div');
  schemRow.className = 'schem-row';
  schemRow.appendChild(buildSchemCol('Input circuit (requested)', c.true_schematic_svg, c.true_schematic_note));
  let predNote = c.pred_schematic_note;
  if (!predNote && !c.pred_schematic_svg) {
    predNote = c.malformed
      ? 'Model did not generate a well-formed netlist -- nothing to draw.'
      : (!c.simulatable ? 'Predicted netlist did not simulate cleanly -- nothing to draw.' : null);
  }
  schemRow.appendChild(buildSchemCol('Predicted circuit (model output)', c.pred_schematic_svg, predNote));
  panel.appendChild(schemRow);

  if (!c.simulatable) {
    const note = document.createElement('div');
    note.className = 'malformed-note';
    note.textContent = c.malformed ? 'Model did not generate a well-formed netlist -- nothing to simulate.'
                                    : 'Predicted netlist did not simulate cleanly (bad node/value tokens) -- nothing to compare.';
    panel.appendChild(note);
    container.appendChild(panel);
    return;
  }

  const stats = document.createElement('div');
  stats.className = 'stat-row';
  stats.innerHTML =
    '<span>Mag error: <b>' + c.mag_err_mean.toFixed(2) + ' dB</b> mean / ' + c.mag_err_max.toFixed(2) + ' dB max</span>' +
    '<span>Phase error: <b>' + c.phase_err_mean.toFixed(1) + '&deg;</b> mean / ' + c.phase_err_max.toFixed(1) + '&deg; max</span>' +
    '<span>Components matched: <b>' + c.n_matched + '/' + c.n_components + '</b></span>' +
    '<span>Node roles: <b>' + (c.node_match ? 'match' : 'differ') + '</b></span>';
  panel.appendChild(stats);

  const row = document.createElement('div');
  row.className = 'charts-row';

  const magCol = document.createElement('div');
  magCol.className = 'chart-col';
  magCol.innerHTML = '<div class="chart-label">Magnitude (dB)</div>' +
    '<div class="legend"><div class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>Requested</div>' +
    '<div class="legend-item"><span class="swatch" style="background:var(--series-2)"></span>Predicted (simulated)</div></div>';
  const magSvg = el('svg', {});
  magCol.appendChild(magSvg);
  row.appendChild(magCol);

  const phCol = document.createElement('div');
  phCol.className = 'chart-col';
  phCol.innerHTML = '<div class="chart-label">Phase (deg)</div>' +
    '<div class="legend"><div class="legend-item"><span class="swatch" style="background:var(--series-1)"></span>Requested</div>' +
    '<div class="legend-item"><span class="swatch" style="background:var(--series-2)"></span>Predicted (simulated)</div></div>';
  const phSvg = el('svg', {});
  phCol.appendChild(phSvg);
  row.appendChild(phCol);

  panel.appendChild(row);
  container.appendChild(panel);

  buildLogXChart(magSvg, c.freqs, c.true_mag, c.pred_mag, 'Requested', 'Predicted', { fmt: v => v.toFixed(1) });
  buildLogXChart(phSvg, c.freqs, c.true_phase, c.pred_phase, 'Requested', 'Predicted', { fmt: v => v.toFixed(0) + '°' });
});

document.getElementById('themeToggle').addEventListener('click', () => {
  const cur = document.documentElement.getAttribute('data-theme');
  if (cur === 'dark') {
    document.documentElement.setAttribute('data-theme', 'light');
    document.getElementById('themeToggle').textContent = 'Dark mode';
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
    document.getElementById('themeToggle').textContent = 'Light mode';
  }
});
</script>
</body>
</html>
"""


class _Stats:
    """Accumulates the aggregate counters (match rates, value error,
    simulated response error) for a set of evaluated circuits."""
    def __init__(self, label):
        self.label = label
        self.count = 0
        self.n_node_match = 0
        self.n_malformed = 0
        self.n_unsimulatable = 0
        self.total_matched = 0
        self.total_true = 0
        self.all_value_rel_err = []
        self.all_mag_err = []
        self.all_phase_err = []

    def print_summary(self):
        n_scored = self.count - self.n_malformed
        n_simulatable = n_scored - self.n_unsimulatable
        print(f"=== Summary: {self.label} ({self.count} circuits) ===")
        print(f"Malformed generations: {self.n_malformed}/{self.count}")
        print(f"Unsimulatable predictions (parsed but bad node/value tokens): "
              f"{self.n_unsimulatable}/{max(n_scored, 1)}")
        if n_scored:
            print(f"IN/GND/OUT node-role match: {100*self.n_node_match/n_scored:.1f}%")
        if self.total_true:
            print(f"Component match rate (type + node pair): "
                  f"{self.total_matched}/{self.total_true} "
                  f"({100*self.total_matched/self.total_true:.1f}%)")
        if self.all_value_rel_err:
            print(f"Value error on matched components: mean "
                  f"{100*sum(self.all_value_rel_err)/len(self.all_value_rel_err):.1f}%, "
                  f"max {100*max(self.all_value_rel_err):.1f}%")
        if self.all_mag_err:
            print(f"Simulated response error across {n_simulatable} predicted circuits: "
                  f"mag {sum(self.all_mag_err)/len(self.all_mag_err):.2f} dB mean, "
                  f"phase {sum(self.all_phase_err)/len(self.all_phase_err):.1f} deg mean")
        print()


def _evaluate_circuit(topo, i, model, vocab, device, stats, report_circuits):
    """Run one topology through the full predict -> compare -> re-simulate
    -> compare pipeline, printing console output and appending a report
    entry, and folding the results into `stats`."""
    freqs, mag_db, phase_deg, engine = simulate_ac_response(
        topo, i, vin_amplitude=1.0, f_start=1.0, f_stop=1e9, points_per_decade=10,
    )
    response = {
        "freq_hz": [float(f) for f in freqs],
        "mag_db": [float(m) for m in mag_db],
        "phase_deg": [float(p) for p in phase_deg],
    }

    true_components = topo["components"]
    max_new_tokens = 1 + 7 + 6 * len(true_components) + 1 + 8

    parsed = predict_topology(model, vocab, response, device, max_new_tokens)

    stage_str = " -> ".join(f"{s['class']}{s['order']}" for s in topo.get("filter_stages", []))
    print(f"=== Circuit {i}  ({len(true_components)} components"
          f"{', ' + stage_str if stage_str else ''}) ===")
    print("True netlist:")
    print(format_true_netlist(topo))
    print("Predicted netlist:")
    print(format_predicted_netlist(parsed))

    true_title = f"Input circuit {i}" + (f" -- {stage_str}" if stage_str else "")
    true_svg, true_svg_note = _render_schematic(topo, true_title)

    stats.count += 1
    base_entry = {
        "index": i, "n_components": len(true_components), "stage_str": stage_str,
        "true_schematic_svg": true_svg, "true_schematic_note": true_svg_note,
    }

    if parsed is None:
        stats.n_malformed += 1
        print("  -> malformed generation (excluded from all comparisons)\n")
        report_circuits.append({**base_entry, "simulatable": False, "malformed": True,
                                 "pred_schematic_svg": None, "pred_schematic_note": None})
        return

    node_match = (parsed["in_node"] == f"N{topo['in_node']}"
                  and parsed["gnd_node"] == f"N{topo['gnd_node']}"
                  and parsed["out_node"] == f"N{topo['out_node']}")
    n_matched, value_rel_errors = _match_components(true_components, parsed["components"])

    stats.n_node_match += int(node_match)
    stats.total_matched += n_matched
    stats.total_true += len(true_components)
    stats.all_value_rel_err += value_rel_errors

    val_str = (f"mean {100*sum(value_rel_errors)/len(value_rel_errors):.1f}%"
               if value_rel_errors else "n/a")
    print(f"  -> IN/GND/OUT match: {'yes' if node_match else 'no'}   "
          f"components matched: {n_matched}/{len(true_components)}   "
          f"value error: {val_str}")

    pred_topo = _parsed_netlist_to_topology(parsed)
    if pred_topo is None:
        stats.n_unsimulatable += 1
        print("  -> predicted netlist could not be simulated (bad node/value tokens)\n")
        report_circuits.append({**base_entry, "simulatable": False, "malformed": False,
                                 "pred_schematic_svg": None, "pred_schematic_note": None})
        return

    pred_title = f"Predicted circuit {i}"
    pred_svg, pred_svg_note = _render_schematic(pred_topo, pred_title)

    pred_freqs, pred_mag, pred_phase, pred_engine = simulate_ac_response(
        pred_topo, i, vin_amplitude=1.0, f_start=1.0, f_stop=1e9, points_per_decade=10,
    )
    mag_errs = [abs(m - p) for m, p in zip(mag_db, pred_mag)]
    phase_errs = [_phase_error_deg(p_true, p_pred) for p_true, p_pred in zip(phase_deg, pred_phase)]
    stats.all_mag_err += mag_errs
    stats.all_phase_err += phase_errs

    print(f"  -> simulated response error: mag {sum(mag_errs)/len(mag_errs):6.2f} dB mean / "
          f"{max(mag_errs):6.2f} dB max   phase {sum(phase_errs)/len(phase_errs):6.1f} deg mean / "
          f"{max(phase_errs):6.1f} deg max\n")

    report_circuits.append({
        **base_entry, "simulatable": True, "malformed": False,
        "node_match": node_match, "n_matched": n_matched,
        "pred_schematic_svg": pred_svg, "pred_schematic_note": pred_svg_note,
        "freqs": [float(f) for f in freqs],
        "true_mag": [float(m) for m in mag_db], "true_phase": [float(p) for p in phase_deg],
        "pred_mag": [float(m) for m in pred_mag], "pred_phase": [float(p) for p in pred_phase],
        "mag_err_mean": sum(mag_errs) / len(mag_errs), "mag_err_max": max(mag_errs),
        "phase_err_mean": sum(phase_errs) / len(phase_errs), "phase_err_max": max(phase_errs),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="inverse_model.pt",
                         help="path to the trained checkpoint from train.py (default: inverse_model.pt)")
    parser.add_argument("--count", type=int, default=50,
                         help="number of brand-new random (in-distribution) circuits to test (default: 50)")
    parser.add_argument("--seed", type=int, default=None,
                         help="random seed for the test circuits (default: nondeterministic)")
    parser.add_argument("--report", default="design_eval_report.html",
                         help="path to write the side-by-side requested-vs-predicted response "
                              "HTML report to (default: design_eval_report.html)")
    parser.add_argument("--no-report", action="store_true",
                         help="skip writing the HTML report (console output only)")
    args = parser.parse_args()
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab, checkpoint = load_model(args.model, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {args.model}  (task={checkpoint.get('task', '?')}, "
          f"{n_params:,} params, vocab={len(vocab)}, device={device})\n")

    report_circuits = []
    standard_stats = _Stats("standard (in-distribution)")

    for i in range(1, args.count + 1):
        topo = random_filter_topology()  # topology_gen's MAX_COMPONENTS / MAX_STAGES
        _evaluate_circuit(topo, i, model, vocab, device, standard_stats, report_circuits)

    standard_stats.print_summary()

    if not args.no_report:
        n_scored_total = standard_stats.count - standard_stats.n_malformed
        meta = {"n_simulatable": n_scored_total - standard_stats.n_unsimulatable,
                 "n_malformed": standard_stats.n_malformed}
        html = _build_report_html(report_circuits, meta)
        with open(args.report, "w") as fh:
            fh.write(html)
        print(f"Side-by-side response report written to {args.report}")


if __name__ == "__main__":
    main()