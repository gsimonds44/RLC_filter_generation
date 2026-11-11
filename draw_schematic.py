"""
Reference circuit-schematic renderer.

Turns a topology dict -- the same shape random_filter_topology() in
topology_gen.py produces -- into an actual circuit schematic: not a
token dump or a node/edge graph, but standard IEC/US-style component
symbols (resistor zigzag, capacitor plates, inductor coil, ground
symbol) laid out as a left-to-right signal-path ladder, with shunt
elements dropping to a shared ground rail underneath.

This module is meant to be imported as well as run directly:

    from draw_schematic import render_schematic_svg, validate_ladder_topology

    ok, reason = validate_ladder_topology(topo)
    if ok:
        svg_markup = render_schematic_svg(topo, title="...")

test.py uses exactly this to render BOTH the true input circuit (the
ground-truth netlist a response was simulated from) and the model's
predicted output circuit, side by side, in its HTML eval report -- see
that file for the integration. render_schematic_svg() doesn't care
whether a topology came straight from the generator or was decoded from
a model's generated token stream, as long as it has the same shape
(in_node/gnd_node/out_node/components, each component a dict with
type/node_a/node_b and either a precomputed 'human' string or a raw
'value_si' number -- see _format_human_value below for the fallback).
validate_ladder_topology() is the thing to call first on anything that
ISN'T guaranteed-well-formed generator output (i.e. a model prediction),
since the layout algorithm below assumes a simple ladder shape -- see
its docstring for exactly what that means and why it matters.

Run as a script, it generates one random topology itself and writes it
out as a standalone SVG file (optionally also a PNG snapshot via
headless Chromium, which requires Playwright and is skipped with a
warning if unavailable) -- handy for iterating on the renderer itself
without needing a trained model or test.py in the loop.

Layout assumptions (true of random_filter_topology's output by
construction, see its docstring): the signal path's node indices are
consecutive integers 1..N with ground as a separate node 0, and every
series component connects two ADJACENT path indices (one filter stage's
in_node/out_node) while every shunt component connects one path node to
ground. Multiple components between the same node pair (e.g. a
bandstop stage's parallel L+C) are drawn as parallel branches; multiple
shunts off the same node (e.g. a 2nd-order stage's C and R both to
ground) are drawn as side-by-side drops. A model-predicted netlist is
NOT guaranteed to hold to this shape (see validate_ladder_topology).

Usage:
    python draw_schematic.py
    python draw_schematic.py --seed 7 --max-components 8 --max-stages 3
    python draw_schematic.py --output my_circuit.svg --png my_circuit.png
"""

import argparse
import random
import re

from topology_gen import random_filter_topology


# ----------------------------------------------------------------------
# Layout constants (all in SVG user units / px)
# ----------------------------------------------------------------------
MIN_STAGE_DX = 170        # minimum horizontal spacing between adjacent path nodes
MARGIN_X = 90             # left/right margin before IN / after OUT
MARGIN_TOP = 70
SIGNAL_Y = 170            # y of the main horizontal signal path
GND_Y = 380               # y of the ground rail
PARALLEL_GAP = 90         # vertical offset for parallel series branches (bandstop L||C) -- matches SHUNT_GAP
PARALLEL_GAP_DENSE = 60   # ...for 3+ parallel branches (predicted netlists only), so the lowest stays above the shunt labels
SHUNT_GAP = 90            # horizontal offset between multiple shunts off the same node
LEAD = 10                 # short straight wire lead into/out of a symbol
BODY_HALF = 18            # half-length of a component's "active" symbol body

# A parallel-branch box (bandstop-style L||C) gets a FIXED width, not one
# that stretches to match however far apart its two nodes happen to be --
# node spacing grows to fit long labels, and a box that stretched with it
# would reach past its own corner and overlap whatever a neighboring node
# is doing there. Plain wire leads fill the rest of the gap on each side,
# so the box's own footprint (and any node's shunt) never overlap no
# matter how wide the surrounding layout gets.
BOX_MAX_WIDTH = 140       # the box never grows wider than this...
BOX_LEAD_MIN_BASE = 40    # ...and always leaves at least this much plain wire on each side...
BOX_MIN_HALF = 30         # ...while the box itself never shrinks narrower than this
BOX_SHUNT_CLEAR = 30      # clear wire between a node's outermost shunt drop and the box's side
BOX_LABEL_DY = 32         # label offset above/below a box branch: clears a capacitor's 16px plates
TERM_STUB = 36            # IN/OUT terminal wire length, measured from the node's outermost shunt drop

# Node spacing is computed dynamically (see _build_layout) from the actual
# width labels will need, so long component values never collide with a
# neighboring node's labels -- fixed constants alone can't guarantee that
# for arbitrary generated values, so we estimate label pixel width instead.
CHAR_W = 7.6              # conservative monospace char width at the 12px label font size
LABEL_GAP = 22            # gap between a shunt drop and where its label text starts
SAFETY_MARGIN = 24        # extra clearance beyond the estimated label reach


# ----------------------------------------------------------------------
# Component symbol templates, all defined in LOCAL coordinates: the
# component sits along the local x axis from -half_len to +half_len,
# centered at the local origin. Each is wrapped in a <g transform=...>
# by _place_symbol() so the same template works for horizontal series
# elements and (rotated 90deg) vertical shunt elements alike.
# ----------------------------------------------------------------------
def _resistor_svg(half_len: float) -> str:
    # classic zigzag: 6 segments inside +/-BODY_HALF, straight leads outside.
    # The first/last segments only rise from 0 to +/-amp (half the vertical
    # travel of the interior peak-to-peak segments), so they get half the
    # horizontal run too -- that keeps every segment's slope identical
    # instead of the end segments looking shallower than the rest.
    zz_half = BODY_HALF
    n_peaks = 6
    amp = 9
    full_dx = (2 * zz_half) / (n_peaks - 1)
    half_dx = full_dx / 2
    pts = [(-zz_half, 0)]
    x = -zz_half
    for i in range(1, n_peaks):
        x += half_dx if i == 1 else full_dx
        y = amp if i % 2 == 1 else -amp
        pts.append((x, y))
    pts.append((zz_half, 0))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return (
        f'<line x1="{-half_len:.1f}" y1="0" x2="{-zz_half:.1f}" y2="0" class="wire"/>'
        f'<polyline points="{poly}" class="symbol-line"/>'
        f'<line x1="{zz_half:.1f}" y1="0" x2="{half_len:.1f}" y2="0" class="wire"/>'
    )


def _capacitor_svg(half_len: float) -> str:
    gap = 5
    plate_h = 16
    return (
        f'<line x1="{-half_len:.1f}" y1="0" x2="{-gap:.1f}" y2="0" class="wire"/>'
        f'<line x1="{-gap:.1f}" y1="{-plate_h:.1f}" x2="{-gap:.1f}" y2="{plate_h:.1f}" class="symbol-line"/>'
        f'<line x1="{gap:.1f}" y1="{-plate_h:.1f}" x2="{gap:.1f}" y2="{plate_h:.1f}" class="symbol-line"/>'
        f'<line x1="{gap:.1f}" y1="0" x2="{half_len:.1f}" y2="0" class="wire"/>'
    )


def _inductor_svg(half_len: float) -> str:
    # three humps drawn as arcs (SVG path 'A'), inside +/-BODY_HALF
    coil_half = BODY_HALF
    n_humps = 3
    step = (2 * coil_half) / n_humps
    r = step / 2
    d_parts = [f"M {-coil_half:.1f} 0"]
    for i in range(n_humps):
        cx = -coil_half + step * i + r
        d_parts.append(f"A {r:.1f} {r:.1f} 0 0 1 {cx + r:.1f} 0")
    path_d = " ".join(d_parts)
    return (
        f'<line x1="{-half_len:.1f}" y1="0" x2="{-coil_half:.1f}" y2="0" class="wire"/>'
        f'<path d="{path_d}" class="symbol-line" fill="none"/>'
        f'<line x1="{coil_half:.1f}" y1="0" x2="{half_len:.1f}" y2="0" class="wire"/>'
    )


_SYMBOLS = {"R": _resistor_svg, "L": _inductor_svg, "C": _capacitor_svg}


def _place_symbol(comp_type: str, cx: float, cy: float, angle_deg: float, half_len: float) -> str:
    body = _SYMBOLS[comp_type](half_len)
    return f'<g transform="translate({cx:.1f},{cy:.1f}) rotate({angle_deg})">{body}</g>'


def _ground_symbol(cx: float, cy: float) -> str:
    """Standard 3-bar ground symbol, drawn hanging below (cx, cy)."""
    bars = [(14, 6), (9, 12), (4, 18)]
    parts = [f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy+4:.1f}" class="wire"/>']
    for w, dy in bars:
        y = cy + dy
        parts.append(f'<line x1="{cx-w/2:.1f}" y1="{y:.1f}" x2="{cx+w/2:.1f}" y2="{y:.1f}" class="symbol-line"/>')
    return "".join(parts)


def _junction_dot(x: float, y: float) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" class="junction"/>'


def _text(x: float, y: float, s: str, cls: str = "label", anchor: str = "middle") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{s}</text>'


# ----------------------------------------------------------------------
# Topology -> layout
# ----------------------------------------------------------------------
def _label_width(s: str) -> float:
    return len(s) * CHAR_W


def _shunt_offsets(node, shunt):
    """x offsets (relative to the node) of each shunt drop off this node,
    centered on the node. Two drops label outward, away from each other,
    so SHUNT_GAP is enough; with three or more, inner drops label into
    the space between drops, so that space has to fit the widest label
    (generator output never has more than two, but a model-PREDICTED
    netlist can)."""
    comps = shunt.get(node, [])
    n = len(comps)
    gap = SHUNT_GAP
    if n >= 3:
        widest = max(_label_width(c.get("human") or _format_human_value(c.get("value_si", 0.0), c["type"]))
                     for c in comps)
        # + 4 chars for the "R10 " ref prefix, as in _build_layout
        gap = max(SHUNT_GAP, LABEL_GAP + widest + 4 * CHAR_W + 12)
    return [(j - (n - 1) / 2) * gap for j in range(n)]


def _required_lead(node, shunt):
    """How much plain wire a parallel-branch box must leave between
    itself and this node: enough to clear the widest tee any shunt on
    this node reaches out to (see the shunt-tee offsets below), so the
    box's own footprint can never fall inside a shunt's horizontal run
    -- not just "usually far enough," but guaranteed by construction."""
    offsets = _shunt_offsets(node, shunt)
    if len(offsets) <= 1:
        return BOX_LEAD_MIN_BASE
    max_shunt_offset = max(offsets)
    # a drop only a few px from the box's vertical side reads as hanging
    # off the box itself, so leave a visible stretch of wire between them
    return max(BOX_LEAD_MIN_BASE, max_shunt_offset + BOX_SHUNT_CLEAR)


def _box_half_width(a, b, x_of, shunt):
    """The fixed-but-clamped half-width of a parallel-branch box between
    nodes a and b, consistent with the lead space _build_layout already
    guaranteed is there."""
    lead = max(_required_lead(a, shunt), _required_lead(b, shunt))
    return min(BOX_MAX_WIDTH / 2, (x_of[b] - x_of[a]) / 2 - lead)


def _box_adjacent_dir(node, path_nodes, node_pos, series):
    """A node that borders a parallel-branch box on only one side reads as
    "crowded" if a centered (off==0) shunt at that node defaults to
    labeling toward the box -- the box's own symbols already occupy that
    side, so a label pointed the same way looks jammed up against it even
    though the wire spacing itself is fine. Returns -1 if the default
    label direction should be forced left (there's a box to the right and
    none to the left), +1 if forced right (box to the left only), 0 if
    neither side (or both) has a box, in which case the caller's own
    default applies."""
    idx = node_pos[node]
    left_box = right_box = False
    if idx > 0:
        prev = path_nodes[idx - 1]
        key = (prev, node) if prev < node else (node, prev)
        left_box = len(series.get(key, [])) >= 2
    if idx < len(path_nodes) - 1:
        nxt = path_nodes[idx + 1]
        key = (node, nxt) if node < nxt else (nxt, node)
        right_box = len(series.get(key, [])) >= 2
    if right_box and not left_box:
        return -1
    if left_box and not right_box:
        return 1
    return 0


def _shunt_reach(node, shunt):
    """How far left/right of this node's own x the outermost shunt tee on
    each side reaches (0 if there's no shunt on that side). Used to center
    a single series component within the actual clear channel between two
    nodes rather than the raw node-to-node midpoint -- if one node's
    shunts spread out toward the component, the component should shift
    away from them to stay visually centered in the open space."""
    offsets = _shunt_offsets(node, shunt)
    if not offsets:
        return 0.0, 0.0
    left_reach = max(0.0, -min(offsets))
    right_reach = max(0.0, max(offsets))
    return left_reach, right_reach


def _build_layout(topo: dict, series: dict, shunt: dict):
    """Returns (path_nodes, x_of): path_nodes is the sorted list of
    non-ground node indices actually used, x_of maps node -> x coord.

    Horizontal spacing between adjacent nodes is computed dynamically
    from the actual shunt-label text each node will draw (rather than a
    fixed constant), so two neighboring nodes' labels can't collide
    regardless of how long the generated component values happen to be.
    """
    node_ids = set()
    for c in topo["components"]:
        node_ids.add(c["node_a"])
        node_ids.add(c["node_b"])
    node_ids.discard(topo["gnd_node"])
    path_nodes = sorted(node_ids)
    node_pos = {n: i for i, n in enumerate(path_nodes)}

    # how far left/right each node's shunt labels reach past the node's own x
    left_need = {n: 0.0 for n in path_nodes}
    right_need = {n: 0.0 for n in path_nodes}
    for node, comps in shunt.items():
        offsets = _shunt_offsets(node, shunt)
        for c, off in zip(comps, offsets):
            # worst-case ref is "R10 " etc -- pad a couple chars beyond the
            # human value text since exact ref numbers aren't assigned yet
            human = c.get("human") or _format_human_value(c.get("value_si", 0.0), c["type"])
            reach = abs(off) + LABEL_GAP + _label_width(human) + 3 * CHAR_W
            if off < 0:
                goes_left = True
            elif off > 0:
                goes_left = False
            else:
                # a centered shunt defaults right, UNLESS this node only
                # borders a parallel box on the right -- then default left,
                # away from the box, to match how a centered shunt on the
                # box's other node (defaulting right, into open space)
                # naturally ends up with generous clearance instead
                goes_left = _box_adjacent_dir(node, path_nodes, node_pos, series) < 0
            if goes_left:
                left_need[node] = max(left_need[node], reach)
            else:
                right_need[node] = max(right_need[node], reach)

    # every node-to-node gap uses the SAME distance (the largest one any
    # single pair actually needs) rather than each pair getting its own
    # size -- a topology where only one pair needs extra room for a long
    # label would otherwise end up with visibly uneven spacing elsewhere
    # for no reason a viewer could see.
    stage_gap = MIN_STAGE_DX
    for i in range(1, len(path_nodes)):
        prev, n = path_nodes[i - 1], path_nodes[i]
        stage_gap = max(stage_gap, right_need[prev] + left_need[n] + SAFETY_MARGIN)

    # a parallel-branch box between two nodes needs enough of a gap that
    # its lead space (see _required_lead) can clear both nodes' shunk-tee
    # reach even at BOX_MIN_HALF, its narrowest allowed width -- if the
    # uniform gap above wouldn't leave that much room, widen it here so
    # the box's fixed-width footprint (see _box_half_width) never has to
    # shrink into a shunt's tee to fit.
    for (pa, pb), pcomps in series.items():
        if len(pcomps) >= 2:
            lead = max(_required_lead(pa, shunt), _required_lead(pb, shunt))
            min_half = BOX_MIN_HALF
            if len(pcomps) >= 3:
                # 3+ branches label INSIDE the box, between branches (see
                # render_schematic_svg), so the box must be wider than its
                # widest label or the text runs across the box's sides
                widest = max(_label_width(c.get("human") or _format_human_value(c.get("value_si", 0.0), c["type"]))
                             + 4 * CHAR_W for c in pcomps)
                min_half = max(min_half, min(BOX_MAX_WIDTH / 2, widest / 2 + 10))
            stage_gap = max(stage_gap, 2 * (lead + min_half))

    x_of = {}
    for i, n in enumerate(path_nodes):
        if i == 0:
            x_of[n] = MARGIN_X + left_need[n]
        else:
            prev = path_nodes[i - 1]
            x_of[n] = x_of[prev] + stage_gap
    return path_nodes, x_of, right_need


def _classify_components(topo: dict):
    """Split into series (dict[(a,b) sorted tuple] -> [components]) and
    shunt (dict[node] -> [components]), preserving generation order."""
    series, shunt = {}, {}
    gnd = topo["gnd_node"]
    for c in topo["components"]:
        a, b = c["node_a"], c["node_b"]
        if a == gnd or b == gnd:
            node = b if a == gnd else a
            shunt.setdefault(node, []).append(c)
        else:
            key = tuple(sorted((a, b)))
            series.setdefault(key, []).append(c)
    return series, shunt


_UNIT_BY_TYPE = {"R": "Ohm", "L": "H", "C": "F"}
_SI_PREFIXES = [
    (1e-12, "p"), (1e-9, "n"), (1e-6, "u"), (1e-3, "m"),
    (1.0, ""), (1e3, "k"), (1e6, "M"), (1e9, "G"),
]


def _format_human_value(value_si: float, comp_type: str) -> str:
    """Render a raw SI value (ohms/farads/henries) the same way
    random_filter_topology's own 'human' strings look (e.g. "180uH",
    "15kOhm", "1.5pF") -- used as a fallback for topologies that don't
    carry a precomputed 'human' field, namely a model-PREDICTED netlist
    decoded straight from its mantissa/decade value tokens, which has
    a value_si but no 'human' string ready-made."""
    unit = _UNIT_BY_TYPE.get(comp_type, "")
    if value_si <= 0:
        return f"0{unit}"
    factor, prefix = _SI_PREFIXES[0]
    for f, p in _SI_PREFIXES:
        if value_si >= f:
            factor, prefix = f, p
    scaled = value_si / factor
    if scaled >= 100:
        s = f"{scaled:.0f}"
    elif scaled >= 10:
        s = f"{scaled:.1f}"
    else:
        s = f"{scaled:.2f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return f"{s}{prefix}{unit}"


def _comp_label(comp: dict, ref: str) -> str:
    human = comp.get("human")
    if human is None:
        human = _format_human_value(comp.get("value_si", 0.0), comp["type"])
    return f"{ref} {human}"


def validate_ladder_topology(topo: dict):
    """Check whether `topo` matches the simple left-to-right ladder shape
    render_schematic_svg() assumes: every non-ground (series) component
    connects two nodes that end up ADJACENT once all node indices are
    sorted. That's always true of random_filter_topology's own output by
    construction (see the module docstring), but it is NOT guaranteed
    for an arbitrary netlist -- in particular a model-PREDICTED netlist
    can name nodes in any order or connect two nodes several hops apart,
    which the ladder layout has no way to draw correctly: rather than
    draw it wrong, render_schematic_svg's main loop would silently omit
    that component (it only ever looks for a series component between
    each *adjacent* pair of sorted node indices). Call this before
    render_schematic_svg() on anything that isn't guaranteed-well-formed
    generator output, and skip rendering (with the returned reason
    shown instead) when it fails.

    Returns (ok: bool, reason: str) -- reason is "" when ok is True.
    """
    if not topo.get("components"):
        return False, "no components"

    gnd = topo["gnd_node"]
    for c in topo["components"]:
        if c["node_a"] == c["node_b"]:
            if c["node_a"] == gnd:
                return False, f"a {c['type']} has both ends on ground"
            return False, f"a {c['type']} has both ends on N{c['node_a']}"

    series, shunt = _classify_components(topo)
    node_ids = set()
    for c in topo["components"]:
        node_ids.add(c["node_a"])
        node_ids.add(c["node_b"])
    node_ids.discard(gnd)
    path_nodes = sorted(node_ids)
    node_pos = {n: i for i, n in enumerate(path_nodes)}

    for (a, b) in series:
        if abs(node_pos[a] - node_pos[b]) != 1:
            return False, f"N{a}-N{b} skips over intermediate nodes (not a straight left-to-right chain)"

    # every adjacent pair needs a series component: the renderer would
    # otherwise join them with a plain wire, drawing a short circuit
    # that isn't in the netlist
    for a, b in zip(path_nodes, path_nodes[1:]):
        if (a, b) not in series:
            return False, f"nothing connects N{a} to N{b} in series (the chain is broken)"

    if topo["in_node"] not in path_nodes:
        return False, "IN node has no component connected to it"
    if topo["out_node"] not in path_nodes:
        return False, "OUT node has no component connected to it"
    # the IN/OUT terminal stubs are drawn outward from the chain's two
    # ends; from a middle node they'd run straight over its neighbors
    if topo["in_node"] != path_nodes[0]:
        return False, f"IN (N{topo['in_node']}) isn't at the start of the chain (N{path_nodes[0]})"
    if topo["out_node"] != path_nodes[-1]:
        return False, f"OUT (N{topo['out_node']}) isn't at the end of the chain (N{path_nodes[-1]})"

    return True, ""


def render_schematic_svg(topo: dict, title: str = "") -> str:
    series, shunt = _classify_components(topo)
    path_nodes, x_of, right_need = _build_layout(topo, series, shunt)
    node_pos = {n: i for i, n in enumerate(path_nodes)}

    elements = []
    ref_counters = {"R": 0, "L": 0, "C": 0}

    def next_ref(ctype):
        ref_counters[ctype] += 1
        return f"{ctype}{ref_counters[ctype]}"

    # ---- main signal wire: draw the base horizontal line across every
    # adjacent pair of path nodes UNLESS that pair's only connection is a
    # parallel-branch pair (k>=2, e.g. bandstop's L||C, which has no
    # direct wire -- the components themselves ARE the only path). ----
    for i in range(len(path_nodes) - 1):
        a, b = path_nodes[i], path_nodes[i + 1]
        key = (a, b) if a < b else (b, a)
        comps = series.get(key, [])
        if len(comps) == 1:
            c = comps[0]
            half_len = BODY_HALF + LEAD
            # center within the actual clear channel, not the raw node
            # midpoint -- if one node's own shunts spread out toward this
            # component, shift away from them so the component still looks
            # centered in the open space rather than crowded to one side
            _, reach_a_right = _shunt_reach(a, shunt)
            reach_b_left, _ = _shunt_reach(b, shunt)
            eff_left = x_of[a] + reach_a_right
            eff_right = x_of[b] - reach_b_left
            cx = (eff_left + eff_right) / 2
            cx = max(x_of[a] + half_len + 5, min(x_of[b] - half_len - 5, cx))
            # leads run only up to the symbol's own footprint -- the symbol
            # draws its own leads inward from there, so the plain wire and
            # the symbol body never occupy the same stretch of the line
            elements.append(f'<line x1="{x_of[a]:.1f}" y1="{SIGNAL_Y}" x2="{cx - half_len:.1f}" y2="{SIGNAL_Y}" class="wire"/>')
            elements.append(f'<line x1="{cx + half_len:.1f}" y1="{SIGNAL_Y}" x2="{x_of[b]:.1f}" y2="{SIGNAL_Y}" class="wire"/>')
            ref = next_ref(c["type"])
            elements.append(_place_symbol(c["type"], cx, SIGNAL_Y, 0, half_len))
            elements.append(_text(cx, SIGNAL_Y - 22, _comp_label(c, ref)))
        elif len(comps) >= 2:
            # parallel branches: no direct wire between a and b -- route
            # each component on its own offset horizontal row. The box
            # itself is a FIXED width regardless of how far apart a and b
            # end up (see BOX_MAX_WIDTH) -- plain wire leads fill the rest
            # of the gap on each side, so the box's own vertical stubs
            # never sit at a node's exact x and can't collide with
            # whatever that node's shunt is doing.
            n = len(comps)
            cx = (x_of[a] + x_of[b]) / 2
            box_half = _box_half_width(a, b, x_of, shunt)
            box_left, box_right = cx - box_half, cx + box_half
            elements.append(f'<line x1="{x_of[a]:.1f}" y1="{SIGNAL_Y}" x2="{box_left:.1f}" y2="{SIGNAL_Y}" class="wire"/>')
            elements.append(f'<line x1="{box_right:.1f}" y1="{SIGNAL_Y}" x2="{x_of[b]:.1f}" y2="{SIGNAL_Y}" class="wire"/>')
            # three or more branches pack closer, each labeled just above
            # itself (in the gap below the branch above): at the 2-branch
            # spacing the lowest one would reach down into the shunt labels
            gap = PARALLEL_GAP if n <= 2 else PARALLEL_GAP_DENSE
            offsets = [
                (j - (n - 1) / 2) * gap for j in range(n)
            ]
            for c, off in zip(comps, offsets):
                y = SIGNAL_Y + off
                half_len = BODY_HALF + LEAD
                elements.append(f'<line x1="{box_left:.1f}" y1="{SIGNAL_Y}" x2="{box_left:.1f}" y2="{y:.1f}" class="wire"/>')
                elements.append(f'<line x1="{box_left:.1f}" y1="{y:.1f}" x2="{cx - half_len:.1f}" y2="{y:.1f}" class="wire"/>')
                elements.append(f'<line x1="{cx + half_len:.1f}" y1="{y:.1f}" x2="{box_right:.1f}" y2="{y:.1f}" class="wire"/>')
                elements.append(f'<line x1="{box_right:.1f}" y1="{y:.1f}" x2="{box_right:.1f}" y2="{SIGNAL_Y}" class="wire"/>')
                ref = next_ref(c["type"])
                elements.append(_place_symbol(c["type"], cx, y, 0, half_len))
                if n >= 3 or off <= 0:
                    label_y = y - BOX_LABEL_DY + 10
                else:
                    label_y = y + BOX_LABEL_DY
                elements.append(_text(cx, label_y, _comp_label(c, ref)))
        else:
            # no series component directly between these two indices at
            # all (shouldn't happen for random_filter_topology's own
            # output, but keep the path electrically connected if it does)
            elements.append(
                f'<line x1="{x_of[a]:.1f}" y1="{SIGNAL_Y}" x2="{x_of[b]:.1f}" y2="{SIGNAL_Y}" class="wire"/>'
            )
        elements.append(_junction_dot(x_of[a], SIGNAL_Y))
    if path_nodes:
        elements.append(_junction_dot(x_of[path_nodes[-1]], SIGNAL_Y))

    # ---- shunt branches: one or more components from a path node down
    # to the ground rail, side by side if more than one hangs off the
    # same node. A parallel-branch box now keeps a fixed width well
    # inset from both of its nodes (see the box-drawing code above), so
    # a node's shunt never shares a column with the box's own stub --
    # every branch can just drop straight off the node with its own tee,
    # full stop, no box-adjacency special-casing needed.
    ground_drop_xs = []
    for node, comps in shunt.items():
        base_x = x_of[node]
        offsets = _shunt_offsets(node, shunt)
        for c, off in zip(comps, offsets):
            drop_x = base_x + off
            ground_drop_xs.append(drop_x)
            top_y = SIGNAL_Y
            if off != 0:
                # independent tee: a plain horizontal run off the node at
                # SIGNAL_Y, ending in its own junction dot, then a straight
                # vertical drop -- a clean 90-degree-only connection that
                # doesn't share a wire with any other branch on this node
                elements.append(f'<line x1="{base_x:.1f}" y1="{SIGNAL_Y:.1f}" x2="{drop_x:.1f}" y2="{SIGNAL_Y:.1f}" class="wire"/>')
                elements.append(_junction_dot(drop_x, SIGNAL_Y))
            cy = (top_y + GND_Y) / 2
            ref = next_ref(c["type"])
            elements.append(f'<line x1="{drop_x:.1f}" y1="{top_y:.1f}" x2="{drop_x:.1f}" y2="{cy - BODY_HALF - LEAD:.1f}" class="wire"/>')
            elements.append(_place_symbol(c["type"], drop_x, cy, 90, BODY_HALF + LEAD))
            elements.append(f'<line x1="{drop_x:.1f}" y1="{cy + BODY_HALF + LEAD:.1f}" x2="{drop_x:.1f}" y2="{GND_Y:.1f}" class="wire"/>')
            elements.append(_junction_dot(drop_x, GND_Y))
            # push each branch's label away in the same direction it was
            # offset (left branches label left, right branches label
            # right) -- keeps multiple shunts off one node from stacking
            # their labels on top of each other or the next branch's symbol.
            # A centered (off==0) branch instead defaults away from any
            # parallel-branch box bordering this node on only one side, so
            # its label doesn't read as jammed up against the box.
            if off < 0:
                goes_left = True
            elif off > 0:
                goes_left = False
            else:
                goes_left = _box_adjacent_dir(node, path_nodes, node_pos, series) < 0
            if goes_left:
                elements.append(_text(drop_x - 22, cy + 4, _comp_label(c, ref), anchor="end"))
            else:
                elements.append(_text(drop_x + 22, cy + 4, _comp_label(c, ref), anchor="start"))

    # ---- ground rail ----
    if ground_drop_xs:
        rail_x0 = min(ground_drop_xs) - 30
        rail_x1 = max(ground_drop_xs) + 30
        elements.append(f'<line x1="{rail_x0:.1f}" y1="{GND_Y:.1f}" x2="{rail_x1:.1f}" y2="{GND_Y:.1f}" class="wire"/>')
        elements.append(_ground_symbol(rail_x0, GND_Y))
        elements.append(_text(rail_x0 - 16, GND_Y + 14, "GND", anchor="end"))

    # ---- IN / OUT terminals ----
    # The terminal stub starts from the node's outermost shunt tee, not the
    # node itself: a 2-shunt OUT node (e.g. a 2nd-order low-pass's C and R)
    # spreads a tee 45px to the right, and a stub measured from the node
    # would put the OUT terminal in the middle of that tee's wire, with
    # the outer shunt hanging beyond the output terminal.
    in_x, out_x = x_of[topo["in_node"]], x_of[topo["out_node"]]
    in_reach, _ = _shunt_reach(topo["in_node"], shunt)
    _, out_reach = _shunt_reach(topo["out_node"], shunt)
    in_term_x = in_x - in_reach - TERM_STUB
    out_term_x = out_x + out_reach + TERM_STUB
    elements.append(f'<line x1="{in_term_x:.1f}" y1="{SIGNAL_Y}" x2="{in_x:.1f}" y2="{SIGNAL_Y}" class="wire"/>')
    elements.append(f'<circle cx="{in_term_x:.1f}" cy="{SIGNAL_Y}" r="4" class="terminal"/>')
    elements.append(_text(in_term_x, SIGNAL_Y - 18, "IN", cls="terminal-label"))

    elements.append(f'<line x1="{out_x:.1f}" y1="{SIGNAL_Y}" x2="{out_term_x:.1f}" y2="{SIGNAL_Y}" class="wire"/>')
    elements.append(f'<circle cx="{out_term_x:.1f}" cy="{SIGNAL_Y}" r="4" class="terminal"/>')
    elements.append(_text(out_term_x, SIGNAL_Y - 18, "OUT", cls="terminal-label"))

    # ---- node index labels (small, muted) above each path node ----
    for n in path_nodes:
        elements.append(_text(x_of[n], SIGNAL_Y - 8, f"N{n}", cls="node-label"))

    last_right_need = right_need.get(path_nodes[-1], 0.0) if path_nodes else 0.0
    width = (max(x_of.values()) if x_of else MARGIN_X) + max(MARGIN_X, last_right_need) + out_reach + TERM_STUB + 20
    height = GND_Y + 60

    title_svg = f'<text x="{width/2:.1f}" y="30" class="title" text-anchor="middle">{title}</text>' if title else ""

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}"
     width="{width:.0f}" height="{height:.0f}" font-family="ui-monospace, Menlo, Consolas, monospace">
  <style>
    .wire {{ stroke: #1a1a1a; stroke-width: 2; }}
    .symbol-line {{ stroke: #1a1a1a; stroke-width: 2; fill: none; }}
    .junction {{ fill: #1a1a1a; }}
    .terminal {{ fill: #ffffff; stroke: #1a1a1a; stroke-width: 2; }}
    .label {{ font-size: 12px; fill: #1a1a1a; }}
    .node-label {{ font-size: 10px; fill: #9a9a9a; }}
    .terminal-label {{ font-size: 13px; font-weight: 700; fill: #1a1a1a; }}
    .title {{ font-size: 15px; font-weight: 700; fill: #1a1a1a; }}
  </style>
  <rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>
  {title_svg}
  {''.join(elements)}
</svg>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: nondeterministic)")
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument("--max-stages", type=int, default=5)
    parser.add_argument("--output", default="random_circuit_schematic.svg", help="SVG output path")
    parser.add_argument("--png", default=None, help="also render a PNG snapshot to this path (needs Playwright)")
    args = parser.parse_args()
    random.seed(args.seed)

    topo = random_filter_topology(max_components=args.max_components, max_stages=args.max_stages)

    stage_str = " -> ".join(f"{s['class']}{s['order']}" for s in topo.get("filter_stages", []))
    title = f"{len(topo['components'])} components" + (f"  |  {stage_str}" if stage_str else "")

    svg = render_schematic_svg(topo, title=title)
    with open(args.output, "w") as fh:
        fh.write(svg)
    print(f"Wrote schematic -> {args.output}")
    print(f"Circuit: IN=N{topo['in_node']}  GND=N{topo['gnd_node']}  OUT=N{topo['out_node']}  "
          f"({len(topo['components'])} components)")
    if stage_str:
        print(f"Filter cascade: {stage_str}")
    for c in topo["components"]:
        print(f"  {c['type']}  N{c['node_a']} - N{c['node_b']}   ({c['human']})")

    if args.png:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print(f"--png requested but Playwright isn't installed -- skipping PNG render.")
            return
        import os
        svg_abs = os.path.abspath(args.output)
        m = re.search(r'width="(\d+)" height="(\d+)"', svg)
        vp_w, vp_h = (int(m.group(1)) + 40, int(m.group(2)) + 40) if m else (1600, 900)
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path="/opt/pw-browsers/chromium") \
                if os.path.exists("/opt/pw-browsers/chromium") else p.chromium.launch()
            page = browser.new_page(viewport={"width": vp_w, "height": vp_h})
            page.goto(f"file://{svg_abs}")
            page.screenshot(path=args.png)
            browser.close()
        print(f"Wrote PNG snapshot -> {args.png}")


if __name__ == "__main__":
    main()