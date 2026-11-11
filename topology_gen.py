"""
Generate standard passive R/L/C filter topologies.

Each topology is built from 1 to 6 canonical, textbook passive filter
STAGES cascaded in series -- not an arbitrary/random graph. Every stage
is one of the four standard filter classes, at 1st or 2nd order:

  lowpass  (1st order RC or RL; 2nd order L-series + C/R shunt)
  highpass (1st order RC or RL; 2nd order C-series + L/R shunt)
  bandpass (2nd order: series R, shunt parallel L+C tank)
  bandstop (2nd order / notch: parallel L+C as the series element, shunt R)

A 1st-order stage is 2 components (one reactive element + one resistor);
a 2nd-order stage is always exactly one R, one L, and one C, arranged
per filter class. Stages are cascaded by feeding one stage's output node
directly into the next stage's input node, up to a total component
budget (MAX_COMPONENTS = 12, at most MAX_STAGES = 6 stages) -- so a generated topology can
range from 2 components (a single 1st-order stage) up to the budget,
never a meaningless/unmotivated wiring of parts.

Netlists are printed in a simple tokenized netlist format:

    R N1 N2 VAL 4.7 x1k      -> 4.7 kOhm resistor between N1 and N2
    L N2 N0 VAL 10 x1m       -> 10 mH inductor between N2 and N0
    C N1 N0 VAL 100 x1n      -> 100 nF capacitor between N1 and N0

Run directly to generate the training dataset (see _parse_args).
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# ----------------------------------------------------------------------
# Circuit size limits -- fixed, not command-line options, so the dataset,
# train.py and test.py can't drift apart. train.py keeps its own copy of
# MAX_COMPONENTS (it sizes the vocab and padded sequence length); change
# both together if you ever regenerate with a different budget.
# ----------------------------------------------------------------------
MAX_COMPONENTS = 12   # total components per topology, across all stages
MAX_STAGES = 6        # cascaded filter stages (6 is reachable only with all 1st-order stages)

# ----------------------------------------------------------------------
# Component value tables (E12 preferred-value series)
# ----------------------------------------------------------------------
E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]

# (decade exponent range, unit symbol, prefix table) per component type
VALUE_RANGES = {
    "R": {"exp_range": (1, 6), "unit": "Ohm"},   # 10 Ohm .. 1 MOhm
    "L": {"exp_range": (-6, -1), "unit": "H"},   # 1 uH .. 100 mH
    "C": {"exp_range": (-12, -4), "unit": "F"},  # 1 pF .. 100 uF
}

PREFIXES = {
    -12: "p", -9: "n", -6: "u", -3: "m", 0: "", 3: "k", 6: "M", 9: "G",
}


def random_value_tokens(comp_type: str):
    """Pick a random E12-series value for a component type.

    Returns (mantissa, decade_token, human_readable_string) where
    decade_token is a coarse "xN" token (e.g. x1k, x1m, x1n) suitable
    for use as a discrete vocabulary entry, and mantissa is one of the
    12 E12 values.
    """
    lo, hi = VALUE_RANGES[comp_type]["exp_range"]
    unit = VALUE_RANGES[comp_type]["unit"]
    exp = random.randint(lo, hi)
    mantissa = random.choice(E12)

    # snap exp down to the nearest metric-prefix decade for the token,
    # and fold the remainder into the mantissa
    prefix_exp = max(p for p in PREFIXES if p <= exp)
    remainder = exp - prefix_exp
    mantissa_scaled = mantissa * (10 ** remainder)

    decade_token = f"x1{PREFIXES[prefix_exp]}" if PREFIXES[prefix_exp] else "x1"
    human = f"{mantissa_scaled:g}{PREFIXES[prefix_exp]}{unit}"
    value_si = mantissa * (10 ** exp)  # actual value in base SI units (Ohm/H/F)
    return round(mantissa_scaled, 3), decade_token, human, value_si


# ----------------------------------------------------------------------
# Standard passive filter synthesis
#
# Instead of wiring components into an arbitrary random graph, every
# topology here is assembled from a small catalog of textbook passive
# filter STAGES -- the same handful of circuits you'd find in any
# analog filter design reference. What's random is which filter classes
# get cascaded, at which order, in which order, and their component
# values -- not the underlying circuit structure.
# ----------------------------------------------------------------------
FILTER_CLASSES = ["lowpass", "highpass", "bandpass", "bandstop"]


def _make_component(comp_type: str, node_a: int, node_b: int) -> dict:
    mantissa, decade_token, human, value_si = random_value_tokens(comp_type)
    return {
        "type": comp_type,
        "node_a": node_a,
        "node_b": node_b,
        "mantissa": mantissa,
        "decade_token": decade_token,
        "human": human,
        "value_si": value_si,
    }


def _build_filter_stage(filter_class: str, order: int, in_node: int, out_node: int) -> list:
    """Build one canonical passive R/L/C filter stage between `in_node`
    and `out_node` (ground is always node 0, used implicitly by every
    shunt element). Every stage below is a standard, well-known filter
    section -- not an arbitrary wiring of parts.

    1st-order stages: one series element + one shunt element (2
    components). 2nd-order stages: exactly one R, one L, and one C,
    arranged per filter class (3 components) -- so component COUNT
    alone tells you the order, and component ROLES always match a
    textbook description of that filter type.
    """
    if filter_class == "lowpass":
        if order == 1:
            if random.random() < 0.5:
                # RC low-pass: series R, shunt C
                return [_make_component("R", in_node, out_node),
                        _make_component("C", out_node, 0)]
            # RL low-pass: series L, shunt R
            return [_make_component("L", in_node, out_node),
                    _make_component("R", out_node, 0)]
        # 2nd order: series L, shunt C, shunt R (resistive load) --
        # a classic (resistively-terminated) LC low-pass ladder section.
        return [_make_component("L", in_node, out_node),
                _make_component("C", out_node, 0),
                _make_component("R", out_node, 0)]

    if filter_class == "highpass":
        if order == 1:
            if random.random() < 0.5:
                # RC high-pass: series C, shunt R
                return [_make_component("C", in_node, out_node),
                        _make_component("R", out_node, 0)]
            # RL high-pass: series R, shunt L
            return [_make_component("R", in_node, out_node),
                    _make_component("L", out_node, 0)]
        # 2nd order: series C, shunt L, shunt R (resistive load)
        return [_make_component("C", in_node, out_node),
                _make_component("L", out_node, 0),
                _make_component("R", out_node, 0)]

    if filter_class == "bandpass":
        # 2nd order only: series R (source), shunt L + shunt C forming a
        # parallel L-C tank to ground. Near resonance the tank's
        # impedance is high, so the signal passes through relatively
        # unattenuated; away from resonance one of L/C dominates and
        # shorts the output to ground -- a standard RLC band-pass.
        return [_make_component("R", in_node, out_node),
                _make_component("L", out_node, 0),
                _make_component("C", out_node, 0)]

    # "bandstop" (notch), 2nd order only: L and C both connected
    # directly between in_node and out_node (a parallel L-C tank used as
    # the SERIES element) plus a shunt R load. Near resonance the tank's
    # impedance is high, blocking transmission -- a standard notch.
    return [_make_component("L", in_node, out_node),
            _make_component("C", in_node, out_node),
            _make_component("R", out_node, 0)]


def random_filter_topology(max_components: int = MAX_COMPONENTS, max_stages: int = MAX_STAGES) -> dict:
    """Cascade 1..max_stages canonical passive filter stages (see
    _build_filter_stage), stopping once the next stage would exceed
    `max_components`. Every stage's output node feeds directly into the
    next stage's input node; ground (N0) is shared by every stage's
    shunt elements, and the overall IN/OUT are the first stage's input
    and the last stage's output.

    Returns a topology dict in exactly the format the rest of this file
    already expects (num_nodes/in_node/gnd_node/out_node/components),
    plus a "filter_stages" field describing the cascade for readability
    and dataset metadata.
    """
    max_components = max(2, max_components)
    n_stages_target = random.randint(1, max(1, max_stages))

    components = []
    filter_stages = []
    node_cursor = 1          # N0 is ground; N1 is the overall IN node
    in_node = node_cursor
    cur_in = in_node

    for _ in range(n_stages_target):
        remaining = max_components - len(components)
        if remaining < 2:
            break  # no budget left for even the smallest possible stage

        if remaining >= 3:
            filter_class = random.choice(FILTER_CLASSES)
            order = 2 if filter_class in ("bandpass", "bandstop") else random.choice([1, 2])
        else:  # remaining == 2: only a 1st-order low/high-pass stage fits
            filter_class = random.choice(["lowpass", "highpass"])
            order = 1

        node_cursor += 1
        cur_out = node_cursor
        stage_components = _build_filter_stage(filter_class, order, cur_in, cur_out)
        components.extend(stage_components)
        filter_stages.append({
            "class": filter_class, "order": order,
            "in_node": cur_in, "out_node": cur_out,
        })
        cur_in = cur_out

    return {
        "num_nodes": node_cursor + 1,   # N0 (ground) .. N{node_cursor}
        "in_node": in_node,
        "gnd_node": 0,
        "out_node": cur_in,
        "components": components,
        "filter_stages": filter_stages,
    }


# ----------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------
def format_netlist(topo: dict, index: int) -> str:
    lines = [f"--- Topology {index} ---"]
    lines.append(
        f"IN=N{topo['in_node']}  GND=N{topo['gnd_node']}  OUT=N{topo['out_node']}"
    )
    if topo.get("filter_stages"):
        stage_str = " -> ".join(
            f"{s['class']}{s['order']}" for s in topo["filter_stages"]
        )
        lines.append(f"Filter cascade ({len(topo['filter_stages'])} stage"
                      f"{'s' if len(topo['filter_stages']) != 1 else ''}): {stage_str}")
    lines.append(
        f"[BOS] IN N{topo['in_node']} GND N{topo['gnd_node']} OUT N{topo['out_node']}"
    )
    for comp in topo["components"]:
        token_line = (
            f"{comp['type']} N{comp['node_a']} N{comp['node_b']} "
            f"VAL {comp['mantissa']} {comp['decade_token']}"
        )
        readable = f"{comp['type']}  N{comp['node_a']} - N{comp['node_b']}  ({comp['human']})"
        lines.append(f"  {token_line:<32} # {readable}")
    lines.append("[SEP]")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# AC (frequency-domain) simulation
#
# Applies an ideal AC voltage source (fixed amplitude, 0 deg reference
# phase) at the IN node w.r.t. GND, and sweeps frequency from 1 Hz to
# 1 GHz at 10 points per decade (matches SPICE's `.ac dec 10 1 1e9`).
# For every sample, solves for the complex node voltages via Modified
# Nodal Analysis (MNA) and reports the OUT-node transfer function
# H(jw) = Vout / Vin as magnitude (dB) and phase (deg).
#
# For ideal linear R/L/C elements this is exactly the linear system a
# SPICE .AC analysis solves internally, so it is numerically equivalent
# to a real SPICE AC sweep. If a real `ngspice` binary is available on
# PATH, `simulate_ac_response` will use it instead automatically.
# ----------------------------------------------------------------------
def ac_sweep_frequencies(f_start: float = 1.0, f_stop: float = 1e9,
                          points_per_decade: int = 10) -> np.ndarray:
    decades = math.log10(f_stop / f_start)
    n_points = int(round(decades * points_per_decade)) + 1
    return np.logspace(math.log10(f_start), math.log10(f_stop), n_points)


# Small shunt conductance added from every non-ground node to ground.
# Real SPICE does exactly this internally (its "gmin" parameter, default
# 1e-12 S) to keep the nodal matrix from going singular. It matters most
# for topologies with no resistor anywhere in the signal path: a pure L/C
# network has zero damping, so at certain frequencies its reactances can
# cancel exactly, driving some node's impedance to true, mathematical
# infinity (a lossless resonance) -- and a Modified Nodal Analysis matrix
# with a node offering zero conductance to the rest of the circuit is
# singular by construction, not from a coding mistake. 1e-12 S is ~1e12
# ohms to ground: negligible next to any real component value in this
# dataset (10 ohm - 1 Mohm), so it doesn't perturb the physics, but it's
# enough to keep the linear solve well-posed exactly at resonance.
GMIN = 1e-12


def _mna_ac_response(topo: dict, freqs: np.ndarray, vin_amplitude: float = 1.0):
    """Internal from-scratch AC solver (Modified Nodal Analysis).

    Builds the complex nodal admittance matrix at each frequency and
    solves for node voltages, exactly like SPICE's .AC analysis does
    for ideal linear R/L/C elements.
    """
    num_nodes = topo["num_nodes"]
    in_node = topo["in_node"]
    out_node = topo["out_node"]
    components = topo["components"]

    n_unknown_nodes = num_nodes - 1          # exclude ground (node 0)
    size = n_unknown_nodes + 1               # + 1 source-current unknown
    src_idx = n_unknown_nodes                # last row/col = source current

    def node_idx(n):
        return None if n == 0 else n - 1

    mag_db = np.empty(len(freqs))
    phase_deg = np.empty(len(freqs))
    vin_complex = complex(vin_amplitude, 0.0)

    for k, f in enumerate(freqs):
        w = 2.0 * math.pi * f
        Y = np.zeros((size, size), dtype=complex)
        b = np.zeros(size, dtype=complex)

        for comp in components:
            ctype, a, bb, val = comp["type"], comp["node_a"], comp["node_b"], comp["value_si"]
            if ctype == "R":
                y = 1.0 / val
            elif ctype == "L":
                y = 1.0 / (1j * w * val) if w > 0 else 1e12  # avoid div-by-zero at DC
            else:  # "C"
                y = 1j * w * val

            ia, ib = node_idx(a), node_idx(bb)
            if ia is not None:
                Y[ia, ia] += y
            if ib is not None:
                Y[ib, ib] += y
            if ia is not None and ib is not None:
                Y[ia, ib] -= y
                Y[ib, ia] -= y

        # gmin: tiny shunt conductance from every non-ground node to
        # ground, to guarantee a well-posed matrix even at an exact
        # lossless resonance (see GMIN comment above)
        for i in range(n_unknown_nodes):
            Y[i, i] += GMIN

        # ideal voltage source stamp: V(in_node) - V(gnd) = vin_complex
        iin = node_idx(in_node)
        if iin is not None:
            Y[iin, src_idx] += 1
            Y[src_idx, iin] += 1
        b[src_idx] = vin_complex

        try:
            x = np.linalg.solve(Y, b)
        except np.linalg.LinAlgError:
            # Even with gmin this shouldn't happen, but fall back to a
            # least-squares (pseudo-inverse) solve rather than crashing
            # the whole sweep over one degenerate sample point.
            x, *_ = np.linalg.lstsq(Y, b, rcond=None)

        iout = node_idx(out_node)
        vout = x[iout] if iout is not None else complex(0.0, 0.0)

        h = vout / vin_complex
        mag = abs(h)
        mag_db[k] = 20.0 * math.log10(mag) if mag > 1e-300 else -300.0
        phase_deg[k] = math.degrees(math.atan2(h.imag, h.real))

    return mag_db, phase_deg


def write_spice_deck(topo: dict, index: int, path: str,
                      vin_amplitude: float = 1.0,
                      f_start: float = 1.0, f_stop: float = 1e9,
                      points_per_decade: int = 10) -> None:
    """Write a real ngspice/LTspice-compatible .cir deck for cross-checking."""
    lines = [f"Topology {index} - cascaded standard R/L/C filter network"]
    lines.append(f"V1 N{topo['in_node']} 0 AC {vin_amplitude}")
    for i, comp in enumerate(topo["components"], start=1):
        prefix = comp["type"]
        lines.append(f"{prefix}{i} N{comp['node_a']} N{comp['node_b']} {comp['value_si']:.6g}")
    lines.append(f".ac dec {points_per_decade} {f_start:g} {f_stop:g}")
    lines.append(".control")
    lines.append("run")
    lines.append(f"wrdata response_{index}.txt vdb(N{topo['out_node']}) vp(N{topo['out_node']})")
    lines.append(".endc")
    lines.append(".end")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _run_real_ngspice(topo: dict, index: int, freqs: np.ndarray, vin_amplitude: float):
    """Try to run a real ngspice binary; return (mag_db, phase_deg) or None."""
    ngspice_path = shutil.which("ngspice")
    if ngspice_path is None:
        return None

    with tempfile.TemporaryDirectory() as tmp:
        deck_path = os.path.join(tmp, f"topo_{index}.cir")
        write_spice_deck(topo, index, deck_path, vin_amplitude=vin_amplitude)
        out_path = os.path.join(tmp, f"response_{index}.txt")
        try:
            subprocess.run(
                [ngspice_path, "-b", deck_path],
                cwd=tmp, capture_output=True, timeout=30, check=True,
            )
            data = np.loadtxt(out_path)
        except Exception:
            return None

        # wrdata columns: freq, vdb, freq(repeat), vp
        f_sim = data[:, 0]
        mag_db = data[:, 1]
        phase_deg = data[:, 3]
        # resample onto our exact frequency grid in case ngspice's grid differs slightly
        mag_db = np.interp(np.log10(freqs), np.log10(f_sim), mag_db)
        phase_deg = np.interp(np.log10(freqs), np.log10(f_sim), phase_deg)
        return mag_db, phase_deg


def simulate_ac_response(topo: dict, index: int, vin_amplitude: float = 1.0,
                          f_start: float = 1.0, f_stop: float = 1e9,
                          points_per_decade: int = 10):
    """Return (freqs, mag_db, phase_deg, engine) for the topology's AC sweep.

    Uses a real ngspice binary if available on PATH, otherwise falls back
    to the built-in Modified Nodal Analysis solver (numerically
    equivalent for these ideal linear R/L/C circuits).
    """
    freqs = ac_sweep_frequencies(f_start, f_stop, points_per_decade)

    result = _run_real_ngspice(topo, index, freqs, vin_amplitude)
    if result is not None:
        mag_db, phase_deg = result
        return freqs, mag_db, phase_deg, "ngspice"

    mag_db, phase_deg = _mna_ac_response(topo, freqs, vin_amplitude)
    return freqs, mag_db, phase_deg, "internal-mna"


# ----------------------------------------------------------------------
# Dataset export
#
# One file is written:
#   - rlc_training_dataset.jsonl : one JSON object per topology, pairing
#                                the full netlist (as both structured
#                                fields and the tokenized sequence used
#                                elsewhere in this project) with its
#                                complete frequency response. This is the
#                                format meant for actually training on --
#                                one line == one (input, output) example,
#                                streamable without loading everything
#                                into memory at once.
# ----------------------------------------------------------------------
def netlist_input_tokens(topo: dict) -> list:
    """The same token stream printed by format_netlist, but as a list of
    discrete string tokens -- i.e. what would actually be fed to the
    model's embedding layer as the input sequence."""
    tokens = [
        "[BOS]",
        "IN", f"N{topo['in_node']}",
        "GND", f"N{topo['gnd_node']}",
        "OUT", f"N{topo['out_node']}",
    ]
    for comp in topo["components"]:
        tokens += [
            comp["type"],
            f"N{comp['node_a']}",
            f"N{comp['node_b']}",
            "VAL",
            str(comp["mantissa"]),
            comp["decade_token"],
        ]
    tokens.append("[SEP]")
    return tokens


def topology_to_dataset_record(topo: dict, index: int, freqs, mag_db, phase_deg,
                                vin_amplitude: float, engine: str) -> dict:
    return {
        "topology_id": index,
        "num_nodes": topo["num_nodes"],
        "in_node": topo["in_node"],
        "gnd_node": topo["gnd_node"],
        "out_node": topo["out_node"],
        "components": [
            {
                "type": c["type"],
                "node_a": c["node_a"],
                "node_b": c["node_b"],
                "value_si": c["value_si"],   # exact value in Ohm / H / F
                "human": c["human"],
            }
            for c in topo["components"]
        ],
        "input_tokens": netlist_input_tokens(topo),
        "filter_stages": topo.get("filter_stages", []),
        "vin_amplitude": vin_amplitude,
        "sim_engine": engine,
        "response": {
            "freq_hz": [float(f) for f in freqs],
            "mag_db": [float(m) for m in mag_db],
            "phase_deg": [float(p) for p in phase_deg],
        },
    }


# ----------------------------------------------------------------------
# Parallel generation
#
# Generating and simulating one topology is entirely independent of every
# other one -- an embarrassingly parallel workload -- and it's all
# CPU-bound Python/numpy (random topology construction is plain Python;
# simulate_ac_response's MNA solve is a handful of tiny -- typically
# well under 15x15 -- numpy.linalg.solve calls per topology). A
# ProcessPoolExecutor (not threads) is the right tool: plain Python
# object construction and small numpy solves both hold the GIL for most
# of their time, so threads would mostly serialize anyway; separate
# processes sidestep that. GPU acceleration was considered and doesn't
# make sense here: each topology's linear solve is minutes-of-work small
# (a handful of complex numbers), topologies have different numbers of
# nodes/components (a ragged shape that resists batching into one dense
# tensor op), and per-call dispatch overhead to the GPU would dwarf the
# actual compute -- you'd only win by batching thousands of
# *same-shaped* circuits into one batched torch.linalg.solve, which is a
# much bigger rewrite than this one-shot dataset-generation script needs.
# ----------------------------------------------------------------------
def _generate_and_simulate(task):
    """Module-level (picklable) unit of work for the process pool: build one
    random topology and simulate its AC response. One call == one
    topology, fully independent of every other call, which is exactly
    what --workers spreads across worker processes.

    `seed` here is a PER-TASK seed derived from (base_seed, index), not
    a single shared stream -- with --seed set, this makes the generated
    dataset reproducible regardless of --workers or which process
    happens to pick up which topology index, which a single global
    random.seed() call (fine in a single sequential process) can't
    guarantee once work is spread across multiple processes.
    """
    i, vin_amplitude, seed = task
    if seed is not None:
        random.seed(seed * 1_000_003 + i)
    topo = random_filter_topology()
    freqs, mag_db, phase_deg, engine = simulate_ac_response(
        topo, i, vin_amplitude=vin_amplitude, f_start=1.0, f_stop=1e9, points_per_decade=10,
    )
    return i, topo, freqs, mag_db, phase_deg, engine


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5_000_000,
                         help="number of random topologies to generate (default: 5,000,000)")
    parser.add_argument("--seed", type=int, default=None,
                         help="random seed (default: nondeterministic; with --workers > 1, "
                              "seeds each topology individually -- see _generate_and_simulate)")
    parser.add_argument("--workers", type=int, default=1,
                         help="worker processes for parallel generation + simulation "
                              "(default: 1, sequential; try your core count for large --count)")
    parser.add_argument("--verbose", action="store_true",
                         help="print the full netlist + sweep preview per topology "
                              "(default: off -- too slow for large --count)")
    parser.add_argument("--quiet", action="store_true",
                         help="suppress the periodic progress line (default: off)")
    parser.add_argument("--progress-every", type=int, default=None,
                         help="print a progress line every N topologies (default: auto)")
    parser.add_argument("--jsonl", default="rlc_training_dataset.jsonl",
                         help="training-ready JSONL output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.workers <= 1:
        random.seed(args.seed)  # None -> nondeterministic; single sequential stream, as before

    n_topologies = args.count
    vin_amplitude = 1.0  # fixed sine amplitude, volts
    jsonl_path = args.jsonl

    # Default output is a periodic progress line, not a full per-topology
    # dump -- too slow at --count in the millions. --verbose restores the
    # full printout; --quiet drops even the progress line.
    progress_every = args.progress_every or max(10, n_topologies // 100)

    # Stream the output to disk as each topology finishes -- peak memory
    # is whatever ONE topology's rows/record cost, regardless of --count.
    engine_used = None
    start_time = time.time()

    tasks = (
        (i, vin_amplitude,
         args.seed if args.workers > 1 else None)
        for i in range(1, n_topologies + 1)
    )

    executor = None
    if args.workers > 1:
        print(f"Generating {n_topologies} topologies across {args.workers} worker processes...\n")
        executor = ProcessPoolExecutor(max_workers=args.workers)
        # Small chunks per IPC round-trip amortize process-communication
        # overhead across many topologies, without making any one chunk so
        # large that a single worker gets stuck holding a disproportionate
        # slice of the work near the end of the run.
        chunksize = max(1, min(64, n_topologies // (args.workers * 4) or 1))
        results_iter = executor.map(_generate_and_simulate, tasks, chunksize=chunksize)
    else:
        results_iter = (_generate_and_simulate(t) for t in tasks)

    try:
        with open(jsonl_path, "w") as jsonl_fh:

            for i, topo, freqs, mag_db, phase_deg, engine in results_iter:
                engine_used = engine

                if args.verbose:
                    print(format_netlist(topo, i))
                    print(f"AC sweep: {len(freqs)} points, 1 Hz - 1 GHz, 10 pts/decade "
                          f"(engine: {engine}, Vin = {vin_amplitude} V)")
                    print(f"{'Freq (Hz)':>14} {'Mag (dB)':>12} {'Phase (deg)':>13}")
                    preview_idx = list(range(5)) + list(range(len(freqs) - 5, len(freqs)))
                    for idx in sorted(set(preview_idx)):
                        print(f"{freqs[idx]:14.4g} {mag_db[idx]:12.3f} {phase_deg[idx]:13.3f}")
                    print(f"  ... ({len(freqs) - 10} more points; full data in JSONL) ...")
                    print()
                elif not args.quiet and (i % progress_every == 0 or i == n_topologies):
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0.0
                    remaining = (n_topologies - i) / rate if rate > 0 else float("inf")
                    print(f"  {i:>10,}/{n_topologies:,} topologies "
                          f"({100 * i / n_topologies:5.1f}%)   "
                          f"{rate:7.1f}/s   elapsed {elapsed:7.1f}s   "
                          f"eta {remaining:7.1f}s   engine={engine}")

                # write this topology's record immediately, then let
                # it (and everything it references) get garbage collected
                # -- nothing from here is kept around for later.
                record = topology_to_dataset_record(
                    topo, i, freqs, mag_db, phase_deg, vin_amplitude, engine
                )
                jsonl_fh.write(json.dumps(record) + "\n")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print(f"Simulation engine used: {engine_used}")
    print(f"Training-ready (netlist + response) records written to {jsonl_path}")