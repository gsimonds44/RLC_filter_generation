"""
INVERSE-ONLY training: response -> topology.

This trains the model to go the opposite direction from the original
setup. The prompt is now a circuit's frequency response (magnitude +
phase, binned/tokenized the same way as before), and the target the
model learns to generate is the netlist that produced it (component
types, node connections, and values). The model is never trained to
predict a response from a netlist in this script -- it only ever learns
response -> topology.

This is a fundamentally harder, one-to-many problem: several different
topologies (different component choices, different node wiring, even
different values) can produce frequency responses that land on the same
binned token sequence, given finite bin resolution (1 dB / 2 deg). So
don't expect the same accuracy numbers as the forward (netlist ->
response) direction -- the held-out evaluation at the end of training
measures this directly: how often the generated netlist's node roles
(IN/GND/OUT) and components (type + node pair, matched greedily) agree
with the ground truth, and how close matched component values come out.

Train/validation split over all available topologies: 10% held out for
validation (subsampled to --val-samples for speed), 90% used for
gradient updates, every epoch printed. Scales to however many topologies
are in the JSONL file -- 10, 100, a million -- with no code changes
needed.

VRAM-limited training: --max-samples caps how many training examples are
tensorized onto the GPU at once, re-sampled fresh every epoch from the
full training pool (see build_inverse_sequence / the training loop
below) so a big dataset gets swept over many epochs instead of
memorizing one fixed slice. Current usage: --dataset generated at 5,000,000
topologies (topology_gen.py --count), --max-samples 15,000 per epoch,
--epochs 1000 -- 15,000,000 total training topologies over the run
(the pool gets swept ~3x), reported as a token total at startup.

STREAMING dataset loading: at multi-million-row scale the old
`records = [json.loads(line) for line in fh]` approach -- parsing the
WHOLE file into a Python list up front -- does not fit in RAM. A 5M-row
dataset at ~7 KB/record on disk is already ~35 GB of raw JSON text, and
the parsed-into-Python-objects form (nested dicts/lists of floats) runs
several times larger than that -- easily 100+ GB resident, before the
model or anything else. JsonlDataset below fixes this: one pass over the
file builds a byte-offset index (one int per line, ~40 MB for 5M rows)
plus the dataset's max component count (needed to size the padded
sequence length), and every record is then read directly off disk by
seeking to its offset and parsing just that one line -- only ever
--max-samples records are resident in memory at once, regardless of how
large --dataset is. See JsonlDataset's docstring.

VARIABLE-LENGTH sequences: topologies now come from topology_gen.py's
standard-filter generator, which cascades 1-6 filter stages up to a
12-component budget (MAX_COMPONENTS below, which MUST match
topology_gen.py's MAX_COMPONENTS) -- so different
topologies have different numbers of components, and their netlist
target segment is a different length. The response prompt segment is
still fixed length (same frequency grid every time), but the total
sequence is not. Every sequence is right-padded with [PAD] up to the
longest POSSIBLE sequence (computed analytically from MAX_COMPONENTS,
not measured by scanning the file -- see seq_len_full below) before
being stacked into a batch tensor; [PAD] positions are excluded from the
loss (see _tokenize_and_pad / _apply_masks below). A plain causal mask
still works unmodified here -- since padding is only ever appended AFTER
the real tokens, causal attention already can't look forward into
padding from an earlier real position, so no separate padding-attention-
mask is needed, only loss masking.

The token vocabulary is built DETERMINISTICALLY (every possible node,
component, value, frequency-index, magnitude-bin, and phase-bin token is
enumerated up front) rather than from whichever tokens happen to appear
in one sampled subset -- required once sampling changes every epoch. The
node-token range depends on MAX_COMPONENTS (see build_deterministic_vocab).

Model capacity (--d-model, --d-ff, --n-heads, --n-layers) is exposed on
the command line. Defaults were bumped from the original 128/4/512/10
(2.1M params) to 224/8/896/10 (6.3M params, ~3x the parameters) -- this
widens the model rather than deepening it, which gets substantially more
capacity for a much smaller compute cost than adding layers would:
attention cost grows with sequence length squared but only linearly with
d_model, so at this project's sequence lengths (~360 tokens) widening
d_model/d_ff is the cheaper way to add parameters. Estimated compute
cost per training step at these defaults is about 1.8x the original
128/4/512/10 config (not 3x) -- comfortably under a 2x budget. See the
module-level comment above main() for the params/compute table this was
chosen from, if you want to pick a different point on that trade-off.
--grad-checkpoint and --cpu-offload are available for memory-constrained
GPUs (see their --help text).

--amp {none,fp16,bf16} (default: none) turns on mixed-precision autocast
training -- NOT weight quantization, see the module comment above
_autocast_ctx for the distinction. Only helps on CUDA; 'bf16' is the
safer default on Ampere-or-newer GPUs.

When training finishes, the model is saved to --save-path (default:
inverse_model.pt) as a torch checkpoint containing the model weights,
the vocabulary, and the config needed to reconstruct the model for
inference later (e.g. a design/search script that generates candidate
topologies for a target response).

Usage:
    python train.py --dataset rlc_training_dataset.jsonl
    python train.py --dataset rlc_training_dataset.jsonl --epochs 200 --batch-size 16
"""

import argparse
import json
import math
import random

import torch
import torch.nn as nn

from model import TenLayerTransformer


# ----------------------------------------------------------------------
# Response tokenization (binning)
# ----------------------------------------------------------------------
MAG_MIN, MAG_MAX, MAG_STEP = -160, 20, 1     # dB
PH_MIN, PH_MAX, PH_STEP = -180, 180, 2       # degrees


def _bin_value(x, lo, hi, step):
    x = max(lo, min(hi, x))
    return int(round(x / step) * step)


def mag_token(mag_db: float) -> str:
    return f"MAG_{_bin_value(mag_db, MAG_MIN, MAG_MAX, MAG_STEP)}"


def phase_token(phase_deg: float) -> str:
    return f"PH_{_bin_value(phase_deg, PH_MIN, PH_MAX, PH_STEP)}"


def decode_mag(token: str) -> float:
    return float(token.split("_", 1)[1])


def decode_phase(token: str) -> float:
    return float(token.split("_", 1)[1])


def response_tokens(response: dict) -> list:
    tokens = []
    for i, (mag, ph) in enumerate(zip(response["mag_db"], response["phase_deg"])):
        tokens += [f"F{i}", mag_token(mag), phase_token(ph)]
    return tokens


# ----------------------------------------------------------------------
# Value-token decoding (inverse of random_value_tokens() in the
# generator): mantissa string + decade token ("x1", "x1k", "x1u", ...)
# -> value in base SI units (Ohm / H / F). Needed to score how close a
# GENERATED component's value comes to the true value.
# ----------------------------------------------------------------------
_PREFIX_EXPONENT = {"": 0, "p": -12, "n": -9, "u": -6, "m": -3, "k": 3, "M": 6, "G": 9}


def _decode_component_value(mantissa_token: str, decade_token: str):
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


def build_inverse_sequence(record: dict):
    """response prompt (+[BOS]/[SEP] wrapper) + [TOPO] + netlist
    description tokens + [EOS].

    This is the reverse of the original netlist -> response ordering:
    here the response is the PROMPT (what the model conditions on) and
    the netlist is the TARGET (what the model learns to generate).
    `record["input_tokens"]` (from the generator) is [BOS] ... [SEP] --
    that wrapper was built for the netlist to serve as a *prompt* in the
    forward direction, so here we strip it and reuse only the inner
    netlist-description tokens as the *target*, wrapped in their own
    [TOPO]...[EOS] markers instead.
    """
    prompt_tokens = ["[BOS]"] + response_tokens(record["response"]) + ["[SEP]"]
    netlist_core = record["input_tokens"][1:-1]  # drop wrapping [BOS] / [SEP]
    target_tokens = ["[TOPO]"] + netlist_core + ["[EOS]"]
    return prompt_tokens + target_tokens, len(prompt_tokens)


class Vocab:
    def __init__(self, tokens):
        uniq = sorted(set(tokens))
        self.itos = ["[PAD]"] + uniq
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def encode(self, tokens):
        return [self.stoi[t] for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]

    def __len__(self):
        return len(self.itos)


# ----------------------------------------------------------------------
# Deterministic vocabulary: every token the netlist/response tokenizer can
# possibly emit, enumerated up front from the fixed generation scheme --
# NOT just whatever happens to appear in one sampled subset. Required
# once sampling changes every epoch -- otherwise a later epoch could draw
# a topology using a value/node combination the vocab was never built to
# recognize.
#
# The E12/decade/value-range constants mirror topology_gen.py's
# generator exactly -- duplicated here rather than imported so this
# script doesn't depend on the generator's filename (yours is
# topology_gen.py).
#
# max_nodes is derived from MAX_COMPONENTS (see build_deterministic_vocab):
# the largest node index the generator can ever produce is reached by the
# smallest possible stages (1st-order, 2 components each) filling the
# whole budget -- max_components // 2 stages, so the node cursor reaches
# 1 + max_components // 2, and node count (including N0) is that + 1.
# Verified empirically against thousands of generated topologies at both
# max_components=10 (max node index 6) and max_components=12 (max node
# index 7) -- this formula is a tight upper bound, not a guess.
# ----------------------------------------------------------------------
MAX_COMPONENTS = 12            # matches topology_gen.py's MAX_COMPONENTS (up to 6 stages)
_E12 = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
_DECADE_PREFIXES = ["", "p", "n", "u", "m", "k", "M", "G"]  # matches generator's PREFIXES values
_N_FREQ_POINTS = 91             # 1 Hz - 1 GHz, 10 points/decade (9 decades + 1)


def build_deterministic_vocab(max_components: int) -> Vocab:
    max_nodes = max_components // 2 + 2  # N0 (ground) .. N{max_components//2 + 1}
    tokens = set()

    # structural / special tokens ([TOPO] marks the start of the
    # generated netlist target; [RESP] is unused now but kept so a
    # dataset/vocab built for the old forward direction still decodes
    # without a KeyError if you ever mix checkpoints).
    tokens.update(["[BOS]", "[SEP]", "[RESP]", "[TOPO]", "[EOS]", "IN", "GND", "OUT", "VAL"])
    tokens.update(["R", "L", "C"])

    # node tokens: N0 (ground) .. N{max nodes - 1}
    tokens.update(f"N{n}" for n in range(max_nodes))

    # decade tokens: x1, x1p, x1n, x1u, x1m, x1k, x1M, x1G
    tokens.update(f"x1{p}" for p in _DECADE_PREFIXES)

    # mantissa tokens: E12 value * 10^{0,1,2} (remainder after snapping the
    # exponent down to the nearest metric-prefix decade), string-formatted
    # exactly as random_value_tokens() does: round(..., 3), then str()
    for m in _E12:
        for r in (0, 1, 2):
            tokens.add(str(round(m * (10 ** r), 3)))

    # frequency-index tokens
    tokens.update(f"F{i}" for i in range(_N_FREQ_POINTS))

    # magnitude bins: -160..20 dB step 1
    tokens.update(mag_token(v) for v in range(MAG_MIN, MAG_MAX + 1, MAG_STEP))

    # phase bins: -180..180 deg step 2
    tokens.update(phase_token(v) for v in range(PH_MIN, PH_MAX + 1, PH_STEP))

    return Vocab(tokens)


# ----------------------------------------------------------------------
# Streaming JSONL dataset
#
# See the module docstring's STREAMING section for why this exists.
# build_index() makes exactly one pass over the file; JsonlDataset then
# reads individual records on demand by seeking to their byte offset --
# nothing beyond that one indexing pass ever holds the whole file's
# parsed content in memory at once.
# ----------------------------------------------------------------------
def build_index(path: str):
    """One pass over the JSONL dataset: record each line's byte offset
    (for O(1) random-access reads later, via JsonlDataset) and the
    dataset's max component count (needed to size the padded sequence
    length up front -- see seq_len_full in main()). Every record is
    parsed once here to read `len(record["components"])`, then
    immediately discarded -- this function never accumulates parsed
    records, only two small integers and the first record (used to
    derive the fixed response-prompt length).

    This is the only full-file pass the whole training run ever makes;
    everything after this reads only the handful of records actually
    needed for one epoch's batch (or the validation sample).
    """
    offsets = []
    max_components = 0
    first_record = None
    with open(path, "rb") as fh:
        offset = fh.tell()
        for raw in fh:
            if raw.strip():
                offsets.append(offset)
                record = json.loads(raw)
                n_comp = len(record["components"])
                if n_comp > max_components:
                    max_components = n_comp
                if first_record is None:
                    first_record = record
            offset = fh.tell()
    return offsets, max_components, first_record


class JsonlDataset:
    """Lazy, random-access view over a JSONL file, backed by the byte
    offsets build_index() computed. __getitem__ seeks to a record's
    offset and parses just that one line -- records are never cached
    here, so holding a JsonlDataset in memory costs only the offset list
    (~8 bytes/record) regardless of dataset size.

    Keeps one open file handle per process, opened lazily on first
    access. __getstate__ drops that handle before pickling (e.g. if this
    object is ever sent across a process boundary), so a fresh handle
    gets opened in whichever process actually reads from it next.
    """

    def __init__(self, path: str, offsets: list):
        self.path = path
        self.offsets = offsets
        self._fh = None

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def _handle(self):
        if self._fh is None:
            self._fh = open(self.path, "r")
        return self._fh

    def __getitem__(self, idx: int) -> dict:
        fh = self._handle()
        fh.seek(self.offsets[idx])
        return json.loads(fh.readline())

    def get_many(self, indices) -> list:
        return [self[i] for i in indices]


# ----------------------------------------------------------------------
# Mixed-precision (AMP) support
#
# "16-bit" here means autocast MIXED precision, not weight quantization:
# model parameters and the optimizer's running stats stay fp32 (the
# "master weights"); autocast just runs the matmul-heavy ops inside each
# forward pass (the attention projections, the FFN) in fp16/bf16 instead,
# which is where GPU tensor cores actually give a real speed and memory
# win. This is NOT the same thing as int8/post-training quantization
# (which shrinks a model for inference deployment, after training is
# done) -- there's no equivalent "quantize during training" technique
# that helps here; autocast is the actual lever for training speed.
#
# fp16 has a narrow exponent range and can under/overflow mid-training,
# so it needs a GradScaler (scales the loss up before backward, then
# unscales gradients before the optimizer step) to stay numerically
# stable. bf16 has the SAME exponent range as fp32 (just less mantissa
# precision), so it never needs a scaler and is the safer default on
# hardware that supports it (NVIDIA Ampere/30xx/40xx/A100 or newer) --
# fp16 usually edges out bf16 on raw throughput on GPUs that support
# both, at the cost of that extra numerical-stability risk.
# ----------------------------------------------------------------------
def _autocast_ctx(device: str, amp_enabled: bool, amp_dtype):
    device_type = "cuda" if device == "cuda" else "cpu"
    return torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled)


def _evaluate_batched(model, input_all, target_all, criterion, ignore_index, batch_size,
                       device, amp_enabled, amp_dtype):
    """Teacher-forced loss/accuracy over a (possibly huge) held-out set, in
    chunks -- NOT one forward call over the whole thing at once (a single
    unbatched forward pass over a large validation split is what causes
    a (N, n_heads, seq_len, seq_len) attention-score OOM)."""
    n = input_all.shape[0]
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_input = input_all[start:start + batch_size]
            batch_target = target_all[start:start + batch_size]

            with _autocast_ctx(device, amp_enabled, amp_dtype):
                logits = model(batch_input, use_causal_mask=True)
                loss = criterion(logits.view(-1, logits.size(-1)), batch_target.view(-1))

            preds = logits.argmax(dim=-1)
            mask = batch_target != ignore_index
            total_loss += loss.item() * batch_input.shape[0]
            total_correct += (preds[mask] == batch_target[mask]).sum().item()
            total_count += mask.sum().item()

    return total_loss / n, total_correct / max(total_count, 1)


def _parse_netlist_tokens(tokens):
    """Parse a flat, generated netlist-description token list of the form
    IN Na GND Nb OUT Nc (TYPE Na Nb VAL mantissa decade)*  into a
    structured dict, or return None if it doesn't match that shape
    (a real risk here, since the model is generating these tokens
    freely -- unlike the forward direction, nothing guarantees a
    well-formed sequence)."""
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
    (type, unordered node pair) -- components are undirected, and order
    in the token stream carries no meaning. Returns (n_matched,
    value_rel_errors) where value_rel_errors covers only matched pairs
    whose generated value token also decoded successfully."""
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


def _generate_and_compare(model, vocab, record, device, amp_enabled, amp_dtype):
    """Autoregressively generate a topology (netlist) from its response
    alone (no teacher forcing) and return (parsed_netlist, node_match,
    n_matched, n_true, value_rel_errors) for comparison against ground
    truth."""
    prompt_tokens = ["[BOS]"] + response_tokens(record["response"]) + ["[SEP]"]
    prompt_ids = torch.tensor(
        vocab.encode(prompt_tokens), dtype=torch.long, device=device
    ).unsqueeze(0)

    true_components = record["components"]
    # exact target length for THIS record is 1 [TOPO] + 7 header +
    # 6*n_components + 1 [EOS] -- n_components now varies per topology
    # (1-6 cascaded filter stages), so this is computed per-record, not
    # a fixed constant; a small buffer covers a model that doesn't emit
    # [EOS] cleanly.
    max_new_tokens = 1 + 7 + 6 * len(true_components) + 1 + 6

    generated = prompt_ids
    with torch.no_grad(), _autocast_ctx(device, amp_enabled, amp_dtype):
        for _ in range(max_new_tokens):
            logits = model(generated, use_causal_mask=True)
            next_id = int(logits[0, -1].argmax())
            generated = torch.cat([generated, torch.tensor([[next_id]], device=device)], dim=1)
            if vocab.itos[next_id] == "[EOS]":
                break

    gen_tokens = vocab.decode(generated[0].tolist())
    gen_target = gen_tokens[len(prompt_tokens):]  # drop the response prompt
    if gen_target and gen_target[0] == "[TOPO]":
        gen_target = gen_target[1:]
    if gen_target and gen_target[-1] == "[EOS]":
        gen_target = gen_target[:-1]

    parsed = _parse_netlist_tokens(gen_target)
    if parsed is None:
        return None, False, 0, len(true_components), []

    node_match = (parsed["in_node"] == f"N{record['in_node']}"
                  and parsed["gnd_node"] == f"N{record['gnd_node']}"
                  and parsed["out_node"] == f"N{record['out_node']}")
    n_matched, value_rel_errors = _match_components(true_components, parsed["components"])

    return parsed, node_match, n_matched, len(true_components), value_rel_errors


def _lr_for_step(base_lr, step, total_steps, warmup_steps, schedule):
    if schedule == "constant":
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    min_lr = base_lr * 0.1
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _tokenize_and_pad(records, vocab, max_len):
    """Encode each record's (variable-length) inverse sequence and
    right-pad with [PAD] up to `max_len`. Tokenizes ON THE FLY from the
    raw record dicts (build_inverse_sequence is cheap pure-Python string
    work) rather than from a precomputed cache -- with the streaming
    dataset, `records` is always just the handful actually needed right
    now (one epoch's --max-samples draw, or the validation sample), so
    there's nothing worth caching across calls. Returns (ids_cpu,
    lengths_cpu) -- ids_cpu: (N, max_len) long tensor; lengths_cpu: (N,)
    long tensor of each record's REAL (unpadded) token count, needed
    downstream to mask out the padding positions from the loss."""
    pad_id = vocab.stoi["[PAD]"]
    ids = torch.full((len(records), max_len), pad_id, dtype=torch.long)
    lengths = torch.empty(len(records), dtype=torch.long)
    for i, r in enumerate(records):
        toks, _ = build_inverse_sequence(r)
        enc = vocab.encode(toks)
        L = len(enc)
        if L > max_len:
            raise ValueError(
                f"topology_id {r.get('topology_id')} tokenizes to {L} tokens, which "
                f"exceeds the padded sequence length max_len={max_len} (derived from "
                f"MAX_COMPONENTS={MAX_COMPONENTS}). The dataset has a topology with more "
                f"components than that -- raise MAX_COMPONENTS in train.py to match "
                f"the generator that produced this file and re-run."
            )
        ids[i, :L] = torch.tensor(enc, dtype=torch.long)
        lengths[i] = L
    return ids, lengths


def _apply_masks(target_ids: torch.Tensor, lengths: torch.Tensor, loss_start: int,
                  ignore_index: int) -> None:
    """Mask target_ids IN PLACE at (a) the prompt region (columns before
    loss_start -- the response prompt the model conditions on, never
    predicted) and (b) any padding beyond each row's real sequence
    length (variable per row now that netlist length varies with
    component count). Positions the model is actually supervised on are
    exactly [loss_start, length-2] per row."""
    target_ids[:, :loss_start] = ignore_index
    T = target_ids.shape[1]
    positions = torch.arange(T, device=target_ids.device).unsqueeze(0)      # (1, T)
    lengths_dev = lengths.to(target_ids.device).unsqueeze(1)                # (N, 1)
    pad_mask = positions >= (lengths_dev - 1)                               # True = padding, mask it
    target_ids[pad_mask] = ignore_index


def _encode_records(records, vocab, loss_start, ignore_index, device, max_len):
    ids_cpu, lengths_cpu = _tokenize_and_pad(records, vocab, max_len)
    ids = ids_cpu.to(device)
    input_ids = ids[:, :-1]
    target_ids = ids[:, 1:].clone()
    _apply_masks(target_ids, lengths_cpu, loss_start, ignore_index)
    return input_ids, target_ids


def _fill_encoded_buffers(records, vocab, loss_start, ignore_index,
                           out_input: torch.Tensor, out_target: torch.Tensor,
                           max_len: int) -> None:
    """Encode `records` (padded to max_len) and copy the result INTO
    pre-allocated GPU buffers (in place) instead of returning new
    tensors -- keeps exactly one allocation alive for the whole run
    regardless of how many epochs re-sample fresh data (avoids CUDA
    allocator fragmentation from a malloc/free cycle every epoch)."""
    ids_cpu, lengths_cpu = _tokenize_and_pad(records, vocab, max_len)
    out_input.copy_(ids_cpu[:, :-1], non_blocking=True)
    target_tmp = ids_cpu[:, 1:].clone()
    _apply_masks(target_tmp, lengths_cpu, loss_start, ignore_index)
    out_target.copy_(target_tmp, non_blocking=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rlc_training_dataset.jsonl")
    parser.add_argument("--val-fraction", type=float, default=0.0001,
                         help="fraction of topologies held out for validation "
                              "(default: 0.10 -- 10%%; rounded, minimum 1 example)")
    parser.add_argument("--val-samples", type=int, default=100,
                         help="cap on held-out topologies actually used for validation "
                              "(default: 100). --val-fraction picks the held-out pool "
                              "(e.g. 10%% of a 5,000,000-row dataset = 500,000 topologies); "
                              "this subsamples THAT pool down to --val-samples for the "
                              "per-epoch teacher-forced loss/accuracy. Pass 0 to validate on "
                              "the full held-out pool instead (reads/tokenizes it from disk "
                              "every epoch -- not recommended at multi-million-row scale).")
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine",
                         help="'cosine' (default) warms up then decays to 10%% of --lr over "
                              "training -- recommended once each epoch trains on fresh data. "
                              "'constant' keeps --lr fixed throughout.")
    parser.add_argument("--batch-size", type=int, default=32,
                         help="mini-batch size for training (default: 32; automatically "
                              "capped to however many training examples exist)")
    parser.add_argument("--seed", type=int, default=0,
                         help="random seed for the train/val split and mini-batch shuffling")
    parser.add_argument("--max-samples", type=int, default=15_000,
                         help="cap on training topologies tensorized onto the GPU PER EPOCH "
                              "(default: 15,000). A fresh random subset of this size is drawn "
                              "from the training pool -- and read off disk via the streaming "
                              "dataset -- every epoch. Pass 0 to use the whole training pool "
                              "every epoch instead (only safe if it fits in VRAM and you're "
                              "fine re-reading/re-tokenizing it from disk every epoch -- at "
                              "multi-million-row scale, keep this capped).")
    parser.add_argument("--d-model", type=int, default=224, help="model embedding dimension")
    parser.add_argument("--d-ff", type=int, default=896, help="feed-forward hidden dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("--n-layers", type=int, default=10, help="number of transformer layers")
    parser.add_argument("--grad-checkpoint", action="store_true",
                         help="trade ~20-30%% more compute for a large cut in activation "
                              "memory. Try this before shrinking --batch-size or --n-layers.")
    parser.add_argument("--cpu-offload", action="store_true",
                         help="move the model to CPU RAM between epochs and back to GPU only "
                              "for that epoch's training + validation.")
    parser.add_argument("--val-gen-samples", type=int, default=50,
                         help="how many held-out topologies to run the (slow, sequential, "
                              "one-token-at-a-time) free-running generation comparison on "
                              "after training, sampled randomly from the validation set "
                              "(default: 50).")
    parser.add_argument("--save-path", default="inverse_model.pt",
                         help="where to save the trained model checkpoint once training "
                              "finishes (default: inverse_model.pt). Contains the model "
                              "weights, vocabulary, and config needed to reload the model "
                              "for inference. Pass an empty string to skip saving.")
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="none",
                         help="mixed-precision training (default: none -- full fp32, unchanged "
                              "from before). 'bf16' is the safer choice on Ampere-or-newer GPUs "
                              "(RTX 30xx/40xx, A100+) -- same exponent range as fp32, no "
                              "numerical-stability tricks needed. 'fp16' can be faster still on "
                              "hardware that supports both, at the cost of needing a loss-scaler "
                              "(handled automatically here) to avoid over/underflow. This is "
                              "mixed-precision AUTOCAST, not weight quantization -- parameters "
                              "stay fp32, only the matmul-heavy ops inside each forward pass run "
                              "in 16-bit. Only helps on CUDA; on CPU this flag is accepted but "
                              "won't give a real speedup.")
    args = parser.parse_args()
    random.seed(args.seed)

    # ---- Streaming dataset: one pass to index byte offsets + measure the
    # dataset's actual max component count (used below to size the padded
    # sequence length exactly, without needing any record's content kept
    # around afterward). See build_index / JsonlDataset docstrings. ----
    print(f"Indexing {args.dataset} (one pass -- byte offsets + max component count only; "
          f"nothing else is kept resident afterward)...")
    offsets, max_components_seen, first_record = build_index(args.dataset)
    n_topologies = len(offsets)
    if n_topologies < 2:
        raise SystemExit("Need at least 2 topologies (>=1 train, 1 val).")
    if max_components_seen > MAX_COMPONENTS:
        raise SystemExit(
            f"Dataset contains a topology with {max_components_seen} components, which "
            f"exceeds MAX_COMPONENTS={MAX_COMPONENTS}. Raise MAX_COMPONENTS in train.py "
            f"to {max_components_seen} (or higher) and re-run."
        )
    dataset = JsonlDataset(args.dataset, offsets)

    # ---- split off 10% (by count, rounded, min 1) for validation, ONCE,
    # by shuffling INDICES (not content) -- nothing is read off disk here
    # except the (small, capped) validation sample below. Everything else
    # is the training pool, which gets a fresh --max-samples-sized subset
    # read off disk every epoch (see the training loop). ----
    all_indices = list(range(n_topologies))
    random.shuffle(all_indices)
    n_val = max(1, round(n_topologies * args.val_fraction))
    n_val = min(n_val, n_topologies - 1)  # keep at least 1 training example
    val_idx_pool = all_indices[:n_val]
    train_idx_pool = all_indices[n_val:]

    if args.val_samples and len(val_idx_pool) > args.val_samples:
        val_sample_idx = random.sample(val_idx_pool, args.val_samples)
    else:
        val_sample_idx = val_idx_pool
    val_records = dataset.get_many(val_sample_idx)
    val_ids = sorted(r["topology_id"] for r in val_records)

    epoch_size = len(train_idx_pool) if not args.max_samples else min(args.max_samples, len(train_idx_pool))
    print(f"Task: INVERSE (response -> topology)")
    print(f"Dataset: {n_topologies} topologies -> {len(train_idx_pool)} in the training pool / "
          f"{len(val_idx_pool)} held out ({args.val_fraction*100:.0f}%), subsampled to "
          f"{len(val_records)} used for validation")
    if epoch_size < len(train_idx_pool):
        print(f"Training pool exceeds --max-samples: drawing a FRESH random {epoch_size:,} "
              f"of {len(train_idx_pool):,} every epoch (over {args.epochs} epochs, that's up to "
              f"{epoch_size * args.epochs:,} draws -- more than the pool size once epochs * "
              f"epoch_size passes {len(train_idx_pool):,}, meaning the run sweeps the pool "
              f"multiple times over).")
    print(f"Held-out validation topology_ids (sample): {val_ids[:20]}"
          f"{' ...' if len(val_ids) > 20 else ''}\n")

    # ---- sequence length: computed ANALYTICALLY from MAX_COMPONENTS,
    # not by scanning/tokenizing the whole file. build_inverse_sequence's
    # target segment is always exactly 8 + 6*n_components tokens (1
    # [TOPO] + 6 header (IN/GND/OUT node tokens) + 6*n_components
    # component tokens + 1 [EOS]) -- see build_inverse_sequence's
    # docstring -- so the true worst case is prompt_len + 8 +
    # 6*max_components, reached at max_components_seen (which build_index
    # already confirmed does not exceed MAX_COMPONENTS). ----
    prompt_len = build_inverse_sequence(first_record)[1]  # identical across all examples (fixed response grid)
    seq_len_full = prompt_len + 8 + 6 * MAX_COMPONENTS
    print(f"Sequence length: up to {seq_len_full} tokens (varies with component count, "
          f"2-{MAX_COMPONENTS} components; padded to {seq_len_full} for batching)   "
          f"Response prompt length: {prompt_len} (fixed)")

    vocab = build_deterministic_vocab(MAX_COMPONENTS)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_start = prompt_len - 1
    IGNORE_INDEX = -100

    amp_enabled = args.amp != "none"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp, torch.float32)
    if amp_enabled and device != "cuda":
        print(f"Note: --amp {args.amp} requested but no CUDA device is available -- autocast "
              f"will still run on CPU but won't give the speedup this flag is for (that comes "
              f"from GPU tensor cores).")
    # GradScaler is a no-op when enabled=False, so it's always safe to run
    # every training step through it -- only fp16 on CUDA actually needs
    # the loss-scaling behavior (bf16 has fp32's exponent range and never
    # over/underflows the way fp16 can).
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and args.amp == "fp16" and device == "cuda"))

    # ---- held-out validation batch: tokenized/tensorized once, reused
    # unchanged every epoch (never resampled, never trained on). ----
    val_input_all, val_target_all = _encode_records(
        val_records, vocab, loss_start, IGNORE_INDEX, device, seq_len_full
    )

    model = TenLayerTransformer(
        vocab_size=len(vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        max_seq_len=seq_len_full + 8,
        grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    batches_per_epoch = math.ceil(epoch_size / min(args.batch_size, epoch_size))
    total_steps = batches_per_epoch * args.epochs
    warmup_steps = max(1, int(0.05 * total_steps))  # 5% warmup
    global_step = 0

    print(f"Model: {n_params:,} parameters (d_model={args.d_model}, d_ff={args.d_ff}, "
          f"n_heads={args.n_heads}, n_layers={args.n_layers}, "
          f"grad_checkpoint={args.grad_checkpoint})")
    print(f"Precision: {args.amp}"
          + (f" (autocast dtype={amp_dtype}, GradScaler {'active' if scaler.is_enabled() else 'inactive'})"
             if amp_enabled else ""))
    print(f"Vocab size: {len(vocab)}   Padded sequence length: {seq_len_full}   "
          f"Model max_seq_len: {seq_len_full + 8}")
    print(f"Training examples/epoch: {epoch_size}   Batch size: {min(args.batch_size, epoch_size)}   "
          f"Batches/epoch: {batches_per_epoch}   LR schedule: {args.lr_schedule}")
    total_topos = epoch_size * args.epochs
    total_tokens = total_topos * (seq_len_full - 1)
    print(f"Total over the run: {total_topos:,} training topologies "
          f"({args.epochs} epochs x {epoch_size:,}/epoch)   "
          f"~{total_tokens:,} training tokens ({total_topos:,} x {seq_len_full - 1} tokens/example)")
    if device == "cuda":
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"CUDA memory after model init: {alloc:.2f} GB allocated, "
              f"{reserved:.2f} GB reserved\n")
    else:
        print()

    # ---- training buffers: allocated ONCE, at the fixed per-epoch size
    # (epoch_size rows x seq_len_full-1 columns), and overwritten in
    # place every epoch (see _fill_encoded_buffers). ----
    train_input_all = torch.empty((epoch_size, seq_len_full - 1), dtype=torch.long, device=device)
    train_target_all = torch.empty_like(train_input_all)
    n_train = epoch_size
    batch_size = min(args.batch_size, n_train)

    def _move_optimizer_state(target_device):
        """Adam's per-parameter state (exp_avg, exp_avg_sq) doesn't follow
        model.to(device) automatically."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(target_device)

    for epoch in range(1, args.epochs + 1):
        if args.cpu_offload:
            model.to(device)
            _move_optimizer_state(device)

        epoch_idx = (train_idx_pool if epoch_size >= len(train_idx_pool)
                     else random.sample(train_idx_pool, epoch_size))
        epoch_records = dataset.get_many(epoch_idx)
        _fill_encoded_buffers(
            epoch_records, vocab, loss_start, IGNORE_INDEX,
            train_input_all, train_target_all, seq_len_full,
        )

        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss_sum, epoch_correct, epoch_total = 0.0, 0, 0

        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            batch_input = train_input_all[idx]
            batch_target = train_target_all[idx]

            lr_now = _lr_for_step(args.lr, global_step, total_steps, warmup_steps, args.lr_schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr_now

            optimizer.zero_grad()
            with _autocast_ctx(device, amp_enabled, amp_dtype):
                logits = model(batch_input, use_causal_mask=True)  # (b, T, V)
                loss = criterion(logits.view(-1, logits.size(-1)), batch_target.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            epoch_loss_sum += loss.item() * idx.numel()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = batch_target != IGNORE_INDEX
                epoch_correct += (preds[mask] == batch_target[mask]).sum().item()
                epoch_total += mask.sum().item()

        train_loss = epoch_loss_sum / n_train
        train_acc = epoch_correct / max(epoch_total, 1)

        model.eval()
        val_loss, val_acc = _evaluate_batched(
            model, val_input_all, val_target_all, criterion, IGNORE_INDEX, batch_size,
            device, amp_enabled, amp_dtype,
        )

        mem_str = ""
        if device == "cuda":
            alloc = torch.cuda.memory_allocated() / 1e9
            peak = torch.cuda.max_memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            mem_str = f"   |   cuda alloc {alloc:.2f}GB  peak {peak:.2f}GB  reserved {reserved:.2f}GB"

        print(f"epoch {epoch:5d}   lr {lr_now:.2e}   train loss {train_loss:.4f}  acc {train_acc*100:6.2f}%   "
              f"|   val (teacher-forced) loss {val_loss:.4f}  acc {val_acc*100:6.2f}%{mem_str}")

        if device == "cuda":
            torch.cuda.empty_cache()

        if args.cpu_offload:
            model.to("cpu")
            _move_optimizer_state("cpu")
            if device == "cuda":
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Real test: autoregressively GENERATE each held-out topology's
    # netlist from its response alone (no teacher forcing) and compare
    # to the ground-truth topology. Teacher-forced val accuracy above
    # can look good even if free-running generation drifts or produces
    # a malformed / structurally different (but response-similar)
    # netlist -- this is the actual generalization check, and the one
    # that matters most given the one-to-many nature of this direction.
    # ------------------------------------------------------------------
    if args.cpu_offload:
        model.to(device)
    model.eval()

    gen_sample = (val_records if len(val_records) <= args.val_gen_samples
                  else random.sample(val_records, args.val_gen_samples))
    print(f"\n=== Held-out topologies: free-running generation vs. ground truth "
          f"({len(gen_sample)} of {len(val_records)} sampled) ===")

    n_malformed = 0
    n_node_match = 0
    total_matched, total_true = 0, 0
    all_value_rel_err = []

    for val_record in gen_sample:
        parsed, node_match, n_matched, n_true, value_rel_errors = _generate_and_compare(
            model, vocab, val_record, device, amp_enabled, amp_dtype
        )

        if parsed is None:
            n_malformed += 1
            print(f"topology {val_record['topology_id']:>4}: model did not generate a "
                  f"well-formed netlist sequence.")
            continue

        n_node_match += int(node_match)
        total_matched += n_matched
        total_true += n_true
        all_value_rel_err += value_rel_errors

        val_err_str = (f"value err mean {100*sum(value_rel_errors)/len(value_rel_errors):5.1f}%"
                        if value_rel_errors else "value err   n/a")
        print(f"topology {val_record['topology_id']:>4}: "
              f"IN/GND/OUT match {'yes' if node_match else 'no ':<3}   "
              f"components matched {n_matched}/{n_true}   {val_err_str}")

    n_scored = len(gen_sample) - n_malformed
    print(f"\nOverall across {len(gen_sample)} sampled held-out topologies "
          f"({n_malformed} malformed generations excluded from the rates below):")
    if n_scored:
        print(f"  IN/GND/OUT node-role match: {100*n_node_match/n_scored:.1f}%")
    if total_true:
        print(f"  Component match rate (type + node pair): "
              f"{total_matched}/{total_true} ({100*total_matched/total_true:.1f}%)")
    if all_value_rel_err:
        print(f"  Value error on matched components: mean "
              f"{100*sum(all_value_rel_err)/len(all_value_rel_err):.1f}%, "
              f"max {100*max(all_value_rel_err):.1f}%")

    # ------------------------------------------------------------------
    # Save the trained model: weights + vocab + config, enough to
    # reconstruct the model and run inference later (e.g. from a
    # generate/verify design-search script) without re-running training.
    # ------------------------------------------------------------------
    if args.save_path:
        if args.cpu_offload:
            model.to("cpu")
        checkpoint = {
            "task": "inverse",  # response -> topology
            "model_state_dict": model.state_dict(),
            "vocab_itos": vocab.itos,
            "config": {
                "vocab_size": len(vocab),
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "d_ff": args.d_ff,
                "n_layers": args.n_layers,
                "max_seq_len": seq_len_full + 8,
            },
            "prompt_len": prompt_len,
            "seq_len_full": seq_len_full,
            "max_components": MAX_COMPONENTS,
        }
        torch.save(checkpoint, args.save_path)
        print(f"\nSaved trained model -> {args.save_path}")


if __name__ == "__main__":
    main()