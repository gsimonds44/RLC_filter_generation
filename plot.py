"""
Visualize what the trained model's token embedding table has learned about
the netlist vocabulary -- component types, node tokens, component values
(mantissa + decade), and structural markers -- by projecting each token's
learned embedding vector down to 2D with UMAP and plotting them as a map.

Nearby points = tokens the model treats as similar in its internal
representation. This is a real look at whether the model has organized,
say, all the "R" tokens near each other and separate from "L"/"C", whether
node tokens cluster by position, whether value tokens are arranged in any
sensible order by magnitude, etc. -- as opposed to a random scatter, which
would suggest the embedding table hasn't learned much structure at all.

By default, only the TOPOLOGY side of the vocabulary is plotted (component
types, nodes, values, structural markers) -- not the frequency-response
tokens (F#, MAG_#, PH_#), since those live in a very different part of
the input (the prompt) and mixing them in dilutes the "map of its
understanding of components/values" picture you actually asked for. Pass
--include-response to add them back in (colored as their own groups).

COLOR: each token is colored by which vocabulary category it belongs to,
using colors sampled evenly along a single perceptually-uniform colormap
(viridis) -- so the palette is a smooth, uniform gradient end-to-end
rather than an arbitrary/rainbow assignment, while still being
distinguishable category to category. Marker SHAPE is layered on top of
color for every category as a redundant identity cue, so categories stay
distinguishable even for colorblind viewers or in grayscale printouts.

Requires: torch (to load the checkpoint), umap-learn, matplotlib, numpy.
Install the one dependency this project doesn't already need:
    pip install umap-learn --break-system-packages

Usage:
    python umap_embedding_map.py --model inverse_model.pt
    python umap_embedding_map.py --model inverse_model.pt --include-response
    python umap_embedding_map.py --n-neighbors 10 --min-dist 0.05 --metric cosine
"""

import argparse
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----------------------------------------------------------------------
# Token -> category classification. Mirrors the vocabulary scheme built
# in train.py's build_deterministic_vocab() / topology_gen.py's token
# emitters -- component types, node tokens, value tokens (mantissa +
# decade), structural markers, and (optionally) the response-side bins.
# ----------------------------------------------------------------------
_STRUCTURAL = {"[PAD]", "[BOS]", "[SEP]", "[RESP]", "[TOPO]", "[EOS]", "IN", "GND", "OUT", "VAL"}
_NODE_RE = re.compile(r"^N\d+$")
_DECADE_RE = re.compile(r"^x1[a-zA-Z]*$")
_FREQ_RE = re.compile(r"^F\d+$")
_MAG_RE = re.compile(r"^MAG_-?\d+$")
_PH_RE = re.compile(r"^PH_-?\d+$")


def _is_float_token(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


def classify_token(tok):
    if tok in ("R", "L", "C"):
        return "component type"
    if _NODE_RE.match(tok):
        return "node"
    if _DECADE_RE.match(tok):
        return "value (decade prefix)"
    if _is_float_token(tok):
        return "value (mantissa)"
    if tok in _STRUCTURAL:
        return "structural marker"
    if _FREQ_RE.match(tok):
        return "response: freq index"
    if _MAG_RE.match(tok):
        return "response: magnitude bin"
    if _PH_RE.match(tok):
        return "response: phase bin"
    return "other"


# Fixed, stable draw order -- NOT re-sorted per run, so a category's color
# stays the same across different --include-response runs and the legend
# order is always the same top-to-bottom.
_CATEGORY_ORDER = [
    "component type",
    "node",
    "value (mantissa)",
    "value (decade prefix)",
    "structural marker",
    "response: freq index",
    "response: magnitude bin",
    "response: phase bin",
    "other",
]
_CATEGORY_MARKERS = ["o", "s", "^", "D", "P", "v", "X", "*", "h"]


def _category_colors(categories_present):
    """Sample `len(categories_present)` colors evenly along viridis (a
    perceptually-uniform colormap) in the fixed category order above, so
    the palette is always a smooth, consistent gradient regardless of
    which categories end up present in a given run."""
    ordered = [c for c in _CATEGORY_ORDER if c in categories_present]
    n = len(ordered)
    cmap = matplotlib.colormaps["viridis"]
    # Evenly spaced samples strictly inside [0, 1] so the ramp's darkest
    # and lightest extremes (which read poorly against a white page) are
    # avoided at the ends.
    positions = np.linspace(0.08, 0.92, n) if n > 1 else np.array([0.5])
    colors = {cat: cmap(p) for cat, p in zip(ordered, positions)}
    markers = {cat: _CATEGORY_MARKERS[i % len(_CATEGORY_MARKERS)] for i, cat in enumerate(ordered)}
    return colors, markers, ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="inverse_model.pt",
                         help="path to the trained checkpoint (default: inverse_model.pt)")
    parser.add_argument("--output", default="token_embedding_umap.png",
                         help="output image path (default: token_embedding_umap.png)")
    parser.add_argument("--include-response", action="store_true",
                         help="also plot the frequency-response tokens (F#, MAG_#, PH_#) -- "
                              "off by default so the plot focuses on the topology/netlist side "
                              "of the vocabulary")
    parser.add_argument("--n-neighbors", type=int, default=15,
                         help="UMAP n_neighbors -- lower values preserve local structure more, "
                              "higher values preserve global structure more (default: 15, but "
                              "the vocabulary here is small -- try 5-10 if points look too "
                              "smeared together)")
    parser.add_argument("--min-dist", type=float, default=0.1,
                         help="UMAP min_dist -- how tightly points are allowed to pack together "
                              "(default: 0.1; lower values give tighter, more cluster-y plots)")
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"],
                         help="UMAP distance metric on the raw embedding vectors (default: cosine, "
                              "the standard choice for comparing learned embedding directions)")
    parser.add_argument("--seed", type=int, default=42, help="UMAP random_state (default: 42)")
    parser.add_argument("--no-labels", action="store_true",
                         help="skip the per-point token-text labels (they're on by default when "
                              "there are <= 150 points, since that's the normal topology-only case)")
    parser.add_argument("--labels", action="store_true",
                         help="force per-point token-text labels on even with a large vocabulary "
                              "(e.g. with --include-response) -- can get crowded")
    args = parser.parse_args()

    try:
        import umap
    except ImportError:
        raise SystemExit(
            "This script needs umap-learn, which isn't installed.\n"
            "Install it with:\n\n    pip install umap-learn --break-system-packages\n"
        )

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    itos = checkpoint["vocab_itos"]

    state = checkpoint["model_state_dict"]
    emb_key = "token_emb.weight"
    if emb_key not in state:
        raise SystemExit(f"Couldn't find '{emb_key}' in the checkpoint's state dict -- "
                          f"available keys starting with 'token_emb': "
                          f"{[k for k in state if k.startswith('token_emb')]}")
    embeddings = state[emb_key].detach().cpu().numpy()  # (vocab_size, d_model)

    if embeddings.shape[0] != len(itos):
        raise SystemExit(f"Embedding table has {embeddings.shape[0]} rows but the checkpoint's "
                          f"vocab has {len(itos)} tokens -- checkpoint looks inconsistent.")

    categories = [classify_token(t) for t in itos]
    keep = [
        i for i, cat in enumerate(categories)
        if args.include_response or not cat.startswith("response:")
    ]
    # [PAD] is never a meaningfully-trained token (every real sequence
    # avoids attending across it via padding masking during training) --
    # excluding it keeps an untrained/near-random vector from distorting
    # the projection of everything else.
    keep = [i for i in keep if itos[i] != "[PAD]"]

    sub_tokens = [itos[i] for i in keep]
    sub_categories = [categories[i] for i in keep]
    sub_embeddings = embeddings[keep]

    print(f"Loaded {args.model}: vocab={len(itos)}, d_model={embeddings.shape[1]}")
    print(f"Plotting {len(sub_tokens)} tokens"
          f"{' (topology vocabulary only)' if not args.include_response else ' (including response bins)'}")
    from collections import Counter
    for cat, n in Counter(sub_categories).most_common():
        print(f"  {cat:28s} {n:4d}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(args.n_neighbors, max(2, len(sub_tokens) - 1)),
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    coords = reducer.fit_transform(sub_embeddings)  # (n_tokens, 2)

    categories_present = set(sub_categories)
    colors, markers, ordered_categories = _category_colors(categories_present)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for cat in ordered_categories:
        idxs = [i for i, c in enumerate(sub_categories) if c == cat]
        ax.scatter(
            coords[idxs, 0], coords[idxs, 1],
            s=60, c=[colors[cat]], marker=markers[cat],
            edgecolors="white", linewidths=0.6,
            label=f"{cat} ({len(idxs)})", zorder=3,
        )

    show_labels = args.labels or (not args.no_labels and len(sub_tokens) <= 150)
    if show_labels:
        for (x, y), tok in zip(coords, sub_tokens):
            ax.annotate(
                tok, (x, y), fontsize=6.5, color="#3a3a38",
                xytext=(3, 3), textcoords="offset points", zorder=4,
            )

    ax.set_title(
        "Token embedding map (UMAP projection)\n"
        f"{'topology + response' if args.include_response else 'topology vocabulary'} tokens, "
        f"n_neighbors={args.n_neighbors}, min_dist={args.min_dist}, metric={args.metric}",
        fontsize=12, color="#0b0b0b", pad=14,
    )
    ax.set_xlabel("UMAP-1", fontsize=10, color="#52514e")
    ax.set_ylabel("UMAP-2", fontsize=10, color="#52514e")
    ax.tick_params(colors="#8a8a86", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#d8d7d0")
    ax.grid(True, color="#e8e7e0", linewidth=0.6, zorder=0)

    legend = ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9,
        frameon=False, title="Token category", title_fontsize=10,
    )
    legend.get_title().set_color("#0b0b0b")
    for text in legend.get_texts():
        text.set_color("#3a3a38")

    fig.tight_layout()
    fig.savefig(args.output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()