# RLC_filter_generation

# RLC Filter Generation Model

This project develops and trains a generative transformer model to design electronic schematics for multi-stage passive RLC (resistor/inductor/capacitor) filter circuits containing between 3 and 12 discrete components. Given a target AC transfer function as a prompt (magnitude and phase response plotted against frequency), the model generates a set of tokens defining the netlist, component types, values, and wiring - whose transfer function approximates the target. This process represents a novel application of transformer models, which are capible of learning the *language* of electronic physics just as they learn any other language. 

The results yeild impressive accuracy for a relitivly small model of around 6 million parameters, evauated as a mediean error of 3.2dB and 2.1 degrees when frequency and phase response of the generated circuit are simulated and compared agasint the prompt from 1 Hz to 1GHz, sampled at 10 points per decade and across 1000 inferance circuits. In some cases, the model generates circuits whose transfer function is nearly identical to the target, but contains fewer components, indicating it has learned to optimize for the physiscs, not just replicate learned circuits.

Filter design is fundementally a one-to-many problem, and normally involves an iterative loop for complex, multi-order transfer functions: pick a topology, simulate it, look at the response, adjust a value, simulate again. The point of training a model in this context is to let it learn the general relationship between a response shape and the topology that produces it.

## Overview

- **Model:** 10-layer transformer, ~6.3M parameters (`d_model=224`, `n_heads=8`, `d_ff=896`)
- **Dataset:** 5,000,000 generated training topologies, each with a full 91-sample simulated AC response
- **Training:** 15,000 random topologies per epoch × 1,000 epochs = 15,000,000 topologies seen, ~5.3 billion training tokens total
- **Sequence length:** ~355 tokens (91-sample response prompt + variable-length netlist target)

The repo includes the circuit simulator/dataset generator, the model and its training loop, an evaluation process, a schematic renderer, and a tool for visualizing the model's learned token embeddings in UMAP.


## Training Data Generation (`topology_gen.py`)

A topology is a plain Python dictionary of nodes, including an `in_node`/`gnd_node`/`out_node`, a list of `components` (type, node pair, value), and `filter_stages` metadata.

Component values come from the E12 preferred-value series over realistic ranges (10 Ω – 1 MΩ, 1 µH – 100 mH, 1 pF – 100 µF). `random_filter_topology()` cascades 1–6 stages which may include lowpass, highpass, bandpass, or bandstop, each a standard 1st-order or 2nd-order section.

Every topology's transfer function is simulated after generation via Modified Nodal Analysis, creating circuit/response pairs representing this ground truth. The data generation is embarresingly parallel, and so 8 seperate threads run the generation concurrently.

An example of a tokenized circuit is shown below:

```
[BOS] IN N1 GND N0 OUT N2 R N1 N2 VAL 56.0 x1k L N1 N2 VAL 39.0 x1u C N2 N0 VAL 68.0 x1n [SEP]
```

This script builds the dataset `rlc_training_dataset.jsonl`

## Model (`model.py`)

A custom GPT-style decoder-only transformer including pre-norm residual blocks, multi-head causal self-attention + GELU MLP, with ten layers.

## Supervised pretraining (`train.py`)

The goal of designing a filter for a desired response is framed as autoregressive generation: a target response and its netlist share one sequence, with the response as the token prefix the netlist conditioned on it:

```
[BOS] F0 MAG_-3 PH_-12 ... [SEP] [TOPO] IN N1 GND N0 OUT N2 R N1 N2 VAL 56.0 x1k ... [EOS]
```

Loss is next-token cross-entropy, masked to the netlist portion only. It is worth being precise about what that rewards, many different netlists can produce the same response, so match rate and value error are a proxy for how well the model learns the grammar, not how well the filter matches the response. The reported accuracy during training will therefore be much lower than the actual accuracy measured in the evaluation, as discussed below.

Adam optimizer, a learning rate of `1e-4` on a cosine schedule, and a batch size of 32 were selected for training. Each epoch reports teacher-forced validation loss/accuracy; after training it free-runs on held-out targets and reports node-role match rate, component match rate, and value error. The checkpoint (weights + vocab + config) is written to `inverse_model.pt`.


## Evaluation (`test.py`)

The test script generates random target topologies, [DOES IT PULL FROM THE JSON?] simulatess their true responses, and has the model run inference to predict a netlist from the response alone. The result is scored in two ways: structurally (component/node match against ground truth) and electrically (re-simulating the *predicted* netlist and comparing its curve to the target — the metric that actually matters, since a netlist can look structurally fine and still fail this). Both write a self-contained HTML report with interactive response charts.


## Schematic rendering (`draw_schematic.py`)

This script contains libraries to render the ladder-topology dictionary as a standard circuit schematic, rendering resistors, capacitors, inductors, junction dots and ground symbols. Series/shunt spacing are handled automatically. 

## Embedding visualization (`umap_embedding_map.py`)

An interactive tool that loads a checkpoint's learned token embeddings (component types, nodes, value tokens, structural markers) and reduces dimensionality to 4D with UMAP, producing a rotatable 3D scatter plus a color axis. It is useful for visualizing how the model learned a coherent geometry over its vocabulary.

## Results

The results below show the model's per



