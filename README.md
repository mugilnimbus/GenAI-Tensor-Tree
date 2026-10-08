# GenAI Tensor Tree

**Live site: https://mugilnimbus.github.io/GenAI-Tensor-Tree/**

An interactive, single-page atlas of LLM and generative-AI architectures: a tree of every technique with a real model
that uses it, to-scale tensor diagrams, every matrix operation step by step, tensor graphs, formulas and sources.

## What is inside

- **Overview tree** — every technique in one picture, each leaf naming a real model that uses it
- **Every operation, step by step** — 25 architectures with to-scale tensors, formulas, and why each step exists
- **Tensor graphs** — vertical and horizontal, with hover details, worked numeric examples and path tracing
- **Concept pages** — diagrams and interactive widgets for attention variants, MoE, state-space models, fine-tuning, diffusion and more
- **Sources** — the primary papers linked on every lesson

## Run it locally

Open `index.html` in a browser. It is fully self-contained: no server, no dependencies.

## Project layout

- `index.html` — the site (generated; do not edit by hand)
- `src/` — the sources; edit there and run `python src/build.py` (see `src/README.md`)
- `assets/` — the original hand-made diagram shown on the overview page
