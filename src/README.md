# LLM Architecture — source

`index.html` (one folder up) is generated from these files. Edit here, then rebuild:

```bash
python src/build.py
```

The output is a single self-contained HTML file: no network access, no dependencies. Python 3 is only needed to build.

The original inputs are kept next to the app: `../assets/llm-arch-3d.png` (shown by the overview page's "original PNG" toggle) and the SVG atlas in `../reference/`.

## Layout

```
src/
  index.template.html     page skeleton; /*__CSS__*/ and /*__JS__*/ are filled in by the build
  build.py                concatenates css + js in a fixed order (the order is listed in the file)
  css/
    01-base.css           layout and components of the concept views
    02-tensor-views.css   step cards, tensor graphs, popup / menu / detail panel
    03-theme.css          black / white theme, purple glow borders, tree, lens, explanations (loaded last, wins)
  js/
    01-core.js            helpers, palette, 3-D tensor blocks, Graph (concept diagrams), Stepper, inspector
    02-state.js           model configuration S, presets, config bar, router
    03-tree.js            architecture tree, breadcrumb, ①②③ lens, lesson order, theme switch
    05-explain.js         WHY: why each transformation exists and what it does to the tensor
    06-formulas.js        FORMULA: the formulas shown in popups, panels and step cards
    10 … 20-view-*.js     concept views (overview, layer, full model, attention, decode, MoE, families, prefill)
    21-view-topics-data.js TOPICS: short concept pages (new architectures, fine-tuning, beyond text)
    07-plots.js           PLOT: small graphs shown with formulas (activations, norms, softmax, RoPE…)
    22-view-topics.js     topic page renderer (stepped diagram + explanation + widget)
    23-topic-diagrams.js  TDIAG: one interactive diagram per topic page
    24-topic-widgets.js   WIDGETS: one small interactive tool per topic page
    30-tensor-core.js     tensor config K (proxied onto S), to-scale drawing, step model, the 9 architectures, numeric demos
    31-tensor-shared.js   steps → graph conversion, shared toolbar (barHTML / bindBar)
    32-tensor-more-archs.js Qwen3.8-Flash-Next and LoRA step builders
    33-tensor-mla.js      multi-head latent attention (DeepSeek) step builder
    40-tensor-steps.js    "every operation, step by step" view + comparison table
    41-tensor-hgraph.js   tensor graph, horizontal (pan / zoom)
    42-tensor-vgraph.js   tensor graph, vertical (aligned compare, hover popup, tracing)
    49-tensor-register.js registers the graph view
    90-ui.js              hover popup, click panel and right-click menu for all concept diagrams
    99-init.js            start-up
```

## Scoping

Files `01`–`20`, `90` and `99` are plain globals in one `<script>`.
Files `30`–`49` are wrapped by the build in one closure, and each of `40`, `41`, `42` in a nested closure,
so every tensor view can use short local names (`renderMain`, `renderBar`, `REG`, …).
They talk to the rest of the page through `VIEWS.steps`, `VIEWS.graph` and `window.TFX`.

## Common changes

- **Add or reword an explanation:** `js/05-explain.js` (`WHY[key] = [why, what]`). Keys are the step keys used in `30-tensor-core.js`.
- **Add an architecture:** add an entry to `ARCH` in `30-tensor-core.js` (a `build(c)` that returns steps) and a node in `TREE` in `03-tree.js`.
- **Change colours:** every tensor is coloured by its role. The role palette is `ROLE` in `01-core.js`; `roleOf()` in `30-tensor-core.js` maps a tensor name to a role. Page colours are the variables at the top of `css/03-theme.css`.

## Added later

- `js/08-sources.js` — `REFS` (papers, arXiv ids checked against the arXiv listing), `SRC` (tree / architecture id → references). A Sources card is appended to every lesson by `afterRoute`; `#refs` lists everything.
- `js/34-tensor-extra.js` — more step-by-step architectures: ALiBi, adapters, prefix / prompt tuning, (IA)³, DoRA, QLoRA (decoder variants that insert steps into the GQA decoder) and Mamba, RWKV, ViT + projector, DiT, VQ tokenizer, masked-diffusion LM (their own sequences, `S.seq = true`).
  Steps there carry their own text: `why`, `what`, `fm`. `table:false` keeps an architecture out of the comparison table. `FIELD_USE` says which toolbar fields belong to which architecture.
- Tree nodes (`js/03-tree.js`): `ex` = a real model that uses the technique (shown on the overview tree); `x:true` = link-only leaf; group `k` = role colour of the branch.
- Overview: `overviewTree(two)` draws a real tree in HTML/CSS (`.rtree`), two-sided when it fits, otherwise one-sided.
