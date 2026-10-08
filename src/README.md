# GenAI Tensor Tree — source

`index.html` (one folder up) is generated from these files. Edit here, then rebuild:

```bash
python src/build.py
```

The output is a single self-contained HTML file with no dependencies. Python 3 is only needed to build.

## Layout

```
src/
  index.template.html       page skeleton; /*__CSS__*/ and /*__JS__*/ are filled in by the build
  build.py                  concatenates css + js in a fixed order (listed in the file)
  css/
    01-base.css             layout and components of the concept views
    02-tensor-views.css     step cards, tensor graphs, popup / menu / detail panel
    03-theme.css            black / white theme, overview tree, explanations, sources (loaded last, wins)
  js/
    01-core.js              helpers, role palette, 3-D tensor blocks, concept-diagram Graph, Stepper, inspector
    02-state.js             model configuration, presets, config bar, router
    03-tree.js              the technique tree, overview tree, breadcrumb, lesson tabs, theme switch
    05-explain.js           WHY: why each step exists and what it does to the tensor
    06-formulas.js          FORMULA: formulas shown in popups, panels and step cards
    07-plots.js             PLOT: small graphs shown with formulas (activations, norms, softmax)
    08-sources.js           REFS and SRC: papers, and which ones back each lesson
    10 … 20-view-*.js       concept views (overview, layer, full model, attention, decode, MoE, families, prefill)
    21-view-topics-data.js  TOPICS: text of the topic pages
    22-view-topics.js       topic page renderer (diagram + explanation + widget)
    23-topic-diagrams.js    TDIAG: one interactive diagram per topic page
    24-topic-widgets.js     WIDGETS: one small interactive tool per topic page
    30-tensor-core.js       tensor drawing, the step model, the base architectures, numeric demos
    31-tensor-shared.js     steps → graph conversion, shared toolbar
    32-tensor-more-archs.js Qwen3.8-Flash-Next and LoRA
    33-tensor-mla.js        multi-head latent attention
    34-tensor-extra.js      ALiBi, adapters, prefix / prompt tuning, (IA)³, DoRA, QLoRA, Mamba, RWKV, ViT, DiT, VQ, diffusion LM
    40-tensor-steps.js      "every operation, step by step" view and comparison table
    41-tensor-hgraph.js     tensor graph, horizontal (pan / zoom)
    42-tensor-vgraph.js     tensor graph, vertical (aligned compare, hover popup, tracing)
    49-tensor-register.js   registers the graph view
    90-ui.js                hover popup, click panel and right-click menu for the concept diagrams
    99-init.js              start-up
```

Files `01`–`24`, `90` and `99` are plain globals. Files `30`–`49` are wrapped by the build in one closure, with `40`, `41` and `42` each in a nested closure.

## Common changes

- **Add a technique to the tree:** add a node to `TREE` in `js/03-tree.js`. `t` = title, `s` = one-line description, `ex` = a real model that uses it, `r` = route, `a` = architecture id for the step and graph tabs.
- **Add an architecture:** push an entry to `ARCH` with a `build(c)` that returns steps (see `js/34-tensor-extra.js`). A step can carry its own `why`, `what` and `fm` (formulas). Set `S.seq = true` when its steps do not line up with the standard decoder.
- **Reword an explanation or formula:** `js/05-explain.js` (`WHY[key] = [why, what]`) and `js/06-formulas.js` (`FORMULA[key]`), keyed by step key.
- **Add a source:** add the paper to `REFS` and list its key under the lesson id in `SRC`, both in `js/08-sources.js`.
- **Change colours:** tensors are coloured by role. `ROLE` in `js/01-core.js` is the palette; `roleOf()` in `js/30-tensor-core.js` maps a tensor name to a role. Page colours are the variables at the top of `css/03-theme.css`.
