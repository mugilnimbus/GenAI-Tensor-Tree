"""Build index.html from the sources in this folder.

    python src/build.py

Everything is inlined into one self-contained HTML file (no network, no dependencies).
Modules are concatenated in the order listed below; see README.md for what each one does.
"""
import os

SRC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SRC, '..', 'index.html')

CSS = ['01-base.css', '02-tensor-views.css', '03-theme.css']

# Plain globals: shared helpers, state, navigation, explanations and the concept views.
GLOBAL = ['01-core', '02-state', '03-tree', '05-explain', '06-formulas', '07-plots', '08-sources',
          '10-view-overview', '11-view-layer-graph', '12-view-layer', '13-view-model-graph', '14-view-model',
          '15-view-attention', '16-view-decode', '17-view-moe', '18-view-families-data', '19-view-families', '20-view-prefill', '21-view-topics-data', '22-view-topics', '23-topic-diagrams', '24-topic-widgets']
# Tensor views live in one closure (SHARED), each view in its own nested closure so that
# they can all use short local names (renderMain, renderBar, REG, ...) without clashing.
TENSOR_SHARED = ['30-tensor-core', '31-tensor-shared', '32-tensor-more-archs', '33-tensor-mla', '34-tensor-extra']
TENSOR_VIEWS = ['40-tensor-steps', '41-tensor-hgraph', '42-tensor-vgraph']
TENSOR_TAIL = ['49-tensor-register']
# Runs last: interactions shared by every diagram, then start-up.
FINAL = ['90-ui', '99-init']


def read(*parts):
    with open(os.path.join(SRC, *parts), encoding='utf-8') as f:
        return f.read().rstrip('\n') + '\n'


def js(name):
    return '\n/* ---- %s.js ---- */\n' % name + read('js', name + '.js')


def build():
    tensor = ('\n(function(){\n' + ''.join(js(n) for n in TENSOR_SHARED)
              + ''.join('\n(function(){' + js(n) + '})();\n' for n in TENSOR_VIEWS)
              + ''.join(js(n) for n in TENSOR_TAIL) + '})();\n')
    script = "'use strict';\n" + ''.join(js(n) for n in GLOBAL) + tensor + ''.join(js(n) for n in FINAL)
    css = ''.join(read('css', n) for n in CSS)
    html = read('index.template.html').replace('/*__CSS__*/', css).replace('/*__JS__*/', script)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write(html)
    print('wrote %s (%d bytes)' % (os.path.normpath(OUT), len(html.encode('utf-8'))))


if __name__ == '__main__':
    build()
