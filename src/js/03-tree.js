/* ================= architecture tree, lens, lesson order, theme ================= */
/* node: id, t = title, s = what it is, ex = a real model that uses it, r = route, a = architecture id for the step / graph lenses.
   x:true = shown in the tree as a link only (it points into another lesson and is not a lesson of its own). */
const TREE=[
  {t:'Start here',c:[{id:'overview',t:'Map of everything',r:'overview'}]},
  {t:'Building blocks',k:'x',c:[
    {id:'layer',t:'Transformer layer',s:'norm → attention → MLP, each added back to the stream',ex:'Llama-2',r:'layer',a:'mha'},
    {id:'model',t:'Full inference pipeline',s:'tokens → embedding → N layers → head → sampling',ex:'Llama-3',r:'model',a:'gqa'},
    {id:'residual4',t:'Multi-branch residual stream',s:'4 parallel streams with gated read / write',ex:'Qwen3.8-Flash-Next',r:'topic/residual4',a:'qnext'},
    {t:'Normalisation',c:[
      {t:'LayerNorm',s:'centre and scale each token',ex:'GPT-2 · BERT',r:'steps/gpt2',x:true},
      {t:'RMSNorm',s:'scale only, cheaper',ex:'Llama · Qwen · Mistral',r:'steps/gqa',x:true}]},
    {t:'Activation',c:[
      {t:'ReLU',s:'max(0, x)',ex:'original Transformer',r:'steps/encdec',x:true},
      {t:'GELU',s:'smooth ReLU',ex:'GPT-2 · BERT',r:'steps/gpt2',x:true},
      {t:'SiLU inside SwiGLU',s:'one branch gates the other',ex:'Llama · Qwen · Mistral',r:'steps/gqa',x:true}]}]},
  {t:'Position information',k:'tok',c:[
    {t:'Learned absolute',s:'a trained vector per position, added at the input',ex:'GPT-2 · BERT',r:'steps/gpt2',x:true},
    {t:'Sinusoidal',s:'fixed sin / cos waves added at the input',ex:'original Transformer',r:'steps/encdec',x:true},
    {t:'Rotary (RoPE)',s:'rotate Q and K in every layer',ex:'Llama-3 · Qwen2.5',r:'steps/gqa',x:true},
    {id:'alibi',t:'ALiBi',s:'distance penalty added to the scores',ex:'BLOOM',r:'steps/alibi',a:'alibi'}]},
  {t:'Token mixing',k:'q',c:[
    {t:'Full attention',c:[
      {id:'mha',t:'Multi-head attention (MHA)',s:'one K/V head per query head',ex:'GPT-2 · Llama-2',r:'attn/mha',a:'mha'},
      {id:'gqa',t:'Grouped-query attention (GQA)',s:'groups of query heads share a K/V head',ex:'Llama-3 · Qwen2.5',r:'attn/gqa',a:'gqa'},
      {id:'mqa',t:'Multi-query attention (MQA)',s:'a single K/V head for all queries',ex:'PaLM · Falcon',r:'attn/mqa',a:'mqa'}]},
    {t:'Compressed attention',c:[
      {id:'mla',t:'Multi-head latent attention (MLA)',s:'cache one small latent instead of K and V',ex:'DeepSeek-V2 · V3',r:'topic/mla',a:'mla'}]},
    {t:'Restricted attention',c:[
      {id:'swa',t:'Sliding-window attention',s:'look only at the last W tokens',ex:'Mistral-7B · Longformer',r:'attn/swa',a:'swa'},
      {id:'qsa',t:'Learned sparse attention',s:'pick the top-512 blocks, attend only there',ex:'Qwen3.8-Flash-Next',r:'topic/qsa',a:'qnext'}]},
    {t:'Recurrent · linear time',c:[
      {id:'gdn',t:'Gated DeltaNet',s:'fixed-size state edited by a delta rule',ex:'Qwen3.8-Flash-Next',r:'topic/gdn',a:'qnext'},
      {id:'ssm',t:'Selective state space',s:'input-dependent recurrence, no attention',ex:'Mamba',r:'topic/ssm',a:'mamba'},
      {id:'rwkv',t:'Time mixing (RNN)',s:'running weighted average with learned decay',ex:'RWKV-4',r:'topic/rwkv',a:'rwkv'}]}]},
  {t:'Per-token computation',k:'hid',c:[
    {id:'dense',t:'Dense MLP · classic',s:'2 matrices, GELU, LayerNorm',ex:'GPT-2',r:'families/dense',a:'gpt2'},
    {id:'modern',t:'Dense SwiGLU · modern',s:'3 matrices, gated, RMSNorm',ex:'Llama-3 · Qwen2.5',r:'families/modern',a:'gqa'},
    {id:'moe',t:'Mixture of Experts',s:'a router picks top-k expert MLPs per token',ex:'Mixtral 8x7B · DeepSeek-V3',r:'moe',a:'moe'}]},
  {t:'Memory beyond the weights',k:'k',c:[
    {id:'decode',t:'KV cache',s:'keep K and V of old tokens, compute one new row',ex:'every autoregressive LLM',r:'decode',a:'decode'},
    {id:'ngram',t:'N-gram lookup table',s:'hash the last 2–3 tokens into a huge table',ex:'Qwen3.8-Flash-Next',r:'topic/ngram',a:'qnext'}]},
  {t:'Model families',k:'attn',c:[
    {id:'enc',t:'Encoder-only',s:'bidirectional, outputs vectors not text',ex:'BERT',r:'families/enc',a:'bert'},
    {id:'encdec',t:'Encoder–decoder',s:'decoder cross-attends to an encoder',ex:'T5 · original Transformer',r:'families/encdec',a:'encdec'},
    {id:'dec',t:'Decoder-only',s:'causal, predicts the next token',ex:'GPT · Llama · Qwen',r:'families/dec',a:'gqa',c:[
      {id:'qnext',t:'Hybrid decoder',s:'recurrent layers + sparse attention + MoE',ex:'Qwen3.8-Flash-Next',r:'topic/qnext',a:'qnext'}]},
    {t:'Non-autoregressive',c:[
      {id:'difflm',t:'Diffusion language model',s:'unmask all positions over a few passes',ex:'LLaDA · Mercury',r:'topic/difflm',a:'mdlm'},
      {id:'jev',t:'Typed decision model',s:'one parallel pass, structured output',ex:'Jev',r:'topic/jev'}]}]},
  {t:'Inference behaviour',k:'s',c:[
    {id:'prefill',t:'Prefill vs decode',s:'same weights, [T × …] versus [1 × …]',ex:'every autoregressive LLM',r:'prefill',a:'decode'},
    {id:'mtp',t:'Multi-token prediction',s:'draft several tokens, verify in one pass',ex:'DeepSeek-V3 · Qwen3.8-Flash-Next',r:'topic/mtp',a:'qnext'}]},
  {t:'Fine-tuning & adaptation',k:'lora',c:[
    {id:'fullft',t:'Full fine-tuning',s:'update every weight',ex:'Llama-2-Chat',r:'topic/fullft',a:'gqa'},
    {t:'Low-rank updates',c:[
      {id:'lora',t:'LoRA',s:'train A·B next to a frozen matrix',ex:'Alpaca-LoRA',r:'topic/lora',a:'lora'},
      {id:'qlora',t:'QLoRA',s:'LoRA on a 4-bit base model',ex:'Guanaco',r:'topic/qlora',a:'qlora'},
      {id:'dora',t:'DoRA',s:'separate magnitude and direction',ex:'paper: LLaMA-7B · LLaVA-1.5',r:'topic/dora',a:'dora'}]},
    {t:'Added modules',c:[
      {id:'adapters',t:'Adapters',s:'small bottleneck MLP after each sub-layer',ex:'paper: BERT',r:'topic/adapters',a:'adapters'},
      {id:'ia3',t:'(IA)³',s:'learned vectors rescale K, V and the MLP',ex:'T-Few (on T0)',r:'topic/ia3',a:'ia3'}]},
    {t:'Learned prompts',c:[
      {id:'prefix',t:'Prefix tuning',s:'trainable rows in K and V of every layer',ex:'paper: GPT-2 · BART',r:'topic/prefix',a:'prefix'},
      {id:'prompt',t:'Prompt tuning',s:'trainable vectors before the input only',ex:'paper: T5',r:'steps/prompt',a:'prompt'}]}]},
  {t:'Beyond text',k:'v',c:[
    {id:'multimodal',t:'Vision encoder + projector',s:'image patches become tokens for the LLM',ex:'LLaVA-1.5',r:'topic/multimodal',a:'vit'},
    {id:'vqtok',t:'Discrete image / audio tokens',s:'nearest entry of a learned codebook',ex:'VQGAN · EnCodec',r:'topic/vqtok',a:'vq'},
    {id:'imgdiff',t:'Latent diffusion',s:'denoise a latent step by step (U-Net or DiT)',ex:'Stable Diffusion · DiT',r:'topic/imgdiff',a:'dit'},
    {id:'vaegan',t:'VAEs and GANs',s:'earlier generators; VAEs still make the latents',ex:'StyleGAN · Stable Diffusion VAE',r:'topic/vaegan'}]},
  {t:'Compare & explore',c:[
    {id:'steps',t:'Every operation, step by step',s:'any architecture, or two side by side',r:'steps'},
    {id:'graph',t:'Tensor graph',s:'all tensors connected',r:'graph'},
    {id:'refs',t:'Sources & papers',s:'every reference used here',r:'refs'}]}
];
const FLAT=[];
(function walk(ns,g,p){ns.forEach(n=>{if(n.r&&!n.x){n.g=g;n.p=p;FLAT.push(n);}if(n.c)walk(n.c,g||n.t,g?n:null);});})(TREE,'',null);
/* the same tree drawn as a real tree on the overview page: root in the middle, branches to both sides */
function overviewTree(two){
  const G=TREE.filter(g=>g.k),cnt=n=>n.c?n.c.reduce((a,c)=>a+cnt(c),n.r?1:0):1;
  const leaf=n=>`<a class="lf" href="#${n.r}"><b>${n.t}</b>${n.s?`<span>${n.s}</span>`:''}${n.ex?`<em title="A real model that uses it">${n.ex}</em>`:''}</a>`;
  const node=n=>`<div class="kid"><div class="tr">${n.r?leaf(n):`<span class="br">${n.t}</span>`}${n.c?`<div class="kids">${n.c.map(node).join('')}</div>`:''}</div></div>`;
  const side=(gs,cls)=>`<div class="kids ${cls}">${gs.map(g=>`<div class="kid" style="--c:${ROLE[g.k]}"><div class="tr"><span class="br top">${g.t}</span><div class="kids">${g.c.map(node).join('')}</div></div></div>`).join('')}</div>`;
  const root='<span class="rt">Generative AI<small>architectures &amp; techniques</small></span>';
  if(!two)return `<div class="rtree one"><div class="tr">${root}${side(G,'R')}</div></div>`;
  /* balance the two sides by number of leaves */
  const tot=G.reduce((a,g)=>a+cnt(g),0);let acc=0,cut=0;
  while(cut<G.length-1&&acc+cnt(G[cut])<=tot/2+2){acc+=cnt(G[cut]);cut++;}
  return `<div class="rtree two"><div class="tr">${side(G.slice(0,cut),'L')}${root}${side(G.slice(cut),'R')}</div></div>`;
}
let CURNODE=FLAT[0];
function resolveNode(){
  const h=location.hash.slice(1)||'overview',[v,sub]=h.split('/');
  if(v==='steps'||v==='graph'){if(sub&&CURNODE.a===sub&&CURNODE.id!=='steps'&&CURNODE.id!=='graph')return CURNODE;const m=sub?FLAT.filter(n=>n.a===sub):[];return FLAT.find(n=>n.r===h)||m.find(n=>n.id===sub||n.r==='steps/'+sub)||(m.length===1?m[0]:null)||FLAT.find(n=>n.id===v);}
  return FLAT.find(n=>n.r===h)||FLAT.find(n=>n.r.split('/')[0]===v)||FLAT[0];
}
function renderNav(){
  CURNODE=resolveNode();
  const item=n=>`<li>${n.r?`<a class="tn" href="#${n.r}"${n===CURNODE?' aria-current="page"':''}><span>${n.t}</span>${n.s?`<small>${n.s}</small>`:''}</a>`:`<span class="tg">${n.t}</span>`}${n.c?`<ul>${n.c.map(item).join('')}</ul>`:''}</li>`;
  const closed=new Set($$('#nav details:not([open])').map(d=>d.dataset.g));
  $('#nav').innerHTML=TREE.map(g=>`<details data-g="${g.t}"${closed.has(g.t)&&g.t!==CURNODE.g?'':' open'}><summary>${g.t}</summary><ul>${g.c.map(item).join('')}</ul></details>`).join('');
}
function afterRoute(id,sub,root){
  const n=CURNODE,view=id==='steps'?'steps':id==='graph'?'graph':'concept';
  $('#crumb').innerHTML=`${n.g}${n.p?' › '+n.p.t:''} › <b>${n.t}</b>${n.a&&view!=='concept'?' › '+(view==='steps'?'every operation':'tensor graph'):''}`;
  if(n.a)root.insertAdjacentHTML('afterbegin',`<div class="lens" role="tablist" aria-label="Ways to study this architecture"><span class="lt">${n.t}</span>${(/^steps\//.test(n.r)?[]:[['concept','① Concept','#'+n.r]]).concat([['steps','② Every operation','#steps/'+n.a],['graph','③ Tensor graph','#graph/'+n.a]]).map(([k,t,h])=>`<a role="tab" href="${h}" aria-selected="${k===view}" class="${k===view?'on':''}">${t}</a>`).join('')}</div>`);
  if(id!=='refs'&&id!=='overview')root.insertAdjacentHTML('beforeend',sourcesHTML([n.id,n.a,view!=='concept'?sub:null]));
  const i=FLAT.indexOf(n),pv=FLAT[i-1],nx=FLAT[i+1];
  root.insertAdjacentHTML('beforeend',`<nav class="pn" aria-label="Lesson order">${pv?`<a href="#${pv.r}">← ${pv.t}</a>`:'<span></span>'}${nx?`<a href="#${nx.r}">${nx.t} →</a>`:'<span></span>'}</nav>`);
  document.body.classList.remove('treeopen');
  document.documentElement.style.setProperty('--hdr',$('header.top').offsetHeight+'px');
}
const LT=()=>document.body.classList.contains('light');
(function(){
  let th='light';try{th=localStorage.getItem('llm-theme')||'light';}catch(e){}
  const apply=()=>{document.body.classList.toggle('light',th==='light');$('#themeBtn').textContent=th==='light'?'● Black theme':'○ White theme';};
  apply();
  $('#themeBtn').addEventListener('click',()=>{th=th==='light'?'dark':'light';try{localStorage.setItem('llm-theme',th);}catch(e){}apply();route();});
  $('#treeBtn').addEventListener('click',()=>document.body.classList.toggle('treeopen'));
})();
