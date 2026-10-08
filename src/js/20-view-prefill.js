/* ================= 7. Prefill vs Decode ================= */
const PF={hw:'h100',te:11,b:1,n:256,sel:null,hov:null};
const HW={a100:{n:'NVIDIA A100 80GB',f:312e12,bw:2.039e12,m:80},h100:{n:'NVIDIA H100 SXM',f:989e12,bw:3.35e12,m:80},h200:{n:'NVIDIA H200',f:989e12,bw:4.8e12,m:141},r4090:{n:'RTX 4090 (24 GB)',f:165e12,bw:1.008e12,m:24}};
function pfCalc(){
  const c=counts(),T=2**PF.te,B=PF.b,by=S.bytes,h=HW[PF.hw],d=S.d,N=S.N,V=S.V;
  const Pm=N*(c.attn+c.mlp),head=V*d,w=(Pm+head)*by,kvTok=2*N*kvH()*dHead()*by;
  const pre={fl:B*(2*Pm*T+N*2*T*T*d+2*head),by:w+B*T*kvTok};
  const dec={fl:B*(2*Pm+N*4*T*d+2*head),by:w+B*T*kvTok};
  [pre,dec].forEach(x=>{x.tc=x.fl/h.f;x.tm=x.by/h.bw;x.t=Math.max(x.tc,x.tm);x.i=x.fl/x.by;x.bound=x.tc>=x.tm?'compute-bound':'memory-bound';});
  return {c,T,B,h,w,kvTok,pre,dec,kv:B*T*kvTok,mem:c.total*by+B*T*kvTok,ridge:h.f/h.bw};
}
function slotRow(cx,y,n,newFrom){
  const sw=19,gap=3,x0=cx-n*(sw+gap)/2;let s='';
  for(let i=0;i<n;i++){const isNew=i>=newFrom;
    s+=`<g ${isNew?'class="pulse"':''}><rect x="${x0+i*(sw+gap)}" y="${y}" width="${sw}" height="14" rx="3" fill="${C.orange}" fill-opacity="${isNew?1:.5}"/><rect x="${x0+i*(sw+gap)}" y="${y+17}" width="${sw}" height="14" rx="3" fill="${C.blue}" fill-opacity="${isNew?1:.5}"/></g>`;}
  return s;
}
function pfPanel(kind){
  const {d,V,N}=S,pre=kind==='pre',hk=kvH(),dh=dHead(),g=new Graph(null,PF.sel||PF.hov),id=s=>kind+s;
  const col=pre?C.green:C.blue;
  g.node({id:id('x'),x:280,y:84,dims:[pre?S.T:1,d],col,label:pre?'Prompt tokens X':'One decode token',shape:pre?`[T, ${d}]`:`[1, ${d}]`});
  const lay=`<rect x="110" y="178" width="340" height="104" rx="14" fill="rgba(214,95,72,.07)" stroke="${C.orange}" stroke-width="1.8" stroke-dasharray="7 5"/><text class="lab" x="280" y="204" text-anchor="middle">All ${N} layers</text><text class="shp" x="280" y="222" text-anchor="middle" style="fill:#f3b9ad">${pre?'Q / K / V for T tokens at once':'new Q / K / V for 1 token  [1, …]'}</text>${cube(280,254,pre?150:36,pre?26:26,18,C.orange)}`;
  g.box({id:id('layers'),x:280,y:230,r:60,col:C.orange,hit:[110,178,340,104],svg:lay,aria:'All layers'});
  const cache=`<rect x="110" y="320" width="340" height="104" rx="14" fill="rgba(69,193,221,.06)" stroke="${C.blue}" stroke-width="1.6"/><text class="lab" x="280" y="344" text-anchor="middle" style="fill:${C.blue}">${pre?'Write KV cache':'KV cache: read old · append new'}</text>${slotRow(280,356,pre?16:16,pre?0:15)}<text class="shp" x="280" y="410" text-anchor="middle">${pre?`per layer: K,V [${hk}, T, ${dh}]`:`read [${hk}, L, ${dh}] · append [${hk}, 1, ${dh}]`}</text>`;
  g.box({id:id('cache'),x:280,y:372,r:60,col:C.blue,hit:[110,320,340,104],svg:cache,aria:'KV cache'});
  g.node({id:id('lg'),x:280,y:510,dims:[1,V],col:C.yellow,label:pre?'First logits':'Next-token logits',shape:'[V]'});
  g.path('M280 142V176',{col:C.gray});g.path('M280 284V318',{col:C.gray});g.path('M280 426V446',{col:C.gray});
  if(!pre){g.path('M96 372C60 372 60 256 106 256',{col:C.orange,w:1.8});g.label(60,318,'read',{col:C.orange});}
  g.label(280,22,pre?'Prompt / Prefill':'Decode',{col:pre?C.green:C.blue});
  return svgWrap(560,580,g.html());
}
function pfInfo(id){
  if(!id)return null;const {d,N}=S,hk=kvH(),dh=dHead(),pre=id.startsWith('pre'),T=2**PF.te;
  const k=id.slice(pre?3:3),I={
    x:pre?{t:'Prompt tokens X',c:C.green,d:`All ${T.toLocaleString()} prompt tokens enter at once as a tall [T, ${d}] matrix. Projections are large matrix–matrix products that keep tensor cores busy → <b>compute-bound</b>.`,r:[['Shape',`[${T}, ${d}]`]]}:{t:'One decode token',c:C.blue,d:`Only the newest token enters: a [1, ${d}] row. Every projection is a matrix–<b>vector</b> product that must stream the whole weight matrix from memory for just one row of work → <b>memory-bound</b>.`,r:[['Shape',`[1, ${d}]`]]},
    layers:pre?{t:'All layers (prefill)',c:C.orange,d:`Q, K and V for every prompt token are computed together; attention builds [H, T, T] score matrices (causal). The K,V of every layer are saved for later.`,r:[['Attention cost','O(T²·d) per layer']]}:{t:'All layers (decode)',c:C.orange,d:`Each layer projects the single new token, appends its K,V, and attends over the cached L entries: one [H, 1, L] score row per layer — cheap per token, but the whole cache is read every step.`,r:[['Attention cost','O(L·d) per layer']]},
    cache:pre?{t:'KV cache — written',c:C.blue,d:'Prefill fills the cache: K and V for all T tokens in every layer, written once.',r:[['Written',fmtB(2*N*hk*dh*T*S.bytes)]]}:{t:'KV cache — read + append',c:C.blue,d:'Decode reads the entire cache (all previous tokens) and appends one new K,V pair per layer. Cache size, not FLOPs, limits how many sequences fit and how fast each token comes out.',r:[['Read per step',fmtB(2*N*hk*dh*T*S.bytes)],['Appended',fmtB(2*N*hk*dh*S.bytes)]]},
    lg:{t:pre?'First logits':'Next-token logits',c:C.yellow,d:pre?'Only the last prompt position is projected to the vocabulary — this produces the first generated token. Time to get here is the <b>time-to-first-token</b> (TTFT).':'Logits for the next token, sampled, then the loop repeats. The gap between tokens is the <b>inter-token latency</b>.',r:[['Shape',`[${S.V}]`]]}
  };
  return I[k]||null;
}
function rooflineSVG(r){
  const x=v=>40+(Math.log10(clamp(v,.5,5e4))+.3)/5*520,y0=130;
  let s=`<line x1="40" y1="${y0}" x2="560" y2="${y0}" stroke="#2a4a73" stroke-width="1.5"/>`;
  [1,10,100,1000,10000].forEach(v=>{s+=`<line x1="${x(v)}" y1="${y0}" x2="${x(v)}" y2="${y0+5}" stroke="#2a4a73"/><text class="shp" x="${x(v)}" y="${y0+20}" text-anchor="middle">${v>=1000?v/1000+'k':v}</text>`;});
  const xr=x(r.ridge);
  s+=`<rect x="40" y="30" width="${xr-40}" height="${y0-30}" fill="rgba(69,193,221,.07)"/><rect x="${xr}" y="30" width="${560-xr}" height="${y0-30}" fill="rgba(214,95,72,.07)"/>`;
  s+=`<line x1="${xr}" y1="24" x2="${xr}" y2="${y0}" stroke="var(--txt)" stroke-dasharray="5 4"/><text class="cap" x="${xr}" y="16" text-anchor="middle">ridge ≈ ${r.ridge.toFixed(0)} FLOP/byte</text>`;
  s+=`<text class="elab" x="${(40+xr)/2}" y="50" text-anchor="middle" style="fill:${C.blue}">memory-bound</text><text class="elab" x="${(xr+560)/2}" y="50" text-anchor="middle" style="fill:${C.orange}">compute-bound</text>`;
  [[r.dec,C.blue,'decode',92],[r.pre,C.green,'prefill',72]].forEach(([p,c,l,yy])=>{s+=`<circle cx="${x(p.i)}" cy="${yy}" r="8" fill="${c}" style="filter:drop-shadow(0 0 6px ${c})"/><text class="lab" x="${x(p.i)+(x(p.i)>420?-14:14)}" y="${yy+4}" text-anchor="${x(p.i)>420?'end':'start'}" style="fill:${c}">${l} · ${p.i<10?p.i.toFixed(1):p.i.toFixed(0)}</text>`;});
  s+=`<text class="cap" x="300" y="${y0+40}" text-anchor="middle">arithmetic intensity = FLOPs per byte moved (log scale)</text>`;
  return svgWrap(600,190,s);
}
VIEWS.prefill={
  show(root){
    root.innerHTML=`${secHead('Prefill vs Decode + KV Cache','Same Transformer weights, very different tensor shapes and runtime behaviour.')}
    <div class="grid2"><div class="card dia"><div id="pf-pre" class="svgwrap" style="min-width:420px"></div></div><div class="card dia"><div id="pf-dec" class="svgwrap" style="min-width:420px"></div></div></div>
    <div class="card pad insp" id="pf-insp"></div>
    <div class="card pad"><h3>Where is the bottleneck? — idealised roofline estimate</h3>
      <div class="toolbar" style="margin:10px 0 14px;gap:10px 22px">
        <label class="f">GPU<select id="pf-hw">${Object.entries(HW).map(([k,v])=>`<option value="${k}">${v.n}</option>`).join('')}</select></label>
        <label class="rg">Prompt / context length <b id="pf-tv"></b><input type="range" id="pf-te" min="6" max="17" step="1"></label>
        <label class="f">Batch size<select id="pf-b">${[1,2,4,8,16,32,64,128].map(v=>`<option>${v}</option>`).join('')}</select></label>
        <label class="rg">Tokens to generate <b id="pf-nv"></b><input type="range" id="pf-n" min="16" max="2048" step="16"></label></div>
      <div class="grid2" style="margin-bottom:0"><div id="pf-res"></div><div><div id="pf-roof" class="svgwrap"></div><div id="pf-mem" style="margin-top:6px"></div></div></div></div>
    <div class="card pad" style="overflow-x:auto"><h3 style="margin-bottom:8px">Side by side</h3><table><thead><tr><th></th><th>Prefill</th><th>Decode</th></tr></thead><tbody>
      <tr><td>Tokens per forward pass</td><td>T (the whole prompt)</td><td>1 per sequence</td></tr>
      <tr><td>Q / K / V</td><td>[T, H, dh] · [T, Hkv, dh]</td><td>[1, H, dh] · [1, Hkv, dh] + cached [L, Hkv, dh]</td></tr>
      <tr><td>Attention scores</td><td>[H, T, T] (causal)</td><td>[H, 1, L]</td></tr>
      <tr><td>KV cache</td><td>written once</td><td>read in full, one entry appended</td></tr>
      <tr><td>Typical bottleneck</td><td>compute (matrix–matrix)</td><td>memory bandwidth (matrix–vector + cache reads)</td></tr>
      <tr><td>User-visible metric</td><td>time-to-first-token</td><td>inter-token latency, tokens/s</td></tr>
      <tr><td>Common optimisations</td><td>FlashAttention, chunked prefill, tensor parallelism</td><td>batching, GQA/MQA, paged KV cache, quantisation, speculative decoding</td></tr></tbody></table></div>`;
    $('#pf-hw').value=PF.hw;$('#pf-te').value=PF.te;$('#pf-b').value=PF.b;$('#pf-n').value=PF.n;
    $('#pf-hw').onchange=e=>{PF.hw=e.target.value;this.calc();};
    $('#pf-te').oninput=e=>{PF.te=+e.target.value;this.calc();};
    $('#pf-b').onchange=e=>{PF.b=+e.target.value;this.calc();};
    $('#pf-n').oninput=e=>{PF.n=+e.target.value;this.calc();};
    ['#pf-pre','#pf-dec'].forEach(sel=>bindNodes($(sel),PF,()=>this.focus()));
    this.update();
  },
  update(){$('#pf-pre').innerHTML=pfPanel('pre');$('#pf-dec').innerHTML=pfPanel('dec');this.focus();this.calc();},
  focus(){$('#pf-insp').innerHTML=inspHTML(pfInfo(PF.sel||PF.hov),'Hover or click a block in either lane. Prefill processes the whole prompt in parallel; decode repeats a one-token step.');},
  calc(){
    const r=pfCalc(),n=PF.n,tot=r.pre.t+n*r.dec.t;
    $('#pf-tv').textContent=r.T.toLocaleString()+' tokens';$('#pf-nv').textContent=n;
    const row=(l,a,b)=>`<tr><td>${l}</td><td class="n">${a}</td><td class="n">${b}</td></tr>`;
    $('#pf-res').innerHTML=`<table><thead><tr><th></th><th class="n" style="color:${C.green}">Prefill</th><th class="n" style="color:${C.blue}">Decode / token</th></tr></thead><tbody>
      ${row('FLOPs',fmtF(r.pre.fl),fmtF(r.dec.fl))}${row('Bytes moved',fmtB(r.pre.by),fmtB(r.dec.by))}${row('FLOP / byte',r.pre.i.toFixed(1),r.dec.i.toFixed(1))}
      ${row('Compute time',fmtS(r.pre.tc),fmtS(r.dec.tc))}${row('Memory time',fmtS(r.pre.tm),fmtS(r.dec.tm))}
      <tr class="hl"><td><b>Time</b></td><td class="n"><b>${fmtS(r.pre.t)}</b></td><td class="n"><b>${fmtS(r.dec.t)}</b></td></tr>${row('Bound by',r.pre.bound,r.dec.bound)}</tbody></table>
    <div class="kv" style="margin-top:12px"><span>Time to first token</span><span><b>${fmtS(r.pre.t)}</b></span><span>Decode speed</span><span><b>${(1/r.dec.t).toFixed(0)}</b> tok/s per sequence · ${(r.B/r.dec.t).toFixed(0)} tok/s across batch</span><span>Latency for ${n} new tokens</span><span><b>${fmtS(tot)}</b></span></div>`;
    $('#pf-roof').innerHTML=rooflineSVG(r);
    const gb=r.h.m*1e9,fit=r.mem<=gb;
    $('#pf-mem').innerHTML=`<div class="callout" style="border-color:${fit?C.green:C.red};background:${fit?'rgba(114,217,69,.07)':'rgba(214,69,155,.08)'}">Weights ${fmtB(r.c.total*S.bytes)} + KV cache ${fmtB(r.kv)} (batch ${r.B} × ${r.T.toLocaleString()} tokens) = <b>${fmtB(r.mem)}</b> — ${fit?'fits in':'does <b>not</b> fit in'} ${r.h.m} GB of ${r.h.n.split(' ').slice(-2).join(' ')} memory.</div><p class="mut" style="font-size:12px;margin-top:6px">Ideal roofline: time = max(FLOPs ÷ peak bf16 FLOP/s, bytes ÷ memory bandwidth). Real systems reach a fraction of peak; specs are approximate.</p>`;
  }
};
