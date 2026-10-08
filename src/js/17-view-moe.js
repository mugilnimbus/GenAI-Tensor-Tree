/* ================= 5. Mixture of Experts ================= */
const MTOK=['The','router','sends','each','token','to','its','experts'];
function moeRoute(){
  const {E,k,seed}=MO,r=rng(seed*977+E*31);
  const bias=Array.from({length:E},()=>(r()-.5)*1.6);
  return MTOK.map(t=>{
    const lg=bias.map(b=>b+(r()+r()+r()-1.5)*2.8);
    const top=[...lg.keys()].sort((a,b)=>lg[b]-lg[a]).slice(0,k);
    const p=softmax(lg),mx=Math.max(...lg),ex=top.map(i=>Math.exp(lg[i]-mx)),z=ex.reduce((a,b)=>a+b,0);
    return {t,lg,p,top,gate:ex.map(v=>v/z)};
  });
}
function moeSVG(R){
  const {E,k,shared,sel}=MO,{d}=S,rowH=clamp(Math.floor(440/E),28,60),top=112,rowsH=E*rowH,extra=shared?rowH:0,H=top+rowsH+extra+80;
  const tk=R[sel],cy=top+rowsH/2,load=new Array(E).fill(0);
  R.forEach(t=>t.top.forEach(i=>load[i]++));
  let s='';
  /* tokens */
  const sp=Math.min(52,(rowsH-34)/7),ty0=top+(rowsH-(7*sp+34))/2;
  R.forEach((t,i)=>{
    const y=ty0+i*sp,on=i===sel;
    s+=`<g class="node${on?' foc sel':''}" data-id="t${i}" tabindex="0" role="button" aria-label="Token ${esc(t.t)}" style="color:${C.yellow}"><rect class="hit" x="30" y="${y-4}" width="150" height="42" rx="8"/><rect x="40" y="${y}" width="126" height="34" rx="8" fill="${on?'rgba(214,204,69,.22)':'rgba(214,204,69,.07)'}" stroke="${C.yellow}" stroke-width="${on?2:1}"/><text x="53" y="${y+22}" style="font:600 13px var(--mono);fill:var(--txt)">${esc(t.t)}</text><text x="158" y="${y+22}" text-anchor="end" class="shp">x${i+1}</text></g>`;
  });
  s+=`<text class="cap" x="40" y="${top-22}">Token states  [T, ${d}]</text>`;
  const tyc=ty0+sel*sp+17;
  s+=`<path class="edge" d="M168 ${tyc}C210 ${tyc} 210 ${cy} 252 ${cy}" stroke="${C.yellow}" stroke-width="2.2" fill="none" marker-end="url(#ar-${C.yellow.slice(1)})"/>`;
  /* router */
  s+=`<rect x="256" y="${top-30}" width="240" height="${rowsH+40}" rx="14" fill="rgba(214,95,72,.06)" stroke="${C.orange}" stroke-width="1.6"/><text class="lab" x="376" y="${top-58}" text-anchor="middle" style="fill:${C.orange}">Router / gate</text><text class="shp" x="376" y="${top-42}" text-anchor="middle">W_r [${d}, ${E}] → scores [T, ${E}]</text>`;
  for(let e=0;e<E;e++){
    const y=top+e*rowH,chosen=tk.top.includes(e),w=Math.max(2,tk.p[e]*150*(E>8?2.2:1.6));
    s+=`<text x="272" y="${y+rowH/2+4}" class="shp" style="fill:${chosen?'#fff':'var(--mut)'}">e${e+1}</text><rect x="302" y="${y+4}" width="${Math.min(w,150)}" height="${rowH-8}" rx="4" fill="${chosen?C.orange:'#38557a'}"/><text x="${302+Math.min(w,150)+6}" y="${y+rowH/2+4}" class="shp" style="fill:${chosen?'#f3b9ad':'var(--mut)'}">${(100*tk.p[e]).toFixed(0)}%</text>`;
  }
  /* experts + lines */
  const ex=640,combX=1010;
  for(let e=0;e<E;e++){
    const y=top+e*rowH+rowH/2,idx=tk.top.indexOf(e),chosen=idx>=0,gt=chosen?tk.gate[idx]:0;
    s+=`<path d="M498 ${y}H${ex}" stroke="${chosen?C.orange:'var(--line2)'}" stroke-width="${chosen?2+gt*5:1}" opacity="${chosen?.95:.5}" fill="none"${chosen?` class="edge" marker-end="url(#ar-${C.orange.slice(1)})"`:''}/>`;
    if(chosen)s+=`<text class="elab" x="${(498+ex)/2}" y="${y-5}" text-anchor="middle" style="fill:#f3b9ad;font-weight:700">g=${gt.toFixed(2)}</text>`;
    s+=`<g class="node${chosen?' foc':''}" data-id="e${e}" tabindex="0" role="button" aria-label="Expert ${e+1}" style="color:${C.purple}"><rect class="hit" x="${ex}" y="${y-rowH/2}" width="210" height="${rowH}" rx="8"/><rect x="${ex+4}" y="${y-rowH/2+3}" width="198" height="${rowH-6}" rx="8" fill="${chosen?'rgba(166,69,220,.28)':'rgba(166,69,220,.07)'}" stroke="${C.purple}" stroke-width="${chosen?2:1}" opacity="${chosen?1:.6}"/>`
      +`<text x="${ex+14}" y="${y+4.5}" style="font:600 13px var(--sans);fill:${chosen?'#fff':'var(--mut)'}">Expert ${e+1}</text><text x="${ex+88}" y="${y+4.5}" class="shp">${rowH>=36?'D→M→D':''}</text>`
      +`<g transform="translate(${ex+160},${y})">${Array.from({length:load[e]},(_,j)=>`<rect x="${j*9-4}" y="-5" width="7" height="10" rx="2" fill="${C.yellow}" opacity=".85"/>`).join('')}</g></g>`;
    if(chosen)s+=`<path class="edge" d="M${ex+210} ${y}C${(ex+210+combX)/2} ${y} ${(ex+210+combX)/2} ${cy} ${combX-34} ${cy}" stroke="${C.orange}" stroke-width="${2+gt*5}" fill="none" marker-end="url(#ar-${C.orange.slice(1)})"/>`;
  }
  s+=`<text class="cap" x="${ex}" y="${top-22}">Expert MLPs (yellow ticks = tokens routed here)</text>`;
  if(shared){const y=top+rowsH+rowH/2;
    s+=`<g class="node" data-id="shared" tabindex="0" role="button" aria-label="Shared expert" style="color:${C.green}"><rect class="hit" x="${ex}" y="${y-rowH/2}" width="210" height="${rowH}" rx="8"/><rect x="${ex+4}" y="${y-rowH/2+3}" width="198" height="${rowH-6}" rx="8" fill="rgba(114,217,69,.18)" stroke="${C.green}" stroke-width="2"/><text x="${ex+14}" y="${y+4.5}" style="font:600 13px var(--sans);fill:var(--txt)">Shared expert (always on)</text></g>`
      +`<path class="edge" d="M${ex+210} ${y}C${(ex+210+combX)/2} ${y} ${(ex+210+combX)/2} ${cy} ${combX-34} ${cy}" stroke="${C.green}" stroke-width="3" fill="none" marker-end="url(#ar-${C.green.slice(1)})"/>`;}
  /* combine + output */
  s+=`<g class="node" data-id="combine" tabindex="0" role="button" aria-label="Weighted combine" style="color:${C.orange}"><rect class="hit" x="${combX-48}" y="${cy-70}" width="96" height="140" rx="12"/><circle class="opc" cx="${combX}" cy="${cy}" r="28"/><text class="sym" x="${combX}" y="${cy+8}" text-anchor="middle" style="font-size:26px">Σ</text><text class="lab" x="${combX}" y="${cy-60}" text-anchor="middle">Weighted</text><text class="lab" x="${combX}" y="${cy-44}" text-anchor="middle">combine</text><text class="shp" x="${combX}" y="${cy+52}" text-anchor="middle">Σ gᵢ·Eᵢ(x)</text></g>`;
  s+=`<path class="edge" d="M${combX+30} ${cy}H1100" stroke="${C.green}" stroke-width="2.2" fill="none" marker-end="url(#ar-${C.green.slice(1)})"/>`;
  const [w,h,dd]=dimsBox([1,d],50);
  s+=`<g style="color:${C.green}">${cube(1135,cy,w,h,dd,C.green)}<text class="lab" x="1135" y="${cy-46}" text-anchor="middle">output y</text><text class="shp" x="1135" y="${cy+51}" text-anchor="middle">[T, ${d}]</text></g>`;
  return svgWrap(1200,H,s);
}
VIEWS.moe={
  show(root){
    root.innerHTML=`${secHead('Mixture-of-Experts (MoE) Transformer MLP','Attention stays shared; the dense MLP is replaced by a router and many expert MLPs. Only the selected experts run for each token.')}
    <div class="card pad toolbar"><span class="lbl">Experts E</span>${segHTML('mo-e',[[4,'4'],[8,'8'],[16,'16']],MO.E)}<span class="lbl">Top-k</span>${segHTML('mo-k',[[1,'1'],[2,'2'],[4,'4']],MO.k)}
      <label class="chk"><input type="checkbox" id="mo-sh"> shared expert</label><button class="btn" id="mo-seed">↻ New router weights</button><span class="mut" style="font-size:12.5px">Click a token on the left to trace its route.</span></div>
    <div class="card dia"><div id="mo-svg" class="svgwrap" style="min-width:1000px"></div></div>
    <div class="grid2"><div class="card pad insp" id="mo-insp"></div><div class="card pad" id="mo-par"></div></div>`;
    MO.hov=null;
    segBind($('#mo-e'),v=>{MO.E=+v;MO.k=Math.min(MO.k,MO.E);this.update();});
    segBind($('#mo-k'),v=>{MO.k=Math.min(+v,MO.E);this.update();});
    $('#mo-sh').checked=MO.shared;$('#mo-sh').onchange=e=>{MO.shared=e.target.checked;this.update();};
    $('#mo-seed').onclick=()=>{MO.seed=(MO.seed*7+3)%1000;this.update();};
    const svg=$('#mo-svg');
    svg.addEventListener('click',e=>{const n=e.target.closest('.node');if(n&&/^t\d$/.test(n.dataset.id)){MO.sel=+n.dataset.id.slice(1);this.update();}});
    svg.addEventListener('mouseover',e=>{const n=e.target.closest('.node');MO.hov=n?n.dataset.id:null;this.focus();});
    svg.addEventListener('mouseleave',()=>{MO.hov=null;this.focus();});
    svg.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){const n=e.target.closest('.node');if(n&&/^t\d$/.test(n.dataset.id)){e.preventDefault();MO.sel=+n.dataset.id.slice(1);this.update();}}});
    this.update();
  },
  update(){
    segSet($('#mo-e'),MO.E);segSet($('#mo-k'),MO.k);
    this.R=moeRoute();
    $('#mo-svg').innerHTML=moeSVG(this.R);this.focus();this.params();
  },
  focus(){
    const id=MO.hov,R=this.R,{d,ff}=S,t=R[MO.sel],pe=3*d*ff;
    let i=null;
    if(id&&/^e\d+$/.test(id)){const e=+id.slice(1),cnt=R.filter(r=>r.top.includes(e)).length;i={t:`Expert ${e+1}`,c:C.purple,d:`An independent SwiGLU MLP (${d} → ${ff} → ${d}) with its own weights. In this batch it receives <b>${cnt}</b> of ${R.length} tokens. In real systems experts are sharded across GPUs (expert parallelism) and tokens are dispatched with all-to-all communication.`,r:[['Params / expert',fmtN(pe)],['This token routed here',t.top.includes(e)?`yes, gate ${t.gate[t.top.indexOf(e)].toFixed(2)}`:'no']]};}
    else if(id==='combine')i={t:'Weighted combine',c:C.orange,d:`The outputs of the selected experts are summed with their gate weights: y = Σ gᵢ · Eᵢ(x)${MO.shared?' + shared(x)':''}. Gates come from a softmax over the top-${MO.k} router scores, so they sum to 1.`,r:[['Token',esc(t.t)],['Experts',t.top.map((e,j)=>`e${e+1} (${t.gate[j].toFixed(2)})`).join(', ')]]};
    else if(id==='shared')i={t:'Shared expert',c:C.green,d:'DeepSeek-style MoE keeps one (or a few) experts that every token always uses, capturing common knowledge so the routed experts can specialise.',r:[['Params',fmtN(pe)]]};
    else if(id&&/^t\d$/.test(id)){const r=R[+id.slice(1)];i={t:`Token “${esc(r.t)}”`,c:C.yellow,d:`Router logits → softmax → top-${MO.k}. Click to trace this token.`,r:[['Chosen experts',r.top.map(e=>'e'+(e+1)).join(', ')]]};}
    $('#mo-insp').innerHTML=inspHTML(i,`Tracing token <b>“${esc(t.t)}”</b>: router picks <b>${t.top.map(e=>'expert '+(e+1)).join(' + ')}</b> with gates ${t.gate.map(g=>g.toFixed(2)).join(' / ')}. Hover an expert or the combine node for details.`);
  },
  params(){
    const {E,k,shared}=MO,{d,ff,N}=S,c=counts(),pe=3*d*ff,router=d*E;
    const tot=N*(E*pe+router+(shared?pe:0)),act=N*((k+(shared?1:0))*pe+router),dense=N*pe;
    const rest=c.emb*(S.tie?1:2)+N*(c.attn+c.norms)+d,load=new Array(E).fill(0);this.R.forEach(t=>t.top.forEach(i=>load[i]++));
    const mean=this.R.length*k/E;
    $('#mo-par').innerHTML=`<h3>Total vs active parameters</h3><div class="kv" style="margin:8px 0">
      <span>One expert (SwiGLU)</span><span>${fmtN(pe)}  (3·${d}·${ff})</span>
      <span>MLP params, all experts</span><span><b>${fmtN(tot)}</b> over ${N} layers</span>
      <span>MLP params used per token</span><span><b>${fmtN(act)}</b> (${((k+(shared?1:0)))} of ${E+(shared?1:0)} experts)</span>
      <span>Whole model</span><span><b>${fmtN(tot+rest)}</b> total · <b>${fmtN(act+rest)}</b> active per token</span>
      <span>vs a dense MLP</span><span>${(tot/dense).toFixed(1)}× capacity at ${((act)/dense).toFixed(1)}× per-token FFN compute</span>
      <span>Load balance (8 tokens)</span><span>max ${Math.max(...load)} vs mean ${mean.toFixed(1)} per expert</span></div>
    <p class="callout">An auxiliary load-balancing loss during training keeps the router from collapsing onto a few popular experts. Try “New router weights” and watch the yellow load ticks.</p>`;
  }
};
