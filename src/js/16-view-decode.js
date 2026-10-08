/* ================= 4. Decode step + KV cache ================= */
const DEC={k:0,t:null,sel:null,hov:null};
const PROMPT=['The','capital','of','France','is'];
const GEN=[' Paris',',',' a',' city',' known',' for',' its',' museums','.'];
const decLen=k=>k===0?0:PROMPT.length+k-1;                 // tokens whose K,V are cached after step k
const decAll=PROMPT.concat(GEN);

function decodeSVG(k){
  const {d,V,N}=S,P=PROMPT.length,hk=kvH(),dh=dHead(),len=decLen(k),g=new Graph(null,DEC.sel||DEC.hov),y=118;
  const cur=k===0?'(press Next)':k===1?'prompt × '+P:decAll[P+k-2].trim()||'␣';
  g.node({id:'xt',x:90,y,dims:[k===1?P:1,d],col:C.green,label:k===1?['Prompt tokens','(prefill)']:'Current token xₜ',shape:`[${k===1?P:1}, ${d}]`,step:0});
  g.box({id:'layer',x:330,y,r:80,col:C.orange,step:1,aria:'Decoder layer using KV cache',hit:[250,y-82,160,166],svg:
    `<rect x="258" y="${y-76}" width="144" height="150" rx="12" fill="rgba(214,95,72,.07)" stroke="${C.orange}" stroke-width="1.8" stroke-dasharray="7 5"/><text class="lab" x="330" y="${y-54}" text-anchor="middle">Decoder Layer</text><text class="shp" x="330" y="${y-38}" text-anchor="middle" style="fill:#f3b9ad">(using KV cache)</text>${cube(330,y+10,44,44,30,C.orange)}<text class="shp" x="330" y="${y+68}" text-anchor="middle">× N = ${N}</text>`});
  g.label(90,y+70,k===0?cur:k===1?cur:'“'+cur+'”',{col:C.green});
  g.node({id:'o',x:520,y,dims:[1,d],col:C.green,label:[],shape:`[${k===1?P:1}, ${d}]`,step:1,r:26,max:50});
  g.node({id:'lm',x:640,y,dims:[d,V],col:C.purple,label:['LM Head','W_lm'],shape:`[${d}, V]`,step:1});
  g.node({id:'lg',x:760,y,dims:[1,V],col:C.yellow,label:'logits',shape:'[1, V]',step:1});
  g.node({id:'smp',x:880,y,dims:[24,24,48],col:C.blue,label:'Sampling',shape:'',step:1,max:50});
  const nxt=k===0?'?':GEN[k-1].trim()||'␣',nid=k===0?'—':hashId(GEN[Math.max(0,k-1)],V);
  g.box({id:'next',x:1030,y,r:50,col:C.orange,step:1,aria:'Next token',hit:[975,y-70,110,140],svg:
    `<text class="lab" x="1030" y="${y-46}" text-anchor="middle">next token</text><rect x="985" y="${y-24}" width="90" height="50" rx="9" fill="rgba(214,95,72,.16)" stroke="${C.orange}" stroke-width="2"/><text x="1030" y="${y+1}" text-anchor="middle" style="font:700 17px var(--mono);fill:var(--txt)">${nid}</text><text class="shp" x="1030" y="${y+19}" text-anchor="middle" style="fill:#f3b9ad">“${esc(nxt)}”</text><text class="shp" x="1030" y="${y+51}" text-anchor="middle">[id]</text>`});
  ['xt','layer','o','lm','lg','smp','next'].forEach((a,i,arr)=>{if(i<arr.length-1)g.edge(a,arr[i+1],{step:i===0?0:1});});
  g.path(`M1030 ${y-72}V26H90V${y-66}`,{col:C.orange,step:1,w:1.8});
  g.label(560,20,'the sampled token becomes xₜ for the next step',{col:C.orange});
  /* ---- KV cache panel ---- */
  const x0=70,sw=76,cy=292,slots=14;
  let cs=`<rect x="40" y="${cy-40}" width="1120" height="190" rx="14" fill="rgba(69,193,221,.05)" stroke="${C.blue}" stroke-width="1.4"/><text class="cap" x="60" y="${cy-18}">KV Cache — keys and values of previous tokens (all ${N} layers, ${hk} KV head${hk>1?'s':''})</text>`;
  const newFrom=k===1?0:len-1;
  for(let i=0;i<slots;i++){
    const x=x0+i*sw,on=i<len,isNew=on&&k>0&&i>=newFrom;
    cs+=`<g ${isNew?'class="pulse"':''}><rect x="${x}" y="${cy}" width="${sw-8}" height="36" rx="6" fill="${on?C.orange:'transparent'}" fill-opacity="${isNew?1:.45}" stroke="${on?C.orange:'var(--line2)'}" stroke-dasharray="${on?'':'4 4'}"/><rect x="${x}" y="${cy+42}" width="${sw-8}" height="36" rx="6" fill="${on?C.blue:'transparent'}" fill-opacity="${isNew?1:.45}" stroke="${on?C.blue:'var(--line2)'}" stroke-dasharray="${on?'':'4 4'}"/>`
      +`<text x="${x+(sw-8)/2}" y="${cy+23}" text-anchor="middle" style="font:700 12px var(--mono);fill:${on?'#1a1000':'var(--dim)'}">K</text><text x="${x+(sw-8)/2}" y="${cy+65}" text-anchor="middle" style="font:700 12px var(--mono);fill:${on?'#001322':'var(--dim)'}">V</text></g>`;
    if(on)cs+=`<text x="${x+(sw-8)/2}" y="${cy+98}" text-anchor="middle" class="shp" style="fill:${i<P?'var(--mut)':C.orange}">${esc((decAll[i]||'').trim()||'␣')}</text><text x="${x+(sw-8)/2}" y="${cy+114}" text-anchor="middle" class="shp" style="fill:var(--dim)">${i+1}</text>`;
  }
  g.under(cs);
  if(k>0){
    g.path(`M300 ${cy-42}V${y+82}`,{col:C.orange,step:null,w:2.2});
    g.label(292,y+104,`Read K,V cache [${N}, ${hk}, ${len}, ${dh}]`,{anchor:'end',col:C.orange});
    const ax=x0+(k===1?(P-1)/2:len-1)*sw+(sw-8)/2;
    g.path(`M368 ${y+80}V${cy-62}H${ax}V${cy-3}`,{col:C.orange,w:2.2});
    const right=ax>800;
    g.label(right?ax-10:ax+10,cy-66,`Append new K,V to cache [${N}, ${hk}, ${k===1?P:1}, ${dh}]`,{anchor:right?'end':'start',col:C.orange});
  }
  return svgWrap(1200,520,g.html());
}
function decodeInfo(id){
  const {d,V,N}=S,hk=kvH(),dh=dHead(),k=DEC.k,len=decLen(k);
  const I={
    xt:{t:k===1?'Prompt tokens (prefill)':'Current token xₜ',c:C.green,d:k===1?`Step 1 is the <b>prefill</b>: all ${PROMPT.length} prompt tokens go through the network in parallel and their K,V are written to the cache.`:'During decoding only the newest token is embedded and processed — a [1, d] tensor, not [T, d]. Everything earlier is read from the KV cache.',r:[['Shape',`[${k===1?PROMPT.length:1}, ${d}]`]]},
    layer:{t:'Decoder layers with KV cache',c:C.orange,d:'Each layer computes Q, K, V for the new token only, appends K,V to its cache, then attends over <b>all</b> cached keys/values. Nothing about earlier tokens is recomputed.',r:[['Attention','q [1, H, dh] × K-cacheᵀ [Hkv, dh, L]'],['Cache read / layer',fmtB(2*hk*dh*len*S.bytes)]]},
    o:{t:'Hidden state of the new token',c:C.green,d:'The output for the newest position — the only row the LM head needs.',r:[['Shape',`[1, ${d}]`]]},
    lm:{t:'LM head',c:C.purple,d:'Scores every vocabulary entry for the next token.',r:[['Weight',`[${d}, ${V}]`],['FLOPs',fmtF(2*d*V)]]},
    lg:{t:'Logits',c:C.yellow,d:'One score per vocabulary entry.',r:[['Shape',`[1, ${V}]`]]},
    smp:{t:'Sampling',c:C.blue,d:'Temperature, top-k / top-p, repetition penalty → one token ID. See the sampling lab on the Full Model page.',r:[]},
    next:{t:'Next token',c:C.orange,d:'The chosen token ID. It is appended to the output and fed back as xₜ for the next step.',r:[['Shape','[1] integer']]}
  };
  return I[id]||null;
}
VIEWS.decode={
  show(root){
    root.innerHTML=`${secHead('Next Token Generation (Decode Step)','Use the KV cache and process one token at a time.')}
    <div class="card pad toolbar"><button class="btn pri" id="d-next">▶ Next token</button><button class="btn" id="d-auto">Auto-play</button><button class="btn" id="d-reset">Reset</button><span class="mut mono" id="d-step" style="font-size:12.5px"></span></div>
    <div class="card pad"><h3>Sequence</h3><div class="seq" id="d-seq"></div></div>
    <div class="card dia"><div id="d-svg" class="svgwrap" style="min-width:1000px"></div></div>
    <div class="grid2"><div class="card pad insp" id="d-insp"></div><div class="card pad" id="d-met"></div></div>`;
    DEC.k=0;DEC.sel=null;DEC.hov=null;
    $('#d-next').onclick=()=>{this.stop();this.step();};
    $('#d-reset').onclick=()=>{this.stop();DEC.k=0;this.update();};
    $('#d-auto').onclick=()=>{
      if(DEC.t){this.stop();return;}
      if(DEC.k>=GEN.length)DEC.k=0;
      this.step();DEC.t=setInterval(()=>{if(DEC.k>=GEN.length){this.stop();return;}this.step();},1500);$('#d-auto').textContent='⏸ Pause';
    };
    bindNodes($('#d-svg'),DEC,()=>this.focus());
    this.update();
  },
  step(){if(DEC.k<GEN.length){DEC.k++;this.update();}},
  stop(){clearInterval(DEC.t);DEC.t=null;const b=$('#d-auto');if(b)b.textContent='Auto-play';},
  hide(){this.stop();},
  update(){
    const k=DEC.k;
    $('#d-svg').innerHTML=decodeSVG(k);this.focus();
    $('#d-step').textContent=k===0?'ready — press Next token to run the prefill':k===1?'step 1 · prefill the prompt':`step ${k} · decode token ${k-1}`;
    $('#d-seq').innerHTML=PROMPT.map(t=>`<span class="tk2">${esc(t)}</span>`).join('')+GEN.slice(0,k).map((t,i)=>`<span class="tk2 ${i===k-1?'n':'g'}">${esc(t)}</span>`).join('')+(k<GEN.length?'<span class="tk2" style="opacity:.5">…</span>':'<span class="tk2 g">⟨end⟩</span>');
    const P=PROMPT.length,withC=k===0?0:k===1?P:1,noC=k===0?0:P+k-1;
    const cumW=k===0?0:P+(k-1),cumN=k*P+k*(k-1)/2,hk=kvH(),cacheB=2*S.N*hk*dHead()*decLen(k)*S.bytes;
    $('#d-met').innerHTML=`<h3>Why the cache matters</h3>
    <div class="kv" style="margin:8px 0">
      <span>Tokens in cache</span><span><b>${decLen(k)}</b>  (${fmtB(cacheB)} for ${S.N} layers)</span>
      <span>Tokens processed this step</span><span><b>${withC}</b> with cache vs ${noC} without</span>
      <span>Cumulative tokens processed</span><span><b>${cumW}</b> with cache vs ${cumN} without${cumW?`  (${(cumN/cumW).toFixed(1)}×)`:''}</span></div>
    <div class="bar" title="with cache"><i style="width:${cumN?100*cumW/cumN:0}%;background:${C.green}"></i></div>
    <div class="leg" style="margin-top:4px"><span style="--c:${C.green}">with KV cache</span><span style="--c:#4a6a8f">without (recompute everything each step)</span></div>
    <p class="mut" style="font-size:12.5px;margin-top:10px">Without a cache, every new token re-runs K and V projections for the whole prefix, so total work grows ~quadratically. With a cache, each step does O(1) projection work and one attention read over L cached entries — which makes decoding <b>memory-bound</b>: the cache and weights are re-read from HBM every token.</p>`;
  },
  focus(){$('#d-insp').innerHTML=inspHTML(decodeInfo(DEC.sel||DEC.hov),'Hover or click a block. Press <b>Next token</b> to run the prefill, then decode one token per click — watch the highlighted slot append to the cache.');}
};
