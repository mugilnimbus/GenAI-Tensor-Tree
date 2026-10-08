/* ================= 3. Attention variants ================= */
const A={ctx:8192,n:40,w:8,g:0,causal:true,wr:4096,row:null};
const ATTN_TXT={
  mha:{t:'Multi-Head Attention (MHA)',s:'Every query head owns its own key and value head.',
    how:'Q, K and V all have H heads: Q [T, H, d<sub>h</sub>], K [T, H, d<sub>h</sub>], V [T, H, d<sub>h</sub>]. Head i attends using only its own K<sub>i</sub> and V<sub>i</sub>, so heads can specialise freely.',
    trade:'Most expressive, but the KV cache grows with H × d<sub>h</sub> per token — it dominates memory at long context and slows decoding, which is memory-bound.',used:['GPT-2 / GPT-3','BERT','Llama-1','Llama-2 7B / 13B']},
  mqa:{t:'Multi-Query Attention (MQA)',s:'All query heads share a single key/value head.',
    how:'Q keeps H heads but K and V have just one: K [T, 1, d<sub>h</sub>], V [T, 1, d<sub>h</sub>]. The single K/V head is broadcast to every query head. Wk and Wv shrink from [d, d] to [d, d<sub>h</sub>].',
    trade:`The KV cache is H× smaller and decoding is much faster, at some cost in quality and training stability.`,used:['PaLM','Falcon-7B','StarCoder','Fast-decoding / on-device models']},
  gqa:{t:'Grouped-Query Attention (GQA)',s:'Query heads are split into groups; each group shares one K/V head.',
    how:'K and V have H<sub>kv</sub> heads (1 < H<sub>kv</sub> < H). Query head i uses K/V head ⌊i / (H / H<sub>kv</sub>)⌋. MHA and MQA are the two extremes of this one idea.',
    trade:'Close to MHA quality with a cache close to MQA size — the default in modern LLMs.',used:['Llama-2 70B','Llama-3','Mistral 7B','Qwen2.5','Gemma-2']}
};
const gqaDefault=()=>{const ds=divisors(S.H).filter(x=>x>1&&x<S.H);return ds.length?ds.reduce((b,x)=>Math.abs(x-S.H/4)<Math.abs(b-S.H/4)?x:b,ds[0]):S.H;};
function headsSVG(){
  const H=S.H,hk=kvH(),g=H/hk,dh=dHead(),W=1200,cw=Math.min(36,1080/H),x0=(W-H*cw)/2,gap=Math.max(2,cw*.16);
  const col=k=>hk===1?C.orange:`hsl(${(k*(360/hk)+190)%360} 85% 62%)`;
  let s=`<text class="cap" x="${x0}" y="34">Query heads — H = ${H}</text><text class="cap" x="${x0}" y="214">Key / Value heads — H_kv = ${hk}${g>1?` · each shared by ${g} query heads`:' · one per query head'}</text>`;
  for(let i=0;i<H;i++){const k=Math.floor(i/g),xq=x0+i*cw+cw/2,xk=x0+(k*g+g/2)*cw;s+=`<path data-g="${k}" class="hl" d="M${r1(xq)} 106C${r1(xq)} 165 ${r1(xk)} 175 ${r1(xk)} 238" stroke="${col(k)}" fill="none" stroke-width="1.3" opacity=".55"/>`;}
  for(let i=0;i<H;i++){const k=Math.floor(i/g);s+=`<g data-g="${k}" data-q="${i}" class="hl" style="cursor:pointer"><rect x="${r1(x0+i*cw+gap/2)}" y="50" width="${r1(cw-gap)}" height="54" rx="5" fill="${col(k)}" opacity=".88"/>${cw>=22?`<text x="${r1(x0+i*cw+cw/2)}" y="82" text-anchor="middle" style="font:600 11px var(--mono);fill:#04101c">${i+1}</text>`:''}</g>`;}
  for(let k=0;k<hk;k++){const x=x0+k*g*cw+gap/2,w=g*cw-gap;
    s+=`<g data-g="${k}" class="hl" style="cursor:pointer"><rect x="${r1(x)}" y="238" width="${r1(w)}" height="26" rx="5" fill="${C.orange}" opacity=".9"/><rect x="${r1(x)}" y="268" width="${r1(w)}" height="26" rx="5" fill="${C.blue}" opacity=".9"/>`
      +`<text x="${r1(x+w/2)}" y="256" text-anchor="middle" style="font:700 12px var(--mono);fill:#04101c">${w>=40?'K'+(k+1):'K'}</text><text x="${r1(x+w/2)}" y="286" text-anchor="middle" style="font:700 12px var(--mono);fill:#04101c">${w>=40?'V'+(k+1):'V'}</text><rect x="${r1(x-gap/2)}" y="236" width="${r1(w+gap)}" height="60" rx="7" fill="none" stroke="${col(k)}" stroke-width="1.5" opacity=".7"/></g>`;}
  const bd=n=>BANDS.slice(0,Math.min(6,Math.max(1,n)));
  s+=plainTensor(270,404,[H,S.T,dh],C.blue,'Q',`[T, ${H}, ${dh}]`,{bands:bd(H),max:72});
  s+=plainTensor(600,404,[hk,S.T,dh],C.blue,'K',`[T, ${hk}, ${dh}]`,{bands:bd(hk),max:72});
  s+=plainTensor(930,404,[hk,S.T,dh],C.blue,'V',`[T, ${hk}, ${dh}]`,{bands:bd(hk),max:72});
  return svgWrap(1200,480,s);
}
function kvRows(){
  const N=S.N,dh=dHead(),by=S.bytes,L=A.ctx,gq=gqaDefault();
  return [['MHA',S.H],['GQA',S.attn==='GQA'?S.Hkv:gq],['MQA',1]].map(([n,hk])=>({n,hk,tok:2*N*hk*dh*by,tot:2*N*hk*dh*by*L,proj:2*S.d*hk*dh}));
}
const swaAllowed=(i,j)=>{
  if(A.causal&&j>i)return 0;
  const lo=Math.floor((A.w-1)/2),hi=Math.ceil((A.w-1)/2);
  const inwin=A.causal?(i-j<A.w):(j>=i-lo&&j<=i+hi);
  const glob=A.causal?(j<A.g):(j<A.g||i<A.g);
  return inwin||glob?1:0;
};
function drawSWA(){
  const c=$('#swa');if(!c)return;
  const size=c.clientWidth||520,dpr=window.devicePixelRatio||1;
  c.width=Math.round(size*dpr);c.height=Math.round(size*dpr);
  const x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);
  const n=A.n,cell=size/n;x.fillStyle=LT()?'#ffffff':'#000000';x.fillRect(0,0,size,size);
  for(let i=0;i<n;i++)for(let j=0;j<n;j++){
    const fut=A.causal&&j>i,ok=swaAllowed(i,j);
    x.fillStyle=fut?(LT()?'#f4eff9':'#0d0912'):ok?(i===A.row?'#f3a8cf':C.red):(LT()?'#d9cdea':'#2a1f3d');
    x.fillRect(j*cell+.5,i*cell+.5,cell-1,cell-1);
  }
  if(A.row!=null){x.strokeStyle=LT()?'#000':'#fff';x.lineWidth=1.5;x.strokeRect(.75,A.row*cell+.75,size-1.5,cell-1.5);}
}
function swaStats(){
  const n=A.n;let cnt=0,row=0;
  for(let i=0;i<n;i++)for(let j=0;j<n;j++)if(swaAllowed(i,j)){cnt++;if(i===A.row)row++;}
  const full=A.causal?n*(n+1)/2:n*n,L=A.ctx,Wr=A.wr,N=S.N;
  const fullL=L*(L+1)/2,winL=L<=Wr?fullL:Wr*(Wr+1)/2+(L-Wr)*Wr;
  const kvOne=2*N*S.Hkv*dHead()*S.bytes;
  $('#swa-stats').innerHTML=`<div class="kv">
    <span>Cells attended</span><span><b>${cnt}</b> of ${full} ${A.causal?'causal':'full'} (${(100*cnt/full).toFixed(0)}%)</span>
    <span>Hovered query</span><span>${A.row==null?'move the mouse over the matrix':`token ${A.row+1} attends to ${row} key${row===1?'':'s'}`}</span></div>
    <h3 style="margin-top:16px">At real scale (per head)</h3>
    <div class="toolbar" style="margin:8px 0"><label class="f">Context length<select id="swa-ctx">${[4096,8192,32768,131072].map(v=>`<option value="${v}" ${v===A.ctx?'selected':''}>${v.toLocaleString()}</option>`).join('')}</select></label>
    <label class="f">Window W<select id="swa-wr">${[512,1024,2048,4096,8192].map(v=>`<option value="${v}" ${v===A.wr?'selected':''}>${v.toLocaleString()}</option>`).join('')}</select></label></div>
    <div class="kv"><span>Full causal scores</span><span>${fmtN(fullL)}</span><span>Sliding window</span><span><b>${fmtN(winL)}</b> (${(fullL/winL).toFixed(1)}× fewer)</span>
    <span>KV cache (all layers)</span><span>${fmtB(kvOne*L)} → <b>${fmtB(kvOne*Math.min(L,Wr))}</b> rolling buffer</span>
    <span>Receptive field</span><span>≈ ${N} layers × W = <b>${fmtN(N*Wr)}</b> tokens (information hops one window per layer)</span></div>`;
  $('#swa-ctx').onchange=e=>{A.ctx=+e.target.value;swaStats();};
  $('#swa-wr').onchange=e=>{A.wr=+e.target.value;swaStats();};
}
VIEWS.attn={
  show(root,sub){
    sub=['mha','mqa','gqa','swa'].includes(sub)?sub:'gqa';this.sub=sub;
    if(sub!=='swa')setAttn(sub.toUpperCase(),true);
    const tabs=[['mha','MHA'],['mqa','MQA'],['gqa','GQA'],['swa','Sliding Window']];
    const nav=`<div class="toolbar" style="margin-top:18px"><nav class="seg" aria-label="Attention variant">${tabs.map(([k,t])=>`<a href="#attn/${k}" class="${k===sub?'on':''}">${t}</a>`).join('')}</nav></div>`;
    if(sub==='swa'){
      root.innerHTML=nav+secHead('Sliding-Window / Local Attention','Each query attends only to a local window instead of the whole sequence: cost ≈ O(T·W), not O(T²).')+`
      <div class="grid2"><div class="card pad"><canvas id="swa" aria-label="Attention mask matrix" style="width:100%;max-width:560px;aspect-ratio:1"></canvas>
        <div class="leg" style="margin-top:10px"><span style="--c:${C.red}">attended</span><span style="--c:#6b5a8a">skipped by window</span><span style="--c:var(--off)">future (causal mask)</span></div></div>
      <div class="card pad"><h3>Mask controls</h3><div class="toolbar" style="margin:10px 0 14px;gap:10px 22px" id="swa-ctl">
        <label class="rg">Sequence length <b id="v-n">${A.n}</b><input type="range" data-k="n" min="12" max="96" step="1" value="${A.n}"></label>
        <label class="rg">Window W <b id="v-w">${A.w}</b><input type="range" data-k="w" min="1" max="96" step="1" value="${A.w}"></label>
        <label class="rg">Global tokens <b id="v-g">${A.g}</b><input type="range" data-k="g" min="0" max="6" step="1" value="${A.g}"></label>
        <label class="chk"><input type="checkbox" id="swa-causal" ${A.causal?'checked':''}> causal (decoder)</label></div>
        <div id="swa-stats"></div>
        <p class="callout" style="margin-top:14px">Used by Mistral-7B (W = 4096), Longformer, and as alternating local/global layers in Gemma-2 and others. Global tokens (Longformer / BigBird) are attended by everyone and attend to everyone.</p></div></div>`;
      const redraw=()=>{drawSWA();swaStats();};
      $('#swa-ctl').addEventListener('input',e=>{const k=e.target.dataset.k;if(!k)return;A[k]=+e.target.value;if(k==='n'&&A.w>A.n){A.w=A.n;}$('#v-n').textContent=A.n;$('#v-w').textContent=A.w;$('#v-g').textContent=A.g;redraw();});
      $('#swa-causal').addEventListener('change',e=>{A.causal=e.target.checked;redraw();});
      const c=$('#swa');
      c.addEventListener('mousemove',e=>{const r=c.getBoundingClientRect();A.row=clamp(Math.floor((e.clientY-r.top)/r.height*A.n),0,A.n-1);redraw();});
      c.addEventListener('mouseleave',()=>{A.row=null;redraw();});
      redraw();
      this.update=()=>{swaStats();};
      return;
    }
    const T=ATTN_TXT[sub];
    root.innerHTML=nav+secHead(T.t,T.s)+`
    <div class="card pad toolbar">${sub==='gqa'?`<label class="f">KV heads (groups)<select id="a-hkv">${divisors(S.H).filter(x=>x>1&&x<S.H).map(x=>`<option value="${x}">${x} KV heads · ${S.H/x} Q per KV</option>`).join('')}</select></label>`:''}
      <label class="f">Context length for cache size<select id="a-ctx">${[2048,4096,8192,32768,131072].map(v=>`<option value="${v}">${v.toLocaleString()} tokens</option>`).join('')}</select></label>
      <span class="mut" style="font-size:12.5px">Hover a query head or a K/V head to see the sharing.</span></div>
    <div class="card dia"><div id="a-svg" class="svgwrap" style="min-width:900px"></div></div>
    <div class="grid2"><div class="card pad insp" id="a-info"></div><div class="card pad" id="a-kv"></div></div>`;
    $('#a-ctx').value=A.ctx;$('#a-ctx').addEventListener('change',e=>{A.ctx=+e.target.value;this.kv();});
    const hk=$('#a-hkv');if(hk){hk.value=S.Hkv;hk.addEventListener('change',e=>{S.Hkv=+e.target.value;S.preset='custom';refresh();});}
    const svg=$('#a-svg');
    svg.addEventListener('mouseover',e=>{
      const el=e.target.closest('[data-g]'),g=el?el.dataset.g:null;
      $$('.hl',svg).forEach(n=>n.style.opacity=g==null||n.dataset.g===g?'':'.18');
      const H=S.H,k=kvH(),gs=H/k;
      $('#a-read').innerHTML=g==null?'':`<b>KV head ${+g+1}</b> serves query heads <b>${+g*gs+1}–${(+g+1)*gs}</b>${gs>1?` (${gs} heads share it)`:''}.`;
    });
    svg.addEventListener('mouseleave',()=>{$$('.hl',svg).forEach(n=>n.style.opacity='');$('#a-read').innerHTML='';});
    this.update=()=>{
      if(this.sub==='gqa'&&$('#a-hkv'))$('#a-hkv').value=S.Hkv;
      svg.innerHTML=headsSVG();this.info();this.kv();
    };
    this.update();
  },
  info(){
    const T=ATTN_TXT[this.sub],H=S.H,hk=kvH(),dh=dHead();
    $('#a-info').innerHTML=`<h3>How it works</h3><p>${T.how}</p><h3 style="margin-top:12px">Trade-off</h3><p>${T.trade}</p>
      <div class="kv" style="margin-top:10px"><span>Q / K / V shapes</span><span>[T, ${H}, ${dh}] · [T, ${hk}, ${dh}] · [T, ${hk}, ${dh}]</span><span>Wq · Wk · Wv</span><span>[${S.d}, ${S.d}] · [${S.d}, ${hk*dh}] · [${S.d}, ${hk*dh}]</span><span>Scores</span><span>[${H}, T, T] — unchanged by K/V sharing</span></div>
      <p style="margin-top:10px">${T.used.map(u=>`<span class="chip">${u}</span>`).join('')}</p><p id="a-read" class="mut" style="min-height:22px;margin-top:6px"></p>`;
  },
  kv(){
    const rows=kvRows(),max=rows[0].tot,L=A.ctx;
    $('#a-kv').innerHTML=`<h3>KV-cache cost</h3><p class="mut" style="font-size:13px">cache = 2 (K and V) × ${S.N} layers × H<sub>kv</sub> × ${dHead()} × ${L.toLocaleString()} tokens × ${S.bytes} B</p>
    <table><thead><tr><th>Variant</th><th class="n">H<sub>kv</sub></th><th class="n">per token</th><th class="n">at ${L>=1024?L/1024+'k':L}</th><th style="width:26%"></th></tr></thead><tbody>
    ${rows.map(r=>`<tr class="${r.n===S.attn?'hl':''}"><td><b>${r.n}</b></td><td class="n">${r.hk}</td><td class="n">${fmtB(r.tok)}</td><td class="n">${fmtB(r.tot)}</td><td><div class="bar"><i style="width:${(100*r.tot/max).toFixed(1)}%;background:${r.n===S.attn?C.orange:'#4a6a8f'}"></i></div></td></tr>`).join('')}</tbody></table>
    <p class="mut" style="font-size:12.5px;margin-top:8px">K/V projection parameters per layer: MHA ${fmtN(rows[0].proj)} · GQA ${fmtN(rows[1].proj)} · MQA ${fmtN(rows[2].proj)}. Smaller caches also mean less memory traffic per generated token.</p>`;
  }
};
