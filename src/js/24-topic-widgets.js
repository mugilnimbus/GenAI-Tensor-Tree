/* ================= small interactive widgets for the topic pages =================
   WIDGETS[topicId](el) fills el. Controls stay in place; only the output area is re-rendered. */
const cssVar=v=>getComputedStyle(document.body).getPropertyValue(v).trim();
const PAL=['#45c1dd','#4a55e0','#a645dc','#d6459b','#d65f48','#d6cc45','#72d945','#45d984'];
/* generic shell: sliders [key,label,min,max,step,fmt?], checkboxes/segments via extra html */
function widget(el,title,st,sliders,render,extra=''){
  el.innerHTML=`<div class="card pad"><h3>${title}</h3><div class="toolbar wctl" style="margin:10px 0;gap:10px 22px">${sliders.map(([k,l,mn,mx,sp])=>`<label class="rg">${l} <b data-o="${k}"></b><input type="range" data-k="${k}" min="${mn}" max="${mx}" step="${sp}" value="${st[k]}"></label>`).join('')}${extra}</div><div class="wout"></div></div>`;
  const out=$('.wout',el),fmt=Object.fromEntries(sliders.map(s=>[s[0],s[5]||(v=>v)]));
  const run=()=>{sliders.forEach(([k])=>{$(`[data-o="${k}"]`,el).textContent=fmt[k](st[k]);});render(out,st);};
  el.addEventListener('input',e=>{const k=e.target.dataset.k;if(k){st[k]=e.target.type==='range'?+e.target.value:e.target.value;run();}});
  el.addEventListener('click',e=>{const b=e.target.closest('[data-seg]');if(b){st[b.dataset.seg]=b.dataset.v;$$(`[data-seg="${b.dataset.seg}"]`,el).forEach(x=>x.classList.toggle('on',x===b));run();}});
  run();
}
const barRow=(l,v,w,c,note='')=>`<div class="wrow"><span>${l}</span><div class="bar"><i style="width:${Math.max(0.5,w).toFixed(1)}%;background:${c}"></i></div><span class="mono">${v}</span>${note?`<small>${note}</small>`:''}</div>`;
function canvasOf(out,w,h){out.insertAdjacentHTML('beforeend',`<canvas width="${w}" height="${h}" style="width:${w}px;max-width:100%"></canvas>`);const c=out.lastElementChild;return [c,c.getContext('2d')];}

/* ---- fine-tuning: trainable parameters of every method, on the configured model ---- */
const PEFT={r:16,m:64,p:20};
function peftWidget(el,cur){
  widget(el,'How much is trained? — every method on the configured model',PEFT,[['r','LoRA / DoRA rank r',1,256,1],['m','Adapter width m',4,512,4],['p','Prefix length p',1,200,1]],(out,s)=>{
    const d=S.d,N=S.N,ff=S.ff,kvd=kvH('GQA')*dHead(),tot=counts('GQA').total,lora=N*s.r*(2*d+d+kvd);
    const rows=[['fullft','Full fine-tuning',tot],['adapters','Adapters (after attention and MLP)',N*2*2*d*s.m],['dora','DoRA (W_Q, W_V)',lora+N*(d+kvd)],['lora','LoRA (W_Q, W_V)',lora],['prefix','Prefix tuning (K, V in every layer)',2*s.p*kvd*N],['ia3','(IA)³',N*(2*kvd+ff)],['prompt','Prompt tuning (input only)',s.p*d]];
    const lg=v=>100*Math.log10(Math.max(v,10))/Math.log10(tot);
    out.innerHTML=rows.map(([k,l,v])=>barRow(k===cur||(cur==='qlora'&&k==='lora')||(cur==='prefix'&&k==='prompt')?`<b>${l}</b>`:l,fmtN(v),lg(v),k===cur||(cur==='qlora'&&k==='lora')?C.purple:'#6b5a8a',(100*v/tot).toFixed(v/tot<.001?4:2)+' %')).join('')+`<p class="mut" style="font-size:12.5px;margin-top:8px">Bars are on a log scale (each step right is ×10). Base model: ${fmtN(tot)} parameters — ${fmtB(tot*2)} at 16-bit, ${fmtB(tot*.5)} at 4-bit.</p>`;
  });
}
/* ---- fading memory of recurrent mixers ---- */
function decayWidget(el,name,sym){
  widget(el,`How long does ${name} remember?`,{a:0.9},[['a',`Retention per token ${sym}`,0.5,0.995,0.005,v=>(+v).toFixed(3)]],(out,s)=>{
    const n=40,hl=Math.log(.5)/Math.log(s.a);
    out.innerHTML=`<div class="wbars">${Array.from({length:n},(_,i)=>{const age=n-1-i,w=Math.pow(s.a,age);return `<i style="height:${(100*w).toFixed(1)}%;background:${C.orange}" title="${age} tokens ago: ${(100*w).toFixed(1)}%"></i>`;}).join('')}</div>
    <div class="leg" style="margin-top:6px"><span style="--c:${C.orange}">share of a past token still in the state (left = 39 tokens ago, right = now)</span></div>
    <div class="kv" style="margin-top:10px"><span>Half-life</span><span>${hl.toFixed(1)} tokens</span><span>After 100 tokens</span><span>${(100*Math.pow(s.a,100)).toFixed(3)} % left</span><span>Attention, for comparison</span><span>every past token is available exactly, at O(T) cost per step</span></div>
    <p class="mut" style="font-size:12.5px">Real models learn a different retention for every channel and make it depend on the input, so some channels hold information for thousands of tokens while others react quickly.</p>`;
  });
}
const fnv1a=(str,mod)=>{let h=2166136261;for(const ch of str){h^=ch.codePointAt(0);h=Math.imul(h,16777619)>>>0;}return h%mod;};
const WIDGETS={
  mla:el=>widget(el,'KV cache: MLA against the other attention types — on the configured model',{dc:Math.max(16,Math.round(S.d/8)),e:15},[['dc','Latent size d_c',16,2048,16],['e','Context length',11,20,1,v=>(2**v).toLocaleString()+' tokens']],(out,s)=>{
    const H=S.H,dh=dHead(),N=S.N,L=2**s.e,by=S.bytes,dR=Math.max(2,Math.round(dh/2)),g=(S.Hkv>1&&S.Hkv<H)?S.Hkv:Math.max(1,Math.round(H/4));
    const rows=[['MHA — K and V for all '+H+' heads',2*H*dh,ROLE.mask],['GQA — '+g+' K/V heads',2*g*dh,ROLE.mask],['MQA — one K/V head',2*dh,ROLE.mask],['MLA — latent '+s.dc+' + RoPE key '+dR,s.dc+dR,ROLE.lora]],mx=rows[0][1];
    out.innerHTML=rows.map(([l,n,c],i)=>barRow(i===3?`<b>${l}</b>`:l,fmtB(n*N*L*by),100*n/mx,c,n.toLocaleString()+' / token·layer')).join('')+`<div class="kv" style="margin-top:10px"><span>MLA vs MHA</span><span><b>${(mx/(s.dc+dR)).toFixed(1)}×</b> smaller cache</span><span>Equivalent to GQA with</span><span>${((s.dc+dR)/(2*dh)).toFixed(2)} K/V heads — but every one of the ${H} heads keeps its own keys and values</span><span>Rank of K and V</span><span>at most ${s.dc} of ${H*dh}</span></div><p class="mut" style="font-size:12.5px;margin-top:8px">${N} layers, ${by} bytes per number. DeepSeek-V2 uses d = 5120, 128 heads of 128, d_c = 512 and a 64-dim RoPE key: 576 numbers per token and layer instead of 32,768.</p>`;
  }),
  fullft:el=>peftWidget(el,'fullft'),dora:el=>peftWidget(el,'dora'),adapters:el=>peftWidget(el,'adapters'),prefix:el=>peftWidget(el,'prefix'),ia3:el=>peftWidget(el,'ia3'),
  lora:el=>{el.innerHTML='<div id="lw1"></div><div id="lw2"></div>';const d=()=>{$('#lw1').innerHTML=loraWidget();};d();
    $('#lw1').addEventListener('input',e=>{const k=e.target.dataset.k;if(k){LR[k]=+e.target.value;e.target.previousElementSibling.textContent=e.target.value;}});
    $('#lw1').addEventListener('change',e=>{const k=e.target.dataset.t;if(k)LR.t[k]=e.target.checked;d();});peftWidget($('#lw2'),'lora');},
  qlora:el=>WIDGETS.lora(el),
  gdn:el=>decayWidget(el,'a Gated DeltaNet state','α'),ssm:el=>decayWidget(el,'a state-space model','Ā'),rwkv:el=>decayWidget(el,'RWKV','e⁻ʷ'),
  qsa:el=>widget(el,'How much of the context is actually attended to?',{e:17},[['e','Context length',11,20,1,v=>(2**v).toLocaleString()+' tokens']],(out,s)=>{
    const T=2**s.e,blocks=T/4,sel=Math.min(blocks,512),frac=sel/blocks,r=rng(7+s.e),cells=256;
    const on=new Set();while(on.size<Math.max(1,Math.round(cells*frac)))on.add(Math.floor(cells*Math.pow(r(),0.6)));
    out.innerHTML=`<div class="wgrid">${Array.from({length:cells},(_,i)=>`<i style="background:${on.has(cells-1-i)?C.red:'var(--off)'}"></i>`).join('')}</div>
    <div class="leg" style="margin-top:6px"><span style="--c:${C.red}">selected blocks</span><span style="--c:var(--off)">skipped (each cell ≈ ${Math.max(1,Math.round(blocks/cells)).toLocaleString()} blocks; oldest at top-left, newest at bottom-right)</span></div>
    <div class="kv" style="margin-top:10px"><span>Blocks of 4 tokens</span><span>${blocks.toLocaleString()}</span><span>Blocks kept per query</span><span>${sel} → ${Math.min(T,sel*4+3).toLocaleString()} tokens (${(100*frac).toFixed(frac<.01?2:1)} % of the context)</span>
    <span>Dense attention would read</span><span>${T.toLocaleString()} keys of 256 × 2 heads per query</span><span>Sparse attention reads</span><span>${blocks.toLocaleString()} block keys of 128 for scoring, then ${Math.min(T,2051).toLocaleString()} full keys</span></div>`;
  }),
  ngram:el=>widget(el,'What does the lookup key look like?',{t:'the quick brown fox jumps over the lazy dog'},[],(out,s)=>{
    const w=s.t.match(/[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu)||[];
    out.innerHTML=`<div style="overflow-x:auto"><table><thead><tr><th>t</th><th>token</th><th>bigram key</th><th>trigram key</th><th>rows (3 of 16 shown)</th></tr></thead><tbody>${w.slice(0,12).map((x,i)=>{const bi=i>0?w[i-1]+' '+x:'—',tri=i>1?w[i-2]+' '+w[i-1]+' '+x:'—';
      return `<tr><td class="n">${i}</td><td><b>${esc(x)}</b></td><td class="mono">${esc(bi)}</td><td class="mono">${esc(tri)}</td><td class="mono">${i>0?[0,1].map(j=>fnv1a(bi+'#'+j,2e7).toLocaleString()).join(' · ')+(i>1?' · '+fnv1a(tri+'#8',2e7).toLocaleString():''):'—'}</td></tr>`;}).join('')}</tbody></table></div>
    <p class="mut" style="font-size:12.5px;margin-top:8px">Row numbers here are computed with FNV-1a (a standard public hash) modulo 20,000,000 rows, with the head number appended to the key; Qwen’s own hash functions are not published. The point: the same two or three tokens always hit the same rows, whatever the wider context — which is why the result is gated by the hidden state before it is used.</p>`;
  },'<label class="f" style="flex:1;min-width:260px">Text<input type="text" data-k="t" value="the quick brown fox jumps over the lazy dog" style="width:100%"></label>'),
  difflm:el=>widget(el,'Unmasking, step by step',{s:0},[['s','Refinement step',0,6,1]],(out,s)=>{
    const w=['The','cat','sat','quietly','on','the','warm','mat','.'],ord=[0,8,5,2,4,1,7,3,6],shown=new Set(ord.slice(0,Math.round(w.length*s.s/6)));
    out.innerHTML=`<div class="seq">${w.map((x,i)=>`<span class="tk2 ${shown.has(i)?'g':''}" style="${shown.has(i)?'':'opacity:.55'}">${shown.has(i)?x:'[MASK]'}</span>`).join('')}</div><p class="mut" style="font-size:13px">${shown.size} of ${w.length} tokens fixed after ${s.s} step${s.s===1?'':'s'}. Tokens appear in order of confidence, not left to right, and several are decided in each step.</p>`;
  }),
  jev:el=>widget(el,'One pass, typed answers — set the scores yourself',{q:6,ty:'enum',a:2,b:1,c:0,d:-1},[['q','Questions in the request',1,20,1],['a','score of option 1',-4,4,0.1],['b','score of option 2',-4,4,0.1],['c','score of option 3',-4,4,0.1],['d','score of option 4',-4,4,0.1]],(out,s)=>{
    const z=[s.a,s.b,s.c,s.d],bin=s.ty==='bin',p=bin?[1/(1+Math.exp(-s.a)),1-1/(1+Math.exp(-s.a))]:softmax(z),L=bin?['yes','no']:s.ty==='ord'?['level 1','level 2','level 3','level 4']:['option 1','option 2','option 3','option 4'];
    const ev=s.ty==='ord'?p.reduce((t,v,i)=>t+v*(i+1),0):null;
    out.innerHTML=p.map((v,i)=>barRow(L[i],(100*v).toFixed(1)+' %',100*v,PAL[(i+2)%8])).join('')+`<div class="kv" style="margin-top:10px"><span>How the numbers are made</span><span>${bin?'P(yes) = sigmoid(score of option 1)':'softmax over the four scores'}</span>${ev!=null?`<span>Expected score</span><span>${ev.toFixed(2)}</span>`:''}<span>Confidence</span><span>${(100*Math.max(...p)).toFixed(1)} %</span><span>Forward passes here</span><span><b>1</b> for all ${s.q} questions</span><span>An LLM writing the answers</span><span>one pass per generated token, for every answer</span></div><p class="mut" style="font-size:12.5px">The scores are yours — move the sliders. Whatever they are, the output can only be one of the declared options with a probability each; that is what “cannot hallucinate” refers to, and it says nothing about the chosen option being correct.</p>`;
  },`<div class="f">Question type<div class="seg">${[['bin','Yes / no'],['enum','Choice'],['ord','Score 1–4']].map(([v,t])=>`<button data-seg="ty" data-v="${v}" class="${v==='enum'?'on':''}">${t}</button>`).join('')}</div></div>`),
  mtp:el=>widget(el,'How much faster?',{n:4,a:0.75},[['n','Draft length',1,8,1],['a','Chance each draft token is right',0.3,0.98,0.01,v=>(100*v).toFixed(0)+' %']],(out,s)=>{
    let acc=0;for(let i=1;i<=s.n;i++)acc+=Math.pow(s.a,i);const per=acc+1;
    out.innerHTML=`<div class="seq">${Array.from({length:s.n},(_,i)=>`<span class="tk2 ${i<Math.round(acc)?'g':''}" style="${i<Math.round(acc)?'':'opacity:.45;text-decoration:line-through'}">draft ${i+1}</span>`).join('')}<span class="tk2 n">+ 1 from the main model</span></div>
    <div class="kv"><span>Expected accepted drafts</span><span>${acc.toFixed(2)}</span><span>Tokens per main-model pass</span><span><b>${per.toFixed(2)}</b> instead of 1</span><span>Speed-up (ignoring draft cost)</span><span>${per.toFixed(2)}×</span></div><p class="mut" style="font-size:12.5px">A draft is accepted only if all drafts before it were accepted, so the gain flattens as drafts get longer.</p>`;
  }),
  multimodal:el=>widget(el,'How many tokens is an image?',{s:448,p:14},[['s','Image side (pixels)',224,1344,56],['p','Patch size',14,32,2]],(out,s)=>{
    const g=Math.floor(s.s/s.p),n=g*g,show=Math.min(g,40);
    out.innerHTML=`<div class="wgrid" style="grid-template-columns:repeat(${show},1fr);max-width:${show*9}px">${Array.from({length:show*show},(_,i)=>`<i style="background:${PAL[(i*7+Math.floor(i/show)*3)%8]};opacity:.75"></i>`).join('')}</div>
    <div class="kv" style="margin-top:10px"><span>Patch grid</span><span>${g} × ${g}</span><span>Image tokens</span><span><b>${n.toLocaleString()}</b></span><span>Equivalent text</span><span>≈ ${Math.round(n*0.75).toLocaleString()} words of context</span><span>Share of an 8k context</span><span>${(100*n/8192).toFixed(0)} %</span></div><p class="mut" style="font-size:12.5px">Many models add a resampler or pooling step to cut this number before the tokens reach the language model.</p>`;
  }),
  imgdiff:el=>widget(el,'The forward process: adding noise',{t:0.5},[['t','Timestep t',0,1,0.02,v=>(+v).toFixed(2)]],(out,s)=>{
    out.innerHTML='';const [c,x]=canvasOf(out,220,220),ab=Math.cos(s.t*Math.PI/2)**2,im=x.createImageData(220,220),r=rng(11);
    for(let i=0;i<220;i++)for(let j=0;j<220;j++){const dx=j-110,dy=i-110,inC=dx*dx+dy*dy<60*60,k=(i*220+j)*4;
      const base=inC?[214,69,155]:[30+j*.5,90+i*.4,200],g=()=>(r()+r()+r()-1.5)*2*90;
      for(let ch=0;ch<3;ch++)im.data[k+ch]=clamp(Math.sqrt(ab)*(base[ch]-128)+Math.sqrt(1-ab)*g()+128,0,255);im.data[k+3]=255;}
    x.putImageData(im,0,0);
    out.insertAdjacentHTML('beforeend',`<div class="kv" style="margin-top:10px"><span>Signal kept √ᾱ</span><span>${Math.sqrt(ab).toFixed(2)}</span><span>Noise added √(1−ᾱ)</span><span>${Math.sqrt(1-ab).toFixed(2)}</span></div><p class="mut" style="font-size:12.5px">Generation runs this in reverse: start at t = 1 (pure noise) and let the denoiser remove a little at every step.</p>`);
  }),
  vqtok:el=>widget(el,'Vector quantisation: snap to the nearest code',{k:8},[['k','Codebook size K',2,32,1]],(out,s)=>{
    out.innerHTML='';const [c,x]=canvasOf(out,420,240),r=rng(5),cen=Array.from({length:32},()=>[20+r()*380,20+r()*200]).slice(0,s.k),r2=rng(9);
    x.fillStyle=cssVar('--bg');x.fillRect(0,0,420,240);
    for(let i=0;i<420;i++){const px=r2()*420,py=r2()*240;let b=0,bd=1e9;cen.forEach(([cx,cy],j)=>{const dd=(px-cx)**2+(py-cy)**2;if(dd<bd){bd=dd;b=j;}});x.fillStyle=PAL[b%8];x.globalAlpha=.75;x.fillRect(px-2,py-2,4,4);}
    x.globalAlpha=1;cen.forEach(([cx,cy],j)=>{x.fillStyle=PAL[j%8];x.strokeStyle=cssVar('--txt');x.lineWidth=2;x.beginPath();x.arc(cx,cy,7,0,7);x.fill();x.stroke();});
    out.insertAdjacentHTML('beforeend',`<div class="kv" style="margin-top:10px"><span>Each vector becomes</span><span>one integer in 0 … ${s.k-1}</span><span>Bits per token</span><span>${Math.log2(s.k).toFixed(1)}</span></div><p class="mut" style="font-size:12.5px">Small dots are encoder outputs, circles are codebook entries. A dot takes the colour — and the token ID — of its nearest circle. More codes mean less information lost.</p>`);
  }),
  vaegan:el=>widget(el,'The VAE latent: z = μ + σ·ε',{m:0.8,s:0.6},[['m','Mean μ',-2,2,0.1,v=>(+v).toFixed(1)],['s','Spread σ',0.1,2,0.05,v=>(+v).toFixed(2)]],(out,st)=>{
    out.innerHTML='';const [c,x]=canvasOf(out,420,180),r=rng(3),bins=new Array(42).fill(0);
    for(let i=0;i<3000;i++){const e=Math.sqrt(-2*Math.log(r()+1e-9))*Math.cos(6.2832*r()),z=st.m+st.s*e,b=Math.floor((z+4.2)/8.4*42);if(b>=0&&b<42)bins[b]++;}
    const mx=Math.max(...bins);x.fillStyle=cssVar('--bg');x.fillRect(0,0,420,180);
    bins.forEach((v,i)=>{x.fillStyle=C.blue;x.fillRect(i*10+1,170-150*v/mx,8,150*v/mx);});
    x.strokeStyle=cssVar('--txt');x.lineWidth=2;x.beginPath();for(let i=0;i<=420;i++){const z=i/420*8.4-4.2,y=170-150*Math.exp(-z*z/2)*(st.s<1?st.s:1);i?x.lineTo(i,y):x.moveTo(i,y);}x.stroke();
    const kl=.5*(st.m*st.m+st.s*st.s-1-Math.log(st.s*st.s));
    out.insertAdjacentHTML('beforeend',`<div class="kv" style="margin-top:10px"><span>KL to the standard normal</span><span><b>${kl.toFixed(3)}</b></span></div><p class="mut" style="font-size:12.5px">Bars: samples of the encoder’s latent for one input. Line: the standard normal prior. The KL term in the loss pulls the bars toward the line (μ → 0, σ → 1), while the reconstruction term pulls them apart so that different inputs stay distinguishable.</p>`);
  }),
  residual4:el=>widget(el,'Reading four branches through a gate',{a:0.7,b:0.2,c:0.6,d:0.1},[['a','gate, branch 1',0,1,0.05],['b','gate, branch 2',0,1,0.05],['c','gate, branch 3',0,1,0.05],['d','gate, branch 4',0,1,0.05]],(out,s)=>{
    const g=[s.a,s.b,s.c,s.d],z=g.reduce((p,q)=>p+q,0)||1;
    out.innerHTML=`<div class="stack" style="height:22px">${g.map((v,i)=>`<i style="flex:${Math.max(v,.001)};background:${PAL[i*2]}"></i>`).join('')}</div><div class="leg">${g.map((v,i)=>`<span style="--c:${PAL[i*2]}">branch ${i+1}: ${(100*v/z).toFixed(0)} % of what the sub-layer sees</span>`).join('')}</div><p class="mut" style="font-size:12.5px;margin-top:8px">In the model the gate is computed from the data and differs per element, so a sub-layer can take syntax from one branch and facts from another. A single residual stream is the special case of one branch with gate 1.</p>`;
  }),
  qnext:el=>{el.innerHTML=`<div class="card pad"><h3>Where do the parameters go for one token? — published figures only</h3><div style="margin-top:10px">${barRow('Main model',fmtN(125e9),100,'#6b5a8a')+barRow('Active for one token',fmtN(6e9),100*6/125,ROLE.w,'≈ 4.8 %')+barRow('N-gram table',fmtN(51.2e9),100*51.2/125,ROLE.lora,'host RAM')+barRow('Read from the table per token',fmtN(16*160),0.5,ROLE.k,'16 rows × 160')+barRow('Draft (MTP) model',fmtN(4e9),100*4/125,ROLE.attn)}</div>
    <div class="kv" style="margin-top:10px"><span>Experts used per token</span><span>10 routed + 1 shared, of 512</span><span>Layers that keep a KV cache</span><span>12 of 48 (the sparse-attention layers)</span></div></div>`;}
};
