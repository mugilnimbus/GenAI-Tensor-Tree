/* ================= views ================= */
let REG=[];
const reg=steps=>{steps.forEach(s=>{s._i=REG.push(s)-1;});return steps;};
const stageTitle=(stage,scope,sn)=>`<div class="stage">${stage}${sn?`<small>${sn}</small>`:scope==='layer'?`<small>inside every layer · × ${K.N}</small>`:scope==='in'?'<small>once, before layer 1</small>':'<small>once, after the last layer</small>'}</div>`;
function archHead(a,steps){
  const mf=a.meta(K);
  if(mf.facts)return `<div class="archhead"><h2>${a.name}</h2>${BACK[a.id]?`<a class="back" href="#${BACK[a.id]}">concept page →</a>`:''}<a class="back" href="#graph/${a.id}">tensor graph →</a><p class="mut" style="font-size:13.5px">${a.desc}</p><div class="kv">${mf.facts.map(([k,v])=>`<span>${k}</span><span>${v}</span>`).join('')}<span>KV cache / token</span><span>${mf.kvText}</span></div></div>`;
  const t=totals(steps),m=a.meta(K),dh=K.d/K.H,b=t.big;
  return `<div class="archhead"><h2>${a.name}</h2>${BACK[a.id]?`<a class="back" href="#${BACK[a.id]}">concept page →</a>`:''}<a class="back" href="#graph/${a.id}">tensor graph →</a><p class="mut" style="font-size:13.5px">${a.desc}</p><div class="kv">
    <span>Parameters / layer</span><span>${fN(t.lp)}${t.act!==t.lp?` (active per token ${fN(t.act)})`:''}</span>
    <span>Whole model</span><span>${fN(t.total)}  = ${K.N} layers + ${fN(t.io)} embedding / head${m.cross?' (decoder side only)':''}</span>
    <span>FLOPs / layer</span><span>${fF(t.lf)}</span>
    <span>Largest activation</span><span>${String(b.n).replace(/<[^>]+>/g,'')} ${b.rep?`${b.rep} × `:''}${shp(b)} = ${fB(b.r*b.c*(b.rep||1)*2)} at 2 B</span>
    <span>KV cache / token</span><span>${m.kvText?m.kvText:m.nocache?'— (not autoregressive)':fB(2*K.N*m.kv*dh*2)+`  (2 · ${K.N} · ${m.kv} · ${dh} · 2 B)`}</span></div></div>`;
}
function flowOne(steps){
  let h='',no=0,stage=null,scope=null,inHead=false;
  const closeHead=()=>{if(inHead){h+='</div>';inHead=false;}};
  steps.forEach(s=>{
    if(s.stage!==stage||s.scope!==scope){closeHead();stage=s.stage;scope=s.scope;h+=stageTitle(stage,scope,s.sn);}
    if(s.head&&!inHead){h+=`<div class="headbox"><div class="cap">For each of the ${K.H} heads — one head shown:<label>head <b class="mono">${clamp(K.h,0,K.H-1)}</b> <input type="range" class="hsel" min="0" max="${K.H-1}" value="${clamp(K.h,0,K.H-1)}" aria-label="Head index"></label><span class="mut" style="font-weight:500">per-head tensors are ${K.d/K.H} wide, not ${K.d}</span></div>`;inHead=true;}
    if(!s.head)closeHead();
    h+=stepHTML(s,++no,false);
  });
  closeHead();return h;
}
function flowCmp(A,B){
  const ma=new Map(A.map(s=>[s.key,s])),mb=new Map(B.map(s=>[s.key,s])),na=new Map(A.map((s,i)=>[s.key,i+1])),nb=new Map(B.map((s,i)=>[s.key,i+1]));
  let h='<div class="cmp">',stage=null,scope=null;
  MASTER.filter(k=>ma.has(k)||mb.has(k)).forEach(k=>{
    const a=ma.get(k),b=mb.get(k),r=a||b;
    if(r.stage!==stage||r.scope!==scope){stage=r.stage;scope=r.scope;h+=stageTitle(stage,scope,r.sn);}
    const diff=!a||!b||a.sig!==b.sig,cell=(s,n,o)=>s?stepHTML(s,n.get(k),diff):`<div class="step none">— no “${o.title}” step in this architecture —</div>`;
    h+=cell(a,na,b)+cell(b,nb,a);
  });
  return h+'</div>';
}
function renderMain(){if(!$('#sf-main'))return;K.px=300;
  REG=[];const a=archOf(K.a),sa=reg(a.build(K));
  if(K.mode==='one'){$('#sf-main').innerHTML=archHead(a,sa)+flowOne(sa);}
  else{const b=archOf(K.b),sb2=reg(b.build(K));
    if(sa.seq||sb2.seq){$('#sf-main').innerHTML=`<p style="margin:10px 0 0" class="mut">These two are different kinds of model, so their steps do not line up one to one; each column shows its own sequence.</p><div class="cols two"><div>${archHead(a,sa)}${flowOne(sa)}</div><div>${archHead(b,sb2)}${flowOne(sb2)}</div></div>`;renderBig();return;}
    const nd=MASTER.filter(k=>{const x=sa.find(s=>s.key===k),y=sb2.find(s=>s.key===k);return (x||y)&&(!x||!y||x.sig!==y.sig);}).length;
    $('#sf-main').innerHTML=`<div class="cols two">${archHead(a,sa)}${archHead(b,sb2)}</div><p style="margin:10px 0 0"><span class="tag d">differs</span> <b>${nd}</b> step${nd===1?'':'s'} differ between these two; everything without an amber outline is identical in formula and tensor shape.</p>`+flowCmp(sa,sb2);}
  renderBig();
}
function renderBig(){
  const get=(S,k)=>S.find(s=>s.key===k),T=s=>s?s.expr.filter(x=>typeof x==='object'):[],last=s=>{const t=T(s);return t.length?shp(t[t.length-1]):'—';};
  const rowsDef=[
    ['Normalisation',(S,m)=>m.norm],['Position information',(S,m)=>m.pos],['Attention mask',(S,m)=>m.mask],
    ['Rows through a layer',S=>String(T(get(S,'emb')).pop().r)],
    ['Q (all heads)',S=>T(get(S,'split'))[0].shape],['K, V (all heads)',S=>T(get(S,'split'))[1].shape],
    ['W_K , W_V',S=>{const t=T(get(S,'k')||get(S,'kc'));return t[1]?shp(t[1]):'—';}],['Score matrix / head',S=>last(get(S,'scores'))+(get(S,'xscores')?' · cross '+last(get(S,'xscores')):'')],
    ['MLP',(S,m)=>m.mlp],['MLP hidden',S=>last(get(S,'mul')||get(S,'actf'))],
    ['Params / layer',(S,m)=>{if(m.facts)return m.facts[0][1];const t=totals(S);return fN(t.lp)+(t.act!==t.lp?` (active ${fN(t.act)})`:'');}],
    ['FLOPs / layer',(S,m)=>m.facts?'—':fF(totals(S).lf)],
    ['KV cache / token',(S,m)=>m.kvText?m.kvText.split('  ')[0]:m.nocache?'—':fB(2*K.N*m.kv*(K.d/K.H)*2)],
    ['Final output',S=>last(get(S,'vsoft'))]
  ];
  const built=ARCH.filter(a=>a.table!==false).map(a=>({a,S:a.build(K),m:a.meta(K)})),sel=[K.a].concat(K.mode==='cmp'?[K.b]:[]);
  $('#big').innerHTML=`<thead><tr><th></th>${built.map(x=>`<th class="${sel.includes(x.a.id)?'sel':''}" data-arch="${x.a.id}" style="cursor:pointer" tabindex="0">${x.a.short}</th>`).join('')}</tr></thead><tbody>`+rowsDef.map(([l,f])=>{
    const v=built.map(x=>f(x.S,x.m));return `<tr><td>${l}</td>${v.map((c,i)=>`<td class="${i&&c!==v[0]?'dv':''}">${c}</td>`).join('')}</tr>`;}).join('')+'</tbody>';
}
function renderBar(){$('#sf-bar').innerHTML=barHTML();}
function mountS(){
  renderBar();renderMain();
  const bar=$('#sf-bar');
  bindBar(bar,()=>{renderBar();renderMain();});
  const main=$('#sf-main');
  main.addEventListener('click',e=>{if(e.target.closest('.hsel,.demo'))return;const el=e.target.closest('.step[data-i]');if(el)openDemo(el,REG[+el.dataset.i]);});
  main.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList.contains('step')&&e.target.dataset.i){e.preventDefault();openDemo(e.target,REG[+e.target.dataset.i]);}});
  main.addEventListener('change',e=>{if(e.target.classList.contains('hsel')){K.h=+e.target.value;const y=scrollY;renderMain();scrollTo(0,y);}});
  main.addEventListener('input',e=>{if(e.target.classList.contains('hsel'))e.target.previousElementSibling.textContent=e.target.value;});
  const pick=e=>{const th=e.target.closest('th[data-arch]');if(!th)return;if(K.mode==='cmp')K.b=th.dataset.arch;else K.a=th.dataset.arch;renderBar();renderMain();$('#sf-main').scrollIntoView({behavior:'smooth'});};
  $('#big').addEventListener('click',pick);$('#big').addEventListener('keydown',e=>{if(e.key==='Enter')pick(e);});
}

TFX.s=()=>REG;
VIEWS.steps={show(root,sub){if(ARCH.some(a=>a.id===sub))K.a=sub;
  root.innerHTML=secHead('Every Operation, Step by Step','One architecture, or two side by side. Each card is one matrix operation with its formula and to-scale tensors; click a card for a worked numeric example.')+'<div class="card pad toolbar" id="sf-bar"></div>'+LEGEND+'<div id="sf-main" class="tf"></div><h3 class="gh">All architectures at a glance</h3><p class="mut" style="font-size:13.5px">Computed from the configuration above. Highlighted cells differ from the first column; click a column header to open that architecture.</p><div class="tblw"><table class="big" id="big"></table></div>';
  mountS();},update(){renderBar();renderMain();}};
