/* ================= tensor graph — vertical layout, hover popup, tracing ================= */
let ctx=null,PICK=null,IBOUND=false,VB=false;
function ensureOverlays(){if($('#detail'))return;document.body.insertAdjacentHTML('beforeend','<aside id="detail" class="tf" hidden aria-label="Step details"></aside><div id="tip" hidden role="tooltip"></div><div id="menu" hidden role="menu"></div><div id="chip" hidden role="status"></div>');}
function hideOverlays(){['detail','tip','menu','chip'].forEach(i=>{const e=$('#'+i);if(e)e.hidden=true;});}
let GR=[];
const KIND={res:'residual stream',act:'activation',w:'learned weight',mask:'scores / mask',prob:'probabilities',tok:'token IDs'};
const nodeOf=el=>{const g=+el.closest('.gbox').dataset.g;return {g,n:GR[g].N[+el.dataset.n]};};
function tipHTML(g,n){
  const N=GR[g].N,s=n.s,t=n.t,el=t.r*t.c*(t.rep||1),cons=N.filter(m=>m.ins.includes(n));
  const li=(m,x)=>`<li><b style="color:${tcol(m.t)}">${m.t.n}</b><span>${shp(m.t)}${x||''}</span></li>`;
  return `<div class="th"><b style="color:${tcol(t)}">${t.n}</b><span>${ROLE_NAME[roleOf(t)]}</span></div>
  <div class="tk"><span>Shape</span><span>${shp(t)}${t.rep?` × ${t.rep} heads`:''}</span><span>Elements</span><span>${fN(el)}</span><span>Memory</span><span>${fB(el*2)} at 2 B${t.kind==='w'?` · ${fN(t.r*t.c)} parameters`:''}</span>
  <span>${n.src?'Used by':'Made by'}</span><span>${s.title}</span><span>Formula</span><span>${s.fx}</span>
  <span>Runs</span><span>${s.sn?s.sn:s.scope==='layer'?`in every layer (× ${K.N})`:s.scope==='in'?'once, before layer 1':'once, after the last layer'}${s.head?` · per head (× ${K.H})`:''}</span>${s.flops?`<span>Step cost</span><span>${fF(s.flops)}</span>`:''}</div>
  ${n.ins.length?`<div class="tl">Inputs — combined by “${esc(n.op||'')}”</div><ul>${n.ins.map(m=>li(m)).join('')}</ul>`:''}
  ${cons.length?`<div class="tl">Feeds</div><ul>${cons.map(m=>li(m,' · '+esc(m.op||''))).join('')}</ul>`:''}
  ${s.note?`<p>${s.note}</p>`:''}${formulaHTML(stepFormulas(s))}${whyHTML(explainStep(s),true)}<div class="tf">Click: worked example · Right-click: trace, find, copy</div>`;
}
function clearTrace(){$$('.gbox.tr').forEach(b=>{b.classList.remove('tr');$$('.tn',b).forEach(x=>x.classList.remove('tn'));$$('.ge.on',b).forEach(x=>x.classList.remove('on'));});const c=$('#chip');if(c)c.hidden=true;}
function trace(g,n,dir){
  clearTrace();const N=GR[g].N,S=new Set([n.id]);
  if(dir!=='down'){const st=[n];while(st.length){const m=st.pop();m.ins.forEach(i=>{if(!S.has(i.id)){S.add(i.id);st.push(i);}});}}
  if(dir!=='up'){const st=[n];while(st.length){const m=st.pop();N.forEach(q=>{if(q.ins.includes(m)&&!S.has(q.id)){S.add(q.id);st.push(q);}});}}
  const box=$(`.gbox[data-g="${g}"]`);box.classList.remove('foc');box.classList.add('tr');
  $$('.gn',box).forEach(x=>x.classList.toggle('tn',S.has(+x.dataset.n)));
  $$('.ge',box).forEach(p=>p.classList.toggle('on',S.has(+p.dataset.a)&&S.has(+p.dataset.b)));
  const c=$('#chip'),nm='<b>'+esc(txt(n.t.n))+'</b>';c.hidden=false;
  c.innerHTML=`<span>${dir==='up'?'Everything '+nm+' depends on':dir==='down'?'Everything '+nm+' feeds':'Full path through '+nm} · ${S.size} tensors</span><button>Clear</button>`;
}
function interactions(main,pick){
  const tip=$('#tip'),menu=$('#menu'),chip=$('#chip');PICK=pick;
  const place=(el,x,y)=>{const w=el.offsetWidth,h=el.offsetHeight;el.style.left=Math.max(8,Math.min(x,innerWidth-w-8))+'px';el.style.top=Math.max(8,y+h+8>innerHeight?y-h-28:y)+'px';};
  const closeMenu=()=>{menu.hidden=true;ctx=null;};
  const say=h=>{chip.hidden=false;chip.innerHTML='<span>'+h+'</span><button>OK</button>';};
  main.addEventListener('mouseover',e=>{
    const el=e.target.closest('.gn');if(!el||!menu.hidden){tip.hidden=true;return;}
    const id=el.closest('.gbox').dataset.g+':'+el.dataset.n;
    if(tip.dataset.n!==id){const {g,n}=nodeOf(el);tip.innerHTML=tipHTML(g,n);tip.dataset.n=id;}
    tip.hidden=false;place(tip,e.clientX+18,e.clientY+18);});
  main.addEventListener('mousemove',e=>{if(!tip.hidden)place(tip,e.clientX+18,e.clientY+18);});
  main.addEventListener('mouseleave',()=>{tip.hidden=true;});
  main.addEventListener('focusin',e=>{const el=e.target.closest('.gn');if(!el||!el.matches(':focus-visible'))return;const {g,n}=nodeOf(el),r=el.getBoundingClientRect();tip.innerHTML=tipHTML(g,n);tip.dataset.n='';tip.hidden=false;place(tip,r.right+12,r.top);});
  main.addEventListener('focusout',()=>{tip.hidden=true;});
  main.addEventListener('contextmenu',e=>{
    const el=e.target.closest('.gn');if(!el)return;e.preventDefault();tip.hidden=true;
    const {g,n}=nodeOf(el);ctx={el,g,n};
    menu.innerHTML=`<div class="mh">${esc(txt(n.t.n))}  ${shp(n.t)}</div>
      <button role="menuitem" data-m="up">⬆ Trace everything this depends on</button><button role="menuitem" data-m="down">⬇ Trace everything this feeds</button><button role="menuitem" data-m="both">⬍ Trace the full path through it</button><hr>
      ${GR.length>1?'<button role="menuitem" data-m="find">⇄ Find this step in the other architecture</button>':''}<button role="menuitem" data-m="ex">▦ Open the worked numeric example</button><button role="menuitem" data-m="copy">⧉ Copy name, shape and formula</button><hr><button role="menuitem" data-m="clear">✕ Clear tracing</button>`;
    menu.hidden=false;const r=el.getBoundingClientRect(),kb=e.clientX===0&&e.clientY===0;place(menu,kb?r.right:e.clientX,kb?r.top:e.clientY);$('button',menu).focus();
  });
  if(IBOUND)return;IBOUND=true;
  menu.addEventListener('click',e=>{
    const b=e.target.closest('button');if(!b||!ctx)return;const {el,g,n}=ctx,m=b.dataset.m;closeMenu();
    if(m==='up'||m==='down'||m==='both')trace(g,n,m);
    else if(m==='clear')clearTrace();
    else if(m==='ex')PICK(el);
    else if(m==='copy'){const s=`${txt(n.t.n)} ${shp(n.t)} — ${n.s.title}: ${txt(n.s.fx)}`;(navigator.clipboard?navigator.clipboard.writeText(s):Promise.reject()).then(()=>say('Copied: '+esc(s)),()=>say('Copy was blocked by the browser: '+esc(s)));}
    else if(m==='find'){
      const N2=GR[1-g].N,t=N2.find(q=>q.s.key===n.s.key&&txt(q.t.n)===txt(n.t.n))||N2.find(q=>q.s.key===n.s.key&&!q.src);
      if(!t){say(`The other architecture has no “${esc(n.s.title)}” step.`);return;}
      const o=$(`.gbox[data-g="${1-g}"] .gn[data-n="${t.id}"]`);o.scrollIntoView({block:'center',behavior:'smooth'});
      [el,o].forEach(x=>{x.classList.remove('flash');void x.getBoundingClientRect();x.classList.add('flash');setTimeout(()=>x.classList.remove('flash'),1700);});
    }
  });
  menu.addEventListener('keydown',e=>{const B=$$('button',menu),i=B.indexOf(document.activeElement);
    if(e.key==='ArrowDown'){e.preventDefault();B[(i+1)%B.length].focus();}
    else if(e.key==='ArrowUp'){e.preventDefault();B[(i-1+B.length)%B.length].focus();}
    else if(e.key==='Escape'){const el=ctx&&ctx.el;closeMenu();if(el)el.focus();}});
  document.addEventListener('pointerdown',e=>{if(!menu.hidden&&!e.target.closest('#menu'))closeMenu();});
  window.addEventListener('scroll',()=>{if(!menu.hidden)closeMenu();tip.hidden=true;},{passive:true});
  chip.addEventListener('click',e=>{if(e.target.closest('button')){clearTrace();chip.hidden=true;}});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&menu.hidden)clearTrace();});
}

/* ================= vertical tensor graph ================= */

let REG=[];
const ROWG={k:'q',v:'q',xk:'xq',xv:'xq',up:'gate',qr:'q',vc:'kc',kr:'kc',catk:'catq'},bandOf=k=>MASTER.indexOf(ROWG[k]||k);
const lblW=n=>Math.max(txt(n.t.n).length*7.8,shp(n.t).length*6.4);
function prep(steps,other){
  const N=buildGraph(steps),om=other?new Map(other.map(s=>[s.key,s])):null,flow=N.filter(n=>!n.src),occ=new Set();
  const sq=steps.seq||(other&&other.seq),bo=steps.seq?k=>steps.findIndex(s=>s.key===k):bandOf;
  N.forEach(n=>{n.diff=!!om&&!sq&&(!om.has(n.s.key)||om.get(n.s.key).sig!==n.s.sig);});
  flow.forEach(n=>{n.band=bo(n.s.key);n.fin=n.ins.filter(i=>!i.src);n.srcs=n.ins.filter(i=>i.src);
    let l=n.fin.length?n.fin[0].lane:(n.s.key==='mem'?1:0);while(occ.has(n.band+':'+l))l++;n.lane=l;occ.add(n.band+':'+l);});
  const bands=[...new Set(flow.map(n=>n.band))].sort((a,b)=>a-b),bi=b=>bands.indexOf(b),E=[],gut={};
  const gkey=(b,id)=>{const m=gut[b]||(gut[b]=new Map());if(!m.has(id))m.set(id,m.size);};
  const busy=(lane,b0,b1)=>flow.some(m=>m.lane===lane&&m.band>b0&&m.band<b1);
  flow.forEach(n=>n.fin.forEach(i=>{
    const e={a:i,b:n,type:i.lane===n.lane&&!busy(i.lane,i.band,n.band)?'d':bi(n.band)-bi(i.band)===1?'g':'b'};E.push(e);
    if(e.type!=='d')gkey(n.band,i.id);
    if(e.type==='b'){e.nb=bands[bi(i.band)+1];gkey(e.nb,i.id);e.side=i.t.kind==='res'?'L':'R';}
  }));
  const tr={L:[],R:[]},tm={L:new Map(),R:new Map()};
  E.filter(e=>e.type==='b').sort((p,q)=>p.a.band-q.a.band).forEach(e=>{const T=tr[e.side],M=tm[e.side];
    if(M.has(e.a.id)){e.track=M.get(e.a.id);T[e.track]=Math.max(T[e.track],e.b.band);return;}
    let k=T.findIndex(end=>end<e.nb);if(k<0){k=T.length;T.push(0);}T[k]=e.b.band;e.track=k;M.set(e.a.id,k);});
  flow.forEach(n=>{
    const inE=E.filter(e=>e.b===n),rt=inE.filter(e=>e.type!=='d'),hasD=inE.length>rt.length;let pm=0;
    const isL=e=>e.type==='b'?e.side==='L':e.a.lane<n.lane;
    if(!hasD&&rt.length===1)rt[0].port=0;
    else{const L=rt.filter(isL).sort((p,q)=>q.a.lane-p.a.lane),R=rt.filter(e=>!isL(e)).sort((p,q)=>p.a.lane-q.a.lane);
      L.forEach((e,k)=>{e.port=-10*(k+1);});R.forEach((e,k)=>{e.port=10*(k+1);});pm=10*Math.max(L.length,R.length);}
    n.pw=n.op?Math.max(28,n.op.length*7.2+16,2*pm+18):0;
  });
  const zA={},nZ={},gC={};
  bands.forEach(b=>{const M=flow.filter(n=>n.band===b);gC[b]=gut[b]?gut[b].size:0;nZ[b]=Math.max(30,...M.map(n=>n.h));zA[b]=Math.max(26,...M.flatMap(n=>n.srcs.map(s=>s.h+34)));});
  return {N,flow,E,bands,gut,tr,zA,nZ,gC,label:steps.layerLabel};
}
function mergeBM(Gs){
  const bands=[...new Set(Gs.flatMap(g=>g.bands))].sort((a,b)=>a-b),y={},gH={},zA={},nZ={};let yy=14;
  bands.forEach(b=>{const mx=f=>Math.max(0,...Gs.map(g=>g[f][b]||0));gH[b]=8+9*mx('gC');zA[b]=mx('zA');nZ[b]=mx('nZ');y[b]=yy;yy+=gH[b]+zA[b]+18+nZ[b]+16;});
  return {bands,y,gH,zA,nZ,H:yy+10};
}
function draw(G,BM){
  const {flow,E,tr}=G,lanes=[...new Set(flow.map(n=>n.lane))].sort((a,b)=>a-b),axis={};
  const BL=tr.L.length?tr.L.length*10+14:0,BR=tr.R.length?tr.R.length*10+14:0;let x=BL+8;
  const srcSlot=s=>Math.max(s.w,lblW(s))+14;
  lanes.forEach(l=>{const M=flow.filter(n=>n.lane===l),lab=Math.max(...M.map(lblW))+12,half=Math.max(...M.map(n=>n.w))/2;
    const right=Math.max(half,...M.map(n=>n.pw/2+(n.srcs.length?16+n.srcs.reduce((a,s)=>a+srcSlot(s),0):0)));
    axis[l]=x+lab+half;x=axis[l]+right+22;});
  const W=x+BR+6,H=BM.H;
  flow.forEach(n=>{const by=BM.y[n.band];n.ax=axis[n.lane];n.x=n.ax-n.w/2;n.pcy=by+BM.gH[n.band]+BM.zA[n.band]/2;n.y=by+BM.gH[n.band]+BM.zA[n.band]+18+(BM.nZ[n.band]-n.h)/2;n.cy=n.y+n.h/2;
    let sx=n.ax+n.pw/2+16;n.srcs.forEach(s=>{const sl=srcSlot(s);s.x=sx+(sl-14-s.w)/2;s.y=n.pcy-s.h/2;s.cx=s.x+s.w/2;s.cy=n.pcy;sx+=sl;});});
  const gy=(b,id)=>BM.y[b]+5+9*G.gut[b].get(id);
  let es='',s='',fr='';
  E.forEach(e=>{
    const {a,b}=e,ab=a.y+a.h+2,pt=b.pcy-13,px=b.ax+(e.port||0),c=tcol(a.t);let p;
    if(e.type==='d')p=[[a.ax,ab],[px,pt]];
    else if(e.type==='g'){const g=gy(b.band,a.id);p=[[a.ax,ab],[a.ax,g],[px,g],[px,pt]];}
    else{const g1=gy(e.nb,a.id),g2=gy(b.band,a.id),bx=e.side==='L'?BL-6-e.track*10:W-BR+6+e.track*10;p=[[a.ax,ab],[a.ax,g1],[bx,g1],[bx,g2],[px,g2],[px,pt]];}
    es+=`<path class="ge" data-a="${a.id}" data-b="${b.id}" d="${rpath(p)}" stroke="${c}" marker-end="url(#m-${tcol(a.t).slice(1)})"/>`;
  });
  const shape=(n,c)=>{const t=n.t;let g=`<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" fill="${c}" fill-opacity="${t.mask?.1:.34}" stroke="${c}" stroke-width="1.6"/>`;
    if(t.mask==='causal')g+=`<path d="M${n.x} ${n.y}L${n.x+n.w} ${n.y+n.h}H${n.x}Z" fill="${c}" fill-opacity=".7"/>`;
    if(t.mask==='window'){const f=clamp(t.frac||.2,.02,1);g+=`<path d="M${n.x} ${n.y}L${n.x+n.w} ${n.y+n.h}H${n.x+n.w*(1-f)}L${n.x} ${n.y+n.h*f}Z" fill="${c}" fill-opacity=".75"/>`;}
    if(t.stripes>1){const m=Math.min(t.stripes,Math.floor(n.w/3));let d='';for(let i=1;i<m;i++)d+=`M${(n.x+i*n.w/m).toFixed(1)} ${n.y}v${n.h}`;if(d)g+=`<path d="${d}" stroke="${c}" stroke-width=".8"/>`;}
    if(t.hiRow!=null)g+=`<rect x="${n.x}" y="${t.hiRow==='last'?n.y+n.h-3:n.y}" width="${n.w}" height="3" fill="#d65f48"/>`;
    if(dimPx(t.c)[1]){const q=(n.h+8)/4,cx=n.x+n.w/2;g+=`<path class="brk" d="M${cx-3} ${n.y-4}l6 ${q}l-6 ${q}l6 ${q}l-6 ${q}"/>`;}
    if(dimPx(t.r)[1]){const q=(n.w+8)/4,cy=n.y+n.h/2;g+=`<path class="brk" d="M${n.x-4} ${cy-3}l${q} 6l${q} -6l${q} 6l${q} -6"/>`;}
    return g;};
  const wrapN=(n,inner,hx,hy,hw,hh)=>`<g class="gn${n.diff?' df':''}" data-n="${n.id}" data-i="${n.s._i}" tabindex="0" role="button" aria-label="${esc(txt(n.t.n)+' '+shp(n.t)+' — '+n.s.title)}"><rect class="gh" x="${hx}" y="${hy}" width="${hw}" height="${hh}" rx="8"/>${inner}</g>`;
  flow.forEach(n=>{
    const c=tcol(n.t),lw=lblW(n);let g='';
    if(n.op){g+=`<path class="ge" data-a="${n.id}" data-b="${n.id}" d="M${n.ax} ${n.pcy+11}V${n.y-3}" stroke="${c}" marker-end="url(#m-${tcol(n.t).slice(1)})"/><rect x="${n.ax-n.pw/2}" y="${n.pcy-11}" width="${n.pw}" height="22" rx="11" class="gb"/><text class="gbt" x="${n.ax}" y="${n.pcy+4}" text-anchor="middle">${esc(n.op)}</text>`;}
    g+=shape(n,c)+`<text class="gl" x="${n.x-8}" y="${n.cy-1}" text-anchor="end" fill="${c}">${svgLabel(n.t.n)}</text><text class="gs" x="${n.x-8}" y="${n.cy+12}" text-anchor="end">${esc(shp(n.t))}</text>`;
    s+=wrapN(n,g,n.x-lw-14,Math.min(n.y,n.cy-15)-4,lw+n.w+20,Math.max(n.h,30)+8);
    n.srcs.forEach(q=>{const cq=tcol(q.t);q.diff=n.diff;
      es+=`<path class="ge" data-a="${q.id}" data-b="${n.id}" d="M${q.x-2} ${q.cy}H${n.ax+n.pw/2+3}" stroke="${cq}" marker-end="url(#m-${tcol(q.t).slice(1)})"/>`;
      const qw=Math.max(q.w,lblW(q));
      s+=wrapN(q,shape(q,cq)+`<text class="gl" x="${q.cx}" y="${q.y-5}" text-anchor="middle" fill="${cq}">${svgLabel(q.t.n)}</text><text class="gs" x="${q.cx}" y="${q.y+q.h+12}" text-anchor="middle">${esc(shp(q.t))}</text>`,q.cx-qw/2-4,q.y-18,qw+8,q.h+34);});
  });
  const bEnd=b=>BM.y[b]+BM.gH[b]+BM.zA[b]+18+BM.nZ[b]+12,LB=flow.filter(n=>n.s.scope==='layer').map(n=>n.band);
  if(LB.length){const y0=BM.y[Math.min(...LB)]-6,y1=bEnd(Math.max(...LB));fr+=`<rect class="fl" x="2" y="${y0}" width="${W-4}" height="${y1-y0}" rx="18"/><text class="flt" x="${W-14}" y="${y0+15}" text-anchor="end">${G.label?G.label.split(' = ')[0]:'ONE LAYER · × '+K.N}</text>`;}
  ['Attention','Cross-attention'].forEach(stg=>{const M=flow.filter(n=>n.s.head&&n.s.stage===stg);if(!M.length)return;
    const x0=Math.min(...M.map(n=>n.x-lblW(n)-18)),x1=Math.max(...M.map(n=>Math.max(n.x+n.w,n.ax+n.pw/2+(n.srcs.length?16+n.srcs.reduce((a,q)=>a+srcSlot(q),0):0))))+8,y0=BM.y[Math.min(...M.map(n=>n.band))]+1,y1=bEnd(Math.max(...M.map(n=>n.band)));
    fr+=`<rect class="fh" x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" rx="14"/><text class="fht" x="${x1-10}" y="${y0+14}" text-anchor="end">one of ${K.H} heads · head ${clamp(K.h,0,K.H-1)}</text>`;});
  return {W,H,inner:fr+es+s};
}
function renderMain(){if(!$('#vg-main'))return;
  const cmp=K.mode==='cmp',ids=cmp?[K.a,K.b]:[K.a],main=$('#vg-main');
  main.innerHTML=`<div class="cols${cmp?' two':''}">${ids.map((_,i)=>`<div class="gcol" id="gc${i}"></div>`).join('')}</div>`;
  const colW=Math.max(320,$('#gc0').clientWidth-4),widthAt=p=>{K.px=p;return Math.max(...ids.map(id=>{const S=archOf(id).build(K),G=prep(S,null);return draw(G,mergeBM([G])).W;}));};
  const w1=widthAt(80),w2=widthAt(160),slope=(w2-w1)/80;
  K.px=Math.round(clamp(80+(Math.min(colW,cmp?1e9:1250)-w1)/slope,cmp?80:90,250));
  document.documentElement.style.setProperty('--hdr',$('header.top').offsetHeight+'px');
  REG=[];const reg=S=>{S.forEach(s=>{s._i=REG.push(s)-1;});return S;};
  const St=ids.map(id=>reg(archOf(id).build(K))),Gs=St.map((S,i)=>prep(S,cmp?St[1-i]:null)),BM=mergeBM(Gs),D=Gs.map(g=>draw(g,BM));
  const sc=Math.min(1,colW/Math.max(...D.map(d=>d.W)));GR=Gs;clearTrace();
  ids.forEach((id,i)=>{const a=archOf(id),t=totals(St[i]),d=D[i];
    $('#gc'+i).innerHTML=`<div class="gcap"><b>${a.name}</b><span class="mut">${a.meta(K).facts?a.meta(K).facts[0][1]:fN(t.lp)+' params / layer · '+fF(t.lf)+' / layer'}</span></div><div class="gbox" data-g="${i}"><svg xmlns="http://www.w3.org/2000/svg" width="${d.W*sc}" height="${d.H*sc}" viewBox="0 0 ${d.W} ${d.H}">${DEFS}${d.inner}</svg></div>`;});
  $('#vg-hint').innerHTML=`Flow runs top → bottom; scroll the page. ${cmp?'The same step sits at the <b>same height</b> in both columns; a blank gap means that architecture has no such step, and a <b style="color:var(--diff)">white dashed</b> outline means the step differs. ':''}<b>Hover</b> a tensor for its details, <b>click</b> for the worked example, <b>right-click</b> to trace its path, find it in the other column or copy it. Tensor scale: ${K.d} columns = ${(K.d*K.px/Math.max(K.d,K.ff)*sc).toFixed(0)} px.`;
}
function showDetail(i){
  const s=REG[i],d=$('#detail');d.hidden=false;
  d.innerHTML=`<button class="x" aria-label="Close details">✕</button><div class="stage">${s.stage}<small>${s.scope==='layer'?`inside every layer · × ${K.N}`:s.scope==='in'?'once, before layer 1':'once, after the last layer'}</small></div>`+stepHTML(s,'•',false);
  openDemo($('.step',d),s);
}
function renderBar(){$('#vg-bar').innerHTML=barHTML({orient:true,head:true});}
function mountV(){
  ensureOverlays();renderBar();renderMain();
  const bar=$('#vg-bar'),main=$('#vg-main'),det=$('#detail');
  bindBar(bar,()=>{renderBar();renderMain();});
  const hl=(w,id)=>{if(w.classList.contains('tr'))return;$$('.ge',w).forEach(p=>p.classList.toggle('on',id!=null&&(p.dataset.a===id||p.dataset.b===id)));w.classList.toggle('foc',id!=null);};
  main.addEventListener('mouseover',e=>{const w=e.target.closest('.gbox');if(!w)return;const n=e.target.closest('.gn');hl(w,n?n.dataset.n:null);});
  main.addEventListener('mouseleave',()=>$$('.gbox',main).forEach(w=>hl(w,null)));
  main.addEventListener('focusin',e=>{const n=e.target.closest('.gn');if(n)hl(n.closest('.gbox'),n.dataset.n);});
  const pick=n=>{$$('.gn.sel').forEach(x=>x.classList.remove('sel'));n.classList.add('sel');showDetail(+n.dataset.i);};
  main.addEventListener('click',e=>{const n=e.target.closest('.gn');if(n)pick(n);});
  main.addEventListener('keydown',e=>{const n=e.target.closest&&e.target.closest('.gn');if(n&&(e.key==='Enter'||e.key===' ')){e.preventDefault();pick(n);}});
  interactions(main,pick);
  if(VB)return;VB=true;
  det.addEventListener('click',e=>{if(e.target.closest('.x')){det.hidden=true;$$('.gn.sel').forEach(x=>x.classList.remove('sel'));return;}if(e.target.closest('.demo'))return;const el=e.target.closest('.step');if(el)openDemo(el,REG[+el.dataset.i]);});
  document.addEventListener('keydown',e=>{if(e.key==='Escape')det.hidden=true;});
  let rt,lw=innerWidth;window.addEventListener('resize',()=>{if(innerWidth===lw)return;lw=innerWidth;clearTimeout(rt);rt=setTimeout(renderMain,200);});
}
TFV.v=mountV;TFV.hide=()=>{hideOverlays();clearTrace();};
