/* ================= tensor graph — horizontal layout with pan and zoom ================= */
/* ================= horizontal tensor graph ================= */
let REG=[];
function layout(steps,other){
  const N=buildGraph(steps,true),om=other?new Map(other.map(s=>[s.key,s])):null;
  N.forEach(n=>{n.diff=!!om&&!steps.seq&&!other.seq&&(!om.has(n.s.key)||om.get(n.s.key).sig!==n.s.sig);});
  /* columns */
  N.forEach(n=>{if(n.src)return;const f=n.ins.filter(i=>!i.src);n.col=f.length?Math.max(...f.map(i=>i.col))+1:0;});
  N.forEach(n=>n.ins.forEach(i=>{if(i.src)i.col=Math.min(i.col==null?1e9:i.col,n.col-1);}));
  N.forEach(n=>{if(n.col==null)n.col=0;});
  /* lanes */
  const occ=new Set(),put=(n,pref,order)=>{for(const d of order){if(!occ.has(n.col+':'+(pref+d))){n.lane=pref+d;occ.add(n.col+':'+n.lane);return;}}n.lane=pref+9;};
  N.filter(n=>!n.src).forEach(n=>{const f=n.ins.find(i=>!i.src);put(n,f?f.lane:0,[0,1,2,3,4,5,6,-1,-2]);});
  N.filter(n=>n.src).forEach(n=>{const c=N.find(m=>m.ins.includes(n));put(n,c?c.lane-1:0,[0,-1,1,2,-2,3,-3,4]);});
  /* geometry */
  const cols=[...new Set(N.map(n=>n.col))].sort((a,b)=>a-b),lanes=[...new Set(N.map(n=>n.lane))].sort((a,b)=>a-b);
  const lblW=n=>Math.max(txt(n.t.n).length*8,shp(n.t).length*6.8),bwOf=n=>Math.max(26,n.op.length*7.2+14);
  const colW={},laneH={},maxBW={},elbow={},riser={};
  cols.forEach(c=>{colW[c]=Math.max(...N.filter(n=>n.col===c).map(n=>Math.max(n.w,lblW(n))));maxBW[c]=Math.max(0,...N.filter(n=>n.col===c&&n.op).map(bwOf));elbow[c]=new Map();riser[c]=new Map();});
  lanes.forEach(l=>{laneH[l]=Math.max(...N.filter(n=>n.lane===l).map(n=>n.h));});
  const edges=[];
  N.forEach(n=>n.ins.forEach(i=>{const straight=i.col===n.col-1&&i.lane===n.lane,long=i.col<n.col-1;edges.push({a:i,b:n,straight,long});
    if(!straight&&!elbow[n.col].has(i.id))elbow[n.col].set(i.id,elbow[n.col].size);
    if(long&&!riser[i.col].has(i.id))riser[i.col].set(i.id,riser[i.col].size);}));
  const colX={},laneY={};let x=0;
  cols.forEach((c,k)=>{const pr=cols[k-1];if(k)x+=(riser[pr]?riser[pr].size*9:0)+34+elbow[c].size*9+(maxBW[c]?maxBW[c]+40:0)+16;colX[c]=x;x+=colW[c];});
  let y=0;lanes.forEach((l,k)=>{if(k)y+=laneH[lanes[k-1]]/2+78+laneH[l]/2;laneY[l]=y;});
  N.forEach(n=>{n.cx=colX[n.col]+colW[n.col]/2;n.cy=laneY[n.lane];n.x=n.cx-n.w/2;n.y=n.cy-n.h/2;
    if(n.op){n.bw=bwOf(n);n.bx=colX[n.col]-22-maxBW[n.col]/2;n.dx0=colX[n.col]-22-maxBW[n.col]-16;}});
  const top=Math.min(...N.map(n=>n.y))-34,bot=Math.max(...N.map(n=>n.y+n.h))+34;
  /* bus tracks */
  const tracks={t:[],b:[]};
  edges.filter(e=>e.long).sort((p,q)=>p.a.col-q.a.col).forEach(e=>{const bus=e.a.t.kind==='res'?'t':'b',T=tracks[bus];let k=T.findIndex(end=>end<e.a.col);if(k<0){k=T.length;T.push(0);}T[k]=e.b.col;e.bus=bus;e.track=k;});
  let s='',es='';
  const colOf=t=>tcol(t);
  edges.forEach(e=>{
    const {a,b}=e,sx=a.x+a.w+2,sy=a.cy,bl=b.bx-b.bw/2,by=b.cy,c=colOf(a.t);let p;
    if(e.straight)p=[[sx,sy],[bl-2,by]];
    else{const dx=b.dx0-elbow[b.col].get(a.id)*9;
      if(!e.long){const ey=by+(sy<by?-6:6);p=[[sx,sy],[dx,sy],[dx,ey],[bl-2,ey]];}
      else{const gx=colX[a.col]+colW[a.col]+16+riser[a.col].get(a.id)*9,busY=e.bus==='t'?top-18-e.track*13:bot+18+e.track*13,ey=by+(e.bus==='t'?-6:6);
        p=[[sx,sy],[gx,sy],[gx,busY],[dx,busY],[dx,ey],[bl-2,ey]];}}
    es+=`<path class="ge" data-a="${a.id}" data-b="${b.id}" d="${rpath(p)}" stroke="${c}" marker-end="url(#m-${tcol(a.t).slice(1)})"/>`;
  });
  N.forEach(n=>{
    const c=colOf(n.t),t=n.t;let g='';
    if(n.op){const bl=n.bx-n.bw/2;g+=`<path class="ge" data-a="${n.id}" data-b="${n.id}" d="M${n.bx+n.bw/2} ${n.cy}H${n.x-3}" stroke="${c}" marker-end="url(#m-${tcol(t).slice(1)})"/><rect x="${bl}" y="${n.cy-11}" width="${n.bw}" height="22" rx="11" class="gb${n.diff?' df':''}"/><text class="gbt" x="${n.bx}" y="${n.cy+4}" text-anchor="middle">${esc(n.op)}</text>`;}
    g+=`<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" fill="${c}" fill-opacity="${t.mask?.06:.18}" stroke="${n.diff?'#ffffff':c}" stroke-width="${n.diff?2.4:1.4}"/>`;
    if(t.mask==='causal')g+=`<path d="M${n.x} ${n.y}L${n.x+n.w} ${n.y+n.h}H${n.x}Z" fill="${c}" fill-opacity=".5"/>`;
    if(t.mask==='window'){const f=clamp(t.frac||.2,.02,1);g+=`<path d="M${n.x} ${n.y}L${n.x+n.w} ${n.y+n.h}H${n.x+n.w*(1-f)}L${n.x} ${n.y+n.h*f}Z" fill="${c}" fill-opacity=".55"/>`;}
    if(t.stripes>1){const m=Math.min(t.stripes,Math.floor(n.w/3));let d='';for(let i=1;i<m;i++)d+=`M${(n.x+i*n.w/m).toFixed(1)} ${n.y}v${n.h}`;if(d)g+=`<path d="${d}" stroke="${c}" stroke-width=".8" opacity=".75"/>`;}
    if(t.hiRow!=null)g+=`<rect x="${n.x}" y="${t.hiRow==='last'?n.y+n.h-3:n.y}" width="${n.w}" height="3" fill="#ff5a5a"/>`;
    if(dimPx(t.c)[1]){const q=(n.h+8)/4;g+=`<path d="M${n.cx-3} ${n.y-4}l6 ${q}l-6 ${q}l6 ${q}l-6 ${q}" stroke="var(--txt)" stroke-width="1.6" fill="none"/>`;}
    if(dimPx(t.r)[1]){const q=(n.w+8)/4;g+=`<path d="M${n.x-4} ${n.cy-3}l${q} 6l${q} -6l${q} 6l${q} -6" stroke="var(--txt)" stroke-width="1.6" fill="none"/>`;}
    g+=`<text class="gl" x="${n.cx}" y="${n.y-8}" text-anchor="middle" fill="${c}">${svgLabel(t.n)}</text><text class="gs" x="${n.cx}" y="${n.y+n.h+15}" text-anchor="middle">${esc(shp(t))}</text>`;
    const hw=Math.max(n.w,lblW(n))+12;
    s+=`<g class="gn" data-n="${n.id}" data-i="${n.s._i}" tabindex="0" role="button" aria-label="${esc(txt(t.n)+' '+shp(t)+' — '+n.s.title)}"><title>${esc(n.s.title+':  '+txt(n.s.fx))}</title><rect class="gh" x="${n.cx-hw/2}" y="${n.y-26}" width="${hw}" height="${n.h+50}" rx="8"/>${g}</g>`;
  });
  /* frames */
  const bb=f=>{const M=N.filter(f);if(!M.length)return null;const x0=Math.min(...M.map(n=>Math.min(n.x,n.op?n.bx-n.bw/2:n.x,n.cx-lblW(n)/2))),x1=Math.max(...M.map(n=>Math.max(n.x+n.w,n.cx+lblW(n)/2))),y0=Math.min(...M.map(n=>n.y-26)),y1=Math.max(...M.map(n=>n.y+n.h+24));return [x0,y0,x1-x0,y1-y0];};
  let fr='';const L=bb(n=>n.s.scope==='layer'),nb=tracks.b.length,nt=tracks.t.length;
  if(L){const p=24,yt=top-18-nt*13-16,yb=bot+18+nb*13+16;fr+=`<rect class="fl" x="${L[0]-p}" y="${yt}" width="${L[2]+2*p}" height="${yb-yt}" rx="26"/><text class="flt" x="${L[0]-p+18}" y="${yt-10}">${steps.layerLabel||'ONE LAYER — repeated × '+K.N+'  (output X feeds the next layer)'}</text>`;}
  ['Attention','Cross-attention'].forEach(stg=>{const hb=bb(n=>n.s.head&&n.s.stage===stg);if(hb){const p=14;fr+=`<rect class="fh" x="${hb[0]-p}" y="${hb[1]-p}" width="${hb[2]+2*p}" height="${hb[3]+2*p}" rx="18"/><text class="fht" x="${hb[0]-p+14}" y="${hb[1]-p+16}">for each of the ${K.H} heads — head ${clamp(K.h,0,K.H-1)} shown</text>`;}});
  const X0=Math.min(...N.map(n=>n.x))-60,X1=Math.max(...N.map(n=>n.x+n.w))+90,Y0=top-18-nt*13-50,Y1=bot+18+nb*13+34;
  return {box:[X0,Y0,X1-X0,Y1-Y0],inner:fr+es+s,n:N.length};
}
function panZoom(wrap,box){
  const svg=$('svg',wrap);let v;
  const set=()=>svg.setAttribute('viewBox',`${v.x} ${v.y} ${v.w} ${v.h}`),R=()=>wrap.getBoundingClientRect();
  const fitH=()=>{const r=R(),sc=Math.min(1.15,r.height/box[3]);v={w:r.width/sc,h:r.height/sc,x:box[0],y:box[1]-(r.height/sc-box[3])/2};set();};
  const fitAll=()=>{const r=R(),sc=Math.min(r.width/box[2],r.height/box[3]);v={w:r.width/sc,h:r.height/sc,x:box[0]-(r.width/sc-box[2])/2,y:box[1]-(r.height/sc-box[3])/2};set();};
  const zoom=(f,px,py)=>{const r=R(),cx=v.x+(px==null?.5:px/r.width)*v.w,cy=v.y+(py==null?.5:py/r.height)*v.h;v.w*=f;v.h*=f;v.x=cx-(px==null?.5:px/r.width)*v.w;v.y=cy-(py==null?.5:py/r.height)*v.h;set();};
  fitH();
  wrap.addEventListener('wheel',e=>{const r=R();if(e.ctrlKey||e.metaKey){e.preventDefault();zoom(e.deltaY>0?1.12:1/1.12,e.clientX-r.left,e.clientY-r.top);}else if(e.shiftKey||Math.abs(e.deltaX)>Math.abs(e.deltaY)){e.preventDefault();v.x+=(e.deltaX||e.deltaY)*v.w/r.width;set();}},{passive:false});
  let drag=null;
  wrap.addEventListener('pointerdown',e=>{if(e.target.closest('.zb'))return;drag={x:e.clientX,y:e.clientY,vx:v.x,vy:v.y,moved:0};});
  window.addEventListener('pointermove',e=>{if(!drag)return;const r=R(),dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag.moved=Math.max(drag.moved,Math.abs(dx)+Math.abs(dy));if(drag.moved>4){wrap.classList.add('drag');v.x=drag.vx-dx*v.w/r.width;v.y=drag.vy-dy*v.h/r.height;set();}});
  window.addEventListener('pointerup',()=>{if(drag){wrap.dataset.moved=drag.moved>4?1:'';drag=null;wrap.classList.remove('drag');}});
  wrap.addEventListener('click',e=>{const b=e.target.closest('.zb button');if(!b)return;({in:()=>zoom(1/1.3),out:()=>zoom(1.3),fit:fitAll,h:fitH,l:()=>{v.x-=v.w*.6;set();},r:()=>{v.x+=v.w*.6;set();}})[b.dataset.z]();});
  wrap.addEventListener('keydown',e=>{if(e.target!==wrap)return;const k={ArrowLeft:[-.2,0],ArrowRight:[.2,0],ArrowUp:[0,-.2],ArrowDown:[0,.2]}[e.key];if(k){e.preventDefault();v.x+=k[0]*v.w;v.y+=k[1]*v.h;set();}});
}
function graphHTML(a,steps,other,id){
  const g=layout(steps,other),t=totals(steps);
  return `<div class="gcap"><b>${a.name}</b><span class="mut">${a.meta(K).facts?a.meta(K).facts[0][1]:fN(t.lp)+' params / layer · '+fF(t.lf)+' / layer'} · ${g.n} tensors</span></div>
  <div class="gwrap" id="${id}" tabindex="0" aria-label="Tensor graph for ${esc(a.short)}; drag to pan, arrow keys to move"><svg xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMinYMin meet" data-box="${g.box.join(',')}">${DEFS}${g.inner}</svg>
  <div class="zb"><button data-z="l" title="Pan left" aria-label="Pan left">◀</button><button data-z="r" title="Pan right" aria-label="Pan right">▶</button><button data-z="out" title="Zoom out" aria-label="Zoom out">−</button><button data-z="in" title="Zoom in" aria-label="Zoom in">+</button><button data-z="h" title="Fit height, start at the left">Fit height</button><button data-z="fit" title="Show the whole graph">Fit all</button></div></div>`;
}
function renderMain(){if(!$('#hg-main'))return;K.px=300;
  REG=[];const reg=S=>{S.forEach(s=>{s._i=REG.push(s)-1;});return S;};
  const a=archOf(K.a),sa=reg(a.build(K)),cmp=K.mode==='cmp',b=cmp?archOf(K.b):null,sb2=cmp?reg(b.build(K)):null;
  $('#hg-main').innerHTML=graphHTML(a,sa,sb2,'gA')+(cmp?graphHTML(b,sb2,sa,'gB'):'')+`<p class="mut" style="font-size:13px;margin:6px 0 0">Drag to pan · Shift + wheel (or ◀ ▶) to move sideways · Ctrl + wheel (or + −) to zoom · click a tensor for its step, formula and a worked numeric example.${cmp?' Amber outlines mark tensors whose step differs between the two architectures.':''}</p><div id="hg-detail"></div>`;
  $$('.gwrap').forEach(w=>{w.style.height=cmp?'44vh':'68vh';panZoom(w,$('svg',w).dataset.box.split(',').map(Number));});
}
function showDetail(i){
  const s=REG[i],d=$('#hg-detail');
  d.innerHTML=`<div class="stage">${s.stage}<small>${s.scope==='layer'?`inside every layer · × ${K.N}`:s.scope==='in'?'once, before layer 1':'once, after the last layer'}</small></div>`+stepHTML(s,'•',false);
  openDemo($('.step',d),s);
}
function renderBar(){$('#hg-bar').innerHTML=barHTML({orient:true,head:true,ab:['Top','Bottom']});}
let HB=false;function mountH(){
  renderBar();renderMain();
  const bar=$('#hg-bar'),main=$('#hg-main');
  bindBar(bar,()=>{renderBar();renderMain();});
  const hl=(w,id)=>{$$('.ge',w).forEach(p=>p.classList.toggle('on',id!=null&&(p.dataset.a===id||p.dataset.b===id)));w.classList.toggle('foc',id!=null);$$('.gn',w).forEach(n=>n.classList.toggle('on',n.dataset.n===id));};
  main.addEventListener('mouseover',e=>{const w=e.target.closest('.gwrap');if(!w)return;const n=e.target.closest('.gn');hl(w,n?n.dataset.n:null);});
  main.addEventListener('focusin',e=>{const n=e.target.closest('.gn');if(n)hl(n.closest('.gwrap'),n.dataset.n);});
  main.addEventListener('click',e=>{
    const n=e.target.closest('.gn');
    if(n){if(n.closest('.gwrap').dataset.moved)return;$$('.gn.sel',main).forEach(x=>x.classList.remove('sel'));n.classList.add('sel');showDetail(+n.dataset.i);return;}
    if(e.target.closest('.demo'))return;
    const el=e.target.closest('#hg-detail .step');if(el)openDemo(el,REG[+el.dataset.i]);
  });
  main.addEventListener('keydown',e=>{const n=e.target.closest&&e.target.closest('.gn');if(n&&(e.key==='Enter'||e.key===' ')){e.preventDefault();$$('.gn.sel',main).forEach(x=>x.classList.remove('sel'));n.classList.add('sel');showDetail(+n.dataset.i);}});
  if(HB)return;HB=true;let rt;window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(renderMain,200);});
}
TFV.h=mountH;TFX.h=()=>REG;
