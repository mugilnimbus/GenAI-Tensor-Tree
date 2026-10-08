/* ================= helpers ================= */
const $=(s,r=document)=>r.querySelector(s);
const $$=(s,r=document)=>Array.from(r.querySelectorAll(s));
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const r1=n=>Math.round(n*10)/10;
const fmtN=n=>{const a=Math.abs(n);if(a>=1e12)return (n/1e12).toFixed(2)+'T';if(a>=1e9)return (n/1e9).toFixed(2)+'B';if(a>=1e6)return (n/1e6).toFixed(1)+'M';if(a>=1e4)return (n/1e3).toFixed(1)+'K';return Math.round(n).toLocaleString('en-US');};
const fmtB=b=>{const u=['B','KiB','MiB','GiB','TiB','PiB'];let i=0;while(b>=1024&&i<u.length-1){b/=1024;i++;}return (i===0||b>=100?b.toFixed(0):b>=10?b.toFixed(1):b.toFixed(2))+' '+u[i];};
const fmtF=f=>{const u=['FLOPs','KFLOPs','MFLOPs','GFLOPs','TFLOPs','PFLOPs','EFLOPs'];let i=0;while(f>=1000&&i<u.length-1){f/=1000;i++;}return (f>=100?f.toFixed(0):f>=10?f.toFixed(1):f.toFixed(2))+' '+u[i];};
const fmtS=s=>s>=1?s.toFixed(2)+' s':s>=1e-3?(s*1e3).toFixed(s>=1e-2?1:2)+' ms':(s*1e6).toFixed(0)+' µs';
const divisors=n=>{const r=[];for(let i=1;i<=n;i++)if(n%i===0)r.push(i);return r;};
const softmax=a=>{const m=Math.max(...a),e=a.map(v=>Math.exp(v-m)),z=e.reduce((x,y)=>x+y,0);return e.map(v=>v/z);};
const rng=seed=>()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};
function hashId(tok,V){let h=2166136261;for(const ch of tok){h^=ch.codePointAt(0);h=Math.imul(h,16777619)>>>0;}return 100+h%Math.max(1000,V-200);}

/* ================= colours ================= */
const C={green:'#72d945',purple:'#a645dc',red:'#d6459b',orange:'#d65f48',blue:'#45c1dd',yellow:'#d6cc45',pink:'#5b66f0',cyan:'#45d984',gray:'#8fa6bd'};
/* one colour per tensor role — used by every diagram, card, graph and legend */
const ROLE={x:'#72d945',xn:'#45d984',xatt:'#a6e87a',q:'#45c1dd',k:'#d65f48',v:'#5b66f0',s:'#d6459b',a:'#f291c9',o:'#7fe0d0',attn:'#2aa7a0',
  gate:'#e0a23f',sg:'#f3c677',up:'#8fb4ff',hid:'#c5d93a',mlp:'#b08cff',router:'#e06fd0',tok:'#d6cc45',prob:'#f2e98a',w:'#a645dc',mask:'#9aa7b4',lora:'#ff9f7a',other:'#c0c7d6'};
const ROLE_NAME={x:'X · residual stream',xn:'X̂ · normalised',xatt:'X_att · after attention',q:'Q · queries',k:'K · keys',v:'V · values',s:'S · scores',a:'A · attention weights',o:'O · head outputs',attn:'Attn · attention output',
  gate:'G · gate',sg:'SiLU(G) · activated gate',up:'U · up',hid:'Hid · gated hidden',mlp:'MLP / expert output',router:'r, g · router',tok:'tokens · logits',prob:'p · probabilities',w:'W · learned weights',mask:'M · mask',lora:'low-rank / latent (LoRA, MLA latent, n-gram memory)',other:'other activations'};
const roleLegend=keys=>`<div class="legend" style="margin:0 0 12px">${(keys||Object.keys(ROLE)).map(k=>`<span style="--c:${ROLE[k]}">${ROLE_NAME[k]}</span>`).join('')}</div>`;
const BANDS=['#45c1dd','#4a55e0','#a645dc','#d6459b','#d65f48','#d6cc45'];
const hex2rgb=h=>{h=h.replace('#','');return [0,2,4].map(i=>parseInt(h.slice(i,i+2),16));};
const rgb2hex=(r,g,b)=>'#'+[r,g,b].map(v=>clamp(Math.round(v),0,255).toString(16).padStart(2,'0')).join('');
const shade=(hex,a)=>{const [r,g,b]=hex2rgb(hex),t=a<0?0:255,p=Math.abs(a);return rgb2hex(r+(t-r)*p,g+(t-g)*p,b+(t-b)*p);};
const rgba=(hex,a)=>{const [r,g,b]=hex2rgb(hex);return `rgba(${r},${g},${b},${a})`;};
$('#gdefs').innerHTML=[...new Set(Object.values(C).concat(Object.values(ROLE)))].map(c=>`<marker id="ar-${c.slice(1)}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9" markerHeight="9" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 0L10 5L0 10z" fill="${c}"/></marker>`).join('');

/* ================= 3-D tensor block ================= */
const lg=n=>Math.log2(Math.max(1,n)+1);
/* dims: [c] | [rows,cols] | [depth,rows,cols]  ->  [w,h,d] in px */
function dimsBox(dims,max=64){
  let a=1,b=1,c=1;
  if(dims.length===1){c=dims[0];}else if(dims.length===2){b=dims[0];c=dims[1];}else{a=dims[0];b=dims[1];c=dims[2];}
  let w=10+lg(c)*4.3,h=10+lg(b)*4.3,d=a<=1?9:8+lg(a)*3.4;
  if(dims.length===1)h=12;
  const k=Math.min(1,max/Math.max(w+d*.6,h+d*.5));
  return [w*k,h*k,d*k];
}
function cube(cx,cy,w,h,d,col,o={}){
  const up=d*.5,rt=d*.6,fx=cx-(w+rt)/2,fy=cy-(h-up)/2,R=r1;
  const nx=o.nx||clamp(Math.round(w/10),2,7),ny=o.ny||clamp(Math.round(h/10),2,7),nz=o.nz||clamp(Math.round(d/9),1,4);
  let s='';const bands=o.bands;
  const rightP=`M${R(fx+w)} ${R(fy)}L${R(fx+w+rt)} ${R(fy-up)}L${R(fx+w+rt)} ${R(fy+h-up)}L${R(fx+w)} ${R(fy+h)}Z`;
  if(bands&&bands.length>1){
    const nb=bands.length,bw=w/nb;
    bands.forEach((bc,i)=>{
      const x0=fx+i*bw,x1=x0+bw;
      s+=`<rect x="${R(x0)}" y="${R(fy)}" width="${R(bw+.4)}" height="${R(h)}" fill="${bc}"/>`;
      s+=`<path d="M${R(x0)} ${R(fy)}L${R(x0+rt)} ${R(fy-up)}L${R(x1+rt)} ${R(fy-up)}L${R(x1)} ${R(fy)}Z" fill="${shade(bc,.3)}"/>`;
    });
    s+=`<path d="${rightP}" fill="${shade(bands[bands.length-1],-.34)}"/>`;
  }else{
    s+=`<rect x="${R(fx)}" y="${R(fy)}" width="${R(w)}" height="${R(h)}" fill="${col}"/>`;
    s+=`<path d="M${R(fx)} ${R(fy)}L${R(fx+rt)} ${R(fy-up)}L${R(fx+w+rt)} ${R(fy-up)}L${R(fx+w)} ${R(fy)}Z" fill="${shade(col,.32)}"/>`;
    s+=`<path d="${rightP}" fill="${shade(col,-.34)}"/>`;
  }
  let g='';
  for(let i=1;i<nx;i++){const x=fx+w*i/nx;g+=`M${R(x)} ${R(fy)}V${R(fy+h)}M${R(x)} ${R(fy)}L${R(x+rt)} ${R(fy-up)}`;}
  for(let j=1;j<ny;j++){const y=fy+h*j/ny;g+=`M${R(fx)} ${R(y)}H${R(fx+w)}M${R(fx+w)} ${R(y)}L${R(fx+w+rt)} ${R(y-up)}`;}
  for(let k=1;k<nz;k++){const t=k/nz;g+=`M${R(fx+rt*t)} ${R(fy-up*t)}H${R(fx+w+rt*t)}M${R(fx+w+rt*t)} ${R(fy-up*t)}V${R(fy+h-up*t)}`;}
  s+=`<path d="${g}" fill="none" stroke="rgba(0,0,0,.34)" stroke-width=".8"/>`;
  s+=`<path d="M${R(fx)} ${R(fy)}H${R(fx+w)}V${R(fy+h)}H${R(fx)}ZM${R(fx)} ${R(fy)}L${R(fx+rt)} ${R(fy-up)}H${R(fx+w+rt)}L${R(fx+w)} ${R(fy)}M${R(fx+w+rt)} ${R(fy-up)}V${R(fy+h-up)}L${R(fx+w)} ${R(fy+h)}" fill="none" stroke="rgba(255,255,255,.28)" stroke-width=".8"/>`;
  return `<g class="cube" style="filter:drop-shadow(0 0 4px ${rgba(col,.5)})">${s}</g>`;
}
/* cube + labels without interactivity */
function plainTensor(x,y,dims,col,label,shape,o={}){
  const [w,h,d]=dimsBox(dims,o.max||64);
  return `<g style="color:${col}">${cube(x,y,w,h,d,col,{bands:o.bands})}<text class="lab" x="${x}" y="${y-46}" text-anchor="middle">${label}</text>${shape?`<text class="shp" x="${x}" y="${y+51}" text-anchor="middle">${shape}</text>`:''}</g>`;
}

/* ================= diagram nodes ================= */
function nodeSVG(o,cls){
  const [w,h,d]=dimsBox(o.dims,o.max||64),lab=[].concat(o.label||[]),n=lab.length;
  let s=`<g class="${cls}" data-id="${o.id}" data-step="${o.step==null?'':o.step}" tabindex="0" role="button" aria-label="${esc(lab.join(' ')||o.id)}" style="color:${o.col}">`;
  s+=`<rect class="hit" x="${o.x-50}" y="${o.y-62}" width="100" height="128" rx="12"/>`;
  s+=cube(o.x,o.y,w,h,d,o.col,{bands:o.bands});
  lab.forEach((t,i)=>{s+=`<text class="lab" x="${o.x}" y="${o.y-46-(n-1-i)*15}" text-anchor="middle">${t}</text>`;});
  if(o.shape)s+=`<text class="shp" x="${o.x}" y="${o.y+51}" text-anchor="middle">${o.shape}</text>`;
  return s+'</g>';
}
function opSVG(o,cls){
  const r=o.r||18,sy=String(o.sym).length;
  return `<g class="${cls}" data-id="${o.id}" data-step="${o.step==null?'':o.step}" tabindex="0" role="button" aria-label="${esc(o.label||o.sym)}" style="color:${o.col||C.gray}"><rect class="hit" x="${o.x-46}" y="${o.y-r-14}" width="92" height="${2*r+52}" rx="12"/><circle class="opc" cx="${o.x}" cy="${o.y}" r="${r}"/><text class="sym" x="${o.x}" y="${o.y+4.5}" text-anchor="middle" style="font-size:${sy>3?10.5:sy>2?11.5:sy>1?13:18}px">${o.sym}</text>${o.label?`<text class="shp" x="${o.x}" y="${o.y+r+17}" text-anchor="middle">${o.label}</text>`:''}</g>`;
}
function pillSVG(o,cls){
  return `<g class="${cls}" data-id="${o.id}" data-step="${o.step==null?'':o.step}" tabindex="0" role="button" aria-label="${esc(o.text)}" style="color:${o.col}"><rect class="pillr" x="${o.x-o.w/2}" y="${o.y-o.h/2}" width="${o.w}" height="${o.h}" rx="9"/><text class="pillt" x="${o.x}" y="${o.y+4.5}" text-anchor="middle">${o.text}</text></g>`;
}
function boxSVG(o,cls){
  const hx=o.hit||[o.x-o.r,o.y-o.r,2*o.r,2*o.r];
  return `<g class="${cls}" data-id="${o.id}" data-step="${o.step==null?'':o.step}" tabindex="0" role="button" aria-label="${esc(o.aria||o.id)}" style="color:${o.col||C.gray}"><rect class="hit" x="${hx[0]}" y="${hx[1]}" width="${hx[2]}" height="${hx[3]}" rx="12"/>${o.svg}</g>`;
}
class Graph{
  constructor(cur,focus){this.cur=cur;this.focus=focus;this.N={};this.u=[];this.e=[];this.n=[];this.o=[];}
  sc(step){return this.cur==null||step==null?'':step>this.cur?'dim':step===this.cur?'act':'';}
  cls(o,extra){return ['node',extra,this.sc(o.step),this.focus===o.id?'foc':''].filter(Boolean).join(' ');}
  reg(o,r){this.N[o.id]={x:o.x,y:o.y,r};}
  node(o){this.reg(o,o.r||36);this.n.push(nodeSVG(o,this.cls(o)));return this;}
  op(o){this.reg(o,o.r||18);this.n.push(opSVG(o,this.cls(o,'op')));return this;}
  pill(o){this.reg(o,o.w/2);this.n.push(pillSVG(o,this.cls(o,'pill')));return this;}
  box(o){this.reg(o,o.r||40);this.n.push(boxSVG(o,this.cls(o,'box')));return this;}
  edge(a,b,o={}){
    const A=this.N[a],B=this.N[b];
    const x1=A.x+A.r+3+(o.dx1||0),y1=A.y+(o.dy1||0),x2=B.x-B.r-3,y2=B.y+(o.dy2||0),m=(x1+x2)/2;
    return this.path(Math.abs(y1-y2)<1?`M${x1} ${y1}H${x2}`:`M${x1} ${y1}C${m} ${y1} ${m} ${y2} ${x2} ${y2}`,o);
  }
  path(d,o={}){
    const col=o.col||C.gray;
    this.e.push(`<path class="${['edge',o.solid?'solid':'',this.sc(o.step)].filter(Boolean).join(' ')}" d="${d}" stroke="${col}" stroke-width="${o.w||2}"${o.noarrow?'':` marker-end="url(#ar-${col.slice(1)})"`}/>`);
    return this;
  }
  label(x,y,t,o={}){this.o.push(`<text class="${['elab',this.sc(o.step)].filter(Boolean).join(' ')}" x="${x}" y="${y}" text-anchor="${o.anchor||'middle'}"${o.col?` style="fill:${o.col}"`:''}>${t}</text>`);return this;}
  under(s){this.u.push(s);return this;}
  html(){return this.u.join('')+this.e.join('')+this.o.join('')+this.n.join('');}
}
const svgWrap=(w,h,inner,extra='')=>`<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" role="img" ${extra}>${inner}</svg>`;

/* ================= UI helpers ================= */
const secHead=(t,s,id)=>`<div class="sec"><h2>${t}</h2>${s?`<p${id?` id="${id}"`:''}>${s}</p>`:''}</div>`;
const segHTML=(id,opts,cur)=>`<div class="seg" id="${id}" role="group">${opts.map(([v,t])=>`<button type="button" data-v="${v}" class="${v===cur?'on':''}" aria-pressed="${v===cur}">${t}</button>`).join('')}</div>`;
const segBind=(el,cb)=>el.addEventListener('click',e=>{const b=e.target.closest('button[data-v]');if(b)cb(b.dataset.v);});
const segSet=(el,val)=>$$('button',el).forEach(b=>{const on=b.dataset.v===String(val);b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);});
let toastT=null;
function toast(msg){const t=$('#toast');t.textContent=msg;t.classList.add('on');clearTimeout(toastT);toastT=setTimeout(()=>t.classList.remove('on'),2800);}

class Stepper{
  constructor(host,n,onChange,ms=1700){
    this.host=host;this.n=n;this.i=null;this.cb=onChange;this.t=null;this.ms=ms;
    host.innerHTML='<button class="btn" data-a="prev" title="Previous step" aria-label="Previous step">◀</button><button class="btn pri" data-a="play">▶ Play steps</button><button class="btn" data-a="next" title="Next step" aria-label="Next step">▶|</button><button class="btn" data-a="all">Show all</button><span class="stepno"></span>';
    host.addEventListener('click',e=>{const b=e.target.closest('[data-a]');if(!b)return;const a=b.dataset.a;if(['prev','next','play','all'].includes(a))this[a]();});
    this.sync();
  }
  prev(){this.stop();this.go(this.i==null?this.n-1:Math.max(0,this.i-1));}
  next(){this.stop();this.go(this.i==null?0:Math.min(this.n-1,this.i+1));}
  all(){this.stop();this.go(null);}
  play(){
    if(this.t){this.stop();return;}
    if(this.i==null||this.i>=this.n-1)this.go(0);
    this.t=setInterval(()=>{if(this.i>=this.n-1){this.stop();return;}this.go(this.i+1);},this.ms);this.sync();
  }
  stop(){clearInterval(this.t);this.t=null;this.sync();}
  go(i){this.i=i;this.cb(i);this.sync();}
  sync(){
    const p=$('[data-a=play]',this.host);if(p)p.textContent=this.t?'⏸ Pause':'▶ Play steps';
    const s=$('.stepno',this.host);if(s)s.textContent=this.i==null?'all steps':`step ${this.i+1} / ${this.n}`;
  }
}
function paintFocus(root,st){
  $$('.node.foc',root).forEach(n=>n.classList.remove('foc','sel'));
  const id=st.sel||st.hov;if(!id)return;
  const n=root.querySelector(`.node[data-id="${id}"]`);
  if(n){n.classList.add('foc');if(st.sel===id)n.classList.add('sel');}
}
function bindNodes(root,st,onFocus){
  const nid=e=>{const n=e.target.closest&&e.target.closest('.node');return n?n.dataset.id:null;};
  root.addEventListener('mouseover',e=>{const id=nid(e);if(id!==st.hov){st.hov=id;paintFocus(root,st);onFocus();}});
  root.addEventListener('mouseleave',()=>{if(st.hov!=null){st.hov=null;paintFocus(root,st);onFocus();}});
  root.addEventListener('focusin',e=>{const id=nid(e);if(id){st.hov=id;paintFocus(root,st);onFocus();}});
  root.addEventListener('click',e=>{const id=nid(e);st.sel=id&&st.sel!==id?id:null;paintFocus(root,st);onFocus();});
  root.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.classList&&e.target.classList.contains('node')){e.preventDefault();e.target.dispatchEvent(new MouseEvent('click',{bubbles:true}));}});
}
const inspHTML=(i,hint)=>!i?`<div class="hint">${hint||'Hover any tensor or operation for details — click to pin it.'}</div>`:
  `<div class="insp"><div class="ih"><i style="background:${i.c};color:${i.c}"></i><b>${i.t}</b></div><p>${i.d}</p>${i.r&&i.r.length?`<dl>${i.r.map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>`:''}${i.fm?formulaHTML(i.fm):''}${i.why?whyHTML(i.why,i.short):''}${i.link?`<p style="margin-top:10px">${i.link}</p>`:''}</div>`;
