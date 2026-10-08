/* ================= shared by the step-by-step and graph views ================= */
/* --- steps → graph of tensors (nodes) joined by operations --- */
const txt=n=>String(n).replace(/<[^>]+>/g,'');
const svgLabel=n=>String(n).replace(/<sub>/g,'\u0001').replace(/<\/sub>/g,'\u0002').replace(/<sup>/g,'\u0003').replace(/<\/sup>/g,'\u0004').replace(/<[^>]*>/g,'').replace(/&(?!\w+;)/g,'&amp;').replace(/\u0001/g,'<tspan dy="4" font-size="9.5">').replace(/\u0002/g,'</tspan><tspan dy="-4">\u200b</tspan>').replace(/\u0003/g,'<tspan dy="-5" font-size="9.5">').replace(/\u0004/g,'</tspan><tspan dy="5">\u200b</tspan>');
function opLabel(s){
  const m={matmul:'×',qkt:'×',av:'×',lookup:'lookup',add:'+',mask:'+',split:'reshape',rope:'RoPE',cache:'append',scale:'÷ √dₕ',softmax:'softmax',concat:'concat',mul:'⊙',topk:'top-k',select:'take row',combine:'Σ gᵢ·'};
  if(s.op==='norm')return s.title;
  if(s.op==='act')return {silu:'SiLU',gelu:'GELU',relu:'ReLU'}[s.fn];
  return m[s.op]||(s.key==='sample'?'sample':'→');
}
const EXACT=new Set(['emb','pos','q','k','v','xq','xk','xv','gate','up','silu','add1','add2','xadd','mul','nghash','nginj','mix1','mix2','mix3','hcw1','hcw2','hcw3','experts','mixout','mtp','lqa','lva','lqadd','lvadd']);
function buildGraph(steps,memAsSource){
  const N=[],latest={};let prev=null,lastNorm=null;
  const add=(t,s,src)=>{const n={id:N.length,t,s,src,ins:[],w:dimPx(t.c)[0],h:dimPx(t.r)[0]};N.push(n);latest[txt(t.n)]=n;return n;};
  const find=(...names)=>{let b=null;names.forEach(x=>{const n=latest[x];if(n&&(!b||n.id>b.id))b=n;});return b;};
  steps.forEach(s=>{
    const ts=s.expr.filter(x=>typeof x==='object'),op=opLabel(s);
    const link=(ins,t)=>{const n=add(t,s,false);n.op=op;n.ins=ins.filter(Boolean);return n;};
    let out;
    if(s.from){out=link(s.from.map(x=>find(x)).concat(ts.slice(0,-1).filter(t=>!s.from.includes(txt(t.n))&&(t.kind==='w'||!find(txt(t.n)))).map(t=>add(t,s,true))),ts[ts.length-1]);}
    else if(ts.length===1){out=add(ts[0],s,!!memAsSource&&s.key!=='tok');out.root=true;if(s.key!=='tok')return;}
    else if(s.op==='split'){ts.forEach(t=>{const i=find(txt(t.n));out=link([i],t);});}
    else if(s.op==='rope'){const q=find('Q'),k=find('K');link([q],ts[1]);out=link([k],ts[2]);}
    else if(s.op==='cache'){const k=find('K′','K');out=link([add(ts[0],s,true),k],ts[2]);}
    else if(s.op==='qkt'){const q=find('Q′','Q'),k=find('K′','K');out=link([q,k],ts[2]);}
    else if(s.op==='av'){out=link([prev,find('V')],ts[2]);}
    else if(s.op==='combine'){out=link([prev,find('g')],ts[1]);}
    else{
      const outT=ts[ts.length-1],ins=ts.slice(0,-1).map((t,i)=>{
        if(t.kind==='w')return add(t,s,true);
        if(i===0&&!EXACT.has(s.key))return prev||find(txt(t.n))||add(t,s,true);
        return find(txt(t.n))||(i===0&&lastNorm)||add(t,s,true);
      });
      out=link(ins,outT);
    }
    prev=out;if(s.op==='norm')lastNorm=out;
  });
  return N;
}
function rpath(p,r=7){
  let d=`M${p[0][0]} ${p[0][1]}`;
  for(let i=1;i<p.length-1;i++){
    const [x0,y0]=p[i-1],[x1,y1]=p[i],[x2,y2]=p[i+1],l1=Math.hypot(x1-x0,y1-y0),l2=Math.hypot(x2-x1,y2-y1),q=Math.min(r,l1/2,l2/2);
    if(q<1){d+=`L${x1} ${y1}`;continue;}
    d+=`L${x1-(x1-x0)/l1*q} ${y1-(y1-y0)/l1*q}Q${x1} ${y1} ${x1+(x2-x1)/l2*q} ${y1+(y2-y1)/l2*q}`;
  }
  const e=p[p.length-1];return d+`L${e[0]} ${e[1]}`;
}
const DEFS=`<defs>${Object.values(ROLE).map(c=>`<marker id="m-${c.slice(1)}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto"><path d="M0 1L10 5L0 9z" fill="${c}"/></marker>`).join('')}</defs>`;
/* --- toolbar shared by all three tensor views --- */
function fixCfg(){
  K.H=clamp(K.H,1,256);
  if(K.d%K.H){const dh=Math.max(8,Math.round(K.d/K.H/8)*8);K.d=dh*K.H;}
  const ds=[];for(let i=1;i<=K.H;i++)if(K.H%i===0)ds.push(i);
  if(!ds.includes(K.Hkv))K.Hkv=ds.reduce((b,x)=>Math.abs(x-K.Hkv)<Math.abs(b-K.Hkv)?x:b,ds[0]);
  K.E=clamp(K.E,2,256);K.k=clamp(K.k,1,K.E);K.T=Math.max(2,K.T);K.S=Math.max(1,K.S);K.W=clamp(K.W,1,K.T);K.h=clamp(K.h,0,K.H-1);
}
function barHTML(o={}){
  const cmp=K.mode==='cmp',ab=o.ab||['Left','Right'];
  const opt=sel=>ARCH.map(a=>`<option value="${a.id}" ${a.id===sel?'selected':''}>${a.name}</option>`).join('');
  const seg=(id,items)=>`<div class="seg" id="${id}">${items.map(([v,t])=>`<button data-v="${v}" class="${K[id]===v?'on':''}">${t}</button>`).join('')}</div>`;
  return `<div class="f">View${seg('mode',[['one','One architecture'],['cmp','Compare two']])}</div>
  <label class="f">${cmp?ab[0]:'Architecture'}<select id="selA">${opt(K.a)}</select></label>${cmp?`<label class="f">${ab[1]}<select id="selB">${opt(K.b)}</select></label>`:''}
  <div class="f">Drawing scale${seg('scale',[['lin','True scale'],['sqrt','√ scale']])}</div>${o.orient?`<div class="f">Layout${seg('orient',[['v','Vertical'],['h','Horizontal']])}</div>`:''}
  <span class="sep"></span>${FIELDS.filter(([k])=>!FIELD_USE[k]||FIELD_USE[k].some(id=>id===K.a||(cmp&&id===K.b))).concat(o.head?[['h','show head']]:[]).map(([k,l])=>`<label class="f">${l}<input type="number" min="${k==='h'?0:1}" data-k="${k}" value="${K[k]}"></label>`).join('')}`;
}
function bindBar(bar,rerender){
  bar.addEventListener('click',e=>{const b=e.target.closest('.seg button');if(!b)return;const id=b.parentNode.id;K[id]=b.dataset.v;if(id==='orient'){TFV.show();return;}rerender();});
  bar.addEventListener('change',e=>{const t=e.target;
    if(t.id==='selA')K.a=t.value;else if(t.id==='selB')K.b=t.value;
    else if(t.dataset.k){const v=Math.round(+t.value);if(isFinite(v)&&v>=(t.dataset.k==='h'?0:1))K[t.dataset.k]=v;fixCfg();}
    rerender();});
}
