/* ================= topic view: stepped diagram (TDIAG) + explanation (TOPICS) + widget (WIDGETS) ================= */
const LR={r:16,a:32,t:{q:true,k:false,v:true,o:false,mlp:false}};
const TP={cur:null,sel:null,hov:null,stp:null,id:null};
function loraWidget(){
  const d=S.d,kvd=kvH('GQA')*dHead(),ff=S.ff,N=S.N,r=LR.r;
  const mats={q:[[d,d]],k:[[d,kvd]],v:[[d,kvd]],o:[[d,d]],mlp:[[d,ff],[d,ff],[ff,d]]};
  let train=0,full=0;
  Object.keys(mats).forEach(k=>{if(LR.t[k])mats[k].forEach(([a,b])=>{train+=r*(a+b);full+=a*b;});});
  train*=N;full*=N;const tot=counts('GQA').total;
  return `<div class="card pad"><h3>LoRA calculator — uses the model configuration above</h3>
  <div class="toolbar" style="margin:10px 0;gap:10px 22px" id="lora-ctl">
    <label class="rg">Rank r <b>${r}</b><input type="range" data-k="r" min="1" max="256" step="1" value="${r}"></label>
    <label class="rg">Alpha α <b>${LR.a}</b><input type="range" data-k="a" min="1" max="512" step="1" value="${LR.a}"></label>
    ${[['q','W_Q'],['k','W_K'],['v','W_V'],['o','W_O'],['mlp','MLP']].map(([k,l])=>`<label class="chk"><input type="checkbox" data-t="${k}" ${LR.t[k]?'checked':''}> ${l}</label>`).join('')}
  </div>
  <div class="kv"><span>Trainable parameters</span><span><b>${fmtN(train)}</b>  (${N} layers)</span>
    <span>Share of the model</span><span>${(100*train/tot).toFixed(3)} % of ${fmtN(tot)}</span>
    <span>Instead of training</span><span>${fmtN(full)} in the adapted matrices (${full?(full/Math.max(train,1)).toFixed(0):0}× more)</span>
    <span>Adapter file</span><span>${fmtB(train*2)} at 16-bit</span>
    <span>Update scale α / r</span><span>${(LR.a/r).toFixed(2)}</span>
    <span>Base model memory</span><span>${fmtB(tot*2)} at 16-bit · ${fmtB(tot*0.5)} at 4-bit (QLoRA)</span></div>
  <p class="mut" style="font-size:12.5px;margin-top:8px">Rank and alpha take effect when you release the slider.</p></div>`;
}
/* topic diagrams name colours loosely; map every name onto the shared role palette */
const TD_ROLE={green:'x',purple:'w',yellow:'tok',blue:'attn',red:'s',orange:'k',pink:'hid',gray:'mask',cyan:'prob'};
const tdCol=c=>ROLE[c]||ROLE[TD_ROLE[c]]||c;
function topicSVG(spec,cur,focus){
  const g=new Graph(cur,focus);
  spec.nodes.forEach(n=>{
    const [id,kind,x,y,label,shape,col,dims,step,bands]=n,c=tdCol(col);
    if(kind==='n')g.node({id,x,y,dims:dims||[64,64],col:c,label,shape,step,bands:bands?BANDS:null});
    else if(kind==='o')g.op({id,x,y,r:String(label).length>2?24:18,sym:label,col:c,label:shape||'',step});
    else if(kind==='p')g.pill({id,x,y,w:Math.max(84,String(label).length*7.4+26),h:30,text:label,col:c,step});
    else{const [w,h]=dims,L=[].concat(label).filter(Boolean);
      g.box({id,x,y,r:w/2,col:c,step,aria:L[0],hit:[x-w/2-4,y-h/2-4,w+8,h+8],svg:`<rect class="pillr" x="${x-w/2}" y="${y-h/2}" width="${w}" height="${h}" rx="10"/>`+L.map((t,i)=>`<text class="${i?'shp':'lab'}" x="${x}" y="${y-(L.length-1)*8+i*16+4.5}" text-anchor="middle">${t}</text>`).join('')});}
  });
  spec.edges.forEach(([a,b,step,o])=>g.edge(a,b,Object.assign({step},o||{})));
  (spec.paths||[]).forEach(([d,step,col])=>g.path(d,{step,col:tdCol(col||'gray')}));
  (spec.labels||[]).forEach(([x,y,t,step,col])=>g.label(x,y,t,{step,col:col?tdCol(col):null}));
  return svgWrap(spec.w,spec.h,g.html());
}
function topicInfo(id){
  const spec=TDIAG[TP.id],i=spec&&spec.info[id];if(!i)return null;
  const n=spec.nodes.find(x=>x[0]===id),k=i[2];
  return {t:i[0],c:tdCol(n[6]),d:i[1]||'',r:n[5]&&n[1]==='n'?[['Shape',n[5]]]:[],fm:k?FORMULA[k]:null,why:k?explainKey(k):null};
}
VIEWS.topic={
  show(root,sub){
    const id=TOPICS[sub]?sub:Object.keys(TOPICS)[0],t=TOPICS[id],spec=TDIAG[id];
    Object.assign(TP,{id,cur:null,sel:null,hov:null});this.id=id;
    root.innerHTML=secHead(t.t,t.lead)+(spec?`
    <div class="card pad toolbar"><span class="mut" style="font-size:13px">Hover a block for details · click to pin · right-click for more</span><span class="sp"></span><div class="stepper" id="tp-stepper"></div></div>
    <div class="card dia"><div id="tp-svg" class="svgwrap" style="min-width:${Math.min(spec.w,900)}px;max-width:${Math.round(spec.w*1.15)}px;margin:0 auto"></div></div>
    <div class="card pad narr" id="tp-narr"></div>`:'')+`
    <div id="topic-widget"></div>
    <div class="grid2" style="grid-template-columns:minmax(0,1.15fr) minmax(0,1fr)">
      <div class="card pad"><h3>How it works</h3><ul class="pts">${t.pts.map(p=>`<li>${p}</li>`).join('')}</ul></div>
      <div><div class="card pad insp" id="tp-insp"></div><div class="card pad">${formulaHTML(t.fm)}</div></div>
    </div>
    ${t.vs?`<div class="card pad" style="overflow-x:auto"><h3 style="margin-bottom:8px">Compared with the usual design</h3><table><thead><tr><th></th><th>Usual</th><th>Here</th></tr></thead><tbody>${t.vs.map(r=>`<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('')}</tbody></table></div>`:''}
    ${t.np?`<p class="callout"><b>Not published:</b> ${t.np}</p>`:''}`;
    if(spec){
      TP.stp=new Stepper($('#tp-stepper'),spec.steps.length,i=>{TP.cur=i;this.draw();},1800);
      bindNodes($('#tp-svg'),TP,()=>this.focus());
      $('#tp-narr').addEventListener('click',e=>{const b=e.target.closest('[data-i]');if(b){TP.stp.stop();TP.stp.go(+b.dataset.i);}});
      this.draw();
    }else this.focus();
    this.update();
  },
  hide(){if(TP.stp)TP.stp.stop();},
  update(){const w=$('#topic-widget'),f=WIDGETS[this.id];if(w){w.innerHTML='';if(f){const el=document.createElement('div');w.appendChild(el);f(el);}}},
  draw(){
    const spec=TDIAG[TP.id],i=TP.cur;
    $('#tp-svg').innerHTML=topicSVG(spec,i,TP.sel||TP.hov);this.focus();
    $('#tp-narr').innerHTML=(i==null?'<b>All steps shown.</b> Press <b>▶ Play steps</b> to walk through it one stage at a time.':`<b>${i+1}. ${spec.steps[i][0]}</b><p>${spec.steps[i][1]}</p>`)
      +`<div class="dots">${spec.steps.map((s,j)=>`<button class="dot ${j===i?'cur':i!=null&&j<i?'on':''}" data-i="${j}" title="${j+1}. ${esc(s[0])}" aria-label="Step ${j+1}: ${esc(s[0])}"></button>`).join('')}</div>`;
  },
  focus(){const p=$('#tp-insp');if(p)p.innerHTML=inspHTML(topicInfo(TP.sel||TP.hov),TDIAG[TP.id]?'Hover or click any block in the diagram above for its role and formulas.':'This page has no diagram.');}
};
