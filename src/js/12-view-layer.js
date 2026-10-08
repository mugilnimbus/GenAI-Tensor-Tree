/* ================= 1. Decoder layer — view ================= */
const L={cur:null,sel:null,hov:null,stp:null};
VIEWS.layer={
  show(root){
    root.innerHTML=`${secHead('Decoder Layer (Multi-Head Attention + MLP)','',"l-sub")}
    <div class="card pad toolbar"><span class="lbl">Attention type</span>${segHTML('l-attn',[['MHA','MHA'],['MQA','MQA'],['GQA','GQA']],S.attn)}<span class="sp"></span><div class="stepper" id="l-stepper"></div></div>
    <div class="card dia"><div id="l-svg" class="svgwrap" style="min-width:1100px"></div></div>
    <div class="card pad narr" id="l-narr"></div>
    <div class="grid2"><div class="card pad insp" id="l-insp"></div><div class="card pad" id="l-tot"></div></div>`;
    L.cur=null;L.sel=null;L.hov=null;
    L.stp=new Stepper($('#l-stepper'),14,i=>{L.cur=i;this.draw();});
    segBind($('#l-attn'),v=>setAttn(v));
    bindNodes($('#l-svg'),L,()=>this.focus());
    $('#l-narr').addEventListener('click',e=>{const b=e.target.closest('[data-i]');if(b){L.stp.stop();L.stp.go(+b.dataset.i);}});
    this.update();
  },
  hide(){if(L.stp)L.stp.stop();},
  update(){
    const sub=$('#l-sub');
    if(sub)sub.innerHTML=`Example: T = ${S.T}, d_model = ${S.d}, n_heads = ${S.H}, d_head = ${dHead()}, d_ff = ${S.ff}`+(S.attn!=='MHA'?`, n_kv_heads = ${kvH()} (${S.attn})`:'');
    segSet($('#l-attn'),S.attn);
    this.draw();this.totals();
  },
  draw(){
    $('#l-svg').innerHTML=layerSVG(L.cur,L.sel||L.hov);
    this.focus();this.narr();
  },
  focus(){$('#l-insp').innerHTML=inspHTML(layerInfo(L.sel||L.hov),'Hover any tensor or operation for its shape, parameters and FLOPs — click to pin it. Use <b>▶ Play steps</b> to follow one forward pass.');},
  narr(){
    const st=layerSteps(),i=L.cur;
    $('#l-narr').innerHTML=(i==null?`<b>All steps shown.</b> Press <b>▶ Play steps</b> (or ▶|) to trace the forward pass one operation at a time.`:`<b>${i+1}. ${st[i][0]}</b><p>${st[i][1]}</p>`)
      +`<div class="dots">${st.map((s,j)=>`<button class="dot ${j===i?'cur':i!=null&&j<i?'on':''}" data-i="${j}" title="${j+1}. ${esc(s[0])}" aria-label="Step ${j+1}: ${esc(s[0])}"></button>`).join('')}</div>`;
  },
  totals(){
    const {T,d,H,ff}=S,dh=d/H,kvd=kvH()*dh,c=counts(),by=S.bytes;
    const f=[['QKV projections',2*T*d*(d+2*kvd),C.purple],['QKᵀ and A·V',4*H*T*T*dh,C.red],['Wo',2*T*d*d,C.orange],['MLP (gate, up, down)',6*T*d*ff,C.pink]];
    const tot=f.reduce((a,b)=>a+b[1],0);
    $('#l-tot').innerHTML=`<h3>Cost of one layer</h3>
    <div class="kv" style="margin:8px 0 12px">
      <span>Attention params</span><span>${fmtN(c.attn)}  (Wq+Wk+Wv+Wo)</span>
      <span>MLP params</span><span>${fmtN(c.mlp)}  (gate+up+down)</span>
      <span>Norm params</span><span>${fmtN(c.norms)}</span>
      <span>Total / layer</span><span><b>${fmtN(c.layer)}</b> × ${S.N} layers = ${fmtN(S.N*c.layer)}</span>
      <span>Forward FLOPs</span><span><b>${fmtF(tot)}</b> for T = ${T} tokens</span>
      <span>Score matrix</span><span>${fmtB(H*T*T*by)}  [${H}, ${T}, ${T}] — why FlashAttention exists</span>
      <span>KV for this layer</span><span>${fmtB(2*T*kvd*by)}  (2 · T · ${kvd})</span>
    </div>
    <div class="stack">${f.map(x=>`<i style="flex:${x[1]};background:${x[2]}" title="${x[0]}"></i>`).join('')}</div>
    <div class="leg">${f.map(x=>`<span style="--c:${x[2]}">${x[0]} ${(100*x[1]/tot).toFixed(0)}%</span>`).join('')}</div>`;
  }
};
