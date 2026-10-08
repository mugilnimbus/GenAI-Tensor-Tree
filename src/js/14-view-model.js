/* ================= 2. Full model — view + sampling lab ================= */
const SM={temp:1,k:12,p:1,rep:1,counts:null,last:null};
const CAND=[[' Paris',9.1],[' the',6.0],[' a',5.4],[' located',5.0],[' known',4.5],[' Lyon',4.3],[' not',3.9],[' France',3.4],[' Berlin',2.9],['⏎',2.2],[' beautiful',1.9],[' Rome',1.6]];
SM.counts=new Array(CAND.length).fill(0);
function sampleDist(){
  const words=tokenWords().map(t=>t.toLowerCase());
  const seen=CAND.map(([t])=>words.includes(t.trim().toLowerCase()));
  const adj=CAND.map(([,l],i)=>seen[i]&&SM.rep!==1?(l>0?l/SM.rep:l*SM.rep):l);
  const p0=softmax(CAND.map(c=>c[1])),p1=softmax(adj.map(v=>v/SM.temp));
  const order=p1.map((_,i)=>i).sort((a,b)=>p1[b]-p1[a]).slice(0,SM.k);
  const zk=order.reduce((a,i)=>a+p1[i],0);
  let acc=0;const keep=[];
  for(const i of order){keep.push(i);acc+=p1[i]/zk;if(acc>=SM.p-1e-9)break;}
  const zf=keep.reduce((a,i)=>a+p1[i],0);
  const p2=p1.map((v,i)=>keep.includes(i)?v/zf:0);
  return {p0,p1,p2,seen,keep};
}
function drawToken(p2){let r=Math.random(),a=0;for(let i=0;i<p2.length;i++){a+=p2[i];if(r<a)return i;}return p2.findIndex(v=>v>0);}

VIEWS.model={
  show(root){
    root.innerHTML=`${secHead('Full Decoder-Only Model (Inference)','Example: Llama / Qwen style — pre-norm, RoPE, GQA optional. Click any block, or step through the whole pipeline.')}
    <div class="card pad toolbar"><label class="f">Position scheme<select id="m-pos">${Object.entries(POS).map(([k,v])=>`<option value="${k}">${v.n}</option>`).join('')}</select></label><span class="sp"></span><div class="stepper" id="m-stepper"></div></div>
    <div class="card dia"><div id="m-svg" class="svgwrap" style="min-width:1200px"></div></div>
    <div class="card pad narr" id="m-narr"></div>
    <div class="grid2"><div class="card pad insp" id="m-insp"></div><div class="card pad" id="m-params"></div></div>
    <div class="grid2">
      <div class="card pad"><h3>Input text → token IDs</h3><p class="mut" style="font-size:13px">Type a prompt. IDs are illustrative hashes into a ${'vocabulary of V'} entries; a real BPE tokenizer splits and numbers differently.</p>
        <input type="text" id="m-text" aria-label="Prompt text" maxlength="120"><div id="m-toks" class="seq" style="margin-top:10px"></div></div>
      <div class="card pad sampl" id="m-samp"></div>
    </div>`;
    M.cur=null;M.sel=null;M.hov=null;
    $('#m-pos').value=M.pos;$('#m-text').value=M.text;
    $('#m-pos').addEventListener('change',e=>{M.pos=e.target.value;this.update();});
    $('#m-text').addEventListener('input',e=>{M.text=e.target.value;this.tokens();this.draw();this.samp();});
    M.stp=new Stepper($('#m-stepper'),7,i=>{M.cur=i;this.draw();},1600);
    bindNodes($('#m-svg'),M,()=>this.focus());
    $('#m-narr').addEventListener('click',e=>{const b=e.target.closest('[data-i]');if(b){M.stp.stop();M.stp.go(+b.dataset.i);}});
    $('#m-samp').addEventListener('input',e=>{const k=e.target.dataset.k;if(k){SM[k]=+e.target.value;this.samp();}});
    $('#m-samp').addEventListener('click',e=>{
      const b=e.target.closest('[data-draw]');if(!b)return;
      const n=+b.dataset.draw;
      if(n===0){SM.counts.fill(0);SM.last=null;}
      else{const {p2}=sampleDist();for(let i=0;i<n;i++){SM.last=drawToken(p2);SM.counts[SM.last]++;}}
      this.samp();
    });
    this.sampInit();
    this.update();
  },
  hide(){if(M.stp)M.stp.stop();},
  update(){this.draw();this.tokens();this.params();this.samp();},
  draw(){$('#m-svg').innerHTML=modelSVG(M.cur,M.sel||M.hov);this.focus();this.narr();},
  focus(){$('#m-insp').innerHTML=inspHTML(modelInfo(M.sel||M.hov),'Hover or click any block. Click a <b>Decoder Layer</b> to jump into the detailed layer diagram.');},
  narr(){
    const st=modelSteps(),i=M.cur;
    $('#m-narr').innerHTML=(i==null?'<b>All stages shown.</b> Press <b>▶ Play steps</b> to follow one token batch from text to the next-token ID.':`<b>${i+1}. ${st[i][0]}</b><p>${st[i][1]}</p>`)
      +`<div class="dots">${st.map((s,j)=>`<button class="dot ${j===i?'cur':i!=null&&j<i?'on':''}" data-i="${j}" title="${j+1}. ${esc(s[0])}" aria-label="Step ${j+1}: ${esc(s[0])}"></button>`).join('')}</div>`;
  },
  tokens(){
    const w=tokenWords();
    $('#m-toks').innerHTML=w.length?w.map(t=>`<span class="tk2" title="${esc(t)}">${esc(t)}<b style="color:var(--yellow);margin-left:6px">${hashId(t,S.V)}</b></span>`).join(''):'<span class="mut">(empty prompt)</span>';
  },
  params(){
    const c=counts(),N=S.N;
    const parts=[['Embedding',c.emb,C.purple],['Attention',N*c.attn,C.red],['MLP',N*c.mlp,C.pink],['Norms',N*c.norms+S.d,C.blue],['LM head',S.tie?0:c.emb,C.yellow]];
    const tot=parts.reduce((a,b)=>a+b[1],0);
    $('#m-params').innerHTML=`<h3>Where the parameters live</h3>
    <p style="font-size:26px;font-weight:700;margin:6px 0 2px">${fmtN(tot)} <span class="mut" style="font-size:14px;font-weight:500">parameters · ${fmtB(tot*S.bytes)} of weights at ${S.bytes} B each</span></p>
    <div class="stack" style="height:18px">${parts.map(p=>`<i style="flex:${Math.max(p[1],1)};background:${p[2]}"></i>`).join('')}</div>
    <table><tbody>${parts.map(p=>`<tr><td><span class="chip" style="border-color:${p[2]};color:${p[2]}">${p[0]}</span></td><td class="n">${fmtN(p[1])}</td><td class="n">${(100*p[1]/tot).toFixed(1)}%</td></tr>`).join('')}</tbody></table>
    <p class="mut" style="font-size:12.5px">Attention = ${N} × (Wq + Wk + Wv + Wo) with ${S.attn} (${kvH()} KV heads). MLP = ${N} × 3 · ${S.d} · ${S.ff}. Toggle “tie embeddings” in the configuration bar to share E with the LM head.</p>`;
  },
  sampInit(){
    $('#m-samp').innerHTML=`<h3>Sampling lab — next token after “<span id="m-st"></span>”</h3>
    <div class="toolbar" style="gap:10px 18px;margin:8px 0">${[['temp','Temperature',0.05,2,0.05],['k','Top-k',1,12,1],['p','Top-p',0.05,1,0.05],['rep','Repetition penalty',1,2,0.05]].map(([k,l,mn,mx,st])=>`<label class="rg">${l} <b id="sv-${k}"></b><input type="range" data-k="${k}" min="${mn}" max="${mx}" step="${st}" value="${SM[k]}"></label>`).join('')}</div>
    <div id="m-srows"></div>
    <div class="toolbar" style="margin-top:10px"><button class="btn pri" data-draw="1">Sample 1</button><button class="btn" data-draw="100">Sample ×100</button><button class="btn" data-draw="0">Reset counts</button><span class="mut mono" id="m-sstat" style="font-size:12px"></span></div>
    <p class="mut" id="m-snote" style="font-size:12.5px;margin-top:8px"></p>`;
  },
  samp(){
    const {p0,p2,seen,keep}=sampleDist(),tot=SM.counts.reduce((a,b)=>a+b,0);
    const ent=-p2.reduce((a,p)=>p>0?a+p*Math.log2(p):a,0);
    $('#m-st').textContent=M.text.slice(-40);
    $('#sv-temp').textContent=SM.temp.toFixed(2);$('#sv-k').textContent=SM.k;$('#sv-p').textContent=SM.p.toFixed(2);$('#sv-rep').textContent=SM.rep.toFixed(2);
    $('#m-srows').innerHTML=CAND.map(([t,l],i)=>`<div class="srow ${keep.includes(i)?'':'off'}"><span class="tk" title="logit ${l}">${esc(t)}${seen[i]?'<sup style="color:var(--orange)"> seen</sup>':''}</span><div class="bar"><span class="gh2" style="width:${(100*p0[i]).toFixed(1)}%"></span><span class="fl" style="width:${(100*p2[i]).toFixed(1)}%"></span></div><span class="pv">${(100*p2[i]).toFixed(1)}%</span><span class="cn">${SM.counts[i]||''}</span></div>`).join('');
    $('#m-sstat').innerHTML=`${SM.last!=null?`last: <b style="color:var(--yellow)">${esc(CAND[SM.last][0])}</b> · `:''}${tot?`${tot} draws · `:''}entropy ${ent.toFixed(2)} bits · ${keep.length} candidate${keep.length>1?'s':''}`;
    $('#m-snote').innerHTML=`Grey = softmax at T = 1 with no filters; yellow = final distribution after penalty → temperature → top-k → top-p. ${SM.temp<=0.1?'Temperature ≈ 0 behaves like <b>greedy</b> decoding. ':''}Tokens already in your prompt are marked “seen” and are penalised.`;
  }
};
