/* ================= tensor core: config, to-scale drawing, step model, architectures, numeric demos ================= */
const fN=n=>{const a=Math.abs(n);return a>=1e12?(n/1e12).toFixed(2)+'T':a>=1e9?(n/1e9).toFixed(2)+'B':a>=1e6?(n/1e6).toFixed(1)+'M':a>=1e4?(n/1e3).toFixed(1)+'K':String(Math.round(n));};
const fF=f=>{if(!f)return '0';const u=['','K','M','G','T','P'];let i=0;while(f>=1000&&i<5){f/=1000;i++;}return (f>=100?f.toFixed(0):f>=10?f.toFixed(1):f.toFixed(2))+' '+u[i]+'FLOPs';};
const fB=b=>{const u=['B','KiB','MiB','GiB','TiB'];let i=0;while(b>=1024&&i<4){b/=1024;i++;}return (b>=100||!i?b.toFixed(0):b.toFixed(1))+' '+u[i];};

/* ---------- configuration (defaults) ---------- */
const K={T:1000,d:4096,H:32,Hkv:8,ff:11008,V:100000,N:32,S:600,W:256,E:8,k:2,C:2,h:0,scale:'lin',mode:'one',a:'mha',b:'gqa'};
const FIELDS=[['S','source S'],['W','window W'],['E','experts E'],['k','top-k']];

/* ---------- tensors ---------- */
const COL={res:'#72d945',act:'#45c1dd',w:'#a645dc',mask:'#d6459b',prob:'#45d984',tok:'#d6cc45'};
const mk=kind=>(n,r,c,o={})=>Object.assign({n,r,c,kind},o);
const act=mk('act'),wgt=mk('w'),res=mk('res'),msk=mk('mask'),prb=mk('prob'),tok=mk('tok');
/* tensor → role → colour */
function roleOf(t){
  if(t.kind==='w')return 'w';
  if(t.kind==='tok')return 'tok';
  const n=String(t.n).replace(/<[^>]+>/g,'').replace(/[′″ᵀ]/g,'');
  if(n==='logits')return 'tok';
  if(t.kind==='prob'&&n==='p')return 'prob';
  if(t.kind==='mask'&&/^M/.test(n))return 'mask';
  if(/^X̂/.test(n)||/^h[₁₂₃]/.test(n))return 'xn';
  if(/^Xatt/.test(n))return 'xatt';
  if(/^(X|R|M$)/.test(n)||/^x/.test(n))return 'x';
  if(/^c[A-Z]/.test(n))return 'lora';
  if(/^[Qq]/.test(n))return 'q';
  if(/^[Kk]/.test(n))return 'k';
  if(/^V/.test(n))return 'v';
  if(/^S/.test(n)||n==='s')return 's';
  if(n==='A')return 'a';
  if(/^(O|\d+ × O)/.test(n))return 'o';
  if(/^(Attn|XAttn|GDN|Mamba|TimeMix)/.test(n))return 'attn';
  if(n==='G')return 'gate';
  if(/^(SiLU|GELU|ReLU)\(/.test(n))return 'sg';
  if(n==='U')return 'up';
  if(/^Hid/.test(n))return 'hid';
  if(/^(MLP|ChanMix|\d+ × E)/.test(n))return 'mlp';
  if(n==='r'||n==='g')return 'router';
  if(/^(Z|Δ|E)/.test(n))return 'lora';
  return 'other';
}
const tcol=t=>ROLE[roleOf(t)];
const shp=t=>t.shape||`[${t.r} × ${t.c}]`;
function dimPx(n){
  const ref=Math.max(K.d,K.ff),P=K.px||300,px=K.scale==='lin'?n*P/ref:Math.sqrt(n)*P/Math.sqrt(ref),CAP=P*1.1;
  return [px>CAP?P*.63:Math.max(px,3),px>CAP];
}
function tsvg(t){
  const [w,bw]=dimPx(t.c),[h,bh]=dimPx(t.r),c=tcol(t);let s='';
  s+=`<rect x="1" y="1" width="${w}" height="${h}" fill="${c}" fill-opacity="${t.mask?.06:.16}" stroke="${c}" stroke-width="1.4"/>`;
  if(t.mask==='causal')s+=`<path d="M1 1L${w+1} ${h+1}H1Z" fill="${c}" fill-opacity=".5"/>`;
  if(t.mask==='full')s+=`<rect x="1" y="1" width="${w}" height="${h}" fill="${c}" fill-opacity=".4"/>`;
  if(t.mask==='window'){const f=clamp(t.frac||.2,0.02,1);s+=`<path d="M1 1L${w+1} ${h+1}H${1+w*(1-f)}L1 ${1+h*f}Z" fill="${c}" fill-opacity=".55"/>`;}
  if(t.stripes>1){const m=Math.min(t.stripes,Math.floor(w/3));let p='';for(let i=1;i<m;i++)p+=`M${(1+i*w/m).toFixed(1)} 1V${h+1}`;if(p)s+=`<path d="${p}" stroke="${c}" stroke-width=".8" opacity=".75"/>`;}
  if(t.hiRow!=null){const y=t.hiRow==='last'?h-2:1;s+=`<rect x="1" y="${y}" width="${w}" height="3" fill="#d65f48"/>`;}
  if(bw){const x=1+w/2;s+=`<path d="M${x-3} -3l6 ${(h+8)/4}l-6 ${(h+8)/4}l6 ${(h+8)/4}l-6 ${(h+8)/4}" stroke="var(--txt)" stroke-width="1.6" fill="none"/>`;}
  if(bh){const y=1+h/2;s+=`<path d="M-3 ${y-3}l${(w+8)/4} 6l${(w+8)/4} -6l${(w+8)/4} 6l${(w+8)/4} -6" stroke="var(--txt)" stroke-width="1.6" fill="none"/>`;}
  return `<figure class="tz" style="--c:${c}"><figcaption>${t.n}</figcaption><svg width="${w+2}" height="${h+2}" role="img" aria-label="${esc(String(t.n).replace(/<[^>]+>/g,''))} ${esc(shp(t))}">${s}</svg><span class="shp">${shp(t)}</span></figure>`;
}

/* ---------- steps ---------- */
function st(key,stage,title,fx,expr,o={}){
  const ts=expr.filter(x=>typeof x==='object');
  const params=o.params!=null?o.params:ts.filter(t=>t.kind==='w').reduce((a,t)=>a+t.r*t.c,0);
  return Object.assign({key,stage,title,fx,expr,params,flops:0,scope:'layer',sig:fx.replace(/<[^>]+>/g,'')+'|'+ts.map(shp).join('')},o);
}
const mmF=(a,w)=>2*a.r*a.c*w.c;
const MASTER=['tok','emb','pos','hc0','nghash','nglook','nggate','nginj','mix1','gproj','gstate','gread','hcw1','mix2','idx','pool','bscore','bsel','norm1','cq','q','qr','lqa','lqb','lqadd','k','v','lva','lvb','lvadd','ckv','kc','vc','kr','split','rope','cache','catq','catk','scores','scale','mask','softmax','av','concat','wo','add1','hcw2','post1','mem','xq','xk','xv','xsplit','xscores','xscale','xsoftmax','xav','xconcat','xwo','xadd','xpost','norm2','mix3','router','topk','experts','gate','up','silu','mul','down','fc1','actf','fc2','combine','add2','hcw3','post2','mixout','final','last','head','vsoft','sample','mtp'];
function stepHTML(s,no,diff){
  const row=s.expr.map(x=>typeof x==='string'?`<span class="op${x.length>2?' s':''}">${x}</span>`:tsvg(x)).join('');
  const ts=s.expr.filter(x=>typeof x==='object'),out=ts[ts.length-1],ins=ts.slice(0,-1);
  const flow=s.flow?s.flow:ins.length?ins.map(shp).join(' , ')+' → <b>'+shp(out)+'</b>':'<b>'+shp(out)+'</b>';
  return `<div class="step${diff?' diff':''}" data-i="${s._i}" data-key="${s.key}" data-op="${s.op||''}" tabindex="0" role="button" aria-expanded="false">
  <div class="sh"><span class="no">${no}</span><b>${s.title}</b><span class="fx">${s.fx}</span>${s.head?'<span class="tag h">one head</span>':''}${s.frozen?'<span class="tag f">frozen</span>':''}${/^(l[qv]|ad\d|p[kv]$|ptok$|ia3|dab$|dmag$)/.test(s.key)?'<span class="tag t">trainable</span>':''}${diff?'<span class="tag d">differs</span>':''}</div>
  <div class="row">${row}</div>
  <div class="meta">${flow}${s.params?` · params <b>${fN(s.params)}</b>${s.active!=null?` (active ${fN(s.active)})`:''}`:''}${s.flops?` · <b>${fF(s.flops)}</b>`:''}${s.mult?` · ×${s.mult} heads`:''}</div>
  ${s.note?`<div class="note">${s.note}</div>`:''}${formulaHTML(stepFormulas(s))}${whyHTML(explainStep(s))}<div class="demo" hidden></div></div>`;
}
function totals(steps){
  const sum=(sc,f)=>steps.filter(s=>s.scope===sc).reduce((a,s)=>a+(s[f]||0),0);
  const lp=sum('layer','params'),lf=sum('layer','flops'),act=steps.filter(s=>s.scope==='layer').reduce((a,s)=>a+(s.active!=null?s.active:s.params),0);
  let big=null;steps.forEach(s=>s.expr.forEach(t=>{if(typeof t==='object'&&t.kind!=='w'&&(!big||t.r*t.c*(t.rep||1)>big.r*big.c*(big.rep||1)))big=t;}));
  return {lp,lf,act,io:sum('in','params')+sum('out','params'),total:sum('in','params')+sum('out','params')+K.N*lp,big};
}

/* ================= building blocks ================= */
const sb=(a,b)=>`${a}<sub>${b}</sub>`;
function inputSteps(S,c,R,pos){
  const {d,V}=c;
  S.push(st('tok','Input','Token IDs','ids = tokenizer(text)',[tok('ids',R,1,{shape:`[${R}]`})],{scope:'in',note:`${R===1?'Only the newest token':'The text'} becomes ${R} integer ID${R>1?'s':''} in 0 … ${V-1}.`}));
  S.push(st('emb','Input','Embedding lookup','X = E[ids]',[tok('ids',R,1,{shape:`[${R}]`}),'→ pick rows of',wgt('E',V,d),'=',res('X',R,d)],{scope:'in',op:'lookup',note:`No multiplication: each ID selects one row of the table. E holds ${fN(V*d)} parameters.`}));
  if(pos==='learned')S.push(st('pos','Input','Add learned positions','X = X + P[0:T]',[res('X',R,d),'+',wgt('P',R,d),'=',res('X',R,d)],{scope:'in',op:'add',note:'Row t of a trained position table is added to token t. Position is injected once, at the input; context length is fixed by the table.'}));
  if(pos==='sin')S.push(st('pos','Input','Add sinusoidal positions','X = X + PE(t)',[res('X',R,d),'+',act('PE',R,d),'=',res('X',R,d)],{scope:'in',op:'add',params:0,note:'Fixed sin/cos waves of many frequencies, one row per position. No parameters.'}));
}
function normStep(S,key,stage,type,inN,outN,R,d,scope){
  const rms=type==='RMSNorm';
  S.push(st(key,stage,type,rms?`${outN} = ${inN} / √(mean(${inN}²)+ε) · γ`:`${outN} = (${inN} − μ)/σ · γ + β`,[res(inN,R,d),`→ ${type} →`,(key.startsWith('post')||key==='xpost'?res:act)(outN,R,d)],{op:'norm',ntype:rms?'rms':'ln',params:rms?d:2*d,scope:scope||'layer',note:rms?'Each row (token) is divided by its own root-mean-square, then scaled by a learned γ of length d. Shape is unchanged.':'Each row is mean-centred, divided by its standard deviation, then scaled and shifted by learned γ and β.'}));
}
function addStep(S,key,stage,fx,a,b,out,R,d,note){
  S.push(st(key,stage,'Residual add',fx,[res(a,R,d),'+',act(b,R,d),'=',res(out,R,d)],{op:'add',note}));
}
/* attention: o={R,L,kv,mask,rope,xin,kin,P,stage,cache} */
function attn(S,c,o){
  const {d,H}=c,dh=d/H,{R,L,kv}=o,kvd=kv*dh,P=o.P||'',stage=o.stage||'Attention',h=clamp(c.h,0,H-1),gsz=H/kv,g=Math.floor(h/gsz);
  const Lk=o.cache?1:L,xin=o.xin,kin=o.kin||xin,cross=P==='x';
  const din=o.din||d,Wq=wgt(sb('W','Q'),din,d),Wk=wgt(sb('W','K'),din,kvd),Wv=wgt(sb('W','V'),din,kvd);
  const share=kv<H?` Only ${kv} K/V head${kv>1?'s':''} of width ${dh} → the matrix is [${d} × ${kvd}] instead of [${d} × ${d}].`:'';
  S.push(st(P+'q',stage,'Query projection',`Q = ${xin}·${sb('W','Q')}`,[act(xin,R,din),'×',Wq,'=',act('Q',R,d)],{op:'matmul',flops:2*R*d*d,note:`Every row of ${xin} (one token) is multiplied by the same weight matrix.`}));
  S.push(st(P+'k',stage,'Key projection',`K = ${kin}·${sb('W','K')}`,[act(kin,Lk,din),'×',Wk,'=',act('K',Lk,kvd)],{op:'matmul',flops:2*Lk*d*kvd,note:(cross?'Keys come from the <b>encoder</b> output M, not from the decoder.':'What each token offers to be matched against.')+share}));
  S.push(st(P+'v',stage,'Value projection',`V = ${kin}·${sb('W','V')}`,[act(kin,Lk,din),'×',Wv,'=',act('V',Lk,kvd)],{op:'matmul',flops:2*Lk*d*kvd,note:(cross?'Values also come from the encoder output M.':'The content each token hands over when it is attended to.')+share}));
  S.push(st(P+'split',stage,'Split into heads',`[·, H·d<sub>h</sub>] → [·, H, d<sub>h</sub>]`,[act('Q',R,d,{stripes:H,shape:`[${R} × ${H} × ${dh}]`}),act('K',Lk,kvd,{stripes:kv,shape:`[${Lk} × ${kv} × ${dh}]`}),act('V',Lk,kvd,{stripes:kv,shape:`[${Lk} × ${kv} × ${dh}]`})],{op:'split',flow:`Q <b>[${R} × ${H} × ${dh}]</b> · K, V <b>[${Lk} × ${kv} × ${dh}]</b>`,note:`Pure reshape — no arithmetic. The ${d} columns of Q are cut into ${H} stripes of ${dh}; K and V into ${kv} stripe${kv>1?'s':''}.${kv<H?` Each K/V stripe will be reused by ${gsz} query heads.`:''}`}));
  if(o.rope)S.push(st('rope',stage,'RoPE — rotate Q and K',`(q<sub>2i</sub>, q<sub>2i+1</sub>) → R(t·θ<sub>i</sub>)·(q<sub>2i</sub>, q<sub>2i+1</sub>)`,[act('Q',R,d,{stripes:H,shape:`[${R} × ${H} × ${dh}]`}),'→ rotate pairs →',act('Q′',R,d,{stripes:H,shape:`[${R} × ${H} × ${dh}]`}),act('K′',Lk,kvd,{stripes:kv,shape:`[${Lk} × ${kv} × ${dh}]`})],{op:'rope',flow:`shapes unchanged · Q′ <b>[${R} × ${H} × ${dh}]</b> · K′ <b>[${Lk} × ${kv} × ${dh}]</b>`,note:`Inside each head, every pair of columns is rotated by an angle t·θ<sub>i</sub> that grows with the token position t. After this, q·k depends only on the distance between two tokens. V is not rotated. No parameters.`}));
  if(o.cache)S.push(st('cache',stage,'Append to KV cache',`K = [K<sub>cache</sub> ; k<sub>new</sub>]`,[act(sb('K','cache'),L-1,kvd,{stripes:kv}),'append',act(sb('k','new'),1,kvd),'=',act('K',L,kvd,{stripes:kv,shape:`[${L} × ${kv} × ${dh}]`})],{op:'cache',note:`Keys and values of the ${L-1} earlier tokens were computed in previous steps and are simply read back; the new row is appended (same for V). This is why decode never recomputes the prefix.`}));
  const Le=o.mask==='window'?Math.min(L,c.W):L,mo=o.mask==='window'?{mask:'window',frac:c.W/L}:o.mask==='causal'?{mask:'causal'}:{};
  const kn=kv<H?sb('K',`g=${g}`):sb('K',`h=${h}`),vn=kv<H?sb('V',`g=${g}`):sb('V',`h=${h}`),qn=sb('Q',`h=${h}`);
  const hd={head:true,mult:H};
  S.push(st(P+'scores',stage,'Scores',`S = ${qn}·${kn}ᵀ`,[act(qn,R,dh),'×',act(kn+'ᵀ',dh,L),'=',msk('S',R,L,{rep:H})],Object.assign({op:'qkt',flops:2*R*Le*dh*H,note:`Row i of Q (query of token i) is dotted with every key: entry S[i, j] says how much token i wants token j.${kv<H?` Query head ${h} uses shared K/V head ${g} (heads ${g*gsz}–${(g+1)*gsz-1} all do).`:''}${o.mask==='window'?` Only the last ${c.W} keys are actually computed per query.`:''} All ${H} heads together: [${H} × ${R} × ${L}].`},hd)));
  S.push(st(P+'scale',stage,'Scale',`S′ = S / √d<sub>h</sub> = S / √${dh}`,[msk('S',R,L),`÷ ${Math.sqrt(dh).toFixed(2)}`,'=',msk('S′',R,L)],Object.assign({op:'scale',note:'Keeps the scores at unit variance so the softmax does not saturate.'},hd)));
  if(o.mask==='causal'||o.mask==='window')S.push(st('mask',stage,o.mask==='window'?'Sliding-window causal mask':'Causal mask',`S″ = S′ + M ,  M[i,j] = ${o.mask==='window'?`0 if i−${c.W} < j ≤ i else −∞`:'0 if j ≤ i else −∞'}`,[msk('S′',R,L),'+',msk('M',R,L,mo),'=',msk('S″',R,L,mo)],Object.assign({op:'mask',note:o.mask==='window'?`Shaded band = allowed. Token i may only look at the previous ${c.W} tokens, so cost is O(T·W) instead of O(T²).`:'Shaded triangle = allowed (j ≤ i). Future positions get −∞ and will become probability 0.'},hd)));
  S.push(st(P+'softmax',stage,'Softmax over keys',`A[i,·] = softmax(S${o.mask==='none'?'′':'″'}[i,·])`,[msk('S',R,L,mo),'→ softmax each row →',prb('A',R,L,mo)],Object.assign({op:'softmax',note:`Each row becomes a probability distribution over the ${L} ${cross?'source':''} positions${o.mask==='none'&&!cross&&R>1?' — left <b>and</b> right, nothing is masked':''}.`},hd)));
  S.push(st(P+'av',stage,'Weighted sum of values',`${sb('O','h')} = A·${vn}`,[prb('A',R,L,mo),'×',act(vn,L,dh),'=',act(sb('O',`h=${h}`),R,dh)],Object.assign({op:'av',flops:2*R*Le*dh*H,note:'Row i of the output is the average of all value vectors, weighted by row i of A.'},hd)));
  S.push(st(P+'concat',stage,`Concat all ${H} heads`,`O = [O<sub>0</sub> | O<sub>1</sub> | … | O<sub>${H-1}</sub>]`,[act(`${H} × O<sub>h</sub>`,R,d,{stripes:H,shape:`${H} × [${R} × ${dh}]`}),'→ concat →',act('O',R,d)],{op:'concat',note:`The ${H} outputs of width ${dh} are laid side by side: ${H} × ${dh} = ${d} columns again.`}));
  S.push(st(P+'wo',stage,'Output projection',`Attn = O·${sb('W','O')}`,[act('O',R,d),'×',wgt(sb('W','O'),d,din),'=',act(cross?'XAttn':'Attn',R,din)],{op:'matmul',flops:2*R*d*d,note:'Mixes information across heads and maps it back into the residual stream.'}));
}
function swiglu(S,c,R,xin,o={}){
  const {d,ff}=c,pm=o.pm||1,am=o.am,tg=o.tag||'',P=p=>({op:'matmul',params:p*pm,active:am?p*am:undefined,flops:2*R*d*ff*pm});
  S.push(st('gate','MLP','Gate projection'+tg,`G = ${xin}·${sb('W','gate')}`,[act(xin,R,d),'×',wgt(sb('W','gate'),d,ff),'=',act('G',R,ff)],Object.assign(P(d*ff),{note:'Expands every token from d to d<sub>ff</sub>. This branch will become the gate.'})));
  S.push(st('up','MLP','Up projection'+tg,`U = ${xin}·${sb('W','up')}`,[act(xin,R,d),'×',wgt(sb('W','up'),d,ff),'=',act('U',R,ff)],Object.assign(P(d*ff),{note:'A second, parallel expansion carrying the content.'})));
  S.push(st('silu','MLP','SiLU'+tg,'σ(G) = G · sigmoid(G)',[act('G',R,ff),'→ SiLU →',act('SiLU(G)',R,ff)],{op:'act',fn:'silu',note:'Applied to every element independently; shape unchanged.'}));
  S.push(st('mul','MLP','Element-wise multiply'+tg,'Hid = SiLU(G) ⊙ U',[act('SiLU(G)',R,ff),'⊙',act('U',R,ff),'=',act('Hid',R,ff)],{op:'mul',note:'Element [i, j] of one matrix times element [i, j] of the other — not a matrix product.'}));
  S.push(st('down','MLP','Down projection'+tg,`MLP = Hid·${sb('W','down')}`,[act('Hid',R,ff),'×',wgt(sb('W','down'),ff,d),'=',act('MLP',R,d)],Object.assign(P(d*ff),{note:'Squeezes the wide hidden state back to model width.'})));
}
function ffn(S,c,R,xin,fn,hid){
  const {d}=c,F={gelu:'GELU',relu:'ReLU'}[fn];
  S.push(st('fc1','MLP','Expand',`Hid = ${xin}·W<sub>1</sub> + b<sub>1</sub>`,[act(xin,R,d),'×',wgt('W<sub>1</sub>',d,hid),'=',act('Hid',R,hid)],{op:'matmul',params:d*hid+hid,flops:2*R*d*hid,note:`One expansion matrix (plus a bias) — compare with the two parallel matrices of SwiGLU. Hidden width here is ${hid} = 4·d.`}));
  S.push(st('actf','MLP',F,fn==='gelu'?'GELU(x) = x·Φ(x)':'ReLU(x) = max(0, x)',[act('Hid',R,hid),`→ ${F} →`,act(`${F}(Hid)`,R,hid)],{op:'act',fn,note:'Applied element by element; no gate branch.'}));
  S.push(st('fc2','MLP','Project back',`MLP = ${F}(Hid)·W<sub>2</sub> + b<sub>2</sub>`,[act(`${F}(Hid)`,R,hid),'×',wgt('W<sub>2</sub>',hid,d),'=',act('MLP',R,d)],{op:'matmul',params:d*hid+d,flops:2*R*d*hid,note:'Back to model width.'}));
}
function outSteps(S,c,R,o={}){
  const {d,V}=c,xn=o.norm?'X̂':'X';
  if(o.norm)normStep(S,'final','Output',o.norm,'X','X̂',R,d,'out');
  if(R>1)S.push(st('last','Output','Take the last token',`x<sub>T</sub> = ${xn}[T−1, :]`,[act(xn,R,d,{hiRow:'last'}),'→ last row →',act(sb('x','T'),1,d)],{scope:'out',op:'select',note:'After N layers only the final position is needed to predict the next token (red row).'}));
  S.push(st('head','Output','LM head',`logits = x<sub>T</sub>·${sb('W','LM')}`,[act(sb('x','T'),1,d),'×',wgt(sb('W','LM'),d,V),'=',msk('logits',1,V)],{scope:'out',op:'matmul',params:o.tied?0:V*d,flops:2*d*V,note:`One score per vocabulary entry.${o.tied?' The matrix is the embedding table transposed (tied weights) — no extra parameters.':''}`}));
  S.push(st('vsoft','Output','Softmax over vocabulary','p = softmax(logits / temperature)',[msk('logits',1,V),'→ softmax →',prb('p',1,V)],{scope:'out',op:'softmax',note:'Turns scores into probabilities that sum to 1.'}));
  S.push(st('sample','Output','Sampling','next = sample(p)',[prb('p',1,V),'→ temperature · top-k · top-p · repetition penalty →',tok('next id',1,1,{shape:'[1]'})],{scope:'out',note:'One token ID is drawn; it is appended to the input and the whole pass repeats.'}));
}
/* modern decoder family: o={kv,mask,mlp,R,cache} */
function modern(c,o){
  const S=[],{d,T}=c,R=o.R||T,L=T;
  inputSteps(S,c,R,null);
  normStep(S,'norm1','Attention','RMSNorm','X','X̂',R,d);
  attn(S,c,{R,L,kv:o.kv,mask:R===1?'none':o.mask,rope:true,xin:'X̂',cache:o.cache});
  addStep(S,'add1','Attention',`${sb('X','att')} = X + Attn`,'X','Attn',sb('X','att'),R,d,'The attention result is added to the untouched input (skip connection).');
  normStep(S,'norm2','MLP','RMSNorm',sb('X','att'),sb('X̂','att'),R,d);
  if(o.mlp==='moe'){
    const {E,k}=c,Re=Math.max(1,Math.round(R*k/E));
    S.push(st('router','MLP','Router',`r = ${sb('X̂','att')}·${sb('W','r')}`,[act(sb('X̂','att'),R,d),'×',wgt(sb('W','r'),d,E),'=',msk('r',R,E)],{op:'matmul',flops:2*R*d*E,note:`A tiny matrix scores all ${E} experts for every token.`}));
    S.push(st('topk','MLP',`Top-${k} + softmax`,`g = softmax(top${sb('','k')}(r))`,[msk('r',R,E),`→ keep top ${k} per row →`,prb('g',R,k)],{op:'topk',note:`Each token keeps its ${k} best experts; their scores are renormalised into gate weights. The other ${E-k} experts do no work for this token.`}));
    swiglu(S,c,Re,sb('x','e'),{pm:E,am:k,tag:` (inside one of ${E} experts)`});
    S.push(st('combine','MLP','Weighted combine',`MLP = Σ<sub>i ∈ top-${k}</sub> g<sub>i</sub> · E<sub>i</sub>(x)`,[act(`${k} × E<sub>i</sub>(x)`,R,d,{shape:`${k} × [${R} × ${d}]`}),'× g, sum',' =',act('MLP',R,d)],{op:'combine',note:`Each expert sees only ≈ ${Re} of the ${R} tokens (T·k/E on average). Total MLP weights are ${E}× a dense MLP, but only ${k}× are used per token.`}));
  }else swiglu(S,c,R,sb('X̂','att'));
  addStep(S,'add2','MLP',`${sb('X','out')} = ${sb('X','att')} + MLP`,sb('X','att'),'MLP',sb('X','out'),R,d,`This [${R} × ${d}] tensor is the input X of the next layer. Repeat ${c.N} times.`);
  outSteps(S,c,R,{norm:'RMSNorm'});
  return S;
}
const ARCH=[
 {id:'mha',name:'Dense decoder · MHA  (Llama-2 style)',short:'Decoder · MHA',desc:'RMSNorm → RoPE → multi-head attention with one K/V head per query head → SwiGLU MLP.',
  meta:c=>({norm:'RMSNorm, pre',pos:'RoPE on Q,K (every layer)',mask:'causal',mlp:'SwiGLU (3 matrices)',kv:c.H}),build:c=>modern(c,{kv:c.H,mask:'causal'})},
 {id:'gqa',name:'Dense decoder · GQA  (Llama-3, Qwen, Mistral)',short:'Decoder · GQA',desc:'Same block, but K and V have only H_kv heads; groups of query heads share one K/V head. W_K and W_V shrink and the KV cache shrinks with them.',
  meta:c=>({norm:'RMSNorm, pre',pos:'RoPE on Q,K (every layer)',mask:'causal',mlp:'SwiGLU (3 matrices)',kv:c.Hkv}),build:c=>modern(c,{kv:c.Hkv,mask:'causal'})},
 {id:'mqa',name:'Dense decoder · MQA  (PaLM, Falcon)',short:'Decoder · MQA',desc:'The extreme case: a single K/V head shared by all query heads.',
  meta:c=>({norm:'RMSNorm, pre',pos:'RoPE on Q,K (every layer)',mask:'causal',mlp:'SwiGLU (3 matrices)',kv:1}),build:c=>modern(c,{kv:1,mask:'causal'})},
 {id:'swa',name:'Sliding-window decoder  (Mistral-7B style, GQA)',short:'Sliding window',desc:'GQA block whose mask also hides tokens further back than W. Scores are a band, not a triangle.',
  meta:c=>({norm:'RMSNorm, pre',pos:'RoPE on Q,K (every layer)',mask:`causal + window W=${c.W}`,mlp:'SwiGLU (3 matrices)',kv:c.Hkv,win:true}),build:c=>modern(c,{kv:c.Hkv,mask:'window'})},
 {id:'moe',name:'MoE decoder  (Mixtral, DeepSeek-MoE; GQA attention)',short:'MoE decoder',desc:'Attention is unchanged; the single dense MLP is replaced by a router and E expert MLPs, of which only top-k run per token.',
  meta:c=>({norm:'RMSNorm, pre',pos:'RoPE on Q,K (every layer)',mask:'causal',mlp:`router + ${c.E} SwiGLU experts, top-${c.k}`,kv:c.Hkv}),build:c=>modern(c,{kv:c.Hkv,mask:'causal',mlp:'moe'})},
 {id:'decode',name:'Decode step with KV cache  (same GQA weights, 1 new token)',short:'Decode step',desc:'The same weights as the GQA decoder, but only one new token flows through: every [T × …] becomes [1 × …] and old K, V are read from the cache.',
  meta:c=>({norm:'RMSNorm, pre',pos:'RoPE on Q,K (every layer)',mask:'none needed (1 query row)',mlp:'SwiGLU (3 matrices)',kv:c.Hkv}),build:c=>modern(c,{kv:c.Hkv,mask:'causal',R:1,cache:true})},
 {id:'gpt2',name:'Classic dense decoder  (GPT-2 style)',short:'Classic GPT-2',desc:'LayerNorm, learned absolute positions added at the input, full multi-head attention, GELU MLP with two matrices, tied LM head.',
  meta:c=>({norm:'LayerNorm, pre',pos:'learned, added at input',mask:'causal',mlp:'GELU (2 matrices, 4·d)',kv:c.H}),build:c=>{const S=[],{d,T}=c;
    inputSteps(S,c,T,'learned');normStep(S,'norm1','Attention','LayerNorm','X','X̂',T,d);
    attn(S,c,{R:T,L:T,kv:c.H,mask:'causal',xin:'X̂'});addStep(S,'add1','Attention',`${sb('X','att')} = X + Attn`,'X','Attn',sb('X','att'),T,d,'Skip connection.');
    normStep(S,'norm2','MLP','LayerNorm',sb('X','att'),sb('X̂','att'),T,d);ffn(S,c,T,sb('X̂','att'),'gelu',4*d);
    addStep(S,'add2','MLP',`${sb('X','out')} = ${sb('X','att')} + MLP`,sb('X','att'),'MLP',sb('X','out'),T,d,`Input of the next layer. Repeat ${c.N} times.`);
    outSteps(S,c,T,{norm:'LayerNorm',tied:true});return S;}},
 {id:'bert',name:'Encoder-only  (BERT style)',short:'Encoder-only',desc:'No causal mask — every token attends left and right. Post-norm (Add & Norm). The output is one contextual vector per token, fed to a small task head; nothing is generated.',
  meta:c=>({norm:'LayerNorm, post (Add & Norm)',pos:'learned, added at input',mask:'none (bidirectional)',mlp:'GELU (2 matrices, 4·d)',kv:c.H,nocache:true}),build:c=>{const S=[],{d,T,C}=c;
    inputSteps(S,c,T,'learned');attn(S,c,{R:T,L:T,kv:c.H,mask:'none',xin:'X'});
    addStep(S,'add1','Attention','X = X + Attn','X','Attn','X',T,d,'Skip connection — the norm comes <b>after</b> the add.');normStep(S,'post1','Attention','LayerNorm','X',sb('X','att'),T,d);
    ffn(S,c,T,sb('X','att'),'gelu',4*d);addStep(S,'add2','MLP',`X = ${sb('X','att')} + MLP`,sb('X','att'),'MLP','X',T,d,'Skip connection.');normStep(S,'post2','MLP','LayerNorm','X',sb('X','out'),T,d);
    S.push(st('last','Output','Take the [CLS] token','x<sub>cls</sub> = X[0, :]',[act('X',T,d,{hiRow:'first'}),'→ first row →',act(sb('x','cls'),1,d)],{scope:'out',op:'select',note:'For classification the first position summarises the sequence. For tagging, all T rows are used instead.'}));
    S.push(st('head','Output','Task head',`logits = x<sub>cls</sub>·${sb('W','cls')}`,[act(sb('x','cls'),1,d),'×',wgt(sb('W','cls'),d,C),'=',msk('logits',1,C)],{scope:'out',op:'matmul',flops:2*d*C,note:`Projects to ${C} classes, not to the vocabulary.`}));
    S.push(st('vsoft','Output','Softmax over classes','p = softmax(logits)',[msk('logits',1,C),'→ softmax →',prb('p',1,C)],{scope:'out',op:'softmax',note:'Class probabilities. There is no sampling loop.'}));return S;}},
 {id:'encdec',name:'Encoder–decoder  (original Transformer style, decoder layer shown)',short:'Encoder–decoder',desc:'The decoder layer has a second attention block: queries come from the decoder (T rows), keys and values from the encoder output (S rows), so scores are [T × S].',
  meta:c=>({norm:'LayerNorm, post (Add & Norm)',pos:'sinusoidal, added at input',mask:'causal (self) · none (cross)',mlp:'ReLU (2 matrices, 4·d)',kv:c.H,cross:true}),build:c=>{const S=[],{d,T}=c,L=c.S;
    inputSteps(S,c,T,'sin');attn(S,c,{R:T,L:T,kv:c.H,mask:'causal',xin:'X'});
    addStep(S,'add1','Attention','X = X + Attn','X','Attn','X',T,d,'Skip connection.');normStep(S,'post1','Attention','LayerNorm','X','X',T,d);
    S.push(st('mem','Cross-attention','Encoder output','M = Encoder(source tokens)',[res('M',L,d)],{params:0,note:`Computed once by ${c.N} encoder layers (each like the encoder-only block) over the ${L} source tokens, then reused by every decoder layer and every decode step.`}));
    attn(S,c,{R:T,L,kv:c.H,mask:'none',xin:'X',kin:'M',P:'x',stage:'Cross-attention'});
    addStep(S,'xadd','Cross-attention','X = X + XAttn','X','XAttn','X',T,d,'Skip connection.');normStep(S,'xpost','Cross-attention','LayerNorm','X',sb('X','att'),T,d);
    ffn(S,c,T,sb('X','att'),'relu',4*d);addStep(S,'add2','MLP',`X = ${sb('X','att')} + MLP`,sb('X','att'),'MLP','X',T,d,'Skip connection.');normStep(S,'post2','MLP','LayerNorm','X',sb('X','out'),T,d);
    outSteps(S,c,T,{tied:true});return S;}}
];
const archOf=id=>ARCH.find(a=>a.id===id)||ARCH[0];

/* ================= tiny numeric demos: how each operation is carried out ================= */
const fmtC=v=>v===-Infinity?'−∞':typeof v!=='number'?v:Number.isInteger(v)?String(v):v.toFixed(2);
const ints=(r,c,rnd)=>Array.from({length:r},()=>Array.from({length:c},()=>Math.floor(rnd()*6)-2));
const flts=(r,c,rnd)=>Array.from({length:r},()=>Array.from({length:c},()=>Math.round((rnd()*4-2)*10)/10));
const sm=row=>{const m=Math.max(...row),e=row.map(v=>v===-Infinity?0:Math.exp(v-m)),z=e.reduce((a,b)=>a+b,0);return e.map(v=>v/z);};
const FN={silu:x=>x/(1+Math.exp(-x)),relu:x=>Math.max(0,x),gelu:x=>{const t=1/(1+0.3275911*Math.abs(x)/Math.SQRT2),y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x/2);return x*0.5*(1+(x>=0?y:-y));}};
const nm=s=>s.expr.filter(x=>typeof x==='object').map(t=>t.n);
function demoFor(s){
  const r=rng(s.key.length*131+7),n=nm(s),col=(M,j)=>M.map((_,i)=>[i,j]),rowc=(M,i)=>M[0].map((_,j)=>[i,j]);
  const elt=(A,B,sym,f)=>{const C=A.map((x,i)=>x.map((v,j)=>f(v,B[i][j])));return {lay:[['a',n[0],A],sym,['b',n[1],B],'=',['o',n[2],C]],link:(i,j)=>({a:[[i,j]],b:[[i,j]],t:`out[${i}][${j}] = ${fmtC(A[i][j])} ${sym} ${fmtC(B[i][j])} = ${fmtC(C[i][j])}`})};};
  switch(s.op){
    case 'matmul':case 'qkt':case 'av':{
      const [a,b,c]=s.op==='qkt'?[4,3,4]:s.op==='av'?[4,4,3]:[3,4,2];
      const A=s.op==='av'?[[1,0,0,0],[.5,.5,0,0],[.2,.3,.5,0],[.1,.2,.3,.4]]:ints(a,b,r),B=ints(b,c,r);
      const Cm=A.map(x=>B[0].map((_,j)=>x.reduce((q,v,k)=>q+v*B[k][j],0)));
      return {head:`A real matrix product, shrunk to [${a} × ${b}] × [${b} × ${c}] → [${a} × ${c}]. Hover an output cell: it is the dot product of one <b style="color:#9aa3ff">row</b> of the left matrix with one <b style="color:#c98bea">column</b> of the right one.`,
        lay:[['a',n[0],A],'×',['b',n[1],B],'=',['o',n[2],Cm]],link:(i,j)=>({a:rowc(A,i),b:col(B,j),t:`out[${i}][${j}] = `+A[i].map((v,k)=>`(${fmtC(v)}·${fmtC(B[k][j])})`).join(' + ')+` = ${fmtC(Cm[i][j])}`})};}
    case 'add':{const d=elt(ints(3,4,r),ints(3,4,r),'+',(x,y)=>x+y);d.head='Element-wise addition: same position, same shape in and out.';return d;}
    case 'mul':{const d=elt(flts(3,4,r),flts(3,4,r),'⊙',(x,y)=>x*y);d.head='Element-wise (Hadamard) product — each cell times the matching cell. Not a matrix product.';return d;}
    case 'act':{const A=flts(3,4,r),f=FN[s.fn],C=A.map(x=>x.map(f));return {head:`${s.fn.toUpperCase()} is applied to every number on its own.`,lay:[['a',n[0],A],'→',['o',n[1],C]],link:(i,j)=>({a:[[i,j]],t:`${s.fn}(${fmtC(A[i][j])}) = ${fmtC(C[i][j])}`})};}
    case 'scale':{const A=ints(4,4,r),C=A.map(x=>x.map(v=>v/2));return {head:'Demo uses d<sub>h</sub> = 4, so every score is divided by √4 = 2.',lay:[['a','S',A],'÷ 2 =',['o','S′',C]],link:(i,j)=>({a:[[i,j]],t:`${fmtC(A[i][j])} / 2 = ${fmtC(C[i][j])}`})};}
    case 'mask':{const A=flts(4,4,r),M=A.map((x,i)=>x.map((_,j)=>j<=i?0:-Infinity)),C=A.map((x,i)=>x.map((v,j)=>v+M[i][j]));return {head:'Row = query token i, column = key token j. Adding −∞ above the diagonal removes every future token.',lay:[['a','S′',A],'+',['b','M',M],'=',['o','S″',C]],link:(i,j)=>({a:[[i,j]],b:[[i,j]],t:`${fmtC(A[i][j])} + ${fmtC(M[i][j])} = ${fmtC(C[i][j])}   (${j<=i?'j ≤ i: visible':'j > i: future, hidden'})`})};}
    case 'softmax':{const sq=s.expr.some(x=>typeof x==='object'&&x.mask==='causal'||x.mask==='window'),one=s.key==='vsoft';
      const A=one?flts(1,5,r):flts(4,4,r).map((x,i)=>x.map((v,j)=>sq&&j>i?-Infinity:v)),C=A.map(sm);
      return {head:'Each row is exponentiated and divided by its own sum, so every row adds up to 1. −∞ becomes exactly 0.',lay:[['a',n[0],A],'→ softmax rows →',['o',n[1],C]],link:(i,j)=>({a:rowc(A,i),t:`A[${i}][${j}] = e^${fmtC(A[i][j])} / (`+A[i].map(v=>'e^'+fmtC(v)).join(' + ')+`) = ${fmtC(C[i][j])}`})};}
    case 'norm':{const A=flts(3,4,r),ln=s.ntype==='ln';const C=A.map(x=>{const mu=ln?x.reduce((a,b)=>a+b,0)/x.length:0,sd=Math.sqrt(x.reduce((a,b)=>a+(b-mu)**2,0)/x.length);return x.map(v=>(v-mu)/sd);});
      return {head:`Each row is normalised using only its own values (demo: γ = 1${ln?', β = 0':''}).`,lay:[['a',n[0],A],'→',['o',n[1],C]],link:(i,j)=>{const x=A[i],mu=ln?x.reduce((a,b)=>a+b,0)/x.length:0,sd=Math.sqrt(x.reduce((a,b)=>a+(b-mu)**2,0)/x.length);return {a:rowc(A,i),t:ln?`μ = ${fmtC(mu)}, σ = ${fmtC(sd)} → (${fmtC(x[j])} − ${fmtC(mu)}) / ${fmtC(sd)} = ${fmtC(C[i][j])}`:`RMS(row ${i}) = √(mean of squares) = ${fmtC(sd)} → ${fmtC(x[j])} / ${fmtC(sd)} = ${fmtC(C[i][j])}`};}};}
    case 'split':case 'concat':{const A=ints(2,6,r),g=[0,0,1,1,2,2];return {head:s.op==='split'?'Demo: 6 columns → 3 heads of width 2. The numbers do not change; only the grouping does (colours = heads).':'Demo: 3 heads of width 2 are placed side by side → 6 columns.',lay:[['a',s.op==='split'?'[2 × 6]':'3 × [2 × 2]',A,s.op==='concat'?g:null],'→',['o',s.op==='split'?'[2 × 3 × 2]':'[2 × 6]',A,s.op==='split'?g:null]],link:(i,j)=>({a:[[i,j]],t:`column ${j} ↔ head ${g[j]}, position ${j%2} inside the head — value ${A[i][j]} unchanged`})};}
    case 'lookup':{const ids=[[3],[0],[2]],E=flts(5,4,r),X=ids.map(([k])=>E[k]);return {head:'Demo vocabulary of 5 entries, d = 4. Token ID k copies row k of E.',lay:[['a','ids',ids],'→',['b','E',E],'=',['o','X',X]],link:(i,j)=>({a:[[i,0]],b:rowc(E,ids[i][0]),t:`X[${i}] = E[${ids[i][0]}]  (whole row copied)`})};}
    case 'topk':{const A=flts(3,5,r),C=A.map(x=>{const idx=[...x.keys()].sort((p,q)=>x[q]-x[p]).slice(0,2),p=sm(idx.map(k=>x[k]));return x.map((_,k)=>idx.includes(k)?p[idx.indexOf(k)]:0);});return {head:'Demo: 5 experts, top-2. Per row the two largest scores survive and are renormalised; the rest become 0 (those experts are not run).',lay:[['a','r',A],'→ top-2 →',['o','g',C]],link:(i,j)=>({a:rowc(A,i),t:C[i][j]?`expert ${j} selected for token ${i}, gate = ${fmtC(C[i][j])}`:`expert ${j} not in the top-2 for token ${i} → skipped`})};}
    case 'rope':{const A=flts(3,4,r),th=[30,10],C=A.map((x,t)=>x.map((_,j)=>{const p=j>>1,a=t*th[p]*Math.PI/180,u=x[2*p],v=x[2*p+1];return j%2?u*Math.sin(a)+v*Math.cos(a):u*Math.cos(a)-v*Math.sin(a);}));return {head:'Demo: one head of width 4 = two pairs, θ = 30° and 10°. Row t is position t; pair i is rotated by t·θ<sub>i</sub>. Row 0 is unchanged.',lay:[['a','Q',A,[0,0,1,1]],'→ rotate →',['o','Q′',C,[0,0,1,1]]],link:(i,j)=>{const p=j>>1,a=i*th[p];return {a:[[i,2*p],[i,2*p+1]],t:`position ${i}, pair ${p}: angle = ${i}·${th[p]}° = ${a}°\n(${fmtC(A[i][2*p])}, ${fmtC(A[i][2*p+1])}) → (${fmtC(C[i][2*p])}, ${fmtC(C[i][2*p+1])})   — length preserved`};}};}
    case 'select':{const A=flts(4,4,r),k=s.expr[0].hiRow==='first'?0:3;return {head:`Only row ${k===0?'0':'T−1'} is kept.`,lay:[['a',n[0],A],'→',['o',n[1],[A[k]]]],link:(i,j)=>({a:rowc(A,k),t:`row ${k} copied`})};}
    case 'cache':{const A=flts(3,3,r),B=flts(1,3,r),C=A.concat(B);return {head:'Old rows are read from memory, the new row is stacked underneath.',lay:[['a',n[0],A],'append',['b',n[1],B],'=',['o',n[2],C]],link:(i,j)=>i<3?{a:[[i,j]],t:`row ${i}: read from cache`}:{b:[[0,j]],t:'row 3: the new token'}};}
  }
  return null;
}
function mxHTML(id,label,data,grp,out){
  return `<div><div class="ml">${label}</div><table class="mx${out?' out':''}" data-m="${id}">${data.map((row,i)=>'<tr>'+row.map((v,j)=>`<td data-r="${i}" data-c="${j}"${grp?` class="g${grp[j]}"`:''}>${fmtC(v)}</td>`).join('')+'</tr>').join('')}</table></div>`;
}
function openDemo(el,s){
  const box=$('.demo',el),open=box.hidden;
  box.hidden=!open;el.classList.toggle('open',open);el.setAttribute('aria-expanded',open);
  if(!open||box.dataset.done)return;
  const d=demoFor(s);box.dataset.done=1;
  if(!d){box.innerHTML='<div class="dh">No arithmetic happens in this step — nothing to work through.</div>';return;}
  box.innerHTML=`<div class="dh">${d.head}</div><div class="mrow">${d.lay.map(x=>typeof x==='string'?`<span class="op${x.length>2?' s':''}">${x}</span>`:mxHTML(x[0],x[1],x[2],x[3],x[0]==='o')).join('')}</div><div class="ex">hover a cell of the result →</div>`;
  const show=(i,j)=>{
    $$('td',box).forEach(t=>t.classList.remove('hi','hj','ho'));
    const L=d.link(i,j);
    [['a','hi'],['b','hj']].forEach(([m,c])=>(L[m]||[]).forEach(([r,k])=>{const t=$(`table[data-m="${m}"] td[data-r="${r}"][data-c="${k}"]`,box);if(t)t.classList.add(c);}));
    const o=$(`table[data-m="o"] td[data-r="${i}"][data-c="${j}"]`,box);if(o)o.classList.add('ho');
    $('.ex',box).textContent=L.t;
  };
  box.addEventListener('mouseover',e=>{const t=e.target.closest('table.out td');if(t)show(+t.dataset.r,+t.dataset.c);});
  box.addEventListener('click',e=>{e.stopPropagation();const t=e.target.closest('table.out td');if(t)show(+t.dataset.r,+t.dataset.c);});
  show(d.lay[d.lay.length-1][2].length-1,0);
}

['T','d','H','Hkv','ff','V','N'].forEach(k=>Object.defineProperty(K,k,{get:()=>S[k],set:v=>{S[k]=v;},enumerable:true,configurable:true}));
['E','k'].forEach(k=>Object.defineProperty(K,k,{get:()=>MO[k],set:v=>{MO[k]=v;},enumerable:true,configurable:true}));
K.orient='v';
window.TFX={tip:(s,label)=>`<div class="th"><b>${label||s.title}</b><span>${s.stage}</span></div><div class="tk"><span>Step</span><span>${s.title}</span><span>Formula</span><span>${s.fx}</span><span>Runs</span><span>${s.sn?s.sn:s.scope==='layer'?'in every layer (× '+K.N+')':s.scope==='in'?'once, before layer 1':'once, after the last layer'}</span>${s.params?`<span>Params</span><span>${fN(s.params)}</span>`:''}${s.flops?`<span>Cost</span><span>${fF(s.flops)}</span>`:''}</div>${s.note?`<p>${s.note}</p>`:''}${formulaHTML(stepFormulas(s))}${whyHTML(explainStep(s),true)}`,
  text:s=>s.title+': '+s.fx.replace(/<[^>]+>/g,'')+' | '+s.expr.filter(x=>typeof x==='object').map(shp).join(' , ')};
const LEGEND=roleLegend()+'<p class="mut" style="font-size:12.5px;margin:-6px 0 12px">⌇ on a tensor = too large to draw to scale</p>';
const BACK={mha:'attn/mha',gqa:'attn/gqa',mqa:'attn/mqa',swa:'attn/swa',moe:'moe',decode:'decode',gpt2:'families/dense',bert:'families/enc',encdec:'families/encdec'};
const TFV={show(){const host=$('#g-host');if(!host)return;if(TFV.hide)TFV.hide();
  host.innerHTML=K.orient==='h'?'<div class="card pad toolbar" id="hg-bar"></div><div id="hg-main" class="tf"></div>':'<div class="card pad toolbar" id="vg-bar"></div><p id="vg-hint" class="mut" style="font-size:13px;margin:0 0 4px"></p><div id="vg-main" class="tf"></div>';
  (K.orient==='h'?TFV.h:TFV.v)();}};
