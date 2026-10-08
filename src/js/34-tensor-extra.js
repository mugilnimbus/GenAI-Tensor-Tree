/* ================= more step-by-step architectures =================
   Decoder variants (ALiBi, adapters, prefix / prompt tuning, (IA)³, DoRA, QLoRA) reuse the GQA decoder and insert steps.
   Other model types (Mamba, RWKV, ViT + projector, DiT, VQ tokenizer, masked-diffusion LM) are their own sequences (S.seq = true).
   Steps here carry their own explanation: o.why / o.what / o.fm. */
K.m=64;K.p=20;
FIELDS.push(['m','adapter m'],['p','prefix p']);
const cfgCopy=c=>({T:c.T,d:c.d,H:c.H,Hkv:c.Hkv,ff:c.ff,V:c.V,N:c.N,S:c.S,W:c.W,E:c.E,k:c.k,C:c.C,h:c.h,r:c.r,m:c.m,p:c.p});
const gqaBase=c=>modern(c,{kv:c.Hkv,mask:'causal'});
const insAfter=(S,key,...steps)=>S.splice(S.findIndex(s=>s.key===key)+1,0,...steps);
const insBefore=(S,key,...steps)=>S.splice(S.findIndex(s=>s.key===key),0,...steps);
const gqaMeta=(c,x)=>Object.assign({norm:'RMSNorm, pre',pos:'RoPE on Q,K (every layer)',mask:'causal',mlp:'SwiGLU (3 matrices)',kv:c.Hkv},x);

const mAfter=(key,...ks)=>MASTER.splice(MASTER.indexOf(key)+1,0,...ks);
mAfter('pos','ptok');mAfter('cq','deq','dab','dsum','dnorm','dmag');mAfter('rope','pk','pv','ia3k','ia3v');mAfter('scale','bias');
mAfter('wo','ad1a','ad1f','ad1b','ad1add');mAfter('mul','ia3f');mAfter('down','ad2a','ad2f','ad2b','ad2add');
Object.assign(SRC,{mamba:['mamba'],mdlm:['mdlm','llada','d3pm'],swa:['longformer','mistral'],decode:['transformer','flash']});
/* ---- ALiBi: no position vectors, a linear distance penalty on the scores ---- */
function alibiDecoder(c){
  const S=gqaBase(c).filter(s=>s.key!=='rope'),T=c.T;
  insAfter(S,'scale',st('bias','Attention','ALiBi distance bias','S′ = S′ − m<sub>h</sub>·(i − j)',[msk('S′',T,T),'+',msk('B',T,T,{mask:'causal'}),'=',msk('S′',T,T)],{op:'add',head:true,mult:c.H,params:0,
    note:'B[i, j] = −m<sub>h</sub>·(i − j). The slope m<sub>h</sub> is fixed per head (a geometric sequence), not learned.',
    why:'Instead of giving tokens a position vector, each head simply subtracts a penalty that grows with the distance between query and key. Nearby tokens are favoured, and because the rule is the same at any distance, the model can be run on sequences longer than it was trained on.',
    what:'Every score is lowered by an amount proportional to how far back the key is. Different heads use different slopes, so some look locally and others far.',
    fm:[['Bias','B<sub>h</sub>[i, j] = −m<sub>h</sub> · (i − j)  for j ≤ i'],['Slopes','m<sub>h</sub> = 2<sup>−8h/H</sup> ,  h = 1 … H'],['Scores','softmax(q<sub>i</sub>k<sub>j</sub><sup>ᵀ</sup>/√d<sub>h</sub> + B<sub>h</sub>[i, j])']]}));
  return S;
}
/* ---- adapters: a small bottleneck MLP after attention and after the MLP ---- */
function adapterDecoder(c){
  const S=gqaBase(c),{T,d}=c,m=c.m;
  const block=(after,p,src,stage)=>insAfter(S,after,
    st(p+'a',stage,'Adapter — down',`Z = ${src}·${sb('W','down')}`,[act(src,T,d),'×',wgt(sb('W','down')+'′',d,m),'=',act('Z'+p,T,m)],{op:'matmul',from:[src],flops:2*T*d*m,note:`Trainable. Bottleneck width m = ${m}.`,
      why:'The frozen sub-layer output is squeezed through a narrow layer so that only a few thousand new parameters per layer are needed to adapt the model.',what:`Width drops from ${d} to ${m}.`,fm:[['Down','Z = h · W<sub>down</sub> ,  W<sub>down</sub> ∈ ℝ<sup>d×m</sup>']]}),
    st(p+'f',stage,'Adapter — non-linearity','Z′ = GELU(Z)',[act('Z'+p,T,m),'→ GELU →',act('Z′'+p,T,m)],{op:'act',fn:'gelu',why:'Without it the adapter would be a plain low-rank linear map; the non-linearity lets it compute task-specific features.',what:'Negative entries are pushed toward zero; shape unchanged.',fm:FORMULA.gelu}),
    st(p+'b',stage,'Adapter — up',`Δ = Z′·${sb('W','up')}`,[act('Z′'+p,T,m),'×',wgt(sb('W','up')+'′',m,d),'=',act('Δ'+p,T,d)],{op:'matmul',flops:2*T*m*d,note:'Trainable, initialised near zero.',why:'Back to model width so the result can be added to the stream.',what:`Width grows from ${m} to ${d}.`,fm:[['Up','Δ = f(Z) · W<sub>up</sub> ,  W<sub>up</sub> ∈ ℝ<sup>m×d</sup>']]}),
    st(p+'add',stage,'Adapter — skip connection',`${src} = ${src} + Δ`,[act(src,T,d),'+',act('Δ'+p,T,d),'=',act(src,T,d)],{op:'add',from:[src,'Δ'+p],why:'With a skip connection and near-zero initialisation the adapter starts as “do nothing”, so training begins from the unchanged model.',what:'Element-wise sum; shape unchanged.',fm:[['Adapter','h ← h + f(h W<sub>down</sub>) W<sub>up</sub>']]}));
  block('wo','ad1','Attn','Attention');block('down','ad2','MLP','MLP');
  S.forEach(s=>{if(!/^ad\d/.test(s.key)&&s.params)s.frozen=true;});
  return S;
}
/* ---- prefix tuning: trainable rows prepended to K and V in every layer ---- */
function prefixDecoder(c){
  const S=gqaBase(c),{T,d,H}=c,dh=d/H,kv=c.Hkv,kvd=kv*dh,p=c.p;
  insAfter(S,'rope',
    st('pk','Attention','Prepend prefix keys',`K = [${sb('P','K')} ; K]`,[wgt(sb('P','K'),p,kvd),'‖',act('K′',T,kvd,{stripes:kv}),'=',act('K′',T+p,kvd,{stripes:kv,shape:`[${T+p} × ${kv} × ${dh}]`})],{op:'cache',from:['K′'],note:`Trainable: ${p} extra key rows per layer.`,
      why:'Rather than changing any weight, the model is given a few extra “virtual tokens” to attend to. Putting them directly into K and V of every layer steers attention at every depth.',what:`K grows from ${T} to ${T+p} rows; the first ${p} are learned vectors, not tokens.`,fm:[['Keys','K′ = [P<sub>K</sub> ; K] ,  P<sub>K</sub> ∈ ℝ<sup>p × H<sub>kv</sub>d<sub>h</sub></sup>']]}),
    st('pv','Attention','Prepend prefix values',`V = [${sb('P','V')} ; V]`,[wgt(sb('P','V'),p,kvd),'‖',act('V',T,kvd,{stripes:kv}),'=',act('V',T+p,kvd,{stripes:kv,shape:`[${T+p} × ${kv} × ${dh}]`})],{op:'cache',from:['V'],note:'Trainable.',
      why:'The prefix needs content to hand over when it is attended to; these rows are that content.',what:`V grows to ${T+p} rows.`,fm:[['Values','V′ = [P<sub>V</sub> ; V]'],['Trainable','2 · p · H<sub>kv</sub> · d<sub>h</sub> per layer']]}));
  S.forEach(s=>{if(['scores','scale','mask','softmax','av'].includes(s.key)){
    s.expr.forEach(t=>{if(typeof t!=='object')return;if(t.r===T&&t.c===T)t.c=T+p;else if(t.r===dh&&t.c===T)t.c=T+p;else if(t.r===T&&t.c===dh&&/^V/.test(t.n))t.r=T+p;});
    s.sig+='|prefix';if(s.key==='scores')s.note+=` The score table is now [${T} × ${T+p}]: every query can also attend to the ${p} prefix rows.`;}
    if(!/^p[kv]$/.test(s.key)&&s.params)s.frozen=true;});
  return S;
}
/* ---- prompt tuning: trainable vectors in front of the input embeddings only ---- */
function promptDecoder(c){
  const p=c.p,cc=cfgCopy(c);cc.T=c.T+p;
  const S=[];inputSteps(S,c,c.T,null);
  S.push(st('ptok','Input','Prepend virtual tokens','X = [P ; X]',[wgt('P',p,c.d),'‖',res('X',c.T,c.d),'=',res('X',c.T+p,c.d)],{scope:'in',op:'cache',from:['X'],note:`The only trainable parameters: ${p} × ${c.d}.`,
    why:'The cheapest way to adapt a model: learn a short sequence of vectors that are not words and put them in front of the real prompt. Everything else is frozen.',what:`The sequence grows from ${c.T} to ${c.T+p} rows; from here on the model treats the virtual tokens like any others.`,fm:[['Input','X′ = [P ; E[ids]] ,  P ∈ ℝ<sup>p×d</sup>'],['Trainable','p · d']]}));
  gqaBase(cc).slice(2).forEach(s=>{if(s.params)s.frozen=true;S.push(s);});
  return S;
}
/* ---- (IA)³: three learned rescaling vectors per layer ---- */
function ia3Decoder(c){
  const S=gqaBase(c),{T,d,H,ff}=c,dh=d/H,kv=c.Hkv,kvd=kv*dh,sh=`[${T} × ${kv} × ${dh}]`;
  const mk3=(key,stage,name,l,w,o)=>st(key,stage,`Rescale ${name}`,`${name} = ${l} ⊙ ${name}`,o.expr,{op:'mul',from:o.from,note:`Trainable: one number per feature (${w}).`,why:o.why,what:'Each column is multiplied by its own learned factor (initialised to 1). Shape unchanged.',fm:[['Rescale',`${name}′ = ${l} ⊙ ${name}`],['Initialisation','l = 1']]});
  insAfter(S,'rope',
    mk3('ia3k','Attention','K',sb('l','k'),kvd,{from:['K′'],expr:[act('K′',T,kvd,{stripes:kv,shape:sh}),'⊙',wgt(sb('l','k'),1,kvd),'=',act('K′',T,kvd,{stripes:kv,shape:sh})],why:'Scaling key features up or down changes which tokens get found, without touching any weight matrix.'}),
    mk3('ia3v','Attention','V',sb('l','v'),kvd,{from:['V'],expr:[act('V',T,kvd,{stripes:kv,shape:sh}),'⊙',wgt(sb('l','v'),1,kvd),'=',act('V',T,kvd,{stripes:kv,shape:sh})],why:'Scaling value features changes what information is passed on when a token is attended to.'}));
  insAfter(S,'mul',mk3('ia3f','MLP','Hid',sb('l','ff'),ff,{from:['Hid'],expr:[act('Hid',T,ff),'⊙',wgt(sb('l','ff'),1,ff),'=',act('Hid',T,ff)],why:'Scaling hidden MLP features amplifies or silences individual learned features for the new task.'}));
  S.forEach(s=>{if(!/^ia3/.test(s.key)&&s.params)s.frozen=true;});
  return S;
}
/* ---- DoRA: W = magnitude × direction; LoRA updates the direction ---- */
function doraDecoder(c){
  const S=gqaBase(c),{T,d}=c,r=c.r,W=(n,a,b)=>wgt(n,a,b);
  insBefore(S,'q',
    st('dab','Attention','Low-rank update','ΔW = A·B',[W('A',d,r),'×',W('B',r,d),'=',W('ΔW',d,d)],{op:'matmul',from:[],params:2*d*r,note:'Trainable A and B, as in LoRA.',why:'The direction of the weight is adapted with a cheap low-rank update.',what:`Two thin matrices multiply into a full [${d} × ${d}] update of rank ≤ ${r}.`,fm:[['Update','ΔW = A · B']]}),
    st('dsum','Attention','Add to the frozen weight',`V′ = ${sb('W','0')} + ΔW`,[W(sb('W','0'),d,d),'+',W('ΔW',d,d),'=',W('V′',d,d)],{op:'add',from:['ΔW'],params:d*d,frozen:true,why:'The update is applied to the frozen matrix before its columns are re-normalised.',what:'Element-wise sum of two matrices.',fm:[['Direction (un-normalised)','V′ = W<sub>0</sub> + A·B']]}),
    st('dnorm','Attention','Column norms','n<sub>j</sub> = ‖V′[:, j]‖',[W('V′',d,d),'→ norm of each column →',W('‖V′‖',1,d)],{from:['V′'],params:0,why:'To separate direction from size, the length of every column is measured so it can be divided out.',what:`One number per column: [1 × ${d}].`,fm:[['Column norm','‖V′‖<sub>c</sub>[j] = √(Σ<sub>i</sub> V′[i, j]²)']]}),
    st('dmag','Attention','Apply the trained magnitude','W′ = m ⊙ V′ / ‖V′‖',[W('V′',d,d),'÷ ‖V′‖ ⊙',W('m',1,d),'=',W('W′',d,d)],{from:['V′','‖V′‖'],params:d,note:'Trainable magnitude vector m.',why:'Full fine-tuning tends to change the size and the direction of weights independently; giving the size its own trainable vector lets a low-rank update match that behaviour.',what:'Each column is scaled to unit length and then to its trained length.',fm:[['DoRA','W′ = m · (W<sub>0</sub> + A·B) / ‖W<sub>0</sub> + A·B‖<sub>c</sub>']]}));
  const q=S.find(s=>s.key==='q');q.expr=[act('X̂',T,d),'×',W('W′',d,d),'=',act('Q',T,d)];q.from=['X̂','W′'];q.params=0;q.fx='Q = X̂·W′';q.sig='Q = X̂·W′|dora';q.note='The adapted matrix is used exactly like the original; after training it can be stored as one ordinary matrix.';
  S.forEach(s=>{if(!/^d(ab|mag)$/.test(s.key)&&s.params&&!s.frozen)s.frozen=true;});
  return S;
}
/* ---- QLoRA: LoRA on a base model stored in 4 bits ---- */
function qloraDecoder(c){
  const S=loraDecoder(c),d=c.d;
  insBefore(S,'q',st('deq','Attention','De-quantise the frozen weight',`${sb('W','Q')} = dequant(${sb('W','Q')}<sup>4-bit</sup>)`,[wgt('W<sub>Q</sub> · 4-bit',d,d,{shape:`[${d} × ${d}] · 0.5 B each`}),'→ de-quantise →',wgt(sb('W','Q'),d,d)],{from:[],params:0,note:'NF4: 16 levels placed for normally distributed weights; the 16-bit copy exists only during the computation.',
    why:'Storing the frozen model in 4 bits cuts its memory about four-fold, which is what lets a large model be fine-tuned on one GPU. The LoRA matrices stay in 16-bit and are the only thing trained.',what:'Each 4-bit code is replaced by its 16-bit value. The numbers are slightly different from the original weights (quantisation error).',fm:[['Forward','h = x · dequant(W<sub>0</sub>) + (α/r) · x · A · B'],['Memory','0.5 byte per frozen weight instead of 2']]}));
  const q=S.find(s=>s.key==='q');q.from=['X̂','WQ'];
  return S;
}
/* ================= other model types (sequential) ================= */
const seq=(S,sn)=>{S.seq=true;if(sn)S.forEach(s=>{if(!s.sn)s.sn=sn;});return S;};
const X=(S,key,stage,title,fx,expr,o)=>S.push(st(key,stage,title,fx,expr,o));
/* ---- Mamba block (expand 2, state 16, conv 4 — the paper's defaults) ---- */
function mambaSteps(c){
  const S=[],{T,d}=c,E=2*d,N=16,R=Math.ceil(d/16),B='Mamba block';
  inputSteps(S,c,T,null);
  X(S,'m_norm',B,'RMSNorm','X̂ = RMSNorm(X)',[res('X',T,d),'→ RMSNorm →',act('X̂',T,d)],{op:'norm',ntype:'rms',params:d,why:WHY.norm[0],what:WHY.norm[1],fm:FORMULA.norm});
  X(S,'m_in',B,'Expand into two branches',`[x | z] = X̂·${sb('W','in')}`,[act('X̂',T,d),'×',wgt(sb('W','in'),d,2*E),'=',act('x | z',T,2*E,{stripes:2})],{op:'matmul',flops:2*T*d*2*E,why:'The block works in a space twice as wide as the model. One half (x) will go through the state-space mixer, the other (z) becomes a gate for its output.',what:`Width grows from ${d} to 2 × ${E}.`,fm:[['Input projection','[x, z] = X̂ · W<sub>in</sub> ,  W<sub>in</sub> ∈ ℝ<sup>d × 2E</sup> ,  E = 2d']]});
  X(S,'m_x',B,'Branch x','x = first half',[act('x | z',T,2*E,{stripes:2}),'→ split →',act('x',T,E)],{from:['x | z'],why:'The branch that carries the sequence through the recurrence.',what:'First E columns.',fm:[['Split','x = [x | z][:, :E]']]});
  X(S,'m_z',B,'Branch z','z = second half',[act('x | z',T,2*E,{stripes:2}),'→ split →',act('z',T,E)],{from:['x | z'],why:'Kept aside; it will gate the mixer output at the end.',what:'Last E columns.',fm:[['Split','z = [x | z][:, E:]']]});
  X(S,'m_conv',B,'Short convolution over time','x′ = conv1d(x), kernel 4',[act('x',T,E),'→ depth-wise conv, k = 4 →',act('x′',T,E)],{from:['x'],params:E*4,why:'A recurrence only sees the past through its state. A small causal convolution first gives every position direct access to its 3 predecessors, so local patterns do not have to be stored in the state.',what:'Each row becomes a learned mix of itself and the three rows before it, per channel.',fm:[['Convolution','x′<sub>t</sub> = Σ<sub>j=0</sub><sup>3</sup> w<sub>j</sub> ⊙ x<sub>t−j</sub>']]});
  X(S,'m_silu',B,'SiLU','u = SiLU(x′)',[act('x′',T,E),'→ SiLU →',act('u',T,E)],{op:'act',fn:'silu',why:WHY.silu[0].split('. ')[0]+'.',what:WHY.silu[1],fm:FORMULA.silu});
  X(S,'m_sel',B,'Selective parameters',`Δ, B, C = u·${sb('W','sel')}`,[act('u',T,E),'×',wgt(sb('W','sel'),E,R+2*N),'=',act('Δ, B, C',T,R+2*N)],{op:'matmul',flops:2*T*E*(R+2*N),why:'This is what makes the state-space model “selective”: the step size and the write and read vectors are computed from the current token, so the model can decide to keep, overwrite or ignore information.',what:`Each token yields a step size Δ and two vectors of ${N} numbers.`,fm:[['Selection','Δ<sub>t</sub> = softplus(linear(u<sub>t</sub>)) ,  B<sub>t</sub>, C<sub>t</sub> = linear(u<sub>t</sub>)']]});
  X(S,'m_state',B,'Update the hidden state','h<sub>t</sub> = Ā<sub>t</sub>·h<sub>t−1</sub> + B̄<sub>t</sub>·u<sub>t</sub>',[act('Δ, B, C',T,R+2*N),'updates',msk(sb('h','t−1'),E,N,{shape:`[${E} × ${N}] fixed`}),'=',msk(sb('h','t'),E,N,{shape:`[${E} × ${N}] fixed`})],{why:'The past is carried in a state of fixed size, so cost per token is constant and no cache grows with the context.',what:'The old state decays by Ā = exp(Δ·A) and the new input is written in through B̄. The state keeps its shape.',fm:[['Recurrence','h<sub>t</sub> = Ā<sub>t</sub> · h<sub>t−1</sub> + B̄<sub>t</sub> · u<sub>t</sub>'],['Discretisation','Ā = exp(Δ·A) ,  B̄ = (Δ·A)<sup>−1</sup>(exp(Δ·A) − I) · Δ·B']],});
  X(S,'m_read',B,'Read out','y<sub>t</sub> = C<sub>t</sub>·h<sub>t</sub>',[msk(sb('h','t'),E,N,{shape:`[${E} × ${N}] fixed`}),'· C','=',act('y',T,E)],{why:'The output is whatever the token chooses to read from the state.',what:`One vector of ${E} numbers per token.`,fm:[['Read-out','y<sub>t</sub> = C<sub>t</sub> · h<sub>t</sub>']]});
  X(S,'m_gate',B,'Gate with the z branch','y′ = y ⊙ SiLU(z)',[act('y',T,E),'⊙',act('z',T,E),'=',act('y′',T,E)],{op:'mul',from:['y','z'],why:'The second branch decides, feature by feature, how much of the mixer output passes — the same gating idea as SwiGLU.',what:'Element-wise product; features with a closed gate are removed.',fm:[['Gate','y′ = y ⊙ SiLU(z)']].concat(FORMULA.silu.slice(0,2))});
  X(S,'m_out',B,'Project back',`Mamba = y′·${sb('W','out')}`,[act('y′',T,E),'×',wgt(sb('W','out'),E,d),'=',act('Mamba',T,d)],{op:'matmul',flops:2*T*E*d,why:'Back to model width so the result can join the residual stream.',what:`Width shrinks from ${E} to ${d}.`,fm:[['Output projection','out = y′ · W<sub>out</sub>']]});
  X(S,'m_add',B,'Residual add','X = X + Mamba',[res('X',T,d),'+',act('Mamba',T,d),'=',res('X',T,d)],{op:'add',from:['X','Mamba'],why:WHY.add[0],what:WHY.add[1],fm:FORMULA.add});
  outSteps(S,c,T,{norm:'RMSNorm'});
  return seq(S);
}
/* ---- RWKV-4 block: time mixing + channel mixing ---- */
function rwkvSteps(c){
  const S=[],{T,d}=c,A='Time mixing',Bc='Channel mixing';
  inputSteps(S,c,T,null);
  X(S,'r_ln1',A,'LayerNorm','X̂ = LN(X)',[res('X',T,d),'→ LayerNorm →',act('X̂',T,d)],{op:'norm',ntype:'ln',params:2*d,why:WHY.norm[0],what:WHY.normpost[1],fm:FORMULA.normln});
  X(S,'r_mix',A,'Blend with the previous token','x<sub>k</sub> = μ<sub>k</sub>⊙x<sub>t</sub> + (1 − μ<sub>k</sub>)⊙x<sub>t−1</sub>',[act('X̂',T,d),'→ mix with row t−1 →',act('xk, xv, xr',T,3*d,{stripes:3,shape:`3 × [${T} × ${d}]`})],{params:3*d,why:'An RNN needs to know what just happened. Three learned blends of the current and previous token give the key, value and receptance paths their own view of “now versus a moment ago”.',what:'Three copies of the input, each a per-channel interpolation between row t and row t−1.',fm:[['Token shift','x<sub>k</sub> = μ<sub>k</sub> ⊙ x<sub>t</sub> + (1 − μ<sub>k</sub>) ⊙ x<sub>t−1</sub>  (same for v and r)']]});
  X(S,'r_kvr',A,'Key, value, receptance','k, v, r = linear(x<sub>k</sub>), linear(x<sub>v</sub>), linear(x<sub>r</sub>)',[act('xk, xv, xr',T,3*d,{stripes:3,shape:`3 × [${T} × ${d}]`}),'×',wgt(sb('W','k,v,r'),d,d,{shape:`3 × [${d} × ${d}]`}),'=',act('k, v, r',T,3*d,{stripes:3,shape:`3 × [${T} × ${d}]`})],{op:'matmul',params:3*d*d,flops:6*T*d*d,why:'k says how important this token is, v what it contributes, and r (receptance) how much of the result this position accepts.',what:'Three linear projections.',fm:[['Projections','k = x<sub>k</sub>W<sub>k</sub> ,  v = x<sub>v</sub>W<sub>v</sub> ,  r = x<sub>r</sub>W<sub>r</sub>']]});
  X(S,'r_state',A,'Update the running sums','a<sub>t</sub> = e<sup>−w</sup>a<sub>t−1</sub> + e<sup>k</sup>v ,  b<sub>t</sub> = e<sup>−w</sup>b<sub>t−1</sub> + e<sup>k</sup>',[act('k, v, r',T,3*d,{stripes:3,shape:`3 × [${T} × ${d}]`}),'updates',msk('a, b',2,d,{shape:`2 × [${d}] fixed`}),'=',msk(sb('a','t')+', '+sb('b','t'),2,d,{shape:`2 × [${d}] fixed`})],{params:2*d,why:'Attention recomputes a weighted average over the whole past for every token. Here the same kind of average is kept as two running sums with a learned decay per channel, so each new token costs a constant amount.',what:'Both sums fade by e⁻ʷ and the current token is added. Their size never changes.',fm:[['Numerator','a<sub>t</sub> = e<sup>−w</sup> · a<sub>t−1</sub> + e<sup>k<sub>t</sub></sup> · v<sub>t</sub>'],['Denominator','b<sub>t</sub> = e<sup>−w</sup> · b<sub>t−1</sub> + e<sup>k<sub>t</sub></sup>']]});
  X(S,'r_wkv',A,'Weighted average of the past','wkv = a / b',[msk(sb('a','t')+', '+sb('b','t'),2,d,{shape:`2 × [${d}] fixed`}),'→ divide →',act('wkv',T,d)],{why:'Dividing the two sums gives an exponentially weighted average of past values — RWKV’s replacement for softmax(QKᵀ)·V.',what:`One ${d}-vector per token.`,fm:[['WKV','wkv<sub>t</sub> = (a<sub>t−1</sub> + e<sup>u + k<sub>t</sub></sup> v<sub>t</sub>) / (b<sub>t−1</sub> + e<sup>u + k<sub>t</sub></sup>)']]});
  X(S,'r_out',A,'Receptance gate and output',`TimeMix = (σ(r) ⊙ wkv)·${sb('W','o')}`,[act('wkv',T,d),'⊙ σ(r) , ×',wgt(sb('W','o'),d,d),'=',act('TimeMix',T,d)],{op:'matmul',flops:2*T*d*d,why:'The sigmoid of r lets each position damp what it receives; the output matrix maps the result into the residual stream.',what:'Element-wise gate between 0 and 1, then a linear projection.',fm:[['Output','o<sub>t</sub> = W<sub>o</sub> · (σ(r<sub>t</sub>) ⊙ wkv<sub>t</sub>)'],['Sigmoid','σ(x) = 1 / (1 + e<sup>−x</sup>)']]});
  X(S,'r_add1',A,'Residual add','X = X + TimeMix',[res('X',T,d),'+',act('TimeMix',T,d),'=',res('X',T,d)],{op:'add',from:['X','TimeMix'],why:WHY.add[0],what:WHY.add[1],fm:FORMULA.add});
  X(S,'r_ln2',Bc,'LayerNorm','X̂ = LN(X)',[res('X',T,d),'→ LayerNorm →',act('X̂',T,d)],{op:'norm',ntype:'ln',params:2*d,why:WHY.norm[0],what:WHY.normpost[1],fm:FORMULA.normln});
  X(S,'r_ck',Bc,'Expand',`k′ = X̂·${sb('W','k')}′`,[act('X̂',T,d),'×',wgt(sb('W','k')+'′',d,4*d),'=',act('k′',T,4*d)],{op:'matmul',flops:8*T*d*d,why:'The per-token MLP of RWKV: expand to four times the width.',what:`Width grows from ${d} to ${4*d}.`,fm:[['Expand','k′ = x · W<sub>k</sub>′']]});
  X(S,'r_sq',Bc,'Squared ReLU','k″ = max(k′, 0)²',[act('k′',T,4*d),'→ ReLU² →',act('k″',T,4*d)],{op:'act',fn:'relu',why:'The non-linearity. Squaring after ReLU makes strong activations stand out even more.',what:'Negatives become 0; positives are squared.',fm:[['Squared ReLU','f(x) = max(0, x)²']]});
  X(S,'r_cv',Bc,'Project back and gate',`ChanMix = σ(r′) ⊙ (k″·${sb('W','v')}′)`,[act('k″',T,4*d),'×',wgt(sb('W','v')+'′',4*d,d),'=',act('ChanMix',T,d)],{op:'matmul',flops:8*T*d*d,why:'Back to model width, gated by a receptance computed from the same input.',what:`Width shrinks from ${4*d} to ${d}.`,fm:[['Channel mixing','o = σ(r′) ⊙ (max(k′, 0)² · W<sub>v</sub>′)']]});
  X(S,'r_add2',Bc,'Residual add','X = X + ChanMix',[res('X',T,d),'+',act('ChanMix',T,d),'=',res('X',T,d)],{op:'add',from:['X','ChanMix'],why:WHY.add[0],what:WHY.add[1],fm:FORMULA.add});
  outSteps(S,c,T,{norm:'LayerNorm'});
  return seq(S);
}
/* ---- image → tokens for a language model (LLaVA-1.5: CLIP ViT-L/14 at 336 px, 2-layer MLP projector) ---- */
function vitSteps(c){
  const S=[],{T,d}=c,P=14,side=336,g=side/P,N=g*g,dv=1024,pd=3*P*P,A='Vision encoder',B='Projector',O='Language model';
  X(S,'v_img',A,'Image','pixels',[tok('image',side,side*3,{shape:`[3 × ${side} × ${side}]`})],{scope:'in',note:'RGB values.',why:'The raw input: a grid of pixels, not a sequence.',what:`3 × ${side} × ${side} numbers.`,fm:[['Input','x ∈ ℝ<sup>3×H×W</sup>']]});
  X(S,'v_patch',A,'Cut into patches',`${N} patches of ${P}×${P}`,[tok('image',side,side*3,{shape:`[3 × ${side} × ${side}]`}),'→ patches →',act('patches',N,pd)],{scope:'in',op:'split',why:'A transformer needs a sequence. Cutting the image into a grid of small squares and flattening each one turns it into a sequence of vectors.',what:`${g} × ${g} = ${N} rows, each holding the ${pd} pixel values of one patch.`,fm:[['Patches','N = (H / P) · (W / P) ,  each ∈ ℝ<sup>3·P²</sup>']]});
  X(S,'v_emb',A,'Patch embedding',`F = patches·${sb('W','p')} + pos`,[act('patches',N,pd),'×',wgt(sb('W','p'),pd,dv),'=',act('F',N,dv)],{scope:'in',op:'matmul',flops:2*N*pd*dv,why:'A linear layer plays the role of the token embedding, and a learned position vector tells the model where each patch was.',what:`Each patch becomes a ${dv}-vector.`,fm:[['Embedding','z<sub>i</sub> = patch<sub>i</sub> · W<sub>p</sub> + p<sub>i</sub>']]});
  X(S,'v_enc',A,'Vision transformer','F′ = ViT(F)  — 24 encoder layers',[act('F',N,dv),'→ 24 × (bidirectional attention + MLP) →',act('F′',N,dv)],{scope:'in',note:'Each layer is the encoder-only block: attention with no mask, then an MLP. Frozen in LLaVA.',why:'Bidirectional attention lets every patch gather context from the whole image, producing features that describe objects rather than pixels.',what:'Same shape; each row now encodes its patch in context.',fm:[['Encoder layer','z ← z + Attn(LN(z)) ;  z ← z + MLP(LN(z))']]});
  X(S,'v_p1',B,'Projector — first layer',`H = F′·${sb('W','1')}`,[act('F′',N,dv),'×',wgt(sb('W','1'),dv,d),'=',act('Hp',N,d)],{scope:'in',op:'matmul',flops:2*N*dv*d,why:'Vision features live in the vision encoder’s space. The projector translates them into the language model’s embedding space — the main part trained first.',what:`Width changes from ${dv} to ${d}.`,fm:[['Projector','tokens = GELU(Z W<sub>1</sub>) W<sub>2</sub>']]});
  X(S,'v_act',B,'GELU','GELU(H)',[act('Hp',N,d),'→ GELU →',act('Hp′',N,d)],{scope:'in',op:'act',fn:'gelu',why:'Makes the projector a small non-linear network instead of a single matrix.',what:WHY.actf[1],fm:FORMULA.gelu});
  X(S,'v_p2',B,'Projector — second layer',`I = GELU(H)·${sb('W','2')}`,[act('Hp′',N,d),'×',wgt(sb('W','2'),d,d),'=',res('I',N,d)],{scope:'in',op:'matmul',flops:2*N*d*d,why:'Produces vectors that the language model can read as if they were word embeddings.',what:`${N} image tokens of width ${d}.`,fm:[['Image tokens','I ∈ ℝ<sup>N×d</sup>']]});
  X(S,'v_cat',O,'Join with the text embeddings','X = [I ; E[ids]]',[res('I',N,d),'‖',res(sb('X','text'),T,d),'=',res('X',N+T,d)],{scope:'in',op:'cache',from:['I'],why:'From here on nothing is special about images: the decoder sees one sequence in which some positions came from pixels.',what:`A sequence of ${N} + ${T} = ${N+T} vectors.`,fm:[['Sequence','X = [image tokens ; text embeddings] ∈ ℝ<sup>(N+T)×d</sup>']]});
  X(S,'v_llm',O,'Decoder-only language model','logits = LLM(X)',[res('X',N+T,d),'→ N decoder layers →',msk('logits',1,c.V)],{scope:'out',note:'An ordinary decoder — see the GQA decoder for every step.',why:'The language model attends over image and text tokens alike and generates the answer.',what:'Next-token logits.',fm:FORMULA.head});
  return seq(S,'once per image');
}
/* ---- one denoising step of a diffusion transformer (DiT-XL/2 on a 64×64×4 latent) ---- */
function ditSteps(c){
  const S=[],C4=4,L=64,p=2,N=(L/p)**2,pd=C4*p*p,dd=1152,A='Denoiser',B='Sampler and decoder';
  X(S,'i_z',A,'Noisy latent','z<sub>t</sub>',[act(sb('z','t'),L,L*C4,{shape:`[${C4} × ${L} × ${L}]`})],{scope:'in',note:'Starts as pure noise at t = T.',why:'Diffusion works on a compressed latent image from a VAE, not on pixels, which makes it about 48× cheaper for a 512-pixel image.',what:`${C4} channels of ${L} × ${L}.`,fm:[['Forward process','z<sub>t</sub> = √ᾱ<sub>t</sub> · z<sub>0</sub> + √(1 − ᾱ<sub>t</sub>) · ε']]});
  X(S,'i_patch',A,'Patchify',`${N} patches of ${p}×${p}`,[act(sb('z','t'),L,L*C4,{shape:`[${C4} × ${L} × ${L}]`}),'→ patches →',act('patches',N,pd)],{op:'split',why:'As in a vision transformer, the latent grid becomes a sequence.',what:`${N} rows of ${pd} numbers.`,fm:[['Patches','N = (64 / 2)² = 1024 ,  each ∈ ℝ<sup>4·2·2</sup>']]});
  X(S,'i_emb',A,'Embed patches',`H = patches·${sb('W','e')} + pos`,[act('patches',N,pd),'×',wgt(sb('W','e'),pd,dd),'=',act('H',N,dd)],{op:'matmul',flops:2*N*pd*dd,why:'Lifts each patch into the transformer’s width and adds a position signal.',what:`Each patch becomes a ${dd}-vector.`,fm:[['Embedding','h<sub>i</sub> = patch<sub>i</sub> · W<sub>e</sub> + pos<sub>i</sub>']]});
  X(S,'i_cond',A,'Condition on timestep and prompt','γ, β = MLP(emb(t) + emb(c))',[act('t, c',1,dd,{shape:'[timestep, condition]'}),'→ embed, MLP →',act('γ, β',2,dd)],{from:[],why:'The network must know how noisy its input is and what image is wanted. In DiT this enters by setting the scale and shift of every layer norm (adaptive LayerNorm).',what:'Two vectors that modulate every block.',fm:[['adaLN','h ← γ(t, c) ⊙ LN(h) + β(t, c)']]});
  X(S,'i_blocks',A,'Transformer blocks','H = DiT blocks(H)  — 28 layers',[act('H',N,dd),'→ 28 × (attention + MLP, adaLN) →',act('H′',N,dd)],{from:['H','γ, β'],note:'Bidirectional attention over all patches; no causal mask.',why:'Every patch attends to every other, so global structure and fine detail are denoised together.',what:'Same shape; each row now encodes what its patch should look like with less noise.',fm:[['Block','h ← h + Attn(adaLN(h)) ;  h ← h + MLP(adaLN(h))']]});
  X(S,'i_out',A,'Final layer',`out = H′·${sb('W','f')}`,[act('H′',N,dd),'×',wgt(sb('W','f'),dd,2*pd),'=',act('out',N,2*pd)],{op:'matmul',flops:2*N*dd*2*pd,why:'Each token is turned back into patch values: a noise estimate and a variance estimate.',what:`${2*pd} numbers per patch (twice the input channels).`,fm:[['Output','[ε̂, Σ̂] per patch']]});
  X(S,'i_unp',A,'Un-patchify','ε̂ = reshape(out)',[act('out',N,2*pd),'→ reshape →',act('ε̂',L,L*C4,{shape:`[${C4} × ${L} × ${L}]`})],{op:'concat',why:'Back to image layout so the estimate lines up with the latent.',what:'Predicted noise with the same shape as the latent.',fm:[['Noise estimate','ε̂ = ε<sub>θ</sub>(z<sub>t</sub>, t, c)']]});
  X(S,'i_step',B,'Sampler step','z<sub>t−1</sub> = step(z<sub>t</sub>, ε̂)',[act('ε̂',L,L*C4,{shape:`[${C4} × ${L} × ${L}]`}),'removed from',act(sb('z','t'),L,L*C4,{shape:`[${C4} × ${L} × ${L}]`}),'=',act(sb('z','t−1'),L,L*C4,{shape:`[${C4} × ${L} × ${L}]`})],{from:['ε̂','zt'],scope:'out',note:'Repeated for every sampling step (typically 20–250).',why:'One network call removes only part of the noise; repeating it walks from pure noise to a clean latent.',what:'A slightly less noisy latent.',fm:[['DDPM step','z<sub>t−1</sub> = (1/√α<sub>t</sub>) · (z<sub>t</sub> − ((1 − α<sub>t</sub>)/√(1 − ᾱ<sub>t</sub>)) · ε̂) + σ<sub>t</sub> · n'],['Guidance','ε̂ = ε<sub>θ</sub>(z, ∅) + w · (ε<sub>θ</sub>(z, c) − ε<sub>θ</sub>(z, ∅))']]});
  X(S,'i_vae',B,'VAE decoder','image = D(z<sub>0</sub>)',[act(sb('z','t−1'),L,L*C4,{shape:`[${C4} × ${L} × ${L}]`}),'→ VAE decoder, × 8 →',tok('image',256,512,{shape:'[3 × 512 × 512]'})],{scope:'out',why:'The clean latent is expanded back into pixels once, at the very end.',what:'Each latent position becomes an 8 × 8 block of pixels.',fm:[['Decode','x = D(z<sub>0</sub>)']]});
  return seq(S,'one network call per sampling step');
}
/* ---- vector-quantised tokenizer (VQGAN, 256-px image, f = 16, codebook 16384 × 256) ---- */
function vqSteps(c){
  const S=[],N=256,D=256,Kc=16384,A='Tokenizer';
  X(S,'q_img',A,'Image','pixels',[tok('image',256,768,{shape:'[3 × 256 × 256]'})],{scope:'in',why:'A continuous signal that an autoregressive transformer cannot predict directly.',what:'RGB values.',fm:[['Input','x ∈ ℝ<sup>3×256×256</sup>']]});
  X(S,'q_enc',A,'Encoder','z = E(x)',[tok('image',256,768,{shape:'[3 × 256 × 256]'}),'→ conv encoder, ÷ 16 →',act('z',N,D)],{scope:'in',why:'Compresses the image into a small grid of feature vectors: 16 × 16 positions instead of 65,536 pixels.',what:`${N} vectors of ${D} numbers.`,fm:[['Encode','z = E(x) ∈ ℝ<sup>16×16×256</sup>']]});
  X(S,'q_dist',A,'Distance to every code','D[i, k] = ‖z<sub>i</sub> − e<sub>k</sub>‖²',[act('z',N,D),'vs',wgt('codebook e',Kc,D),'=',msk('dist',N,Kc)],{scope:'in',flops:3*N*Kc*D,why:'To turn a vector into a discrete symbol, find which entry of a learned dictionary it is closest to.',what:`A table of ${N} × ${Kc} distances.`,fm:[['Distance','D[i, k] = ‖z<sub>i</sub>‖² − 2·z<sub>i</sub>·e<sub>k</sub> + ‖e<sub>k</sub>‖²']]});
  X(S,'q_arg',A,'Pick the nearest code','ids[i] = argmin<sub>k</sub> D[i, k]',[msk('dist',N,Kc),'→ argmin →',tok('ids',N,1,{shape:`[${N}]`})],{scope:'in',why:'The index of the nearest code is the token. These integers are what a language model predicts when it generates an image.',what:`${N} integers between 0 and ${Kc-1} (14 bits each).`,fm:[['Quantise','k<sub>i</sub> = argmin<sub>k</sub> ‖z<sub>i</sub> − e<sub>k</sub>‖']]});
  X(S,'q_look',A,'Look the codes back up','z<sub>q</sub> = e[ids]',[tok('ids',N,1,{shape:`[${N}]`}),'→ rows of the codebook →',act(sb('z','q'),N,D)],{scope:'out',op:'lookup',why:'To reconstruct, each token is replaced by its dictionary vector — the part of the information that survived.',what:'The same shape as z, but every row is now one of the codebook entries.',fm:[['Lookup','z<sub>q,i</sub> = e<sub>k<sub>i</sub></sub>'],['Training loss','ℒ = ‖x − x̂‖² + ‖sg[z] − e‖² + β‖z − sg[e]‖²']]});
  X(S,'q_dec',A,'Decoder','x̂ = G(z<sub>q</sub>)',[act(sb('z','q'),N,D),'→ conv decoder, × 16 →',tok('x̂',256,768,{shape:'[3 × 256 × 256]'})],{scope:'out',why:'Turns the token grid back into pixels. In VQGAN it is trained with a perceptual and an adversarial loss so the output looks sharp.',what:'A reconstructed image.',fm:[['Decode','x̂ = G(z<sub>q</sub>)']]});
  return seq(S,'once per image');
}
/* ---- one refinement step of a masked-diffusion language model (LLaDA / MDLM style) ---- */
function mdlmSteps(c){
  const S=[],{T,d,V}=c,A='One refinement step';
  X(S,'d_ids',A,'Partly masked sequence','x<sub>t</sub>',[tok(sb('x','t'),T,1,{shape:`[${T}] · some [MASK]`})],{scope:'in',why:'Generation starts with every position masked. Each step fills some in.',what:'Token IDs, with a special mask ID at undecided positions.',fm:[['Forward process','each token is replaced by [MASK] with probability t']]});
  X(S,'d_emb',A,'Embedding','X = E[x<sub>t</sub>]',[tok(sb('x','t'),T,1,{shape:`[${T}]`}),'→ pick rows of',wgt('E',V,d),'=',res('X',T,d)],{scope:'in',op:'lookup',why:WHY.emb[0],what:'The mask token has its own embedding row.',fm:FORMULA.emb});
  X(S,'d_net',A,'Bidirectional transformer','H = Transformer(X)  — no causal mask',[res('X',T,d),`→ ${c.N} × (full attention + MLP) →`,act('H',T,d)],{note:'The layers are the encoder-only block: every position attends to every other.',why:'A masked position needs context from both sides to be filled in, so the causal mask of an autoregressive model is dropped.',what:'One contextual vector per position.',fm:[['Layer','h ← h + Attn(Norm(h)) ;  h ← h + MLP(Norm(h))']]});
  X(S,'d_head',A,'Predict every position',`logits = H·${sb('W','LM')}`,[act('H',T,d),'×',wgt(sb('W','LM'),d,V),'=',msk('logits',T,V)],{op:'matmul',flops:2*T*d*V,why:'Unlike an autoregressive model, all positions are scored in one pass.',what:`A [${T} × V] table of scores.`,fm:FORMULA.head});
  X(S,'d_soft',A,'Softmax','p = softmax(logits)',[msk('logits',T,V),'→ softmax each row →',prb('p',T,V)],{op:'softmax',why:WHY.vsoft[0].split('. ')[0]+'.',what:'One distribution per position.',fm:FORMULA.softmax});
  X(S,'d_un',A,'Unmask the most confident','x<sub>t−Δ</sub> = fill(x<sub>t</sub>, p)',[prb('p',T,V),'→ sample, keep the most confident, re-mask the rest →',tok(sb('x','t−Δ'),T,1,{shape:`[${T}] · fewer [MASK]`})],{scope:'out',note:'Repeated for a fixed number of steps until no masks remain.',why:'Committing only to the predictions the model is sure about, and asking again for the rest, lets later steps use the newly fixed tokens as context.',what:'A sequence with fewer masks.',fm:[['Training loss','ℒ = 𝔼[(1/t) · Σ<sub>masked i</sub> −log p<sub>θ</sub>(x<sub>0</sub><sup>i</sup> | x<sub>t</sub>)]']]});
  return seq(S,'one pass per refinement step');
}
const notDecoder=(facts,x)=>()=>Object.assign({norm:'—',pos:'—',mask:'—',mlp:'—',kv:1,kvText:'no KV cache',facts},x);
ARCH.push(
 {id:'alibi',name:'Decoder with ALiBi position bias  (BLOOM style)',short:'ALiBi',desc:'No position vectors and no RoPE: each head subtracts a fixed, linear distance penalty from its attention scores.',meta:c=>gqaMeta(c,{pos:'ALiBi bias on the scores'}),build:alibiDecoder},
 {id:'adapters',name:'Adapter fine-tuning  (bottleneck layers in a frozen decoder)',short:'Adapters',table:false,desc:'A small trainable down–up network with a skip connection is inserted after attention and after the MLP of every frozen layer.',meta:c=>gqaMeta(c),build:adapterDecoder},
 {id:'prefix',name:'Prefix tuning  (trainable rows in K and V)',short:'Prefix tuning',table:false,desc:'Every weight is frozen. A few trainable key and value rows are prepended in every layer, so attention can read from learned “virtual tokens”.',meta:c=>gqaMeta(c),build:prefixDecoder},
 {id:'prompt',name:'Prompt tuning  (trainable vectors before the input)',short:'Prompt tuning',table:false,desc:'Only a short sequence of input vectors is trained; the frozen model processes them like ordinary tokens.',meta:c=>gqaMeta(c),build:promptDecoder},
 {id:'ia3',name:'(IA)³  (learned rescaling of K, V and the MLP hidden state)',short:'(IA)³',table:false,desc:'Three trainable vectors per layer multiply keys, values and the MLP hidden activations element by element.',meta:c=>gqaMeta(c),build:ia3Decoder},
 {id:'dora',name:'DoRA  (magnitude + direction, low-rank update on W_Q)',short:'DoRA',table:false,desc:'The weight is split into column lengths and directions. LoRA updates the direction; a trainable vector sets the lengths.',meta:c=>gqaMeta(c),build:doraDecoder},
 {id:'qlora',name:'QLoRA  (LoRA on a 4-bit base model)',short:'QLoRA',table:false,desc:'The frozen weights are stored in 4 bits and de-quantised on the fly; ordinary 16-bit LoRA matrices are trained on top.',meta:c=>gqaMeta(c),build:qloraDecoder},
 {id:'mamba',name:'Mamba block  (selective state-space model)',short:'Mamba',table:false,desc:'Token mixing by an input-dependent recurrence over a fixed-size state. Sizes follow the paper’s defaults: expansion 2, state 16, convolution 4.',meta:notDecoder([['Token mixing','selective state-space recurrence, O(T)'],['State per layer','[2d × 16], fixed size'],['Defaults','expand = 2 · d_state = 16 · d_conv = 4']]),build:mambaSteps},
 {id:'rwkv',name:'RWKV-4 block  (time mixing + channel mixing)',short:'RWKV',table:false,desc:'An RNN whose time-mixing step keeps an exponentially decayed weighted average of the past as two running sums.',meta:notDecoder([['Token mixing','running numerator / denominator with learned decay'],['State per layer','a few vectors of width d, fixed size']]),build:rwkvSteps},
 {id:'vit',name:'Image → tokens for an LLM  (LLaVA-1.5: ViT-L/14 at 336 px + MLP projector)',short:'ViT + projector',table:false,desc:'A vision transformer encodes image patches; a two-layer MLP projects them into the language model’s embedding space, where they join the text tokens.',meta:notDecoder([['Vision encoder','CLIP ViT-L/14, 336 px → 576 patch tokens of width 1024'],['Projector','2-layer MLP with GELU'],['Language model','an ordinary decoder; width from the configuration bar']]),build:vitSteps},
 {id:'dit',name:'Diffusion transformer — one denoising step  (DiT-XL/2)',short:'DiT',table:false,desc:'A noisy latent is cut into patches, processed by a bidirectional transformer conditioned on timestep and prompt, and turned into a noise estimate that the sampler removes.',meta:notDecoder([['Latent','4 × 64 × 64 for a 512-px image (VAE, ÷ 8)'],['DiT-XL/2','patch 2 → 1024 tokens · width 1152 · 28 layers'],['Sampling','20 – 250 network calls per image']]),build:ditSteps},
 {id:'vq',name:'Vector-quantised tokenizer  (VQGAN, f = 16)',short:'VQ tokenizer',table:false,desc:'An encoder produces a grid of vectors; each is replaced by the index of its nearest codebook entry. Those integers are the tokens.',meta:notDecoder([['Compression','256 × 256 pixels → 16 × 16 = 256 tokens'],['Codebook','16,384 entries of 256 dims (VQGAN)']]),build:vqSteps},
 {id:'mdlm',name:'Masked-diffusion language model — one refinement step',short:'Diffusion LM',table:false,desc:'All positions are predicted at once by a bidirectional transformer; the most confident predictions are kept and the rest stay masked for the next step.',meta:notDecoder([['Order','any order, several tokens per step'],['Attention mask','none (bidirectional)'],['Passes','one per refinement step, not one per token']]),build:mdlmSteps}
);
Object.assign(BACK,{alibi:'model',adapters:'topic/adapters',prefix:'topic/prefix',prompt:'topic/prefix',ia3:'topic/ia3',dora:'topic/dora',qlora:'topic/qlora',mamba:'topic/ssm',rwkv:'topic/rwkv',vit:'topic/multimodal',dit:'topic/imgdiff',vq:'topic/vqtok',mdlm:'topic/difflm'});
/* which toolbar fields matter for which architecture */
const FIELD_USE={S:['encdec'],W:['swa'],E:['moe'],k:['moe'],r:['lora','qlora','dora'],m:['adapters'],p:['prefix','prompt']};
