/* ================= 1. Decoder layer — graph, info, steps ================= */
function layerSVG(cur,focus){
  const {T,d,H,ff}=S,dh=d/H,kvh=kvH(),kvd=kvh*dh,g=new Graph(cur,focus),tD=`[T, ${d}]`;
  g.under(`<rect class="frame" x="16" y="44" width="1488" height="420" rx="18"/><rect class="frame" x="16" y="492" width="1488" height="300" rx="18"/><text class="frt" x="150" y="68">ATTENTION</text><text class="frt" x="100" y="516">MLP · SWIGLU</text>`);
  /* ---- attention row ---- */
  g.node({id:'x',x:70,y:250,dims:[T,d],col:ROLE.x,label:'X',shape:tD,step:0});
  g.node({id:'norm1',x:190,y:250,dims:[T,d],col:ROLE.xn,label:'RMSNorm',shape:tD,step:1});
  [['q',110,H],['k',250,kvh],['v',390,kvh]].forEach(([n,y,hh])=>{
    const wc=n==='q'?d:kvd,N=n.toUpperCase();
    g.node({id:'w'+n,x:320,y,dims:[d,wc],col:C.purple,label:'W'+n,shape:`[${d}, ${wc}]`,step:2});
    g.node({id:n,x:430,y,dims:[T,wc],col:ROLE[n],label:N,shape:`[T, ${wc}]`,step:2});
    g.node({id:n+'h',x:555,y,dims:[hh,T,dh],bands:BANDS.slice(0,Math.min(6,Math.max(1,hh))),col:ROLE[n],label:N+' heads',shape:`[T, ${hh}, ${dh}]`,step:3});
  });
  g.op({id:'qk',x:665,y:180,r:22,sym:'QKᵀ',col:C.red,label:`÷ √${dh}`,step:4});
  g.node({id:'scores',x:790,y:108,dims:[H,T,T],col:ROLE.s,label:'Attn scores',shape:`[${H}, T, T]`,step:4});
  g.pill({id:'smx',x:790,y:196,w:176,h:28,text:'Scale + Mask + Softmax',col:C.blue,step:5});
  g.node({id:'a',x:790,y:300,dims:[H,T,T],col:ROLE.a,label:'A  (weights)',shape:`[${H}, T, T]`,step:5});
  g.op({id:'av',x:915,y:300,r:22,sym:'A×V',col:C.red,label:'',step:6});
  g.node({id:'oh',x:1030,y:300,dims:[H,T,dh],bands:BANDS,col:C.blue,label:'Head outputs',shape:`[T, ${H}, ${dh}]`,step:6});
  g.node({id:'cat',x:1140,y:300,dims:[T,d],col:ROLE.o,label:['Concat','heads'],shape:tD,step:7});
  g.node({id:'wo',x:1245,y:300,dims:[d,d],col:C.purple,label:'Wo',shape:`[${d}, ${d}]`,step:7});
  g.node({id:'ao',x:1350,y:300,dims:[T,d],col:ROLE.attn,label:'Attn out',shape:tD,step:7});
  g.op({id:'add1',x:1440,y:300,r:18,sym:'+',col:C.green,label:'',step:8});
  g.edge('x','norm1',{step:1});
  ['q','k','v'].forEach(n=>{g.edge('norm1','w'+n,{step:2});g.edge('w'+n,n,{step:2});g.edge(n,n+'h',{step:3});});
  g.label(492,102,'reshape',{step:3});g.label(492,242,'reshape',{step:3});g.label(492,382,'reshape',{step:3});
  g.edge('qh','qk',{step:4,dy2:-9});g.edge('kh','qk',{step:4,dy2:9});g.edge('qk','scores',{step:4});
  if(kvh<H)g.label(665,236,`K,V broadcast ×${H/kvh}`,{step:4,col:C.orange});
  g.path('M790 164V180',{step:5});g.path('M790 212V240',{step:5});
  g.edge('a','av',{step:6});g.path('M597 390H915V326',{step:6});
  g.edge('av','oh',{step:6});g.edge('oh','cat',{step:7});g.edge('cat','wo',{step:7});g.edge('wo','ao',{step:7});g.edge('ao','add1',{step:8});
  g.path('M130 250V30H1440V278',{col:C.green,w:3,step:0,solid:true});
  g.label(790,22,'Residual stream  [T, '+d+']',{col:C.green});
  g.path('M1440 322V478H70V640H149',{col:C.green,w:3,step:8,solid:true});
  g.label(760,468,'h = X + attention(norm(X))   [T, '+d+']',{col:C.green,step:8});
  /* ---- MLP row ---- */
  g.node({id:'norm2',x:190,y:640,dims:[T,d],col:ROLE.xn,label:'RMSNorm',shape:tD,step:9});
  g.node({id:'wg',x:330,y:570,dims:[d,ff],col:C.purple,label:'W_gate',shape:`[${d}, ${ff}]`,step:10});
  g.node({id:'wu',x:330,y:715,dims:[d,ff],col:C.purple,label:'W_up',shape:`[${d}, ${ff}]`,step:10});
  g.node({id:'gt',x:445,y:570,dims:[T,ff],col:ROLE.gate,label:'gate',shape:`[T, ${ff}]`,step:10});
  g.node({id:'up',x:445,y:715,dims:[T,ff],col:ROLE.up,label:'up',shape:`[T, ${ff}]`,step:10});
  g.op({id:'silu',x:555,y:570,r:22,sym:'SiLU',col:C.orange,label:'',step:11});
  g.op({id:'mul',x:655,y:640,r:18,sym:'⊙',col:C.orange,label:'',step:11});
  g.node({id:'prod',x:765,y:640,dims:[T,ff],col:ROLE.hid,label:['SiLU(gate)','⊙ up'],shape:`[T, ${ff}]`,step:11});
  g.node({id:'wd',x:880,y:640,dims:[ff,d],col:C.purple,label:'W_down',shape:`[${ff}, ${d}]`,step:12});
  g.node({id:'mo',x:995,y:640,dims:[T,d],col:ROLE.mlp,label:'MLP out',shape:tD,step:12});
  g.op({id:'add2',x:1095,y:640,r:18,sym:'+',col:C.green,label:'',step:13});
  g.node({id:'out',x:1210,y:640,dims:[T,d],col:ROLE.x,label:'Output',shape:tD,step:13});
  g.edge('norm2','wg',{step:10});g.edge('norm2','wu',{step:10});g.edge('wg','gt',{step:10});g.edge('wu','up',{step:10});
  g.edge('gt','silu',{step:11});g.edge('silu','mul',{step:11,dy2:-7});g.edge('up','mul',{step:11,dy2:7});
  g.edge('mul','prod',{step:11});g.edge('prod','wd',{step:12});g.edge('wd','mo',{step:12});g.edge('mo','add2',{step:13});g.edge('add2','out',{step:13});
  g.path('M1095 478V620',{col:C.green,w:3,step:13,solid:true});
  g.path('M1253 640H1318',{col:C.green,step:13});g.label(1326,645,'next layer',{anchor:'start',step:13});
  return svgWrap(1520,800,g.html());
}

function layerInfoBase(id){
  if(!id)return null;
  const {T,d,H,ff}=S,dh=d/H,kvh=kvH(),kvd=kvh*dh,by=S.bytes,g=H/kvh;
  const sh=(...a)=>'['+a.join(', ')+']',mem=n=>fmtB(n*by);
  const mm=(m,k,n)=>fmtF(2*m*k*n);
  const wIn={q:d,k:kvd,v:kvd};
  const gen={
    x:{t:'X — layer input',c:C.green,d:'Hidden states entering the layer: one vector of width d per token. This is the <b>residual stream</b> — every sub-layer reads it and adds its result back.',r:[['Shape',sh('T',d)+` = ${sh(T,d)}`],['Memory',mem(T*d)]]},
    norm1:{t:'RMSNorm (pre-attention)',c:C.blue,d:'y = x / √(mean(x²)+ε) · γ. Rescales each token vector to unit RMS with a learned gain γ. Compared with LayerNorm there is no mean-centering and no bias, so it is cheaper and just as stable.',r:[['Shape',sh('T',d)+' → '+sh('T',d)],['Params',fmtN(d)+' (γ)']]},
    qk:{t:'Q·Kᵀ — attention scores',c:C.red,d:`For every head, each query is dotted with every key and scaled by 1/√${dh}. This produces a T×T score matrix per head — the source of attention's O(T²) cost.${kvh<H?` With ${S.attn}, ${g} query heads reuse the same K head.`:''}`,r:[['Per head',`${sh('T',dh)} × ${sh(dh,'T')} → ${sh('T','T')}`],['FLOPs (all heads)',fmtF(2*H*T*T*dh)]]},
    scores:{t:'Attention scores',c:C.red,d:'Raw similarity between every query position and every key position, one T×T matrix per head. Stored in full this is the biggest activation in the layer — FlashAttention avoids ever materialising it.',r:[['Shape',sh(H,'T','T')+` = ${sh(H,T,T)}`],['Memory',mem(H*T*T)]]},
    smx:{t:'Scale + causal mask + softmax',c:C.blue,d:'Scores are divided by √d_head, future positions (j > i) are set to −∞ so a token cannot see later tokens, and a row-wise softmax turns each row into weights that sum to 1.',r:[['Shape',sh(H,'T','T')+' → '+sh(H,'T','T')],['Cost','~5·H·T² element ops']]},
    a:{t:'A — attention weights',c:C.red,d:'Row i holds how much token i attends to every earlier token j ≤ i. Rows sum to 1; masked entries are exactly 0.',r:[['Shape',sh(H,'T','T')+` = ${sh(H,T,T)}`],['Memory',mem(H*T*T)]]},
    av:{t:'A × V — weighted sum of values',c:C.red,d:'Each token takes a weighted average of the value vectors it attends to, per head.',r:[['Per head',`${sh('T','T')} × ${sh('T',dh)} → ${sh('T',dh)}`],['FLOPs (all heads)',fmtF(2*H*T*T*dh)]]},
    oh:{t:'Head outputs',c:C.blue,d:'One [T, d_head] result per head, still separate. Heads can specialise: syntax, coreference, position, copying…',r:[['Shape',sh('T',H,dh)]]},
    cat:{t:'Concat heads',c:C.blue,d:`The ${H} head outputs of width ${dh} are laid side by side to rebuild a vector of width ${d} per token. Pure reshape — no FLOPs.`,r:[['Shape',sh('T',H,dh)+' → '+sh('T',d)]]},
    wo:{t:'Wo — output projection',c:C.purple,d:'Mixes information across heads and maps back into the residual stream.',r:[['Weight',sh(d,d)],['Params',fmtN(d*d)],['FLOPs',mm(T,d,d)]]},
    ao:{t:'Attention output',c:C.blue,d:'What this sub-layer wants to add to the residual stream.',r:[['Shape',sh('T',d)]]},
    add1:{t:'Residual add ⊕',c:C.green,d:'h = X + attention_out. The skip connection lets information and gradients bypass the sub-layer, which is what makes deep stacks trainable.',r:[['Shape',sh('T',d)+' + '+sh('T',d)]]},
    norm2:{t:'RMSNorm (pre-MLP)',c:C.blue,d:'Second pre-norm, this time in front of the feed-forward block.',r:[['Shape',sh('T',d)],['Params',fmtN(d)+' (γ)']]},
    wg:{t:'W_gate',c:C.purple,d:'Expands each token from d to d_ff. After SiLU this branch acts as a learned, per-feature gate.',r:[['Weight',sh(d,ff)],['Params',fmtN(d*ff)],['FLOPs',mm(T,d,ff)]]},
    wu:{t:'W_up',c:C.purple,d:'Second expansion d → d_ff carrying the content that the gate lets through.',r:[['Weight',sh(d,ff)],['Params',fmtN(d*ff)],['FLOPs',mm(T,d,ff)]]},
    gt:{t:'gate activations',c:C.pink,d:'h · W_gate — the raw gate values before SiLU.',r:[['Shape',sh('T',ff)],['Memory',mem(T*ff)]]},
    up:{t:'up activations',c:C.pink,d:'h · W_up — the content stream.',r:[['Shape',sh('T',ff)],['Memory',mem(T*ff)]]},
    silu:{t:'SiLU activation',c:C.orange,d:'SiLU(x) = x · sigmoid(x), a smooth ReLU-like non-linearity (also called swish).',r:[['Shape',sh('T',ff)+' → '+sh('T',ff)]]},
    mul:{t:'Element-wise product ⊙',c:C.orange,d:'SwiGLU: SiLU(gate) ⊙ up. The gate scales each feature of the up projection element by element.',r:[['Shape',sh('T',ff)+' ⊙ '+sh('T',ff)]]},
    prod:{t:'Gated features',c:C.pink,d:'The wide hidden state of the MLP, where most of the layer’s parameters and FLOPs live.',r:[['Shape',sh('T',ff)],['Memory',mem(T*ff)]]},
    wd:{t:'W_down',c:C.purple,d:'Projects the wide hidden state back to model width d.',r:[['Weight',sh(ff,d)],['Params',fmtN(ff*d)],['FLOPs',mm(T,ff,d)]]},
    mo:{t:'MLP output',c:C.blue,d:'What the MLP wants to add to the residual stream.',r:[['Shape',sh('T',d)]]},
    add2:{t:'Residual add ⊕',c:C.green,d:'out = h + mlp_out — the layer output, ready for the next decoder layer.',r:[['Shape',sh('T',d)]]},
    out:{t:'Layer output',c:C.green,d:'Same shape as the input, so layers stack: the output of layer ℓ is the input of layer ℓ+1.',r:[['Shape',sh('T',d)+` = ${sh(T,d)}`],['Memory',mem(T*d)]]}
  };
  if(gen[id])return gen[id];
  const m=id.match(/^w([qkv])$/);
  if(m){const n=m[1].toUpperCase(),wc=wIn[m[1]];return {t:'W'+m[1]+' — '+n+' projection',c:C.purple,d:`Learned matrix that turns every normalised token into its ${n==='Q'?'query':n==='K'?'key':'value'} vector.${n!=='Q'&&kvh<H?` Only ${kvh} K/V head${kvh>1?'s':''} (${kvd} wide) instead of ${H}, so the ${S.attn} KV cache is ${H/kvh}× smaller.`:''}`,r:[['Weight',sh(d,wc)],['Params',fmtN(d*wc)],['FLOPs',mm(T,d,wc)]]};}
  const p=id.match(/^([qkv])$/);
  if(p){const n=p[1].toUpperCase(),wc=wIn[p[1]];return {t:n+' — '+(n==='Q'?'queries':n==='K'?'keys':'values'),c:C.green,d:n==='Q'?'What each token is looking for.':n==='K'?'What each token offers to be matched against.':'The information each token hands over when attended to.',r:[['Shape',sh('T',wc)+` = ${sh(T,wc)}`],['Memory',mem(T*wc)]]};}
  const q=id.match(/^([qkv])h$/);
  if(q){const n=q[1].toUpperCase(),hh=q[1]==='q'?H:kvh;return {t:n+' heads',c:C.blue,d:`Reshape [T, ${hh*dh}] → [T, ${hh}, ${dh}]: the width is split into ${hh} independent heads of ${dh} dims. Heads become a batch dimension.${q[1]!=='q'&&kvh<H?` Each is shared by ${g} query heads.`:''}`,r:[['Shape',sh('T',hh,dh)],['In memory',sh(hh,'T',dh)]]};}
  return null;
}

function layerSteps(){
  const {T,d,H,ff}=S,dh=d/H,kvh=kvH(),kvd=kvh*dh;
  return [
    ['Input X',`The layer receives X — ${T} token vectors of width ${d}. This tensor is the residual stream: every sub-layer reads it and adds its result back.`],
    ['RMSNorm (pre-norm)',`Each token vector is rescaled by its RMS and a learned gain so the projections see well-conditioned inputs. Shape stays [T, ${d}].`],
    ['Q, K, V projections',`Three independent matmuls: Q = X·Wq [${d}, ${d}], K = X·Wk [${d}, ${kvd}], V = X·Wv [${d}, ${kvd}].${kvh<H?` ${S.attn} gives K and V only ${kvh} head(s), shared by ${H/kvh} query heads each.`:''}`],
    ['Split into heads',`Reshape Q to [T, ${H}, ${dh}] and K, V to [T, ${kvh}, ${dh}]. Heads become a batch dimension and run in parallel.`],
    ['Scores = Q·Kᵀ',`Every query dots every key: [${H}, T, ${dh}] × [${H}, ${dh}, T] → scores [${H}, T, T]. A T×T matrix per head is what makes attention O(T²).`],
    ['Scale · mask · softmax',`Divide by √${dh}, set future positions to −∞ (causal mask) and softmax each row → attention weights A; every row sums to 1.`],
    ['A × V',`Each token takes a weighted average of value vectors: [${H}, T, T] × [${H}, T, ${dh}] → [${H}, T, ${dh}].`],
    ['Concat heads · Wo',`Heads are concatenated back to [T, ${d}] and mixed by the output projection Wo [${d}, ${d}].`],
    ['Residual add',`h = X + attention_out. The skip connection lets information and gradients bypass the sub-layer.`],
    ['RMSNorm #2',`A second pre-norm, now in front of the MLP. h is also kept aside for the next skip connection.`],
    ['Gate & up projections',`Two parallel expansions d → d_ff: gate = h·W_gate and up = h·W_up, both [T, ${ff}].`],
    ['SwiGLU',`SiLU(gate) ⊙ up — the gate decides, feature by feature, how much of "up" passes through.`],
    ['Down projection',`W_down [${ff}, ${d}] squeezes the wide hidden state back to model width: [T, ${ff}] → [T, ${d}].`],
    ['Residual add → output',`out = h + mlp_out. This [T, ${d}] tensor is the input of the next decoder layer (or of the final norm after layer N).`]
  ];
}

const LAYER_ROLE={x:'x',norm1:'xn',q:'q',k:'k',v:'v',qh:'q',kh:'k',vh:'v',scores:'s',a:'a',oh:'o',cat:'o',ao:'attn',norm2:'xn',gt:'gate',up:'up',prod:'hid',mo:'mlp',out:'x',wq:'w',wk:'w',wv:'w',wo:'w',wg:'w',wu:'w',wd:'w'};
const layerInfo=id=>{let i=layerInfoBase(id);if(i&&LAYER_ROLE[id])i=Object.assign({},i,{c:ROLE[LAYER_ROLE[id]]});return i&&LAYER_WHY[id]?Object.assign({},i,{why:explainKey(LAYER_WHY[id]),fm:formulasFor(LAYER_WHY[id])}):i;};
