/* ================= additional architectures: Qwen3.8-Flash-Next and LoRA fine-tuning ================= */

/* ---- Qwen3.8-Flash-Next: hybrid decoder (Gated DeltaNet + sparse attention), ultra-sparse MoE, n-gram memory.
   Uses the published dimensions (hidden 2560, 24/2 heads of 256, 512 experts top-10, 16 × 160 n-gram rows).
   Only T and V follow the configuration bar. Unpublished sizes are drawn with placeholders and labelled. ---- */
function qwenNext(c){
  const S=[],T=c.T,D=2560,V=c.V,NP='not published';
  const R4=()=>res('R',T,4*D,{stripes:4,shape:`[${T} × 4 × ${D}]`});
  const SN={ng:'once, at decoder block 2',gdn:'in 3 of every 4 layers · 36 of 48',qsa:'in every 4th layer · 12 of 48',moe:'in every one of the 48 layers',out:'once, after layer 48'};
  inputSteps(S,{d:D,V},T,null);
  S.push(st('hc0','Input','Widen to 4 residual branches','R = branches(X)',[res('X',T,D),'→ 4 parallel streams →',R4()],{scope:'in',note:'The residual stream is four parallel branches (“gated residual”). How they are initialised from the embedding is '+NP+'.'}));
  /* n-gram memory */
  S.push(st('nghash','N-gram memory','Hash the last 2 and 3 tokens','rows = hash(x<sub>t−1</sub>, x<sub>t</sub>) ∪ hash(x<sub>t−2</sub>, x<sub>t−1</sub>, x<sub>t</sub>)',[tok('ids',T,1,{shape:`[${T}]`}),'→ 8 bigram + 8 trigram hashes →',tok('rows',T,16,{shape:`[${T} × 16]`})],{scope:'in',sn:SN.ng,note:'Eight hash heads over the last two tokens and eight over the last three give 16 row numbers per position. The lookup key is known from token IDs alone, before any layer has run.'}));
  S.push(st('nglook','N-gram memory','Look up 16 rows and concatenate','E<sub>ng</sub> = [N<sub>1</sub>[row<sub>1</sub>] | … | N<sub>16</sub>[row<sub>16</sub>]]',[tok('rows',T,16,{shape:`[${T} × 16]`}),'→ pick rows of',wgt('N',20000000,160,{shape:'16×[20M × 160]'}),'=',act(sb('E','ng'),T,D)],{scope:'in',sn:SN.ng,op:'lookup',params:51.2e9,note:'16 × 160 = 2560 numbers per token. The table holds about 51B parameters and is kept in host RAM, prefetched while the GPU computes.'}));
  S.push(st('nggate','N-gram memory','Gate and refine','U = g ⊙ E<sub>ng</sub> ;  Δ = U + SiLU(DWConv(RMSNorm(U)))',[act(sb('E','ng'),T,D),'→ gate per branch, conv, SiLU →',act('Δ',T,4*D,{stripes:4,shape:`[${T} × 4 × ${D}]`})],{scope:'in',sn:SN.ng,note:'A gate g ∈ ℝ<sup>4×1</sup> computed from the hidden state decides how much memory each branch receives; a depth-wise convolution refines it.'}));
  S.push(st('nginj','N-gram memory','Inject into the residual branches','R = R + Δ',[R4(),'+',act('Δ',T,4*D,{stripes:4,shape:`[${T} × 4 × ${D}]`}),'=',R4()],{scope:'in',sn:SN.ng,op:'add',note:'Done once. Every later layer sees the memorised n-gram information.'}));
  /* Gated DeltaNet sub-layer */
  S.push(st('mix1','Gated DeltaNet','Read the residual branches','h = Mix(R)',[R4(),'→ gated read →',act('h₁',T,D)],{sn:SN.gdn,note:'Each branch is normalised and blended through a data-dependent element-wise gate (exact form '+NP+').'}));
  S.push(st('gproj','Gated DeltaNet','Project to key, value, query and gates','q, k, v, α, β = f(h)',[act('h₁',T,D),'→ project →',act('q, k, v, α, β',T,768,{shape:'sizes n/p'})],{sn:SN.gdn,note:'α (forget) and β (write strength) are between 0 and 1.'}));
  S.push(st('gstate','Gated DeltaNet','Update the fixed-size state','S<sub>t</sub> = α<sub>t</sub>·S<sub>t−1</sub>·(I − β<sub>t</sub>k<sub>t</sub>k<sub>t</sub><sup>ᵀ</sup>) + β<sub>t</sub>v<sub>t</sub>k<sub>t</sub><sup>ᵀ</sup>',[act('q, k, v, α, β',T,768,{shape:'sizes n/p'}),'edits',msk(sb('S','t−1'),256,256,{shape:'[dₖ × dᵥ] fixed'}),'=',msk(sb('S','t'),256,256,{shape:'[dₖ × dᵥ] fixed'})],{sn:SN.gdn,note:'The state never grows with the context: no KV cache for these layers. The update rule is the Gated DeltaNet form; Qwen’s exact variant and state size are '+NP+'.'}));
  S.push(st('gread','Gated DeltaNet','Read from the state','o<sub>t</sub> = S<sub>t</sub>·q<sub>t</sub>',[msk(sb('S','t'),256,256,{shape:'[dₖ × dᵥ] fixed'}),'× q, project','=',act('GDN',T,D)],{sn:SN.gdn,note:'Constant work per token, whatever the context length.'}));
  S.push(st('hcw1','Gated DeltaNet','Write back to the branches','R[b] = R[b] + c<sub>b</sub>·GDN',[R4(),'+ c ⊙',act('GDN',T,D),'=',R4()],{sn:SN.gdn,op:'add',note:'Four data-dependent coefficients, one per branch.'}));
  /* sparse attention sub-layer */
  const nb=Math.max(1,Math.ceil(T/4)),sel=Math.min(T,2051);
  S.push(st('mix2','Sparse attention','Read the residual branches','h = Mix(R)',[R4(),'→ gated read →',act('h₂',T,D)],{sn:SN.qsa}));
  S.push(st('idx','Sparse attention','Index vectors','q<sup>I</sup>, k<sup>I</sup> = index(h)',[act('h₂',T,D),'→ small projections →',act('q<sup>I</sup> k<sup>I</sup>',T,640,{stripes:5,shape:`[${T} × 5 × 128]`})],{sn:SN.qsa,note:'Four 128-d query heads and one shared key head, used only to decide where to look.'}));
  S.push(st('pool','Sparse attention','Pool keys into blocks of 4','k̄<sub>b</sub> = RoPE(norm(mean of 4 keys))',[act('q<sup>I</sup> k<sup>I</sup>',T,640,{stripes:5,shape:`[${T} × 5 × 128]`}),'→ avg-pool 4, RoPE →',act('k̄',nb,128)],{sn:SN.qsa,note:'One compressed key per four tokens. This index cache is a quarter the length of the context.'}));
  S.push(st('bscore','Sparse attention','Score every block','s<sub>t,b</sub> = (1/√128)·Σ<sub>h</sub> ReLU(⟨q<sup>I</sup><sub>t,h</sub>, k̄<sub>b</sub>⟩)',[act('k̄',nb,128),'· q<sup>I</sup>, ReLU, Σ heads','=',msk('s',T,nb)],{sn:SN.qsa,note:'Cheap: 128-dimensional vectors against a quarter as many keys.'}));
  S.push(st('bsel','Sparse attention','Keep the top 512 blocks','sel<sub>t</sub> = top-512 blocks → 2048 tokens (+ ≤ 3)',[msk('s',T,nb),'→ top-512 →',msk(sb('M','sel'),T,T,{mask:'window',frac:sel/T,shape:`[${T} × ≤${sel}]`})],{sn:SN.qsa,note:'Each query attends to at most 2051 positions, so the real attention has a fixed cost even at 1M tokens of context.'}));
  const i0=S.length;
  attn(S,{d:6144,H:24,h:c.h,W:sel},{R:T,L:T,kv:2,mask:'window',rope:true,xin:'h₂',din:D,stage:'Sparse attention'});
  S.slice(i0).forEach(s=>{s.sn=SN.qsa;});
  const m=S.find(s=>s.key==='mask');
  Object.assign(m,{title:'Sparse block mask',fx:'S″ = S′ + M<sub>sel</sub> ,  M[i,j] = 0 if j is in a selected block of i (and j ≤ i) else −∞',note:'The band is only an illustration — the selected blocks can be anywhere in the past.'});
  m.expr[2]=msk(sb('M','sel'),T,T,{mask:'window',frac:sel/T});m.sig=m.fx+'|sparse';
  S.push(st('hcw2','Sparse attention','Write back to the branches','R[b] = R[b] + c<sub>b</sub>·Attn',[R4(),'+ c ⊙',act('Attn',T,D),'=',R4()],{sn:SN.qsa,op:'add'}));
  /* MoE */
  S.push(st('mix3','MoE','Read the residual branches','h = Mix(R)',[R4(),'→ gated read →',act('h₃',T,D)],{sn:SN.moe}));
  S.push(st('router','MoE','Router',`r = h·${sb('W','r')}`,[act('h₃',T,D),'×',wgt(sb('W','r'),D,512),'=',msk('r',T,512)],{sn:SN.moe,op:'matmul',flops:2*T*D*512,note:'One score per expert, for 512 experts.'}));
  S.push(st('topk','MoE','Top-10 + softmax','g = softmax(top<sub>10</sub>(r))',[msk('r',T,512),'→ keep top 10 per row →',prb('g',T,10)],{sn:SN.moe,op:'topk',note:'About 2% of the experts run for each token.'}));
  S.push(st('experts','MoE','Run the chosen experts','E<sub>i</sub>(h) for the 10 routed experts + 1 shared expert',[act('h₃',T,D),'→ 10 routed + 1 shared →',act('11 × E<sub>i</sub>(h)',T,D,{shape:`11 × [${T} × ${D}]`})],{sn:SN.moe,note:'Expert hidden width and activation are '+NP+'.'}));
  S.push(st('combine','MoE','Weighted combine','MLP = Σ g<sub>i</sub>·E<sub>i</sub>(h) + E<sub>shared</sub>(h)',[act('11 × E<sub>i</sub>(h)',T,D,{shape:`11 × [${T} × ${D}]`}),'× g, sum',' =',act('MLP',T,D)],{sn:SN.moe,op:'combine'}));
  S.push(st('hcw3','MoE','Write back to the branches','R[b] = R[b] + c<sub>b</sub>·MLP',[R4(),'+ c ⊙',act('MLP',T,D),'=',R4()],{sn:SN.moe,op:'add',note:'End of one layer. 48 layers in total: 12 × (3 Gated DeltaNet layers + 1 sparse-attention layer), each followed by this MoE.'}));
  /* output */
  S.push(st('mixout','Output','Read the final hidden state','X = Norm(Mix(R))',[R4(),'→ gated read, norm →',res('X',T,D)],{scope:'out',sn:SN.out,note:'Norm type is '+NP+'.'}));
  const j0=S.length;outSteps(S,{d:D,V},T,{});S.slice(j0).forEach(s=>{s.sn=SN.out;});
  S.push(st('mtp','Output','Multi-token draft','draft = MTP(x<sub>T</sub>)',[act(sb('x','T'),1,D),'→ 1-layer draft model →',tok('draft ids',1,3,{shape:'[a few tokens]'})],{scope:'out',sn:SN.out,note:'A one-layer draft model (≈ 4B parameters) proposes several next tokens; the main model verifies them in one pass.'}));
  S.layerLabel='ONE LAYER · 48 in total = 12 × (3 Gated DeltaNet + 1 sparse attention), MoE in every layer';
  return S;
}

/* ---- LoRA: the GQA decoder with frozen weights and a trainable low-rank update on W_Q and W_V ---- */
function loraDecoder(c){
  const S=modern(c,{kv:c.Hkv,mask:'causal'}),{d,T}=c,r=c.r,kvd=c.Hkv*(d/c.H);
  const ins=(after,p,name,dout)=>{
    const i=S.findIndex(s=>s.key===after),N=name.toUpperCase();
    S.splice(i+1,0,
      st('l'+p+'a','Attention',`LoRA down (${N})`,`Z = X̂·${sb('A',name)}`,[act('X̂',T,d),'×',wgt(sb('A',name),d,r),'=',act(sb('Z',name),T,r)],{op:'matmul',flops:2*T*d*r,note:`Trainable. Rank r = ${r}: every token is squeezed to ${r} numbers.`}),
      st('l'+p+'b','Attention',`LoRA up (${N})`,`Δ${N} = Z·${sb('B',name)}`,[act(sb('Z',name),T,r),'×',wgt(sb('B',name),r,dout),'=',act('Δ'+N,T,dout)],{op:'matmul',flops:2*T*r*dout,note:'Trainable, initialised to zero so training starts from the unchanged model.'}),
      st('l'+p+'add','Attention',`Add the update (${N})`,`${N} = ${N} + (α/r)·Δ${N}`,[act(N,T,dout),'+',act('Δ'+N,T,dout),'=',act(N,T,dout)],{op:'add',note:`The frozen projection plus the low-rank correction. ${fN(r*(d+dout))} trainable numbers instead of ${fN(d*dout)}.`}));
  };
  ins('q','q','q',d);ins('v','v','v',kvd);
  S.forEach(s=>{if(!/^l[qv]/.test(s.key)&&s.params){s.frozen=true;}});
  return S;
}
K.r=16;
FIELDS.push(['r','LoRA rank']);
ARCH.push(
 {id:'qnext',name:'Qwen3.8-Flash-Next  (Gated DeltaNet + sparse attention + MoE + n-gram memory)',short:'Qwen3.8-Next',
  desc:'A hybrid decoder: three of every four layers mix tokens with a fixed-size recurrent state (Gated DeltaNet), the fourth with sparse attention over the top-512 blocks. The MLP is a 512-expert MoE and a 51B-parameter n-gram table adds memory at block 2. Dimensions shown are the published ones; only T and V follow the configuration bar.',
  meta:()=>({norm:'per-branch norm (type not published)',pos:'RoPE, in the sparse-attention layers',mask:'causal · top-512 blocks (≤ 2051 tokens)',mlp:'MoE: 512 experts, top-10 + 1 shared',kv:2,
    kvText:fB(2*12*2*256*2)+'  (only the 12 sparse-attention layers keep K,V: 2 heads × 256) + a fixed-size state per Gated DeltaNet layer',
    facts:[['Published size','125B main model + 51B n-gram table + ≈ 4B draft (MTP) model'],['Active per token','≈ 6B parameters'],['Layers','48 = 12 × (3 Gated DeltaNet + 1 sparse attention)'],['Hidden size','2560 (implied by the published tensor shapes)'],['Not published','Gated DeltaNet state size and exact update, expert width and activation, norm type, vocabulary size']]}),
  build:qwenNext},
 {id:'lora',name:'LoRA fine-tuning  (GQA decoder, low-rank update on W_Q and W_V)',short:'LoRA',
  desc:'The base model is frozen. Two small matrices A and B are trained next to W_Q and W_V; their product is a low-rank correction added to the projection. Set the rank with “LoRA rank”.',
  meta:c=>({norm:'RMSNorm, pre',pos:'RoPE on Q,K (every layer)',mask:'causal',mlp:'SwiGLU (3 matrices), frozen',kv:c.Hkv}),build:loraDecoder}
);
BACK.qnext='topic/qnext';BACK.lora='topic/lora';
