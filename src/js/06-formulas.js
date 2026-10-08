/* ================= formulas for every transformation (shown in popups, panels and step cards) =================
   FORMULA[key] = [[label, formula], ...]. Keys are the canonical keys produced by canonKey() in 05-explain.js,
   plus variants: normln (LayerNorm), gelu / relu (classic MLP activations). */
const FORMULA={
  tok:[['Tokenise','ids = tokenizer(text) ,  ids[t] ∈ {0, …, V − 1}']],
  mem:[['Encoder output','M = Encoder(x<sub>source</sub>) ∈ ℝ<sup>S×d</sup>']],
  emb:[['Lookup','X[t] = E[ids[t]]'],['Shapes','E ∈ ℝ<sup>V×d</sup> ,  X ∈ ℝ<sup>T×d</sup>']],
  pos:[['Learned','X[t] ← X[t] + P[t]'],['Sinusoidal','PE(t, 2i) = sin(t / 10000<sup>2i/d</sup>) ,  PE(t, 2i+1) = cos(t / 10000<sup>2i/d</sup>)']],
  norm:[['RMSNorm','y = x / RMS(x) · γ'],['RMS','RMS(x) = √( (1/d) · Σ<sub>j</sub> x<sub>j</sub>² + ε )'],['Per row','applied to each token vector x ∈ ℝ<sup>d</sup> ;  γ ∈ ℝ<sup>d</sup> learned']],
  normln:[['LayerNorm','y = (x − μ) / √(σ² + ε) · γ + β'],['Mean','μ = (1/d) · Σ<sub>j</sub> x<sub>j</sub>'],['Variance','σ² = (1/d) · Σ<sub>j</sub> (x<sub>j</sub> − μ)²']],
  q:[['Projection','Q = X̂ · W<sub>Q</sub>'],['Entry','Q[t, j] = Σ<sub>i</sub> X̂[t, i] · W<sub>Q</sub>[i, j]']],
  k:[['Projection','K = X̂ · W<sub>K</sub>'],['Width','W<sub>K</sub> ∈ ℝ<sup>d × (H<sub>kv</sub>·d<sub>h</sub>)</sup>']],
  v:[['Projection','V = X̂ · W<sub>V</sub>'],['Width','W<sub>V</sub> ∈ ℝ<sup>d × (H<sub>kv</sub>·d<sub>h</sub>)</sup>']],
  split:[['Reshape','Q[t, h, j] = Q[t, h·d<sub>h</sub> + j]'],['Head size','d<sub>h</sub> = d / H']],
  rope:[['Rotation','[q′<sub>2i</sub> ; q′<sub>2i+1</sub>] = [cos tθ<sub>i</sub>  −sin tθ<sub>i</sub> ; sin tθ<sub>i</sub>  cos tθ<sub>i</sub>] · [q<sub>2i</sub> ; q<sub>2i+1</sub>]'],['Frequencies','θ<sub>i</sub> = 10000<sup>−2i/d<sub>h</sub></sup>'],['Result','⟨q′<sub>m</sub>, k′<sub>n</sub>⟩ depends only on m − n']],
  cache:[['Append','K<sub>t</sub> = [K<sub>t−1</sub> ; k<sub>t</sub>] ,  V<sub>t</sub> = [V<sub>t−1</sub> ; v<sub>t</sub>]'],['Size','2 · N<sub>layers</sub> · H<sub>kv</sub> · d<sub>h</sub> numbers per token']],
  scores:[['Dot product','S[i, j] = ⟨q<sub>i</sub>, k<sub>j</sub>⟩ = Σ<sub>m</sub> Q<sub>h</sub>[i, m] · K<sub>g</sub>[j, m]'],['Matrix form','S = Q<sub>h</sub> · K<sub>g</sub><sup>ᵀ</sup>'],['Shared K/V head','g = ⌊h / (H / H<sub>kv</sub>)⌋']],
  scale:[['Scale','S′ = S / √d<sub>h</sub>'],['Reason','Var(⟨q, k⟩) ≈ d<sub>h</sub> for unit-variance entries']],
  mask:[['Causal mask','M[i, j] = 0 if j ≤ i , −∞ otherwise'],['Apply','S″ = S′ + M']],
  maskwin:[['Window mask','M[i, j] = 0 if i − W &lt; j ≤ i , −∞ otherwise'],['Apply','S″ = S′ + M'],['Cost','O(T · W) instead of O(T²)']],
  softmax:[['Softmax','A[i, j] = exp(S″[i, j]) / Σ<sub>k</sub> exp(S″[i, k])'],['Properties','A[i, j] ≥ 0 ,  Σ<sub>j</sub> A[i, j] = 1 ,  exp(−∞) = 0']],
  av:[['Weighted sum','O<sub>h</sub>[i] = Σ<sub>j</sub> A[i, j] · V<sub>g</sub>[j]'],['Matrix form','O<sub>h</sub> = A · V<sub>g</sub>'],['Whole head','O<sub>h</sub> = softmax(Q<sub>h</sub>K<sub>g</sub><sup>ᵀ</sup> / √d<sub>h</sub> + M) · V<sub>g</sub>']],
  concat:[['Concatenate','O[t] = [O<sub>0</sub>[t] | O<sub>1</sub>[t] | … | O<sub>H−1</sub>[t]] ∈ ℝ<sup>H·d<sub>h</sub></sup>']],
  wo:[['Projection','Attn = O · W<sub>O</sub>'],['Full sub-layer','Attn(X) = Concat<sub>h</sub>(softmax(Q<sub>h</sub>K<sub>g</sub><sup>ᵀ</sup>/√d<sub>h</sub> + M) V<sub>g</sub>) · W<sub>O</sub>']],
  add:[['Residual','X<sub>out</sub> = X + F(Norm(X))'],['Gradient','∂X<sub>out</sub>/∂X = I + ∂F/∂X  (identity path always present)']],
  gate:[['Gate branch','G = X̂ · W<sub>gate</sub> ,  W<sub>gate</sub> ∈ ℝ<sup>d × d<sub>ff</sub></sup>']],
  up:[['Up branch','U = X̂ · W<sub>up</sub> ,  W<sub>up</sub> ∈ ℝ<sup>d × d<sub>ff</sub></sup>']],
  silu:[['SiLU (swish)','SiLU(x) = x · σ(x)'],['Sigmoid','σ(x) = 1 / (1 + e<sup>−x</sup>)'],['Derivative','SiLU′(x) = σ(x) · (1 + x · (1 − σ(x)))'],['Limits','x → +∞: SiLU(x) → x ;  x → −∞: SiLU(x) → 0 ;  minimum ≈ −0.278 at x ≈ −1.278']],
  mul:[['Gating','Hid = SiLU(G) ⊙ U ,  Hid[t, j] = SiLU(G[t, j]) · U[t, j]']],
  down:[['Down projection','MLP = Hid · W<sub>down</sub> ,  W<sub>down</sub> ∈ ℝ<sup>d<sub>ff</sub> × d</sup>'],['Whole SwiGLU MLP','MLP(x) = (SiLU(x W<sub>gate</sub>) ⊙ x W<sub>up</sub>) · W<sub>down</sub>']],
  fc1:[['Expand','Hid = X̂ · W<sub>1</sub> + b<sub>1</sub> ,  W<sub>1</sub> ∈ ℝ<sup>d × 4d</sup>']],
  gelu:[['GELU','GELU(x) = x · Φ(x) = (x / 2) · (1 + erf(x / √2))'],['Common approximation','GELU(x) ≈ 0.5 x · (1 + tanh(√(2/π) · (x + 0.044715 x³)))'],['Limits','x → +∞: → x ;  x → −∞: → 0 ;  minimum ≈ −0.170 at x ≈ −0.75']],
  relu:[['ReLU','ReLU(x) = max(0, x)'],['Derivative','1 if x &gt; 0 , 0 if x &lt; 0']],
  fc2:[['Project back','MLP = act(Hid) · W<sub>2</sub> + b<sub>2</sub>'],['Whole MLP','MLP(x) = act(x W<sub>1</sub> + b<sub>1</sub>) · W<sub>2</sub> + b<sub>2</sub>']],
  router:[['Router scores','r = x · W<sub>r</sub> ,  W<sub>r</sub> ∈ ℝ<sup>d × E</sup>']],
  topk:[['Selection','𝒯 = indices of the k largest r<sub>e</sub>'],['Gate weights','g<sub>e</sub> = exp(r<sub>e</sub>) / Σ<sub>j ∈ 𝒯</sub> exp(r<sub>j</sub>)  for e ∈ 𝒯 ,  0 otherwise']],
  experts:[['Each expert','E<sub>e</sub>(x) = MLP<sub>e</sub>(x)  with its own weights'],['Work done','only experts in 𝒯 (plus any shared expert) are evaluated']],
  combine:[['Mixture','MLP(x) = Σ<sub>e ∈ 𝒯</sub> g<sub>e</sub> · E<sub>e</sub>(x)  (+ E<sub>shared</sub>(x))']],
  last:[['Select','x<sub>T</sub> = X̂[T − 1, :]']],
  cls:[['Select','x<sub>cls</sub> = X[0, :]']],
  head:[['Logits','z = x<sub>T</sub> · W<sub>LM</sub> ,  z<sub>v</sub> = ⟨x<sub>T</sub>, W<sub>LM</sub>[:, v]⟩'],['Tied weights','W<sub>LM</sub> = E<sup>ᵀ</sup>  (when input and output embeddings are shared)']],
  vsoft:[['Softmax with temperature','p<sub>v</sub> = exp(z<sub>v</sub> / τ) / Σ<sub>u</sub> exp(z<sub>u</sub> / τ)'],['Training loss','ℒ = −log p<sub>target</sub>  (cross-entropy)']],
  sample:[['Top-k','keep the k largest p<sub>v</sub> , renormalise'],['Top-p (nucleus)','keep the smallest set with Σ p<sub>v</sub> ≥ p , renormalise'],['Repetition penalty','z<sub>v</sub> ← z<sub>v</sub> / ρ if z<sub>v</sub> &gt; 0 else z<sub>v</sub> · ρ  for already-used tokens'],['Draw','next ~ Categorical(p)']],
  /* ---- Qwen3.8-Flash-Next ---- */
  hc0:[['Residual state','R<sub>t</sub> ∈ ℝ<sup>4×2560</sup>  (four parallel branches per token)']],
  nghash:[['Bigram rows','row<sub>j</sub> = hash<sub>j</sub>(x<sub>t−1</sub>, x<sub>t</sub>) ,  j = 1…8'],['Trigram rows','row<sub>8+j</sub> = hash<sub>8+j</sub>(x<sub>t−2</sub>, x<sub>t−1</sub>, x<sub>t</sub>) ,  j = 1…8'],['Note','the hash functions themselves are not published']],
  nglook:[['Lookup','E<sub>t</sub> = [N<sub>1</sub>[row<sub>1</sub>] | … | N<sub>16</sub>[row<sub>16</sub>]] ∈ ℝ<sup>16×160 = 2560</sup>'],['Table','16 tables × 20M rows × 160 ≈ 51.2B parameters']],
  nggate:[['Gate','g<sub>t</sub> = Gate(Norm(Q<sub>t</sub>), Norm(K<sub>t</sub>)) ∈ ℝ<sup>4×1</sup>'],['Gated value','U<sub>t</sub> = g<sub>t</sub> ⊙ V<sub>t</sub>'],['Refine','Δ<sub>t</sub> = U<sub>t</sub> + SiLU(DWConv(RMSNorm(U<sub>t</sub>)))'],['SiLU','SiLU(x) = x / (1 + e<sup>−x</sup>)']],
  nginj:[['Inject','R̃<sub>t</sub> = R<sub>t</sub> + Δ<sub>t</sub>']],
  mix:[['Read','h<sub>t</sub> = Mix(R<sub>t</sub>) ∈ ℝ<sup>2560</sup>'],['Note','each branch is normalised and read through a data-dependent element-wise gate; the exact gate is not published']],
  gproj:[['Projections','q<sub>t</sub>, k<sub>t</sub>, v<sub>t</sub> = linear(h<sub>t</sub>) ;  α<sub>t</sub>, β<sub>t</sub> ∈ (0, 1) from gates on h<sub>t</sub>']],
  gstate:[['Gated delta rule','S<sub>t</sub> = α<sub>t</sub> · S<sub>t−1</sub> · (I − β<sub>t</sub> k<sub>t</sub> k<sub>t</sub><sup>ᵀ</sup>) + β<sub>t</sub> v<sub>t</sub> k<sub>t</sub><sup>ᵀ</sup>'],['Reading it','α<sub>t</sub>: forget ·  (I − β k kᵀ): erase the old value stored under key k ·  β v kᵀ: write the new one'],['Source','form from the Gated DeltaNet paper; Qwen’s exact variant and state size are not published']],
  gread:[['Read-out','o<sub>t</sub> = S<sub>t</sub> · q<sub>t</sub>'],['Cost','O(1) per token — the state has a fixed size, there is no per-token cache']],
  hcw:[['Write back','R<sub>t</sub>[b] ← R<sub>t</sub>[b] + c<sub>t,b</sub> · out<sub>t</sub> ,  b = 1…4'],['Coefficients','c<sub>t</sub> ∈ ℝ<sup>4</sup> from a learned projection of the normalised branches']],
  idx:[['Index vectors','q<sup>I</sup><sub>t,h</sub> ∈ ℝ<sup>128</sup> for h = 1…4 ;  k<sup>I</sup><sub>t</sub> ∈ ℝ<sup>128</sup>']],
  pool:[['Block key','k̄<sup>I</sup><sub>b</sub> = RoPE( normalise( ¼ · Σ<sub>t ∈ block b</sub> k<sup>I</sup><sub>t</sub> ) )']],
  bscore:[['Block score','s<sub>t,b</sub> = (1/√128) · Σ<sub>h=1</sub><sup>4</sup> ReLU(⟨q<sup>I</sup><sub>t,h</sub>, k̄<sup>I</sup><sub>b</sub>⟩)'],['ReLU','ReLU(x) = max(0, x)']],
  bsel:[['Selection','𝓑<sub>t</sub> = top-512 blocks by s<sub>t,b</sub>  → 2048 tokens (+ up to 3 from the unfinished block)'],['Mask','M[t, j] = 0 if j ∈ tokens(𝓑<sub>t</sub>) , −∞ otherwise']],
  /* ---- multi-head latent attention ---- */
  cq:[['Query bottleneck','c<sub>Q</sub> = X̂ · W<sub>DQ</sub> ,  W<sub>DQ</sub> ∈ ℝ<sup>d × d<sub>c</sub>′</sup>']],
  qr:[['Position part','q<sup>R</sup> = RoPE(c<sub>Q</sub> · W<sub>QR</sub>) ,  W<sub>QR</sub> ∈ ℝ<sup>d<sub>c</sub>′ × H·d<sub>R</sub></sup>']],
  ckv:[['Latent','c<sub>KV</sub> = X̂ · W<sub>DKV</sub> ,  W<sub>DKV</sub> ∈ ℝ<sup>d × d<sub>c</sub></sup> ,  d<sub>c</sub> ≪ H·d<sub>h</sub>'],['Cached per token and layer','d<sub>c</sub> + d<sub>R</sub>   instead of   2 · H · d<sub>h</sub>']],
  kc:[['Keys from the latent','k<sup>C</sup> = c<sub>KV</sub> · W<sub>UK</sub> ,  W<sub>UK</sub> ∈ ℝ<sup>d<sub>c</sub> × H·d<sub>h</sub></sup>'],['Folding at inference','q<sup>C</sup> · (c<sub>KV</sub> W<sub>UK</sub>)<sup>ᵀ</sup> = (q<sup>C</sup> W<sub>UK</sub><sup>ᵀ</sup>) · c<sub>KV</sub><sup>ᵀ</sup>']],
  vc:[['Values from the latent','v = c<sub>KV</sub> · W<sub>UV</sub> ,  W<sub>UV</sub> ∈ ℝ<sup>d<sub>c</sub> × H·d<sub>h</sub></sup>'],['Folding at inference','(A · c<sub>KV</sub> W<sub>UV</sub>) · W<sub>O</sub> = A · c<sub>KV</sub> · (W<sub>UV</sub> W<sub>O</sub>)']],
  kr:[['Position part','k<sup>R</sup> = RoPE(X̂ · W<sub>KR</sub>) ,  W<sub>KR</sub> ∈ ℝ<sup>d × d<sub>R</sub></sup>  (shared by all heads)']],
  catqk:[['Joined vectors','q̃<sub>h</sub> = [q<sup>C</sup><sub>h</sub> ; q<sup>R</sup><sub>h</sub>] ,  k̃<sub>h</sub> = [k<sup>C</sup><sub>h</sub> ; k<sup>R</sup>]'],['Score','⟨q̃<sub>h</sub>, k̃<sub>h</sub>⟩ = ⟨q<sup>C</sup><sub>h</sub>, k<sup>C</sup><sub>h</sub>⟩ + ⟨q<sup>R</sup><sub>h</sub>, k<sup>R</sup>⟩'],['Attention','o<sub>h</sub> = softmax( q̃<sub>h</sub> k̃<sub>h</sub><sup>ᵀ</sup> / √(d<sub>h</sub> + d<sub>R</sub>) + M ) · v<sub>h</sub>']],
  /* ---- LoRA ---- */
  loraA:[['Down projection','Z = X̂ · A ,  A ∈ ℝ<sup>d × r</sup> ,  r ≪ d'],['Initialisation','A ~ 𝒩(0, σ²)']],
  loraB:[['Up projection','Δ = Z · B ,  B ∈ ℝ<sup>r × d<sub>out</sub></sup>'],['Initialisation','B = 0  → the update starts at zero']],
  loraAdd:[['LoRA','h = X̂ · W<sub>0</sub> + (α / r) · X̂ · A · B'],['Merged','W′ = W<sub>0</sub> + (α / r) · A · B  (no extra cost at inference)'],['Trainable','r · (d + d<sub>out</sub>) instead of d · d<sub>out</sub>']],
  mtp:[['Draft','x̂<sub>t+1…t+n</sub> = MTP(h<sub>t</sub>)'],['Verify','the main model checks the drafted tokens; accepted ones are kept']]
};
function formulasFor(key,o={}){
  let k=canonKey(key,o);
  if(k==='norm'||k==='normfinal')k=o.ln?'normln':'norm';
  else if(k==='normpost')k='normln';
  else if(k==='actf')k=o.fn==='relu'?'relu':'gelu';
  return FORMULA[k]||null;
}
const stepFormulas=s=>s.fm||formulasFor(s.key,{window:/window/i.test(s.title),cls:/CLS/.test(s.title),ln:s.ntype==='ln',fn:s.fn});
const formulaHTML=f=>!f?'':`<div class="fm"><h4>Formulas</h4>${f.map(([l,x])=>`<div class="fr"><span>${l}</span><code>${x}</code></div>`).join('')}${f.plot?`<div class="pl">${f.plot()}</div>`:''}</div>`;
