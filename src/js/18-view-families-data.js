/* ================= 6. Architecture families — data + stack renderer ================= */
const F={key:'dec',sel:null,hov:null,ffn:'swiglu',local:false};
const it=(id,label,shape,col,d,o={})=>Object.assign({id,label,shape,col,d},o);
const grp=(label,items)=>({grp:label,items});
const FAMS=[['enc','Encoder-only'],['dec','Decoder-only'],['encdec','Encoder–Decoder'],['dense','Dense Decoder'],['modern','Modern Decoder']];
const FMETA={
  enc:{name:'Encoder-only (BERT-style)',sub:'Every token may attend to tokens on both sides (subject to padding masks).',masks:[['Bidirectional self-attention','full']],
    models:['BERT','RoBERTa','DeBERTa','ELECTRA','ModernBERT'],tasks:['Classification','NER / tagging','Sentence embeddings','Retrieval & reranking'],train:'Masked-language modelling — hide ~15% of tokens and predict them from both sides. It outputs a contextual vector per token, so it cannot generate text on its own.'},
  dec:{name:'Decoder-only (GPT / Llama-style)',sub:'Causal self-attention: token t can only see tokens ≤ t, so the model can be trained to predict the next token and generate autoregressively.',masks:[['Causal self-attention','causal']],
    models:['GPT-2/3/4','Llama 1–3','Qwen','Mistral','Gemma','DeepSeek'],tasks:['Chat & instruction following','Code generation','Reasoning','Few-shot prompting'],train:'Next-token prediction on huge text corpora, then instruction tuning and preference optimisation. One stack, one objective — simple to scale.'},
  encdec:{name:'Encoder–Decoder (T5 / translation-style)',sub:'The encoder reads the whole source bidirectionally; the decoder generates the target causally and cross-attends to the encoder states in every layer.',masks:[['Encoder self-attention','full'],['Decoder self-attention','causal'],['Cross-attention (T × S)','cross']],
    models:['Original Transformer','T5 / mT5 / Flan-T5','BART','Whisper (audio encoder)','NLLB'],tasks:['Translation','Summarisation','Speech recognition','Structured seq2seq'],train:'Sequence-to-sequence denoising (T5 span corruption, BART) or supervised pairs. Cross-attention makes the source a first-class input.'},
  dense:{name:'Dense Decoder (classic GPT-style block)',sub:'“Dense” means every parameter — attention and MLP — is used for every token. This is the baseline that MoE sparsifies.',masks:[['Causal self-attention','causal']],
    models:['GPT-2','GPT-3','OPT','BLOOM','Llama-1'],tasks:['General language modelling'],train:'Next-token prediction. Compared with modern LLMs it uses LayerNorm, learned absolute positions, full multi-head attention and a GELU MLP.'},
  modern:{name:'Modern Decoder Block — RoPE + GQA + RMSNorm + SwiGLU',sub:'The recipe shared by most current open LLMs. Toggle the options to see where sliding-window attention and MoE slot in.',masks:[['Causal self-attention','causal']],
    models:['Llama 2/3','Mistral','Qwen2.5','Gemma','DeepSeek','Mixtral (MoE)'],tasks:['Chat & instruction following','Long-context tasks','Code & reasoning'],train:'Same objective as GPT, but with cheaper norms, relative positions, a smaller KV cache and a gated MLP — better quality per FLOP and per byte.'}
};
function famData(key){
  const hkv=(S.Hkv>1&&S.Hkv<S.H)?S.Hkv:gqaDefault(),gqa=`Q:${S.H}  KV:${hkv}`;
  const cols={
    enc:[{cx:210,w:380,title:'Encoder-only',items:[
      it('tok','Token IDs','[T]',C.yellow,'Integer IDs from the tokenizer (plus [CLS]/[SEP] special tokens in BERT).'),
      it('emb','Embedding + position','[T, D]',C.purple,'Token embedding + learned position embedding (+ segment embedding in BERT). Position is added once at the input.'),
      grp('× N',[
        it('mha','Bidirectional MHA','[T, D]',C.red,'Multi-head self-attention with <b>no causal mask</b>: every token attends to every other token, left and right, except padding.',{skipS:1}),
        it('an1','Add + Norm','[T, D]',C.green,'Residual add followed by LayerNorm (post-LN in the original BERT).',{skipE:1,h:38}),
        it('ffn','FFN','[T,D]→[T,M]→[T,D]',C.pink,'Position-wise two-layer MLP with GELU, M ≈ 4D, applied to each token independently.',{skipS:1}),
        it('an2','Add + Norm','[T, D]',C.green,'Second residual + LayerNorm.',{skipE:1,h:38})]),
      it('head','Task head','classification / token',C.yellow,'A small head on top of the contextual vectors: a classifier on [CLS], a per-token tagger, or a pooled embedding for retrieval.')]}],
    dec:[{cx:210,w:380,title:'Decoder-only',items:[
      it('tok','Token IDs','[T]',C.yellow,'Integer IDs from the tokenizer.'),
      it('emb','Embedding','[T, D]',C.purple,'Lookup in E [V, D]. Position is usually injected inside attention (RoPE) in modern models.'),
      grp('× N',[
        it('n1','Norm','[T, D]',C.blue,'Pre-norm (LayerNorm or RMSNorm) in front of attention.',{skipS:1,h:38}),
        it('mha','Causal MHA / GQA','[T, D]',C.red,'Self-attention with a <b>causal mask</b>: token t attends only to tokens ≤ t. Enables autoregressive generation and the KV cache.'),
        it('a1','Residual add','[T, D]',C.green,'x + attention(norm(x)).',{skipE:1,h:34}),
        it('n2','Norm','[T, D]',C.blue,'Pre-norm in front of the MLP.',{skipS:1,h:38}),
        it('ffn','FFN / SwiGLU','[T,D]→[T,M]→[T,D]',C.pink,'Position-wise MLP; in modern models a gated SwiGLU with M ≈ 8/3·D … 3.5·D.'),
        it('a2','Residual add','[T, D]',C.green,'h + mlp(norm(h)).',{skipE:1,h:34})]),
      it('nf','Final norm','[T, D]',C.blue,'One last norm before the vocabulary projection.',{h:38}),
      it('head','LM head','[D] → [V]',C.yellow,'Linear map to vocabulary logits — often tied to the embedding table.'),
      it('out','Next-token distribution','[V]',C.yellow,'Softmax over the last position’s logits; sample or take argmax.',{h:38})]}],
    encdec:[
      {cx:200,w:340,title:'Encoder',items:[
        it('src','Source tokens','[S]',C.yellow,'The input sequence (e.g. a sentence to translate).'),
        it('eemb','Encoder embed','[S, D]',C.purple,'Embeddings for the source tokens (T5 uses relative position biases instead of absolute vectors).'),
        grp('× N',[
          it('esa','Bidirectional self-attn','[S, D]',C.red,'Every source token sees every other source token.',{skipS:1}),
          it('ean1','Add + Norm','[S, D]',C.green,'Residual + norm.',{skipE:1,h:36}),
          it('effn','FFN','[S, D]',C.pink,'Position-wise MLP.',{skipS:1}),
          it('ean2','Add + Norm','[S, D]',C.green,'Residual + norm.',{skipE:1,h:36})]),
        it('encstates','Encoder states','[S, D]',C.orange,'The final encoder output. It is computed once and then read (as keys and values) by cross-attention in <b>every</b> decoder layer.')]},
      {cx:640,w:340,title:'Decoder',items:[
        it('tgt','Target tokens (shifted)','[T]',C.yellow,'The output generated so far, shifted right by one position (teacher forcing during training).'),
        it('demb','Decoder embed','[T, D]',C.purple,'Embeddings for the target tokens.'),
        grp('× N',[
          it('dsa','Masked self-attn','[T, D]',C.red,'Causal self-attention over the target generated so far.',{skipS:1}),
          it('dan1','Add + Norm','[T, D]',C.green,'Residual + norm.',{skipE:1,h:36}),
          it('cross','Cross-attention','Q=dec  K,V=enc',C.orange,'Queries come from the decoder; keys and values come from the encoder states. This is how the output “looks at” the source. Scores are [H, T, S].',{skipS:1}),
          it('dan2','Add + Norm','[T, D]',C.green,'Residual + norm.',{skipE:1,h:36}),
          it('dffn','FFN','[T, D]',C.pink,'Position-wise MLP.',{skipS:1}),
          it('dan3','Add + Norm','[T, D]',C.green,'Residual + norm.',{skipE:1,h:36})]),
        it('head','LM head','[D] → [V]',C.yellow,'Projects decoder states to vocabulary logits.')]}],
    dense:[{cx:210,w:380,title:'Dense decoder (GPT-2 style)',items:[
      it('tok','Token IDs','[T]',C.yellow,'Integer IDs from a BPE tokenizer.'),
      it('emb','Token + learned position','[T, D]',C.purple,'Token embedding plus a learned absolute position vector P[t]; the context window is fixed by the size of P.'),
      grp('× N',[
        it('ln1','LayerNorm','[T, D]',C.blue,'Pre-LN: subtract mean, divide by std, learned γ and β.',{skipS:1,h:38}),
        it('mha','Causal MHA','[T, D]',C.red,`${S.H} heads, each with its own Q/K/V. All parameters are used for every token.`),
        it('a1','Residual add','[T, D]',C.green,'x + attention(norm(x)).',{skipE:1,h:34}),
        it('ln2','LayerNorm','[T, D]',C.blue,'Pre-LN in front of the MLP.',{skipS:1,h:38}),
        it('ffn','FFN · GELU','[D→4D→D]',C.pink,'Two linear layers with a GELU in between, hidden width 4·D. Every token uses every weight — the definition of a <b>dense</b> layer.'),
        it('a2','Residual add','[T, D]',C.green,'h + mlp(norm(h)).',{skipE:1,h:34})]),
      it('lnf','Final LayerNorm','[T, D]',C.blue,'Normalise before the output projection.',{h:38}),
      it('head','LM head','[D] → [V]',C.yellow,'Often weight-tied with the embedding table.'),
      it('out','Next-token distribution','[V]',C.yellow,'Softmax over logits.',{h:38})]}],
    modern:[{cx:210,w:400,title:'Modern decoder block',items:[
      it('tok','Token IDs','[T]',C.yellow,'Integer IDs from a BPE / SentencePiece tokenizer (vocabularies of 32k–256k).'),
      it('emb','Token embedding','[T, D]',C.purple,'Embedding lookup only — no position vectors are added here because RoPE acts inside attention.'),
      grp('× N',[
        it('n1','RMSNorm','[T, D]',C.blue,'x / RMS(x) · γ — no mean subtraction, no bias. Cheaper than LayerNorm.',{skipS:1,h:38}),
        it('qkv','Q + grouped K/V',gqa,C.purple,`Projections Wq [D, H·dh] and Wk, Wv [D, Hkv·dh]. ${S.H} query heads share ${hkv} K/V heads, so the KV cache is ${(S.H/hkv).toFixed(0)}× smaller than MHA. See the GQA diagram.`),
        it('rope','RoPE','Q, K',C.orange,'Rotary position embedding: rotates pairs of dimensions of Q and K by a position-dependent angle so attention scores depend on relative distance. No parameters.',{h:38}),
        it('att',F.local?'Sliding-window GQA':'Causal GQA','[T, D]',C.red,F.local?'Grouped-query attention restricted to the last W tokens: O(T·W) compute and a rolling KV cache (Mistral, Gemma-2 local layers).':'Grouped-query attention with a causal mask over all previous tokens.'),
        it('a1','Residual add','[T, D]',C.green,'x + attention(norm(x)).',{skipE:1,h:34}),
        it('n2','RMSNorm','[T, D]',C.blue,'Pre-norm in front of the MLP.',{skipS:1,h:38}),
        it('ffn',F.ffn==='moe'?'MoE · top-k SwiGLU experts':'SwiGLU','D → M → D',C.pink,F.ffn==='moe'?'A router picks top-k of E SwiGLU experts per token (Mixtral, DeepSeek-MoE, Qwen-MoE). Total parameters grow with E, per-token compute only with k.':'SiLU(x·W_gate) ⊙ (x·W_up) → W_down. Three matrices, M ≈ 8/3·D … 3.5·D. No biases.'),
        it('a2','Residual add → next layer','[T, D]',C.green,'h + mlp(norm(h)); becomes the input of the next block.',{skipE:1,h:34})]),
      it('nf','Final RMSNorm','[T, D]',C.blue,'Last norm before the LM head.',{h:38}),
      it('head','LM head','[D] → [V]',C.yellow,'Projects to the vocabulary; tied with the embedding in small models.')]}]
  };
  return {cols:cols[key],cross:key==='encdec'?{from:'encstates',to:'cross'}:null};
}
function findItem(cols,id){
  for(const c of cols)for(const x of c.items){
    if(x.grp){const f=x.items.find(b=>b.id===id);if(f)return f;}else if(x.id===id)return x;
  }
  return null;
}
function stackSVG(cols,focus,cross){
  const GAP=26;let s='',under='',maxY=0;const pos={};
  cols.forEach(col=>{
    let y=48,skipStart=null;const bw=col.w,x=col.cx-bw/2;
    s+=`<text class="cap" x="${col.cx}" y="26" text-anchor="middle">${col.title}</text>`;
    const conn=()=>{s+=`<path class="conn" d="M${col.cx} ${y}V${y+GAP-4}" marker-end="url(#ar-8fa6bd)"/>`;y+=GAP;};
    const block=(b,ix,iw)=>{
      const h=b.h||44;pos[b.id]={x:ix,y,w:iw,h};
      if(b.skipS)skipStart=y-GAP/2;
      s+=`<g class="node${focus===b.id?' foc':''}" data-id="${b.id}" tabindex="0" role="button" aria-label="${esc(b.label)}" style="color:${b.col}"><rect class="hit" x="${ix-4}" y="${y-4}" width="${iw+8}" height="${h+8}" rx="12"/><rect class="blk" x="${ix}" y="${y}" width="${iw}" height="${h}" rx="10"/><text class="blab" x="${ix+14}" y="${y+h/2+5}">${b.label}</text>${b.shape?`<text class="bshp" x="${ix+iw-12}" y="${y+h/2+4.5}" text-anchor="end">${b.shape}</text>`:''}</g>`;
      if(b.skipE&&skipStart!=null){const xr=ix+iw;under+=`<path class="skip" d="M${xr+3} ${skipStart}H${xr+22}V${y+h/2}H${xr+4}" marker-end="url(#ar-72d945)"/>`;skipStart=null;}
      y+=h;
    };
    col.items.forEach((item,idx,arr)=>{
      if(item.grp){
        const pad=34,top=y;y+=30;
        item.items.forEach((b,j)=>{block(b,x+pad,bw-2*pad);if(j<item.items.length-1)conn();});
        y+=14;
        under+=`<rect class="grp" x="${x}" y="${top}" width="${bw}" height="${y-top}" rx="14"/><text class="grpl" x="${x+14}" y="${top+20}">${item.grp}</text>`;
      }else block(item,x,bw);
      if(idx<arr.length-1)conn();
    });
    maxY=Math.max(maxY,y);
  });
  if(cross){
    const a=pos[cross.from],b=pos[cross.to];
    if(a&&b){const x1=a.x+a.w,y1=a.y+a.h/2,x2=b.x,y2=b.y+b.h/2,m=(x1+x2)/2;
      under+=`<path class="edge" d="M${x1} ${y1}C${m} ${y1} ${m} ${y2} ${x2-3} ${y2}" stroke="${C.orange}" fill="none" stroke-width="2.4" marker-end="url(#ar-${C.orange.slice(1)})"/><text class="elab" x="${m}" y="${(y1+y2)/2-8}" text-anchor="middle" style="fill:${C.orange};font-weight:700">K, V from encoder</text>`;}
  }
  const W=Math.max(...cols.map(c=>c.cx+c.w/2))+50;
  return svgWrap(W,maxY+30,under+s,`style="max-width:${Math.round(W*1.2)}px;margin:0 auto"`);
}
function maskSVG(kind,label){
  const n=9,cs=13,w=kind==='cross'?6:n,col=kind==='causal'?C.green:kind==='cross'?C.orange:C.blue;let s='';
  for(let i=0;i<n;i++)for(let j=0;j<w;j++){const on=kind==='causal'?j<=i:true;s+=`<rect x="${j*cs+1}" y="${i*cs+1}" width="${cs-1.5}" height="${cs-1.5}" fill="${on?col:'var(--off)'}" opacity="${on?.9:1}"/>`;}
  return `<figure style="margin:0;text-align:center"><svg viewBox="0 0 ${w*cs+2} ${n*cs+2}" width="${w*cs*1.35}" aria-hidden="true">${s}</svg><figcaption class="mut" style="font-size:11.5px;max-width:130px">${label}</figcaption></figure>`;
}
