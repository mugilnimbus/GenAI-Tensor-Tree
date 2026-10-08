/* ================= why each transformation exists, and what it does to the numbers =================
   WHY[key] = [why it is needed, what changes inside the tensor].
   explainKey(key, opts) resolves variants (cross-attention, pre/post norm, window mask, activation type). */
const WHY={
  tok:['A network can only do arithmetic, so text has to become numbers first. Sub-word tokenisation keeps the vocabulary finite (tens of thousands of pieces) while still being able to spell any string, including words never seen in training.',
       'A string becomes T integers. These numbers carry no geometry yet: ID 500 is not “closer” to ID 501 than to ID 9000 — they are just row addresses for the next step.'],
  emb:['An integer ID says nothing about meaning. The embedding table gives every token a learned point in d-dimensional space, where tokens used in similar ways end up near each other. This [T × d] tensor is the residual stream that every later layer reads and edits.',
       'T integers become T rows of d real numbers, each copied verbatim from one row of E. Two occurrences of the same token get identical rows — there is no context and no position in the tensor yet.'],
  pos:['Attention treats its input as a set: without position information “dog bites man” and “man bites dog” would produce exactly the same result. Position has to be injected somewhere; here it is added once, at the input.',
       'Every row is shifted by a vector that depends only on its index t. The same token at two positions now has two different rows. Shape and scale stay the same.'],
  norm:['By the time a layer runs, the residual stream is the sum of many earlier updates, so the length of each token vector drifts and differs wildly between tokens. Dot products and softmax downstream are very sensitive to scale: too large and attention saturates, too small and nothing is distinguishable. Normalising first gives every sub-layer inputs of predictable size, which is what keeps gradients stable through dozens of layers. Doing it on a copy (pre-norm) leaves the skip path untouched.',
        'Each row is divided by its own RMS, so every token vector ends up with the same overall length; then each column is multiplied by a learned gain γ. The direction of the vector — the relative sizes of its features — is preserved. Tokens are still independent: no information moves between rows.'],
  normpost:['In the original (post-norm) design the normalisation comes after the residual add, to stop the summed stream from growing layer after layer. It works, but the skip path now passes through a norm in every layer, which makes very deep stacks harder to train — the reason modern models moved the norm in front of the sub-layer.',
        'Each row of the summed tensor is mean-centred and rescaled to unit variance, then scaled and shifted by learned γ and β. Unlike RMSNorm, the mean of every row is removed as well.'],
  normfinal:['The output of the last layer is still a raw, un-normalised sum of all updates. The LM head scores tokens with dot products, so it needs vectors of controlled length — otherwise the logits (and therefore the confidence of the model) would depend on how long the vector happened to be.',
        'Every row is rescaled to the same length and multiplied by a learned gain; direction is kept. Shape is unchanged.'],
  q:['One hidden vector has to play three different roles in attention. W_Q extracts the first: “what am I looking for?”. It is a separate matrix from W_K so that asking and offering can differ — a verb can look for its subject without the subject having to look for the verb.',
     'Every row is linearly re-mixed: each output number is a weighted sum of all d input features of the same token. Think of it as rotating and stretching the token vector into “query space”. Tokens are still not mixed with each other.'],
  k:['W_K extracts the second role: “what can I be found by?”. A query and a key that point the same way will produce a large score in the next steps, so this matrix decides which tokens are discoverable for which kinds of question.',
     'Each row is re-mixed into “key space”. With MHA the width stays d; with GQA/MQA the matrix has fewer columns, so every token is described by a shorter key vector (fewer K/V heads) — the source of the smaller KV cache.'],
  v:['W_V extracts the third role: “what do I hand over if someone attends to me?”. Keeping values separate from keys lets a token be found by one property (e.g. being a noun) and deliver a different one (e.g. its meaning).',
     'Each row is re-mixed into “value space”. These numbers are not used for matching at all; they are the payload that will be averaged and copied to other tokens.'],
  xqkv:['In cross-attention the queries still come from the decoder, but keys and values come from the encoder output. That is how each target position looks up the source positions it needs — the learned alignment between input and output.',
     'Q has one row per target token (T rows); K and V have one row per source token (S rows). The two sides now have different lengths.'],
  split:['A single softmax can only concentrate on one pattern at a time. Cutting the vector into H independent sub-spaces lets a token attend to several things at once — syntax in one head, the previous token in another, a matching bracket in a third — for the same total cost.',
     'No number changes. The d columns are only regrouped into H blocks of d_h. From here on, dot products are taken inside one block at a time, so each block behaves like its own small attention.'],
  rope:['In a RoPE model nothing was added at the input, so the tensor has no idea where each token sits. Rotating queries and keys by an angle proportional to their position makes the dot product q·k depend on the angle difference — that is, on how far apart the two tokens are. It needs no parameters and behaves the same at any absolute position. Values are not rotated: position should influence where to look, not what is carried.',
     'Inside each head, every pair of columns (2i, 2i+1) is rotated by t·θᵢ. The length of every vector is unchanged and row 0 is untouched. Fast-rotating pairs encode fine position, slow ones coarse position.'],
  cache:['With a causal mask, a past token never sees later tokens, so its key and value never change. Recomputing them for every new token would be pure waste; storing them turns generation from quadratic to linear work.',
     'K (and V) grow by exactly one row. All earlier rows are bit-for-bit the ones computed in previous steps.'],
  scores:['This is the first moment tokens interact. The dot product between token i’s query and token j’s key measures how well what i is looking for matches what j offers. Every pair is compared, which is why attention costs T × T.',
     'Two [T × d_h] tensors collapse into a [T × T] table of single numbers: the feature dimension disappears and a second token dimension appears. Values are unbounded; large positive means “relevant”, negative means “ignore”.'],
  xscores:['Each target position is compared with every source position, so the model can decide which part of the input matters for the token it is about to produce.',
     'The table is rectangular, [T × S]: one row per target token, one column per source token.'],
  scale:['A dot product is a sum of d_h products, so its typical size grows like √d_h. Left alone, scores would be large enough to push the softmax into a near one-hot regime where gradients vanish. Dividing by √d_h brings them back to roughly unit variance, whatever the head size.',
     'Every entry shrinks by the same factor (√d_h). The ranking of the scores is unchanged; only their spread is reduced, which makes the following softmax softer.'],
  mask:['During training all T positions are predicted in parallel. If token i could look at tokens after it, it would simply copy the answer. The mask enforces the same information flow that exists at generation time — and it is also what makes the KV cache valid, because past tokens never depend on future ones.',
     'Entries above the diagonal (j > i) become −∞; everything on or below it is unchanged. Row i now only has i + 1 usable entries.'],
  maskwin:['On top of hiding the future, tokens further back than W are hidden too. Each query then compares against at most W keys, so cost grows linearly with length instead of quadratically, and the KV cache can be a rolling buffer. Distant information still arrives indirectly, one window per layer.',
     'Entries outside the band i − W < j ≤ i become −∞. Only a diagonal stripe of the table remains usable.'],
  softmax:['The scores must become weights for an average: non-negative, summing to 1, and differentiable. Softmax does this and exaggerates differences — the exponential turns a gap in score into a ratio in weight — so a token can focus sharply when one match stands out, or spread out when several are similar.',
     'Each row turns from arbitrary real numbers into probabilities between 0 and 1 that add up to 1. A −∞ entry becomes exactly 0. A score that is higher by 2 gets about 7.4× the weight. Rows are independent of each other.'],
  av:['The scores only decided where to look. This is the step that actually moves information: every token collects a weighted average of the value vectors of the tokens it attends to. It is the only place in the whole network where one token’s vector receives content from another token.',
     'Row i becomes a blend of the V rows it attends to, so its numbers stay within the range spanned by those rows. The [T × T] table is consumed and the result is [T × d_h] again — but each row now contains context from other positions.'],
  concat:['The H heads worked in separate sub-spaces. Laying their outputs side by side gives the next matrix a single vector in which it can use everything all heads found.',
     'No arithmetic. H blocks of width d_h become one row of width d.'],
  wo:['Head outputs live in each head’s private coordinates, in a fixed block order. The residual stream has one shared coordinate system that every layer reads. W_O translates the head results into that system, lets results of different heads be combined, and learns how strongly each head writes.',
     'Each row is re-mixed across all head blocks. The result is typically small compared with X — it is an update to the stream, not a replacement.'],
  add:['The skip connection makes every sub-layer learn a correction instead of a whole new representation. The untouched path lets gradients reach early layers directly through dozens of blocks, and lets information such as token identity survive without every layer having to copy it forward.',
     'Element-wise sum: the original vector is kept and nudged by the update. Row lengths grow a little with every add, which is exactly why the next sub-layer starts with a normalisation.'],
  gate:['Attention can only move and average information between tokens — a linear operation. To compute genuinely new features, each token must be processed non-linearly on its own. The first move is to expand into a much wider space (d_ff ≈ 2.7 × d), where many more patterns can each have their own direction. The gate branch computes, for every one of those patterns, how strongly it is present in this token.',
     'Width grows from d to d_ff. Each of the d_ff columns is one learned detector: the dot product of the token with a pattern stored in W_gate. Values are signed and unbounded at this point.'],
  up:['A second, independent expansion supplies the content that the gate will let through. Separating “whether a feature fires” (gate) from “what value it carries” (up) gives a multiplicative interaction, which is more expressive than a single activation for the same number of parameters — the idea behind SwiGLU.',
     'Same shape as the gate branch, but different learned detectors, so the numbers are unrelated to G until the two are multiplied.'],
  silu:['Without a non-linearity, W_down·(W_up·x) would collapse into one single matrix — the wide hidden layer would compute nothing a plain projection could not. SiLU bends the gate values so each hidden unit acts as a soft switch: clearly negative means “off”, clearly positive means “on, pass the value through”. It comes right after W_gate because the gate must become a selective signal before it multiplies the other branch; a raw linear gate would select nothing. Being smooth (unlike ReLU) it never has an exactly-zero gradient, which makes training easier.',
     'Applied to every number on its own. Large positive x stays ≈ x; 0 stays 0; large negative x is squashed to ≈ 0 (slightly below, never less than −0.28). Roughly half of the d_ff entries are pushed toward zero, leaving a pattern of active features that is specific to this token. Shape is unchanged.'],
  mul:['This is the gating itself: features whose gate is closed are removed from the content branch, open ones pass through scaled. Multiplying two learned projections also creates products of input features (feature A and feature B present together), something a single linear layer cannot express.',
     'Entry [i, j] of one tensor times entry [i, j] of the other. Wherever SiLU(G) ≈ 0 the result is ≈ 0 no matter what U holds; elsewhere U is scaled, and its sign can flip.'],
  down:['The hidden state is far too wide to keep and lives in the MLP’s private feature space. W_down reads which features are active and writes their combined effect back into the d-dimensional stream: each hidden feature owns a direction saying “if I fire, add this to the token”. It works like a key → value memory, and is where much of the model’s stored knowledge lives.',
     'Width shrinks from d_ff back to d. Each output number is a weighted sum of all d_ff gated features of the same token.'],
  fc1:['Attention only averages information linearly; new features need per-token non-linear processing. The first matrix expands each token into a wider space (4 × d) where patterns can be separated, and scores how strongly each pattern is present.',
     'Width grows from d to 4·d. Each column is one learned detector; values are signed and unbounded. A bias shifts every detector’s threshold.'],
  actf:['Two linear layers in a row would collapse into one matrix. The activation in between is what makes the MLP able to compute something new: it switches each hidden feature off when its detector is negative and passes it when positive.',
     'Applied element by element. ReLU sets every negative number to exactly 0; GELU does the same smoothly, letting small negatives through slightly. About half of the hidden entries end up at or near zero.'],
  fc2:['The wide hidden features must be written back into the residual stream. The second matrix turns the set of active features into a d-dimensional update — each feature contributes its own learned direction.',
     'Width shrinks from 4·d back to d; each output number is a weighted sum of the active hidden features of the same token.'],
  router:['A dense MLP makes every token pay for all of its weights. A mixture of experts keeps many specialised MLPs and lets a tiny matrix decide, token by token, which specialists are relevant — so capacity can grow without the per-token cost growing.',
     '[T × d] becomes [T × E]: one relevance score per expert for every token.'],
  topk:['Running every expert would cost as much as one enormous dense layer. Keeping only the best k makes the computation sparse; a softmax over the survivors turns their scores into mixing weights.',
     'In each row all but k entries are dropped; the remaining ones are renormalised so they add up to 1.'],
  combine:['The k chosen experts each propose an update. Weighting them by the router’s confidence produces one result per token, so the rest of the network sees an ordinary MLP output.',
     'A weighted sum of k [T × d] tensors gives a single [T × d] tensor.'],
  mem:['The decoder needs the complete source sentence at every step of generation. The encoder reads it once, with full left and right context, and the result is kept as a fixed memory.',
     'A constant [S × d] tensor. The decoder reads it in every layer but never modifies it.'],
  last:['With a causal mask only the final position has seen the whole prefix, so its vector is the summary used to predict what comes next. (During training every row is used, each predicting its own next token.)',
     '[T × d] becomes [1 × d]: one row is kept, the rest are discarded.'],
  cls:['For classification one vector must stand for the whole sequence. Because attention is bidirectional, the first position can gather information from every token, and the model is trained to put its summary there.',
     '[T × d] becomes [1 × d]: only row 0 is kept.'],
  head:['The hidden vector lives in the model’s abstract feature space; to say something it must be turned back into words. The dot product with each vocabulary row measures how compatible the predicted “next meaning” is with that token.',
     '[1 × d] becomes [1 × V]: one unbounded score (logit) per vocabulary entry.'],
  vsoft:['Sampling and training both need a proper probability distribution. Dividing the logits by a temperature first controls how peaked it is: below 1 sharpens toward the top choice, above 1 flattens it.',
     'V arbitrary real numbers become V probabilities between 0 and 1 that add up to 1.'],
  /* ---- Qwen3.8-Flash-Next ---- */
  hc0:['A single residual stream forces every layer to read and write the same vector. Widening it to four parallel branches gives layers separate channels to keep information apart, and lets each layer choose what to read and where to write.',
       'One [T × 2560] tensor becomes four of them per token. How the branches are initialised from the embedding is not published.'],
  nghash:['A normal embedding knows only the current token. Very common local patterns — names, idioms, code tokens — can simply be memorised if the lookup key also includes the one or two tokens before it. Hashing turns those short sequences into row numbers without needing a table entry for every possible n-gram.',
       'Three token IDs become 16 integers per position: 8 hashes of the last two tokens and 8 of the last three. Different n-grams can collide on one row; using several hash heads makes a full collision unlikely.'],
  nglook:['This is memory that costs almost no computation: 51B parameters sit in a table, but each token touches only 16 short rows. Because the rows depend only on token IDs, they can be fetched from ordinary RAM ahead of time while the GPU is busy, so the table does not have to live on the accelerator.',
       '16 rows of 160 numbers are copied out and laid side by side into one 2560-wide vector per token — the same width as the hidden state. No multiplication is involved.'],
  nggate:['A memorised n-gram vector is not always useful — the same three tokens can mean different things in different contexts. A gate computed from the current hidden state decides how much of it to let in, and a small convolution over neighbouring positions smooths it.',
       'The looked-up vector is scaled per branch by a gate value, then a refined copy (normalise → depth-wise convolution → SiLU) is added to itself. The result has one 2560-vector for each of the four branches.'],
  nginj:['The memory is added to the residual stream once, at the second block, so every later layer can build on it without paying for the lookup again.',
       'Element-wise sum into the four residual branches. Shape is unchanged.'],
  mix:['With four residual branches, a sub-layer must first decide what to read. A learned, data-dependent gate blends the normalised branches into the single vector the sub-layer works on.',
       '[T × 4 × 2560] is reduced to [T × 2560]. Each branch is normalised separately and weighted element by element; the exact gate is not published.'],
  gproj:['Gated DeltaNet replaces attention’s growing key/value list with a small memory matrix. From each token it needs a key (where to write), a value (what to write), a query (what to read) and two gates: how much to forget and how strongly to write.',
       'Each token vector is projected into q, k, v and two gate values between 0 and 1. Sizes are not published.'],
  gstate:['Attention keeps every past key and value, so memory and compute grow with the length of the context. Here the past is compressed into a matrix of fixed size that is edited token by token: decay the old content, erase whatever was stored under the current key, then write the new value there. That makes cost per token constant — the reason 3 of every 4 layers use it.',
       'The state matrix keeps its shape forever. Each token multiplies it by a forget factor, subtracts the component along the current key, and adds the outer product of the new value and key.'],
  gread:['The token reads from the memory by multiplying the state with its query: whatever was stored under similar keys comes back. This plays the role of “softmax(QKᵀ)·V” in attention, at constant cost.',
       'A matrix–vector product turns the fixed-size state into one output vector per token, which is then projected to the hidden width.'],
  hcw:['The sub-layer’s result has to be written back into the four branches. Four learned, data-dependent coefficients decide how much goes into each branch — the multi-branch version of the residual add.',
       'The same [T × 2560] update is added to each branch, scaled by that branch’s coefficient. [T × 4 × 2560] in and out.'],
  idx:['Full attention over a million tokens is far too expensive. Before attending, a tiny separate set of 128-dimensional vectors is computed whose only job is to find which parts of the context are worth attending to.',
       'Each token gets four small query vectors and one small key vector. These are not the real attention Q and K.'],
  pool:['Scoring every single token would still be T × T. Averaging index keys in groups of four makes one key per block, cutting the search by 4× and the index cache with it. RoPE is applied after pooling so that vectors with different rotations are not averaged together.',
       'T key vectors become T/4 block keys of the same width.'],
  bscore:['Each query scores every block cheaply to decide where to look. ReLU keeps only positive matches from each of the four index heads, so one strong match is not cancelled by the others.',
       'A [T × T/4] table of non-negative scores: one per query and block.'],
  bsel:['Only the best 512 blocks (2048 tokens) are kept for each query, so the real attention that follows has a fixed cost no matter how long the context is. The softmax and the values still use the original, uncompressed keys and values of the selected tokens.',
       'Scores become a sparse mask: at most 2051 positions per row are allowed, the rest are −∞.'],
  experts:['Only the experts chosen by the router run. With 512 experts and 10 chosen, about 2% of the MLP weights are used per token — that is how a 125B-parameter model activates only about 6B.',
       'Each token is processed by its 10 routed experts and the shared expert, giving 11 candidate updates. Expert width and activation are not published.'],
  mtp:['Generating one token per forward pass of a 48-layer model is slow. A small one-layer draft model proposes several next tokens at once; the main model then verifies them in a single pass and keeps the ones it agrees with.',
       'One hidden vector produces a short run of draft token IDs. Accepted drafts skip their own full forward pass.'],
  /* ---- multi-head latent attention ---- */
  cq:['The query projections are among the largest matrices in the layer. Passing through a smaller bottleneck first makes them low-rank, which cuts parameters and activation memory during training. Queries are never cached, so this compression is only about cost, not about the cache.',
      'Each token vector shrinks from d to d_c′ numbers. Information that cannot be expressed in d_c′ directions is dropped before the queries are formed.'],
  qr:['If RoPE were applied to keys rebuilt from the latent, the rotation would sit between the two matrices and they could no longer be merged — the cached keys would have to be recomputed for every new token. So position is moved into its own small part of the query (and key) that is rotated separately.',
      'A second, short query of d_R numbers per head is produced and rotated by position. The content query is left unrotated.'],
  ckv:['In ordinary attention the cache holds a key and a value for every head of every token — the biggest memory cost at long context. Keys and values of all heads are highly redundant, so they are generated from one shared low-dimensional latent per token. Only that latent has to be stored.',
       'd numbers become d_c numbers (about an eighth). This single short vector is the token’s entry in the cache; every key head and value head will be rebuilt from it.'],
  kc:['The latent must be turned back into one key per head so that the usual per-head dot products can be taken. Because this is a plain matrix with no position rotation in the way, it can be folded into the query projection at inference, so the expanded keys never need to exist in memory.',
      'd_c numbers are expanded to H × d_h. Every key is a linear combination of the same d_c latent directions — keys of different heads are no longer independent.'],
  vc:['Values are rebuilt from the same latent as the keys, which is what lets one cached vector serve both. The expansion matrix can be merged into the output projection at inference.',
      'd_c numbers are expanded to H × d_h values. The payload each token can hand over is limited to what the latent holds.'],
  kr:['The key needs a position part to match the query’s. It is small and shared by all heads, so it adds very little to the cache while keeping relative-position information exact.',
      'One vector of d_R numbers per token, rotated by its position. It is cached next to the latent.'],
  catqk:['Content and position were computed separately; joining them lets a single dot product score both at once: “is this the kind of token I want?” plus “is it at the right distance?”.',
      'Each head’s vector grows from d_h to d_h + d_R. For keys the same positional vector is appended to every head.'],
  /* ---- LoRA ---- */
  loraA:['Fine-tuning every weight of a large matrix is expensive and produces a full copy of the model per task. LoRA freezes the original matrix and learns only a small correction that is forced to be low-rank. The first factor, A, compresses each token to r numbers.',
       '[T × d] becomes [T × r] with r typically 4–64. Only the directions A has learned to look at survive.'],
  loraB:['The second factor expands the r numbers back to the full width, so the correction has the same shape as the frozen projection’s output. B starts at zero, so at the beginning of training the model behaves exactly like the original.',
       '[T × r] becomes [T × d]. Every output is a combination of just r learned directions — that is the low-rank constraint.'],
  loraAdd:['The correction is added to the frozen projection’s output, scaled by α / r so that changing the rank does not change the size of the update. After training, A·B can be merged into W, so inference costs nothing extra.',
       'Element-wise sum of the frozen result and the scaled low-rank update. Shape is unchanged.'],
  sample:['Always taking the most likely token produces repetitive, flat text. Sampling adds controlled variety; top-k and top-p cut off the long unreliable tail, and a repetition penalty lowers tokens already used.',
     'A distribution over V entries collapses to one integer. It is appended to the sequence and the whole forward pass runs again.']
};
/* step key → canonical key shared by WHY and FORMULA */
function canonKey(key,o={}){
  let k=/^x(q|k|v|split|scores|scale|softmax|av|concat|wo|add|post)$/.test(key)?key.slice(1):key;
  if(/^(norm1|norm2)$/.test(k))k='norm';
  else if(/^(post1|post2|post)$/.test(k))k='normpost';
  else if(k==='final')k='normfinal';
  else if(/^add\d?$/.test(k))k='add';
  else if(/^mix(\d|out)$/.test(k))k='mix';
  else if(/^hcw\d$/.test(k))k='hcw';
  else if(/^cat[qk]$/.test(k))k='catqk';
  else if(/^l[qv]a$/.test(k))k='loraA';
  else if(/^l[qv]b$/.test(k))k='loraB';
  else if(/^l[qv]add$/.test(k))k='loraAdd';
  else if(k==='mask'&&o.window)k='maskwin';
  else if(k==='last'&&o.cls)k='cls';
  return k;
}
function explainKey(key,o={}){
  if(/^x(q|k|v)$/.test(key)){const b=WHY[key.slice(1)],x=WHY.xqkv;return {why:b[0]+' '+x[0],what:b[1]+' '+x[1]};}
  if(key==='xscores'){const b=WHY.scores,x=WHY.xscores;return {why:b[0]+' '+x[0],what:x[1]+' '+b[1].split('. ').slice(1).join('. ')};}
  const w=WHY[canonKey(key,o)];return w?{why:w[0],what:w[1]}:null;
}
const explainStep=s=>s.why?{why:s.why,what:s.what||''}:explainKey(s.key,{window:/window/i.test(s.title),cls:/CLS/.test(s.title)});
const whyHTML=(e,short)=>!e?'':`<div class="why"><h4>Why this step is needed</h4><p>${e.why}</p>${short?'':`<h4>What changes inside the tensor</h4><p>${e.what}</p>`}</div>`;
/* concept-diagram blocks → explanation key */
const LAYER_WHY={norm1:'norm1',wq:'q',q:'q',wk:'k',k:'k',wv:'v',v:'v',qh:'split',kh:'split',vh:'split',qk:'scores',scores:'scores',smx:'softmax',a:'softmax',av:'av',oh:'av',cat:'concat',wo:'wo',ao:'wo',add1:'add',norm2:'norm2',wg:'gate',gt:'gate',wu:'up',up:'up',silu:'silu',mul:'mul',prod:'mul',wd:'down',mo:'down',add2:'add',out:'add'};
const MODEL_WHY={tok:'tok',emb:'emb',e0:'emb',pos:'rope',norm:'final',normOut:'final',lm:'head',logits:'head',sample:'sample'};
