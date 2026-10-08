/* ================= 2. Full model — graph + info ================= */
const M={cur:null,sel:null,hov:null,stp:null,pos:'rope',text:'The capital of France is'};
const POS={
  rope:{n:'RoPE (rotary)',box:['Positional','Encoding','(RoPE)'],note:['RoPE(Q,K)','(during attention)'],add:false,d:'Rotary embeddings: each (even, odd) pair of dimensions in Q and K is rotated by an angle proportional to the token position, so q·k depends on the <b>relative</b> distance. Nothing is added to the embeddings — the rotation happens inside every attention layer. Used by Llama, Qwen, Mistral, Gemma…'},
  learned:{n:'Learned absolute',box:['Learned','positions','P[Tmax, D]'],note:['added once','at the input'],add:true,d:'A trainable table P of shape [T_max, D]; row t is added to the embedding of token t. Simple, but the context length is fixed by the table size (GPT-2, BERT).'},
  sin:{n:'Sinusoidal',box:['Sinusoidal','encoding','(fixed)'],note:['added once','at the input'],add:true,d:'Fixed sin/cos waves of different frequencies added to the embeddings — the original Transformer. No parameters, in principle usable at unseen lengths.'},
  alibi:{n:'ALiBi',box:['ALiBi','(no position','vectors)'],note:['bias −m·|i−j|','on attention scores'],add:false,d:'No position vectors at all: each head subtracts a linear penalty m·(i−j) from its attention scores, so distant tokens are down-weighted. Extrapolates well to longer contexts (BLOOM, MPT).'}
};
const tokenWords=()=>M.text.match(/[\p{L}\p{N}_]+|[^\s\p{L}\p{N}_]/gu)||[];

function modelSVG(cur,focus){
  const {d,V,N}=S,g=new Graph(cur,focus),y=205,P=POS[M.pos];
  const ids=tokenWords().map(t=>hashId(t,V));
  const rows=ids.length<=5?ids:[ids[0],ids[1],ids[2],'⋮',ids[ids.length-1]];
  let ts='';
  rows.forEach((v,i)=>{ts+=`<rect x="28" y="${y-50+i*20}" width="60" height="20" fill="rgba(214,204,69,.14)" stroke="${C.yellow}" stroke-width="1"/><text x="58" y="${y-35+i*20}" text-anchor="middle" class="shp" style="fill:var(--txt)">${v}</text>`;});
  ts+=`<text class="lab" x="58" y="${y-62}" text-anchor="middle">Input Tokens</text><text class="shp" x="58" y="${y+66}" text-anchor="middle">[T]</text>`;
  g.box({id:'tok',x:58,y,r:30,svg:ts,col:C.yellow,step:0,aria:'Input tokens',hit:[22,y-72,72,150]});
  g.node({id:'emb',x:165,y,dims:[V,d],col:C.purple,label:['Embedding','E'],shape:`[V, ${d}]`,step:1});
  g.node({id:'e0',x:262,y,dims:[S.T,d],col:C.green,label:[],shape:`[T, ${d}]`,step:1,r:26,max:46});
  g.op({id:'plus',x:332,y,r:15,sym:'+',col:C.green,label:'',step:2});
  const box=(x,yy,w,h,lines,col)=>`<rect x="${x-w/2}" y="${yy-h/2}" width="${w}" height="${h}" rx="10" fill="${rgba(col,.1)}" stroke="${col}" stroke-width="1.5"/>`+lines.map((t,i)=>`<text class="${i===0?'lab':'shp'}" x="${x}" y="${yy-(lines.length-1)*8+i*16+4}" text-anchor="middle" style="${i===0?'':'fill:var(--txt2)'}">${t}</text>`).join('');
  g.box({id:'pos',x:332,y:335,r:46,svg:box(332,335,116,66,P.box,C.blue),col:C.blue,step:2,aria:'Positional encoding',hit:[272,298,120,74]});
  g.path(`M332 301V${y+22}`,{col:C.blue,step:2,solid:!P.add,w:P.add?2:1.5});
  g.box({id:'note',x:470,y:385,r:55,svg:box(470,385,112,40,P.note,C.orange),col:C.orange,step:2,aria:'Position note',hit:[412,362,116,46]});
  g.path('M390 345C402 345 402 385 414 385',{col:C.blue,step:2,solid:true,noarrow:true,w:1.2});
  if(!P.add){
    g.path('M527 385H878',{col:C.orange,step:3,w:1.6});
    [420,624,875].forEach(x=>g.path(`M${x} 385V292`,{col:C.orange,step:3,w:1.4}));
  }
  const layer=(id,x,title)=>g.box({id,x,y,r:47,col:C.orange,step:3,aria:title,hit:[x-52,y-92,104,184],svg:
    `<rect x="${x-44}" y="${y-84}" width="88" height="168" rx="10" fill="rgba(214,95,72,.07)" stroke="${C.orange}" stroke-width="1.8" stroke-dasharray="7 5"/>`
    +`<text class="lab" x="${x}" y="${y-62}" text-anchor="middle">${title.split(' ').slice(0,1).join(' ')}</text><text class="lab" x="${x}" y="${y-46}" text-anchor="middle">${title.split(' ').slice(1).join(' ')}</text>`
    +cube(x,y+14,38,38,26,C.orange)+`<text class="shp" x="${x}" y="${y+66}" text-anchor="middle">attn + MLP</text>`});
  layer('L1',420,'Decoder Layer 1');layer('L2',624,'Decoder Layer 2');layer('LN',875,'Decoder Layer N');
  [[524,'o1'],[728,'o2'],[979,'oN']].forEach(([x,id])=>g.node({id,x,y,dims:[S.T,d],col:C.green,label:[],shape:`[T, ${d}]`,step:3,r:26,max:46}));
  g.box({id:'dots',x:790,y,r:14,svg:`<text x="790" y="${y+10}" text-anchor="middle" style="font:700 30px var(--sans);fill:var(--txt2)">⋯</text>`,col:C.gray,step:3,aria:'more layers',hit:[770,y-30,40,60]});
  g.label(650,y-112,`× N = ${N} blocks, each with its own weights`,{step:3,col:C.orange});
  g.node({id:'norm',x:1073,y,dims:[S.T,d],col:C.blue,label:['Final','RMSNorm'],shape:`[T, ${d}]`,step:4,r:30,max:56});
  g.node({id:'normOut',x:1161,y,dims:[S.T,d],col:C.green,label:[],shape:`[T, ${d}]`,step:4,r:26,max:46});
  g.node({id:'lm',x:1251,y,dims:[d,V],col:C.purple,label:['LM Head','W_lm'],shape:`[${d}, V]`,step:5});
  g.node({id:'logits',x:1349,y,dims:[S.T,V],col:C.yellow,label:'Logits',shape:'[T, V]',step:5});
  g.label(1349,y+68,'only the last row is used',{step:5,col:C.yellow});
  let sm=`<rect x="1415" y="${y-100}" width="150" height="200" rx="12" fill="rgba(69,193,221,.07)" stroke="${C.blue}" stroke-width="1.5"/><text class="lab" x="1490" y="${y-78}" text-anchor="middle">Sampling</text><text class="shp" x="1490" y="${y-62}" text-anchor="middle" style="fill:var(--mut)">(last token)</text>`;
  ['Temperature','Top-k / Top-p','Repetition penalty','Softmax'].forEach((t,i)=>{sm+=`<text class="shp" x="1490" y="${y-36+i*20}" text-anchor="middle" style="fill:var(--txt2)">${t}</text>`;});
  [12,28,44,14,6].forEach((h,i)=>{sm+=`<rect x="${1448+i*17}" y="${y+88-h}" width="11" height="${h}" fill="${[C.yellow,C.orange,C.orange,C.yellow,C.yellow][i]}"/>`;});
  g.box({id:'sample',x:1490,y,r:75,svg:sm,col:C.blue,step:6,aria:'Sampling',hit:[1415,y-100,150,200]});
  const ch=['tok','emb','e0','plus','L1','o1','L2','o2','dots','LN','oN','norm','normOut','lm','logits','sample'],stp=[1,1,2,3,3,3,3,3,3,3,4,4,5,5,6];
  ch.slice(0,-1).forEach((a,i)=>g.edge(a,ch[i+1],{step:stp[i]??6}));
  return svgWrap(1585,360,`<g transform="translate(0,-60)">${g.html()}</g>`);
}

function modelInfoBase(id){
  if(!id)return null;
  const {d,V,N,T}=S,by=S.bytes,c=counts(),sh=(...a)=>'['+a.join(', ')+']',P=POS[M.pos];
  const lay=n=>({t:`Decoder Layer ${n}`,c:C.orange,d:`One full transformer block (norm → attention → add → norm → MLP → add). All ${N} layers share the same structure but have different weights; each reads and writes the [T, ${d}] residual stream.`,r:[['In → out',sh('T',d)+' → '+sh('T',d)],['Params / layer',fmtN(c.layer)],['All layers',fmtN(N*c.layer)]],link:'<a href="#layer">Open the layer in detail →</a>'});
  const o=n=>({t:`Hidden states after layer ${n}`,c:C.green,d:'Same shape in and out — that is what allows layers to be stacked.',r:[['Shape',sh('T',d)]]});
  const I={
    tok:{t:'Input tokens',c:C.yellow,d:'The tokenizer (BPE / SentencePiece) cuts the text into sub-word pieces and maps each to an integer ID. IDs shown here are illustrative hashes — a real tokenizer will differ.',r:[['Shape','[T]  integers in 0 … V−1'],['Vocabulary',fmtN(V)]]},
    emb:{t:'Embedding table E',c:C.purple,d:'A learned lookup table with one d-dimensional row per vocabulary entry. Token ID i simply selects row i — no matmul.',r:[['Weight',sh('V',d)+` = ${sh(V,d)}`],['Params',fmtN(c.emb)],['Memory',fmtB(c.emb*by)]]},
    e0:{t:'Token embeddings',c:C.green,d:'One vector of width d per input token — the start of the residual stream.',r:[['Shape',sh('T',d)]]},
    plus:{t:P.add?'Add position vectors':'Position is not added here',c:C.green,d:P.add?'The position vector for index t is added to token t’s embedding.':`With ${P.n} nothing is added at the input; position information enters inside attention.`,r:[['Scheme',P.n]]},
    pos:{t:'Positional encoding — '+P.n,c:C.blue,d:P.d,r:[['Scheme',P.n]]},
    note:{t:'Where position is applied',c:C.orange,d:P.add?'Applied once, before the first layer.':'Applied inside every attention layer, to Q and K (RoPE) or to the scores (ALiBi), so every layer sees position.',r:[]},
    L1:lay(1),L2:lay(2),LN:lay('N'),o1:o(1),o2:o(2),oN:o('N'),
    dots:{t:`… layers 3 to N−1`,c:C.gray,d:`Layers repeat identically. This model has N = ${N}.`,r:[]},
    norm:{t:'Final RMSNorm',c:C.blue,d:'One last normalisation of the residual stream before projecting to the vocabulary.',r:[['Shape',sh('T',d)],['Params',fmtN(d)]]},
    normOut:{t:'Final hidden states',c:C.green,d:'During generation only the last row (the newest token) is needed.',r:[['Shape',sh('T',d)]]},
    lm:{t:'LM head  W_lm',c:C.purple,d:S.tie?'Tied with the embedding table (same weights, transposed) — saves V·d parameters.':'A separate [d, V] matrix that scores every vocabulary entry. Many models tie it to the embedding table.',r:[['Weight',sh(d,'V')+` = ${sh(d,V)}`],['Params',S.tie?'0 extra (tied)':fmtN(c.emb)],['FLOPs / token',fmtF(2*d*V)]]},
    logits:{t:'Logits',c:C.yellow,d:'One unnormalised score per vocabulary entry for each position. At inference only the last row [1, V] matters — it scores the next token.',r:[['Shape',sh('T','V')],['Last row',sh(1,V)+` = ${fmtB(V*by)}`]]},
    sample:{t:'Sampling',c:C.blue,d:'Turn the last row of logits into one token: apply repetition penalty, divide by temperature, softmax, keep the top-k / top-p candidates, then draw. Try it in the lab below.',r:[['Input',sh(1,'V')],['Output','1 token ID']]}
  };
  return I[id]||null;
}
const modelSteps=()=>[
  ['Tokenize','The prompt text becomes a list of T integer token IDs.'],
  ['Embedding lookup',`Each ID selects a row of E [V, ${S.d}] → one ${S.d}-dim vector per token.`],
  ['Position information',`Scheme: ${POS[M.pos].n}. ${POS[M.pos].add?'Position vectors are added to the embeddings.':'Nothing is added here — position enters inside attention.'}`],
  ['N decoder layers',`The [T, ${S.d}] residual stream flows through ${S.N} identical blocks, each adding attention and MLP updates.`],
  ['Final RMSNorm','A last normalisation before the output projection.'],
  ['LM head → logits',`W_lm [${S.d}, V] turns each hidden vector into V scores; only the last row is used for the next token.`],
  ['Sampling','Penalty → temperature → softmax → top-k / top-p → draw one token ID. It is appended to the input and the loop repeats.']
];

const modelInfo=id=>{const i=modelInfoBase(id),k=id==='pos'&&M.pos!=='rope'?'pos':MODEL_WHY[id];return i&&k?Object.assign({},i,{why:explainKey(k),fm:formulasFor(k)}):i;};
