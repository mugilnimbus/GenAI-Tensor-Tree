/* ================= Multi-head Latent Attention (DeepSeek-V2 / V3) =================
   K and V are produced from one small latent per token; only that latent (plus a small RoPE key) is cached.
   Ratios follow DeepSeek-V2 (d = 5120: d_c = 512, d_c' = 1536, d_R = 64): d_c = d/8, d_c' = 3·d_c, d_R = d_h/2. */
const mlaDims=c=>{const dh=c.d/c.H,dc=Math.max(16,Math.round(c.d/8)),dR=Math.max(2,Math.round(dh/2));return {dh,dc,dq:3*dc,dR,hw:dh+dR};};
function mlaDecoder(c){
  const base=modern(c,{kv:c.H,mask:'causal'}),{d,H,T}=c,{dh,dc,dq,dR,hw}=mlaDims(c),h=clamp(c.h,0,H-1),A='Attention';
  const S=base.slice(0,base.findIndex(s=>s.key==='q')),by=k=>base.find(s=>s.key===k);
  const P=(key,title,fx,expr,o)=>S.push(st(key,A,title,fx,expr,o));
  const cQ=()=>act(sb('c','Q'),T,dq),cKV=()=>act(sb('c','KV'),T,dc),X=()=>act('X̂',T,d);
  const QR=()=>act(sb('Q','R'),T,H*dR,{stripes:H,shape:`[${T} × ${H} × ${dR}]`}),KR=()=>act(sb('K','R'),T,dR);
  P('cq','Compress for the queries',`${sb('c','Q')} = X̂·${sb('W','DQ')}`,[X(),'×',wgt(sb('W','DQ'),d,dq),'=',cQ()],{op:'matmul',from:['X̂'],flops:2*T*d*dq,note:`Each token is squeezed from ${d} to ${dq} numbers before the query projections. This is not cached; it only makes the query matrices smaller.`});
  P('q','Query — content part',`${sb('Q','C')} = ${sb('c','Q')}·${sb('W','UQ')}`,[cQ(),'×',wgt(sb('W','UQ'),dq,d),'=',act('Q',T,d)],{op:'matmul',from:['cQ'],flops:2*T*dq*d,note:`Expanded back to ${H} heads × ${dh}. This part carries no position information.`});
  P('qr','Query — position part',`${sb('Q','R')} = RoPE(${sb('c','Q')}·${sb('W','QR')})`,[cQ(),'×',wgt(sb('W','QR'),dq,H*dR),'=',QR()],{op:'matmul',from:['cQ'],flops:2*T*dq*H*dR,note:`A separate, small query of ${dR} dims per head. RoPE is applied only here.`});
  P('ckv','Compress keys and values into one latent',`${sb('c','KV')} = X̂·${sb('W','DKV')}`,[X(),'×',wgt(sb('W','DKV'),d,dc),'=',cKV()],{op:'matmul',from:['X̂'],flops:2*T*d*dc,note:`<b>This is what gets cached:</b> ${dc} numbers per token, instead of ${2*d} for all keys and values of ${H} heads.`});
  P('kc','Key — content part',`${sb('K','C')} = ${sb('c','KV')}·${sb('W','UK')}`,[cKV(),'×',wgt(sb('W','UK'),dc,d),'=',act('K',T,d)],{op:'matmul',from:['cKV'],flops:2*T*dc*d,note:`All ${H} key heads are re-created from the latent. At inference ${sb('W','UK')} can be folded into the query side, so this is never materialised.`});
  P('vc','Values',`V = ${sb('c','KV')}·${sb('W','UV')}`,[cKV(),'×',wgt(sb('W','UV'),dc,d),'=',act('V',T,d)],{op:'matmul',from:['cKV'],flops:2*T*dc*d,note:`Values come from the same latent. ${sb('W','UV')} can be folded into ${sb('W','O')} at inference.`});
  P('kr','Key — position part',`${sb('K','R')} = RoPE(X̂·${sb('W','KR')})`,[X(),'×',wgt(sb('W','KR'),d,dR),'=',KR()],{op:'matmul',from:['X̂'],flops:2*T*d*dR,note:`One ${dR}-dim positional key per token, shared by all heads. Also cached.`});
  S.push(by('split'));
  P('catq','Join content and position — queries',`q̃<sub>h</sub> = [${sb('Q','C,h')} ; ${sb('Q','R,h')}]`,[act('Q',T,d,{stripes:H,shape:`[${T} × ${H} × ${dh}]`}),'‖',QR(),'=',act('Q̃',T,H*hw,{stripes:H,shape:`[${T} × ${H} × ${hw}]`})],{op:'concat',from:['Q','QR'],note:`Per head: ${dh} content dims followed by ${dR} rotated dims.`});
  P('catk','Join content and position — keys',`k̃<sub>h</sub> = [${sb('K','C,h')} ; ${sb('K','R')}]`,[act('K',T,d,{stripes:H,shape:`[${T} × ${H} × ${dh}]`}),'‖',KR(),'=',act('K̃',T,H*hw,{stripes:H,shape:`[${T} × ${H} × ${hw}]`})],{op:'concat',from:['K','KR'],note:`The same positional key is appended to every head’s content key.`});
  const hd={head:true,mult:H};
  S.push(st('scores',A,'Scores',`S = q̃<sub>h</sub>·k̃<sub>h</sub>ᵀ`,[act(sb('Q̃',`h=${h}`),T,hw),'×',act(sb('K̃',`h=${h}`)+'ᵀ',hw,T),'=',msk('S',T,T,{rep:H})],Object.assign({op:'qkt',from:['Q̃','K̃'],flops:2*T*T*hw*H,note:`The dot product is the sum of a content match (${dh} dims) and a position match (${dR} dims).`},hd)));
  S.push(st('scale',A,'Scale',`S′ = S / √(d<sub>h</sub> + d<sub>R</sub>) = S / √${hw}`,[msk('S',T,T),`÷ ${Math.sqrt(hw).toFixed(2)}`,'=',msk('S′',T,T)],Object.assign({op:'scale',note:'The vectors are longer than in ordinary attention, so the divisor uses the joined width.'},hd)));
  ['mask','softmax','av'].forEach(k=>S.push(by(k)));
  base.slice(base.findIndex(s=>s.key==='concat')).forEach(s=>S.push(s));
  return S;
}
ARCH.push({id:'mla',name:'Multi-head latent attention  (DeepSeek-V2 / V3 style)',short:'MLA',
  desc:'Keys and values of all heads are rebuilt from one small latent vector per token, and only that latent is cached. Position is carried by a separate small RoPE part of the query and key. Sizes follow DeepSeek-V2’s ratios, scaled to the configured d_model.',
  meta:c=>{const m=mlaDims(c);return {norm:'RMSNorm, pre',pos:`RoPE on a separate ${m.dR}-dim part of Q and K`,mask:'causal',mlp:'SwiGLU (3 matrices)',kv:c.H,
    kvText:fB(c.N*(m.dc+m.dR)*2)+`  (${c.N} layers × (${m.dc} latent + ${m.dR} RoPE key) × 2 B)`};},build:mlaDecoder});
BACK.mla='topic/mla';
