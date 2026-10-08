/* ================= global model state ================= */
const S={T:1000,d:4096,H:32,Hkv:8,ff:11008,V:32000,N:32,bytes:2,attn:'MHA',tie:false,preset:'image'};
const MO={E:8,k:2,shared:false,sel:0,seed:7};
const VIEWS={};
let curView=null;
const kvH=(v=S.attn)=>v==='MHA'?S.H:v==='MQA'?1:S.Hkv;
const dHead=()=>S.d/S.H;
function counts(v=S.attn){
  const d=S.d,kv=kvH(v)*dHead();
  const attn=2*d*d+2*d*kv,mlp=3*d*S.ff,norms=2*d,layer=attn+mlp+norms,emb=S.V*d;
  return {attn,mlp,norms,layer,emb,kv,total:emb+S.N*layer+d+(S.tie?0:emb)};
}
const PRESETS=[
  ['image','Image example (T=1000, 4096 · 32 heads · 11008)',{T:1000,d:4096,H:32,Hkv:8,ff:11008,V:32000,N:32,attn:'MHA'}],
  ['llama2','Llama-2-7B · MHA',{T:4096,d:4096,H:32,Hkv:32,ff:11008,V:32000,N:32,attn:'MHA'}],
  ['llama3','Llama-3-8B · GQA',{T:4096,d:4096,H:32,Hkv:8,ff:14336,V:128256,N:32,attn:'GQA'}],
  ['llama70','Llama-3-70B · GQA',{T:4096,d:8192,H:64,Hkv:8,ff:28672,V:128256,N:80,attn:'GQA'}],
  ['mistral','Mistral-7B · GQA + sliding window',{T:4096,d:4096,H:32,Hkv:8,ff:14336,V:32000,N:32,attn:'GQA'}],
  ['qwen','Qwen2.5-7B · GQA',{T:4096,d:3584,H:28,Hkv:4,ff:18944,V:152064,N:28,attn:'GQA'}],
  ['mixtral','Mixtral-8x7B · GQA + MoE',{T:4096,d:4096,H:32,Hkv:8,ff:14336,V:32000,N:32,attn:'GQA',moe:[8,2]}],
  ['tiny','Tiny (GPT-2-small sized) · MHA',{T:512,d:768,H:12,Hkv:12,ff:3072,V:50257,N:12,attn:'MHA'}]
];
const CFG=[['T','Tokens T',1,1048576],['d','d_model',64,65536],['H','Heads',1,512],['Hkv','KV heads',1,512],['ff','d_ff',64,262144],['V','Vocab',1000,1000000],['N','Layers',1,256]];

function fixKV(){
  const ds=divisors(S.H);
  if(!ds.includes(S.Hkv))S.Hkv=ds.reduce((b,x)=>Math.abs(x-S.Hkv)<Math.abs(b-S.Hkv)?x:b,ds[0]);
}
function groupKV(){ // sensible GQA value strictly between 1 and H
  const ds=divisors(S.H).filter(x=>x>1&&x<S.H);
  if(ds.length)S.Hkv=ds.reduce((b,x)=>Math.abs(x-S.H/4)<Math.abs(b-S.H/4)?x:b,ds[0]);
}
function applyCfg(k,raw){
  let v=Math.round(Number(raw));
  if(!isFinite(v)||v<1){syncCfg();return;}
  S[k]=v;S.preset='custom';
  if(k==='H'||k==='d'){
    if(S.d%S.H){const dh=Math.max(8,Math.round(S.d/S.H/8)*8);S.d=dh*S.H;toast(`d_model set to ${S.d} so it splits evenly into ${S.H} heads of ${dh}`);}
  }
  if(k==='H')fixKV();
  if(k==='Hkv'){fixKV();S.attn=S.Hkv>=S.H?'MHA':S.Hkv===1?'MQA':'GQA';}
  syncCfg();refresh();
}
function setAttn(v,quiet){
  S.attn=v;
  if(v==='GQA'&&(S.Hkv>=S.H||S.Hkv<=1))groupKV();
  syncCfg();if(!quiet)refresh();
}
function applyPreset(id){
  const p=PRESETS.find(x=>x[0]===id);if(!p)return;
  Object.assign(S,p[2]);S.preset=id;
  if(p[2].moe){MO.E=p[2].moe[0];MO.k=p[2].moe[1];}
  syncCfg();refresh();
}
function buildCfg(){
  $('#cfg').innerHTML=`<label class="f">Preset<select id="preset">${PRESETS.map(p=>`<option value="${p[0]}">${p[1]}</option>`).join('')}<option value="custom">Custom</option></select></label>`
    +CFG.map(([k,l,mn,mx])=>`<label class="f">${l}<input type="number" id="c-${k}" min="${mn}" max="${mx}" inputmode="numeric"></label>`).join('')
    +`<label class="f">Precision<select id="c-bytes"><option value="2">bf16 / fp16 · 2 B</option><option value="1">fp8 / int8 · 1 B</option><option value="0.5">int4 · 0.5 B</option><option value="4">fp32 · 4 B</option></select></label>`
    +`<label class="chk"><input type="checkbox" id="c-tie"> tie embeddings</label><span class="derived" id="derived"></span>`;
  CFG.forEach(([k])=>$('#c-'+k).addEventListener('change',e=>applyCfg(k,e.target.value)));
  $('#preset').addEventListener('change',e=>applyPreset(e.target.value));
  $('#c-bytes').addEventListener('change',e=>{S.bytes=Number(e.target.value);refresh();});
  $('#c-tie').addEventListener('change',e=>{S.tie=e.target.checked;refresh();});
  syncCfg();
}
function syncCfg(){
  CFG.forEach(([k])=>{const i=$('#c-'+k);if(i&&document.activeElement!==i)i.value=S[k];});
  $('#preset').value=PRESETS.some(p=>p[0]===S.preset)?S.preset:'custom';
  $('#c-bytes').value=String(S.bytes);$('#c-tie').checked=S.tie;
  const g=S.H/kvH();
  $('#derived').textContent=`d_head = ${dHead()} · attention = ${S.attn} (${kvH()} KV heads${g>1?`, ${g} Q per KV`:''}) · ${fmtN(counts().total)} params`;
}
function refresh(){syncCfg();const v=VIEWS[curView];if(v&&v.update)v.update();}

/* ================= navigation ================= */
function route(){
  const [v,sub]=(location.hash.slice(1)||'overview').split('/');
  const id=VIEWS[v]?v:'overview',prev=curView;
  if(prev&&VIEWS[prev].hide)VIEWS[prev].hide();
  curView=id;renderNav();
  $('#cfgwrap').hidden=(id==='overview'||id==='refs'||id==='families'||(id==='topic'&&!/^(lora|qlora)$/.test(sub||'')));
  const root=$('#main');root.innerHTML='';
  VIEWS[id].show(root,sub);
  afterRoute(id,sub,root);
  document.title='Generative AI Atlas — '+(typeof CURNODE!=='undefined'&&CURNODE?CURNODE.t:id);
  if(prev!==id)window.scrollTo(0,0);
}
