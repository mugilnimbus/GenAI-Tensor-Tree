/* ================= small graphs shown with the formulas (activations, norms, softmax, RoPE, …) =================
   PLOT[canonical key]() returns an inline SVG. They are attached to FORMULA[key].plot at the bottom. */
function curvePlot(curves,xr,yr,note){
  const W=300,H=150,L=28,R=8,T=8,B=20,sx=x=>L+(x-xr[0])/(xr[1]-xr[0])*(W-L-R),sy=y=>T+(yr[1]-y)/(yr[1]-yr[0])*(H-T-B);
  let s=`<line x1="${L}" y1="${sy(0)}" x2="${W-R}" y2="${sy(0)}" class="pax"/><line x1="${sx(0)}" y1="${T}" x2="${sx(0)}" y2="${H-B}" class="pax"/>`;
  for(let x=Math.ceil(xr[0]);x<=xr[1];x++)if(x)s+=`<text class="ptk" x="${sx(x)}" y="${sy(0)+11}" text-anchor="middle">${x}</text>`;
  for(let y=Math.ceil(yr[0]);y<=yr[1];y++)if(y)s+=`<text class="ptk" x="${sx(0)-4}" y="${sy(y)+3}" text-anchor="end">${y}</text>`;
  curves.forEach(([f,c,dash])=>{let d='';for(let i=0;i<=120;i++){const x=xr[0]+(xr[1]-xr[0])*i/120,y=clamp(f(x),yr[0]-1,yr[1]+1);d+=(i?'L':'M')+sx(x).toFixed(1)+' '+sy(y).toFixed(1);}
    s+=`<path d="${d}" fill="none" stroke="${c}" stroke-width="${dash?1.4:2.4}"${dash?' stroke-dasharray="4 3"':''}/>`;});
  const lg=curves.map(([,c,,l],i)=>`<tspan x="${L+6}" dy="${i?12:0}" fill="${c}">${l}</tspan>`).join('');
  return `<svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="${note||'function graph'}"><clipPath id="pc"><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}"/></clipPath><g clip-path="url(#pc)">${s}</g><text class="plg" x="${L+6}" y="${T+10}">${lg}</text></svg>`;
}
function barPlot(groups){
  const n=groups[0][1].length,W=300,H=140,gw=(W-20)/groups.length,mx=Math.max(...groups.flatMap(g=>g[1].map(Math.abs)))||1,mid=groups.some(g=>g[1].some(v=>v<0))?H*0.52:H-24,sc=(mid-22)/mx;
  let s='';
  groups.forEach(([label,vals,col],gi)=>{const x0=10+gi*gw,bw=(gw-16)/n;
    s+=`<line x1="${x0}" y1="${mid}" x2="${x0+gw-10}" y2="${mid}" class="pax"/><text class="ptk" x="${x0+(gw-10)/2}" y="${H-4}" text-anchor="middle">${label}</text>`;
    vals.forEach((v,i)=>{const h=Math.abs(v)*sc;s+=`<rect x="${x0+3+i*bw}" y="${v>=0?mid-h:mid}" width="${bw-3}" height="${Math.max(h,.6)}" fill="${col}"/><text class="ptk" x="${x0+3+i*bw+(bw-3)/2}" y="${v>=0?mid-h-3:mid+h+9}" text-anchor="middle">${Math.abs(v)<10&&!Number.isInteger(v)?v.toFixed(2):v}</text>`;});});
  return `<svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="before and after values">${s}</svg>`;
}
const _sig=x=>1/(1+Math.exp(-x)),_silu=x=>x*_sig(x),_gelu=x=>0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*x*x*x)));
const _sm=(a,t=1)=>{const m=Math.max(...a),e=a.map(v=>Math.exp((v-m)/t)),z=e.reduce((p,q)=>p+q,0);return e.map(v=>v/z);};
const _ex=[3,-1,4,0.5,-2];
const PLOT={
  silu:()=>curvePlot([[x=>x,'#8a8aa0',1,'y = x'],[_sig,'#45c1dd',1,'σ(x)'],[_silu,'#d6459b',0,'SiLU(x)']],[-6,4],[-1,4],'SiLU and sigmoid'),
  gelu:()=>curvePlot([[x=>Math.max(0,x),'#8a8aa0',1,'ReLU'],[_silu,'#45c1dd',1,'SiLU'],[_gelu,'#d6459b',0,'GELU(x)']],[-5,4],[-1,4],'GELU compared with ReLU and SiLU'),
  relu:()=>curvePlot([[x=>Math.max(0,x),'#d6459b',0,'ReLU(x) = max(0, x)']],[-5,4],[-1,4],'ReLU'),
  norm:()=>{const r=Math.sqrt(_ex.reduce((a,v)=>a+v*v,0)/_ex.length);return barPlot([['one token vector x',_ex,'#8a8aa0'],[`x / RMS  (RMS = ${r.toFixed(2)})`,_ex.map(v=>v/r),'#45c1dd']]);},
  normln:()=>{const mu=_ex.reduce((a,v)=>a+v,0)/_ex.length,sd=Math.sqrt(_ex.reduce((a,v)=>a+(v-mu)**2,0)/_ex.length);return barPlot([['one token vector x',_ex,'#8a8aa0'],[`(x − μ)/σ  (μ = ${mu.toFixed(2)}, σ = ${sd.toFixed(2)})`,_ex.map(v=>(v-mu)/sd),'#45c1dd']]);},
  softmax:()=>{const z=[2,1,0.2,-1];return barPlot([['scores',z,'#d6459b'],['softmax → weights',_sm(z),'#45d984']]);},
  vsoft:()=>{const z=[3,2,1,0,-1];return barPlot([['τ = 0.5 (sharper)',_sm(z,.5),'#d6459b'],['τ = 1',_sm(z,1),'#45d984'],['τ = 2 (flatter)',_sm(z,2),'#45c1dd']]);},
  scale:()=>{const z=[9,-4,14,2];return barPlot([['raw scores S',z,'#8a8aa0'],['S / √d_h  (d_h = 16)',z.map(v=>v/4),'#d6459b']]);},
  sample:()=>{const p=[.42,.23,.14,.09,.06,.04,.02];return barPlot([['sorted probabilities',p,'#45d984'],['top-p = 0.8 keeps the first 4',[...p.slice(0,4).map(v=>v/.88),0,0,0],'#d6cc45']]);},
  mask:()=>{let s='';for(let i=0;i<6;i++)for(let j=0;j<6;j++)s+=`<rect x="${90+j*20}" y="${10+i*20}" width="18" height="18" fill="${j<=i?'#d6459b':'var(--off)'}"/>`;return `<svg class="plot" viewBox="0 0 300 150" role="img" aria-label="causal mask">${s}<text class="ptk" x="150" y="142" text-anchor="middle">row = query i · column = key j · filled = allowed (j ≤ i)</text></svg>`;},
  rope:()=>{let s=`<circle cx="150" cy="72" r="56" fill="none" class="pax"/>`;[0,1,2,3,4].forEach(t=>{const a=t*0.5,x=150+56*Math.cos(a),y=72-56*Math.sin(a);s+=`<line x1="150" y1="72" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${['#45c1dd','#4a55e0','#a645dc','#d6459b','#d65f48'][t]}" stroke-width="2.2"/><text class="ptk" x="${(150+70*Math.cos(a)).toFixed(1)}" y="${(72-70*Math.sin(a)+3).toFixed(1)}" text-anchor="middle">t=${t}</text>`;});
    return `<svg class="plot" viewBox="0 0 300 150" role="img" aria-label="RoPE rotation">${s}<text class="ptk" x="150" y="146" text-anchor="middle">one (2i, 2i+1) pair: same vector, rotated by t·θ</text></svg>`;},
  gstate:()=>curvePlot([[x=>Math.pow(0.9,x),'#d6cc45',0,'α = 0.9'],[x=>Math.pow(0.7,x),'#d65f48',0,'α = 0.7'],[x=>Math.pow(0.98,x),'#45d984',0,'α = 0.98']],[0,30],[0,1],'memory left after t tokens'),
  topk:()=>{const r=[1.2,-0.4,2.1,0.3,1.7,-1];const top=[2,4];return barPlot([['router scores',r,'#8a8aa0'],['top-2, renormalised',r.map((v,i)=>top.includes(i)?_sm([2.1,1.7])[top.indexOf(i)]:0),'#d65f48']]);},
  add:()=>barPlot([['X',[2,-1,3,1],'#72d945'],['+ update',[0.4,0.6,-0.5,0.2],'#45c1dd'],['= X_out',[2.4,-0.4,2.5,1.2],'#72d945']]),
  mul:()=>barPlot([['SiLU(G)',[0.02,1.8,-0.1,2.4],'#d65f48'],['U',[1.5,-0.9,2.2,0.7],'#45c1dd'],['product',[0.03,-1.62,-0.22,1.68],'#5b66f0']])
};
PLOT.nggate=PLOT.silu;PLOT.bscore=PLOT.relu;PLOT.gread=PLOT.gstate;PLOT.normfinal=PLOT.norm;
Object.keys(PLOT).forEach(k=>{if(FORMULA[k])FORMULA[k].plot=PLOT[k];});
