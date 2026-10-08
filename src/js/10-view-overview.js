/* ================= 0. Overview ================= */
VIEWS.overview={
  show(root,sub,mode){
    const two=mode?mode==='two':root.clientWidth>=1250;this.two=root.clientWidth>=1250;
    root.innerHTML=`
    <section class="ovhead">
      <h1>Generative AI Architecture Atlas</h1>
      <p>The whole field on one page. Follow a branch from the root to a technique: every leaf says what the technique does and names <em class="exk">a real model that uses it</em>. Click a leaf for its diagram, every tensor operation step by step, and the tensor graph.</p>
    </section>
    ${overviewTree(two)}
    <section class="hero" style="margin-top:26px"><div>
      <h3 class="gh" style="margin-top:0">How to read the lessons</h3>
      <p>Every tensor shape and cost is recomputed live from the model configuration. Each tensor keeps one colour everywhere:</p>
      ${roleLegend(['x','xn','q','k','v','s','a','attn','gate','up','hid','mlp','w','tok'])}
      <p class="mut" style="font-size:13px;margin-top:12px"><b>Hover</b> a tensor or operation for details, formulas and the reason it exists; <b>click</b> to pin it or open a worked numeric example; <b>right-click</b> for tracing and more; press <b>▶ Play steps</b> to animate data flow.</p>
    </div><div id="hero-art" class="svgwrap"></div></section>
    <details class="card pad" style="margin-top:20px"><summary style="cursor:pointer;font-weight:600">Show the original PNG for comparison</summary>
      <img src="assets/llm-arch-3d.png" alt="Original LLM architecture diagram" style="max-width:100%;border-radius:10px;margin-top:12px" onerror="this.replaceWith(Object.assign(document.createElement('p'),{className:'mut',textContent:'The original PNG was not found at assets/llm-arch-3d.png.'}))">
    </details>`;
    /* fit the tree: first stack each leaf on two lines, then fall back to a one-sided tree */
    const rt=$('.rtree',root),over=()=>rt.scrollWidth>rt.clientWidth+2;
    if(over())rt.classList.add('two2');
    if(over()&&two){this.show(root,sub,'one');return;}
    const x=[[90,C.green,[1000,4096]],[250,C.blue,[1000,4096]],[410,C.purple,[4096,4096]],[570,C.red,[32,1000,1000]],[730,C.orange,[32,1000,128]],[890,C.yellow,[1000,32000]]];
    let s='';x.forEach(([cx,c,dm],i)=>{const [w,h,d]=dimsBox(dm,70);s+=cube(cx,120,w,h,d,c,{bands:i===4?BANDS:null});if(i<x.length-1)s+=`<path d="M${cx+50} 120H${cx+110}" stroke="#6f86a0" stroke-width="2" stroke-dasharray="6 5" class="edge" marker-end="url(#ar-8fa6bd)"/>`;});
    $('#hero-art').innerHTML=svgWrap(980,230,s+`<text class="cap" x="490" y="210" text-anchor="middle">X → norm → W → scores → heads → logits</text>`);
  },
  resize(){const r=$('#main');clearTimeout(this.rt);this.rt=setTimeout(()=>this.show(r),150);}
};
window.addEventListener('resize',()=>{if(curView==='overview')VIEWS.overview.resize();});
