/* ================= shared hover popup, click panel and right-click menu for every diagram ================= */
(function(){
  if(!$('#detail'))document.body.insertAdjacentHTML('beforeend','<aside id="detail" class="tf" hidden aria-label="Details"></aside><div id="tip" hidden role="tooltip"></div><div id="menu" hidden role="menu"></div><div id="chip" hidden role="status"></div>');
  const tip=$('#tip'),menu=$('#menu'),chip=$('#chip'),det=$('#detail'),main=$('#main');
  let actx=null;
  const INFO={topic:id=>topicInfo(id),layer:id=>layerInfo(id),model:id=>modelInfo(id),decode:id=>decodeInfo(id),prefill:id=>pfInfo(id),
    families:id=>{const b=VIEWS.families.fd&&findItem(VIEWS.families.fd.cols,id);return b?{t:b.label,c:b.col,d:b.d,r:b.shape?[['Shape',esc(b.shape)]]:[]}:null;}};
  const STATE={topic:()=>TP,layer:()=>L,model:()=>M,decode:()=>DEC,prefill:()=>PF,families:()=>F};
  const STEPPER={topic:()=>TP.stp,layer:()=>L.stp,model:()=>M.stp};
  const plain=h=>{const d=document.createElement('div');d.innerHTML=h;return d.textContent.replace(/\s+/g,' ').trim();};
  const place=(el,x,y)=>{const w=el.offsetWidth,h=el.offsetHeight;el.style.left=Math.max(8,Math.min(x,innerWidth-w-8))+'px';el.style.top=Math.max(8,y+h+8>innerHeight?y-h-28:y)+'px';};
  const say=h=>{chip.hidden=false;chip.innerHTML='<span>'+h+'</span><button>OK</button>';};
  const copy=s=>(navigator.clipboard?navigator.clipboard.writeText(s):Promise.reject()).then(()=>say('Copied: '+esc(s.slice(0,140))),()=>say('Copy was blocked by the browser: '+esc(s.slice(0,140))));
  const closeMenu=()=>{menu.hidden=true;actx=null;};
  const hideAll=()=>{tip.hidden=true;closeMenu();det.hidden=true;};
  /* what is under the pointer? vertical-graph nodes are handled by their own module */
  function target(e){
    if(e.target.closest('#vg-main'))return null;
    let el=e.target.closest('#hg-main .gn');if(el)return {el,k:'hg'};
    el=e.target.closest('.node');if(el&&main.contains(el))return {el,k:'node'};
    el=e.target.closest('#a-svg .hl[data-g]');if(el)return {el,k:'head'};
    el=e.target.closest('#sf-main .step[data-i]');if(el)return {el,k:'step'};
    return null;
  }
  function infoHTML(t){
    const el=t.el;
    if(t.k==='hg'){const s=TFX.h&&TFX.h()[+el.dataset.i];return s?TFX.tip(s,el.getAttribute('aria-label').split(' — ')[0]):'';}
    if(t.k==='head'){const H=S.H,k=kvH(),gs=H/k,g=+el.dataset.g,q=el.dataset.q;
      return `<div class="th"><b>${q!=null?'Query head '+(+q+1):'K/V head '+(g+1)}</b><span>${S.attn}</span></div><div class="tk"><span>Shape</span><span>[T, ${dHead()}] per head</span><span>K/V head</span><span>${g+1} of ${k}</span><span>Shared by</span><span>query heads ${g*gs+1}–${(g+1)*gs}${gs>1?` (${gs} heads)`:''}</span><span>Cache / token</span><span>${fmtB(2*dHead()*S.bytes)} per layer for this K/V head</span></div>`;}
    if(t.k==='node'){const f=INFO[curView];
      if(f){const i=f(el.dataset.id);return i?inspHTML(Object.assign({short:true},i)):'';}
      if(curView==='moe'){const p=$('#mo-insp');return p&&p.querySelector('.ih')?p.innerHTML:`<div class="th"><b>${esc(el.getAttribute('aria-label')||'')}</b><span>MoE</span></div><p>Click a token to trace its route through the router and experts.</p>`;}
    }
    return '';
  }
  const hint=k=>`<div class="tf">${k==='head'?'Hover highlights the sharing group':k==='hg'?'Click: worked example · Right-click: more':'Click: pin details · Right-click: more actions'}</div>`;
  main.addEventListener('mouseover',e=>{
    const t=target(e);if(!t||t.k==='step'||!menu.hidden){if(!e.target.closest('#vg-main'))tip.hidden=true;return;}
    const h=infoHTML(t);if(!h){tip.hidden=true;return;}
    tip.innerHTML=h+hint(t.k);tip.dataset.n='';tip.hidden=false;place(tip,e.clientX+18,e.clientY+18);
  });
  main.addEventListener('mousemove',e=>{if(!tip.hidden&&!e.target.closest('#vg-main'))place(tip,e.clientX+18,e.clientY+18);});
  main.addEventListener('mouseleave',()=>{tip.hidden=true;});
  /* left click on a concept-diagram block: floating pinned panel with next actions */
  function panel(el){
    const st=STATE[curView]&&STATE[curView](),f=INFO[curView],id=el.dataset.id;
    if(!st||!f||st.sel!==id){if(!$('#vg-main'))det.hidden=true;return;}
    const i=f(id);if(!i){det.hidden=true;return;}
    const step=el.dataset.step,a=CURNODE.a;
    det.innerHTML=`<button class="x" aria-label="Close details">✕</button><div style="padding-top:10px">${inspHTML(i)}</div><div class="toolbar" style="margin-top:12px">${STEPPER[curView]&&step!==''&&step!=null?`<button class="btn pri" data-da="step" data-s="${step}">▶ Show this step in the walkthrough</button>`:''}${a?`<a class="btn" href="#steps/${a}" style="text-decoration:none">② Every operation</a><a class="btn" href="#graph/${a}" style="text-decoration:none">③ Tensor graph</a>`:''}</div>`;
    det.hidden=false;
  }
  main.addEventListener('click',e=>{const t=target(e);if(t&&t.k==='node'){tip.hidden=true;panel(t.el);}});
  const unpin=()=>{const st=STATE[curView]&&STATE[curView]();if(st&&st.sel){st.sel=null;$$('.node.sel',main).forEach(n=>n.classList.remove('sel','foc'));const v=VIEWS[curView];if(v&&v.focus)v.focus();}};
  det.addEventListener('click',e=>{
    if(e.target.closest('.x')){det.hidden=true;unpin();return;}
    const b=e.target.closest('[data-da=step]');if(b&&STEPPER[curView]){const sp=STEPPER[curView]();sp.stop();sp.go(+b.dataset.s);}
  });
  /* right click */
  main.addEventListener('contextmenu',e=>{
    const t=target(e);if(!t)return;e.preventDefault();tip.hidden=true;actx=t;
    const el=t.el,a=CURNODE.a,B=(m,l)=>`<button role="menuitem" data-am="${m}">${l}</button>`;let h='',title='';
    if(t.k==='node'){
      const st=STATE[curView]&&STATE[curView](),pinned=st&&st.sel===el.dataset.id,sp=STEPPER[curView]&&STEPPER[curView](),step=el.dataset.step;
      title=el.getAttribute('aria-label')||el.dataset.id;
      if(INFO[curView])h+=B('pin',pinned?'📌 Unpin details':'📌 Pin details');
      if(curView==='moe'&&/^t\d$/.test(el.dataset.id))h+=B('pin','↝ Trace this token');
      if(sp&&step!==''&&step!=null)h+=B('step','▶ Show this step in the walkthrough')+(sp.i!=null?B('all','◼ Show all steps again'):'');
      if(a)h+='<hr>'+B('steps','② Open every operation, step by step')+B('graph','③ Open the tensor graph');
      h+='<hr>'+B('copy','⧉ Copy name and details');
    }else if(t.k==='hg'){title=el.getAttribute('aria-label').split(' — ')[0];h=B('ex','▦ Open the worked numeric example')+B('vert','⇅ Switch to vertical layout (tracing)')+'<hr>'+B('copy','⧉ Copy name, shape and formula');}
    else if(t.k==='head'){title='Attention head';h=B('layer','◧ See this attention type in the layer diagram')+B('steps','② Open every operation, step by step')+B('graph','③ Open the tensor graph');}
    else{title=plain($('.sh b',el).innerHTML);h=B('ex',el.classList.contains('open')?'▦ Close the worked example':'▦ Open the worked numeric example')+B('sgraph','③ Open this architecture as a tensor graph')+'<hr>'+B('copy','⧉ Copy step and formula');}
    menu.innerHTML=`<div class="mh">${esc(title)}</div>`+h;menu.hidden=false;
    const r=el.getBoundingClientRect(),kb=e.clientX===0&&e.clientY===0;place(menu,kb?r.right:e.clientX,kb?r.top:e.clientY);$('button',menu).focus();
  });
  menu.addEventListener('click',e=>{
    const b=e.target.closest('button[data-am]');if(!b||!actx)return;const {el,k}=actx,m=b.dataset.am,a=CURNODE.a;closeMenu();
    if(m==='pin'||m==='ex')el.dispatchEvent(new MouseEvent('click',{bubbles:true}));
    else if(m==='step'){const sp=STEPPER[curView]();sp.stop();sp.go(+el.dataset.step);}
    else if(m==='all'){STEPPER[curView]().all();}
    else if(m==='steps')location.hash='#steps/'+(k==='head'?S.attn.toLowerCase():a);
    else if(m==='graph')location.hash='#graph/'+(k==='head'?S.attn.toLowerCase():a);
    else if(m==='layer')location.hash='#layer';
    else if(m==='sgraph'){const s=$('#sf-bar #selA');location.hash='#graph/'+(s?s.value:'');}
    else if(m==='vert'){const v=$('#hg-bar #orient [data-v=v]');if(v)v.click();}
    else if(m==='copy'){
      if(k==='hg'){const s=TFX.h()[+el.dataset.i];copy(el.getAttribute('aria-label').split(' — ')[0]+' — '+TFX.text(s));}
      else if(k==='step'){const s=TFX.s()[+el.dataset.i];copy(TFX.text(s));}
      else copy(plain(infoHTML(actx||{el,k}))||el.getAttribute('aria-label')||'');
    }
  });
  menu.addEventListener('keydown',e=>{if(!actx)return;const B=$$('button',menu),i=B.indexOf(document.activeElement);
    if(e.key==='ArrowDown'){e.preventDefault();B[(i+1)%B.length].focus();}
    else if(e.key==='ArrowUp'){e.preventDefault();B[(i-1+B.length)%B.length].focus();}
    else if(e.key==='Escape'){const el=actx.el;closeMenu();el.focus&&el.focus();}});
  document.addEventListener('pointerdown',e=>{if(actx&&!e.target.closest('#menu'))closeMenu();});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('#vg-main')){if(!det.hidden){det.hidden=true;unpin();}tip.hidden=true;}});
  window.addEventListener('scroll',()=>{tip.hidden=true;if(actx)closeMenu();},{passive:true});
  window.addEventListener('hashchange',hideAll);
  chip.addEventListener('click',e=>{if(e.target.closest('button')&&!$('.gbox.tr'))chip.hidden=true;});
})();
