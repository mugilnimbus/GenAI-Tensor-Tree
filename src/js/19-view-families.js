/* ================= 6. Architecture families — view ================= */
const FAM_TABLE=[
  ['enc','Encoder-only','Bidirectional self-attention','Yes — all tokens','Contextual vectors → task head','Masked LM','BERT, RoBERTa, DeBERTa'],
  ['dec','Decoder-only','Causal self-attention','No — only the past','Next-token logits','Next-token prediction','GPT, Llama, Qwen, Mistral'],
  ['encdec','Encoder–Decoder','Bidirectional (enc) + causal (dec) + cross','Encoder: yes · decoder: no','Target tokens conditioned on source','Span corruption / seq2seq','T5, BART, Whisper']
];
VIEWS.families={
  show(root,sub){
    F.key=FAMS.some(f=>f[0]===sub)?sub:F.key;F.sel=null;F.hov=null;
    const m=FMETA[F.key];
    root.innerHTML=`<div class="toolbar" style="margin-top:18px"><nav class="seg" aria-label="Architecture family">${FAMS.map(([k,t])=>`<a href="#families/${k}" class="${k===F.key?'on':''}">${t}</a>`).join('')}</nav></div>
    ${secHead(''+m.name,m.sub)}
    ${F.key==='modern'?`<div class="card pad toolbar"><span class="lbl">MLP</span>${segHTML('f-ffn',[['swiglu','SwiGLU (dense)'],['moe','MoE (sparse)']],F.ffn)}<label class="chk"><input type="checkbox" id="f-local" ${F.local?'checked':''}> sliding-window attention</label></div>`:''}
    <div class="grid2" style="grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);align-items:start">
      <div class="card dia"><div id="f-svg" class="svgwrap" style="min-width:520px"></div></div>
      <div><div class="card pad insp" id="f-insp"></div><div class="card pad"><h3>Attention mask</h3><div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:10px">${m.masks.map(([l,k])=>maskSVG(k,l)).join('')}</div></div>
        <div class="card pad"><h3>Models</h3><p>${m.models.map(x=>`<span class="chip">${x}</span>`).join('')}</p><h3 style="margin-top:10px">Typical tasks</h3><p>${m.tasks.map(x=>`<span class="chip">${x}</span>`).join('')}</p><h3 style="margin-top:10px">Training</h3><p style="font-size:14px;color:var(--txt2)">${m.train}</p></div></div>
    </div>
    ${F.key==='dense'||F.key==='modern'?this.evolution():''}
    <div class="card pad" style="overflow-x:auto"><h3 style="margin-bottom:8px">The three families side by side</h3><table><thead><tr><th>Family</th><th>Attention pattern</th><th>Sees the future?</th><th>Output</th><th>Pre-training</th><th>Examples</th></tr></thead><tbody>
      ${FAM_TABLE.map(r=>`<tr class="${r[0]===F.key||(r[0]==='dec'&&(F.key==='dense'||F.key==='modern'))?'hl':''}"><td><b>${r[1]}</b></td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td><td>${r[5]}</td><td>${r[6]}</td></tr>`).join('')}</tbody></table></div>`;
    const svg=$('#f-svg');
    bindNodes(svg,F,()=>this.focus());
    if(F.key==='modern'){
      segBind($('#f-ffn'),v=>{F.ffn=v;this.update();});
      $('#f-local').onchange=e=>{F.local=e.target.checked;this.update();};
    }
    this.update();
  },
  update(){
    const fd=famData(F.key);this.fd=fd;
    if($('#f-ffn'))segSet($('#f-ffn'),F.ffn);
    $('#f-svg').innerHTML=stackSVG(fd.cols,F.sel||F.hov,fd.cross);
    this.focus();
  },
  focus(){
    const b=findItem(this.fd.cols,F.sel||F.hov);
    $('#f-insp').innerHTML=inspHTML(b?{t:b.label.replace(/<[^>]+>/g,''),c:b.col,d:b.d,r:b.shape?[['Shape',esc(b.shape)]]:[]}:null,`<b>${FMETA[F.key].name}</b> — hover or click a block. Green dashed arcs are the residual (skip) connections; the dashed box repeats N times.`);
  },
  evolution(){
    const rows=[
      ['Normalisation','LayerNorm (mean + variance, γ and β)','RMSNorm (RMS only, γ)','Cheaper, equally stable'],
      ['Position','Learned absolute vectors added at the input','RoPE rotating Q and K inside attention','Relative distance; better length extrapolation'],
      ['Attention','MHA — H K/V heads','GQA — H<sub>kv</sub> < H K/V heads (optionally windowed)','Much smaller KV cache, faster decoding'],
      ['MLP','GELU, D → 4D → D (2 matrices)','SwiGLU, D → M → D (3 matrices), or MoE','Better quality per FLOP; MoE adds capacity'],
      ['Biases','Linear layers have biases','No biases','Simpler, slightly faster'],
      ['Embeddings','Tied input/output embeddings','Tied only in small models','Large models benefit from separate LM head']
    ];
    return `<div class="card pad" style="overflow-x:auto"><h3 style="margin-bottom:8px">Classic dense decoder → modern decoder</h3><table><thead><tr><th>Component</th><th>Classic (GPT-2)</th><th>Modern (Llama / Qwen / Mistral)</th><th>Why it changed</th></tr></thead><tbody>${rows.map(r=>`<tr><td><b>${r[0]}</b></td><td>${r[1]}</td><td>${r[2]}</td><td class="mut">${r[3]}</td></tr>`).join('')}</tbody></table></div>`;
  }
};
