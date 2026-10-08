/* ================= sources =================
   REFS: primary papers and reports (arXiv ids were checked against the arXiv API when this file was written).
   SRC:  which references back each tree entry / architecture. Shown at the bottom of every page and on the Sources page. */
const ax=id=>'https://arxiv.org/abs/'+id;
const REFS={
  transformer:['Attention Is All You Need','Vaswani et al., 2017',ax('1706.03762')],
  bert:['BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding','Devlin et al., 2018',ax('1810.04805')],
  albert:['ALBERT: A Lite BERT for Self-supervised Learning of Language Representations','Lan et al., 2019',ax('1909.11942')],
  t5:['Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)','Raffel et al., 2019',ax('1910.10683')],
  gpt2:['Language Models are Unsupervised Multitask Learners (GPT-2)','Radford et al., 2019','https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf'],
  llama:['LLaMA: Open and Efficient Foundation Language Models','Touvron et al., 2023',ax('2302.13971')],
  llama2:['Llama 2: Open Foundation and Fine-Tuned Chat Models','Touvron et al., 2023',ax('2307.09288')],
  mistral:['Mistral 7B','Jiang et al., 2023',ax('2310.06825')],
  mixtral:['Mixtral of Experts','Jiang et al., 2024',ax('2401.04088')],
  qwen25:['Qwen2.5 Technical Report','Qwen team, 2024',ax('2412.15115')],
  dsv2:['DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model','DeepSeek-AI, 2024',ax('2405.04434')],
  dsv3:['DeepSeek-V3 Technical Report','DeepSeek-AI, 2024',ax('2412.19437')],
  rope:['RoFormer: Enhanced Transformer with Rotary Position Embedding','Su et al., 2021',ax('2104.09864')],
  alibi:['Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)','Press et al., 2021',ax('2108.12409')],
  rmsnorm:['Root Mean Square Layer Normalization','Zhang & Sennrich, 2019',ax('1910.07467')],
  layernorm:['Layer Normalization','Ba et al., 2016',ax('1607.06450')],
  swiglu:['GLU Variants Improve Transformer (SwiGLU)','Shazeer, 2020',ax('2002.05202')],
  gelu:['Gaussian Error Linear Units (GELUs)','Hendrycks & Gimpel, 2016',ax('1606.08415')],
  swish:['Searching for Activation Functions (Swish / SiLU)','Ramachandran et al., 2017',ax('1710.05941')],
  gqa:['GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints','Ainslie et al., 2023',ax('2305.13245')],
  mqa:['Fast Transformer Decoding: One Write-Head is All You Need (multi-query attention)','Shazeer, 2019',ax('1911.02150')],
  longformer:['Longformer: The Long-Document Transformer','Beltagy et al., 2020',ax('2004.05150')],
  flash:['FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness','Dao et al., 2022',ax('2205.14135')],
  moe17:['Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer','Shazeer et al., 2017',ax('1701.06538')],
  switch:['Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity','Fedus et al., 2021',ax('2101.03961')],
  lora:['LoRA: Low-Rank Adaptation of Large Language Models','Hu et al., 2021',ax('2106.09685')],
  qlora:['QLoRA: Efficient Finetuning of Quantized LLMs','Dettmers et al., 2023',ax('2305.14314')],
  dora:['DoRA: Weight-Decomposed Low-Rank Adaptation','Liu et al., 2024',ax('2402.09353')],
  adapters:['Parameter-Efficient Transfer Learning for NLP (adapters)','Houlsby et al., 2019',ax('1902.00751')],
  prefix:['Prefix-Tuning: Optimizing Continuous Prompts for Generation','Li & Liang, 2021',ax('2101.00190')],
  prompt:['The Power of Scale for Parameter-Efficient Prompt Tuning','Lester et al., 2021',ax('2104.08691')],
  ia3:['Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning ((IA)³)','Liu et al., 2022',ax('2205.05638')],
  bitfit:['BitFit: Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models','Ben-Zaken et al., 2021',ax('2106.10199')],
  mamba:['Mamba: Linear-Time Sequence Modeling with Selective State Spaces','Gu & Dao, 2023',ax('2312.00752')],
  rwkv:['RWKV: Reinventing RNNs for the Transformer Era','Peng et al., 2023',ax('2305.13048')],
  gdn:['Gated Delta Networks: Improving Mamba2 with Delta Rule','Yang et al., 2024',ax('2412.06464')],
  deltanet:['Parallelizing Linear Transformers with the Delta Rule over Sequence Length','Yang et al., 2024',ax('2406.06484')],
  specdec:['Fast Inference from Transformers via Speculative Decoding','Leviathan et al., 2022',ax('2211.17192')],
  nucleus:['The Curious Case of Neural Text Degeneration (nucleus / top-p sampling)','Holtzman et al., 2019',ax('1904.09751')],
  vit:['An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)','Dosovitskiy et al., 2020',ax('2010.11929')],
  clip:['Learning Transferable Visual Models From Natural Language Supervision (CLIP)','Radford et al., 2021',ax('2103.00020')],
  llava:['Visual Instruction Tuning (LLaVA)','Liu et al., 2023',ax('2304.08485')],
  llava15:['Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)','Liu et al., 2023',ax('2310.03744')],
  ldm:['High-Resolution Image Synthesis with Latent Diffusion Models','Rombach et al., 2021',ax('2112.10752')],
  ddpm:['Denoising Diffusion Probabilistic Models','Ho et al., 2020',ax('2006.11239')],
  cfg:['Classifier-Free Diffusion Guidance','Ho & Salimans, 2022',ax('2207.12598')],
  dit:['Scalable Diffusion Models with Transformers (DiT)','Peebles & Xie, 2022',ax('2212.09748')],
  flow:['Flow Matching for Generative Modeling','Lipman et al., 2022',ax('2210.02747')],
  vqvae:['Neural Discrete Representation Learning (VQ-VAE)','van den Oord et al., 2017',ax('1711.00937')],
  vqgan:['Taming Transformers for High-Resolution Image Synthesis (VQGAN)','Esser et al., 2020',ax('2012.09841')],
  soundstream:['SoundStream: An End-to-End Neural Audio Codec (residual VQ)','Zeghidour et al., 2021',ax('2107.03312')],
  encodec:['High Fidelity Neural Audio Compression (EnCodec)','Défossez et al., 2022',ax('2210.13438')],
  vae:['Auto-Encoding Variational Bayes','Kingma & Welling, 2013',ax('1312.6114')],
  gan:['Generative Adversarial Networks','Goodfellow et al., 2014',ax('1406.2661')],
  mdlm:['Simple and Effective Masked Diffusion Language Models','Sahoo et al., 2024',ax('2406.07524')],
  d3pm:['Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)','Austin et al., 2021',ax('2107.03006')],
  llada:['Large Language Diffusion Models (LLaDA)','Nie et al., 2025',ax('2502.09992')],
  chinchilla:['Training Compute-Optimal Large Language Models','Hoffmann et al., 2022',ax('2203.15556')],
  qnextblog:['Qwen3.8-Flash-Next: blog post','Qwen team, 2026','https://qwen.ai/blog?id=qwen3.8-flash-next'],
  qnextpaper:['On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability','Qwen team, 2026',ax('2608.30320')],
  qnextsglang:['Qwen3.8-Flash-Next: Day-0 Support in SGLang','LMSYS, 2026','https://www.lmsys.org/blog/2026-08-26-qwen-flash-next/'],
  jev:['Jev model entry (secondary source; TypeSafe AI has not published the architecture)','DataLearner, 2026','https://www.datalearner.com/en/ai-models/pretrained-models/jev']
};
const SRC={
  layer:['transformer','llama','rmsnorm','rope','swiglu','swish'],model:['transformer','llama','nucleus'],residual4:['qnextpaper','qnextsglang'],
  mha:['transformer','llama2'],gqa:['gqa','llama2','mistral','qwen25'],mqa:['mqa','gqa'],swa:['longformer','mistral'],mla:['dsv2','dsv3'],qsa:['qnextpaper','qnextsglang'],
  gdn:['gdn','deltanet','qnextpaper'],ssm:['mamba'],rwkv:['rwkv'],dense:['gpt2','transformer','layernorm','gelu'],modern:['llama','llama2','rmsnorm','rope','swiglu','gqa'],
  moe:['moe17','switch','mixtral','dsv2'],decode:['transformer','flash'],ngram:['qnextblog','qnextpaper','qnextsglang'],enc:['bert','albert'],encdec:['transformer','t5'],dec:['gpt2','llama','llama2','qwen25'],
  qnext:['qnextblog','qnextpaper','qnextsglang','gdn'],jev:['jev'],difflm:['mdlm','d3pm','llada'],prefill:['flash','gqa'],mtp:['specdec','dsv3','qnextsglang'],
  fullft:['bitfit','bert'],lora:['lora'],qlora:['qlora','lora'],dora:['dora','lora'],adapters:['adapters'],prefix:['prefix','prompt'],prompt:['prompt'],ia3:['ia3'],
  multimodal:['vit','clip','llava','llava15'],vqtok:['vqvae','vqgan','soundstream','encodec'],imgdiff:['ldm','ddpm','cfg','dit','flow'],vaegan:['vae','gan'],
  gpt2:['gpt2','layernorm','gelu'],bert:['bert'],alibi:['alibi'],vit:['vit','llava15'],dit:['dit','ldm'],vq:['vqvae','vqgan']
};
const refChip=k=>{const r=REFS[k];return r?`<a class="ref" href="${r[2]}" target="_blank" rel="noopener"><b>${r[0]}</b><small>${r[1]} ↗</small></a>`:'';};
function sourcesHTML(ids){
  const keys=[...new Set(ids.filter(Boolean).flatMap(i=>SRC[i]||[]))];
  return keys.length?`<div class="card pad srcs"><h3>Sources</h3><div class="refs">${keys.map(refChip).join('')}</div><p class="mut" style="font-size:12px;margin-top:8px">Primary papers and reports for what is on this page. <a href="#refs">All sources →</a></p></div>`:'';
}
VIEWS.refs={show(root){
  root.innerHTML=secHead('Sources & papers','Every reference used in this atlas. arXiv identifiers were checked against the arXiv listing when the page was built; other links are the publishers’ own pages.')+
    `<div class="card pad"><div class="refs wide">${Object.keys(REFS).map(refChip).join('')}</div></div>`;
}};
