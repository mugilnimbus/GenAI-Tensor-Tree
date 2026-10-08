VIEWS.graph={show(root,sub){if(ARCH.some(a=>a.id===sub))K.a=sub;
  root.innerHTML=secHead('Tensor Graph','Every tensor of an architecture as one connected graph: rectangles are tensors drawn to scale, pills are operations, arrows carry tensors between them. Hover for details, click for a worked example, right-click (vertical layout) to trace paths.')+LEGEND+'<div id="g-host"></div>';
  TFV.show();},update(){TFV.show();},hide(){if(TFV.hide)TFV.hide();}};
