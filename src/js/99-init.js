/* ================= init ================= */
(function init(){
  $('#logo').innerHTML=svgWrap(38,34,cube(12,22,12,12,9,C.blue)+cube(26,16,10,10,8,C.purple)+cube(22,28,10,8,7,C.orange),'width="38" height="34"');
  buildCfg();renderNav();
  window.addEventListener('hashchange',route);
  route();
})();
