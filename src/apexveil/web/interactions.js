/* Studio controls and presentation. No image processing runs in the browser. */
const effortNames={low:'Low',medium:'Medium',high:'High',xhigh:'Extra High',max:'Max'};
let advisorEfforts=['low','medium','high','xhigh','max'];
let progressAnimation=0,progressSample=null,progressReceipt=0,progressStateKey=null;
let progressWasActive=false,progressFinishing=false,progressFinishUntil=0,connectionHealthy=true;
const progressMotion=new ApexProgress.SmoothProgress();

function advisorPayload(){return {ai_model:$('aiModel').value,ai_effort:$('aiEffort').value||'medium'};}
function updateAdvisorLabel(){
  const effort=advisorPayload().ai_effort;
  $('aiEffort').setAttribute('aria-label','思考等级');
  $('aiEffort').title=effortNames[effort]||'Medium';
}
function setAdvisor(model,effort){
  const options=config?.ai_models||[];
  const chosen=options.find(m=>m.id===model)||options.find(m=>m.id==='gpt-6.1-sol')||options[0];
  if(!chosen)return;
  $('aiModel').value=chosen.id;
  advisorEfforts=chosen.efforts;
  $('aiEffort').replaceChildren(...advisorEfforts.map(e=>new Option(effortNames[e],e)));
  $('aiEffort').value=advisorEfforts.includes(effort)?effort:'medium';
  updateAdvisorLabel();
}
function initAdvisor(){
  $('aiModel').replaceChildren(...(config.ai_models||[]).map(m=>new Option(m.label,m.id)));
  let preferred=config.ai_defaults||{model:'gpt-6.1-sol',effort:'medium'};
  try{const saved=JSON.parse(localStorage.getItem('apexveil.advisor.v1'));if(saved)preferred=saved;}catch{}
  setAdvisor(preferred.model,preferred.effort);
}
function restoreAdvisor(record){
  setAdvisor(record.ai_model||config.ai_defaults?.model||'gpt-6.1-sol',record.ai_effort||'medium');
}
function changeAdvisor(){
  updateAdvisorLabel();
  try{localStorage.setItem('apexveil.advisor.v1',JSON.stringify({model:advisorPayload().ai_model,effort:advisorPayload().ai_effort}));}catch{}
  renderSettingsChanged();
}
function syncAdvisorControls(){
  const enabled=$('aiAssist').checked&&mode==='replace'&&Boolean(config?.ai_available);
  $('advisorCard').hidden=!enabled;
  for(const id of ['aiModel','aiEffort','resetAdvisor'])$(id).disabled=busy||uploading||!enabled;
}

function initTooltips(){
  let anchor=null,pinned=false;
  const tip=$('studioTooltip');
  function close(){if(anchor)anchor.setAttribute('aria-expanded','false');anchor=null;pinned=false;tip.hidden=true;}
  function show(button){
    if(anchor&&anchor!==button)anchor.setAttribute('aria-expanded','false');
    anchor=button;tip.textContent=button.dataset.tip;tip.hidden=false;button.setAttribute('aria-expanded','true');
    const r=button.getBoundingClientRect(),w=tip.offsetWidth,h=tip.offsetHeight;
    let left=r.left-w-10;if(left<10)left=Math.min(innerWidth-w-10,r.right+10);
    tip.style.left=Math.max(10,left)+'px';tip.style.top=Math.max(10,Math.min(innerHeight-h-10,r.top-h/2+r.height/2))+'px';
  }
  for(const button of document.querySelectorAll('[data-tip]')){
    button.setAttribute('aria-describedby','studioTooltip');button.setAttribute('aria-expanded','false');
    button.addEventListener('pointerenter',()=>show(button));
    button.addEventListener('pointerleave',()=>{if(!pinned&&document.activeElement!==button)close();});
    button.addEventListener('focus',()=>show(button));
    button.addEventListener('blur',()=>{if(!pinned)close();});
    button.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();if(pinned&&anchor===button)close();else{show(button);pinned=true;}});
  }
  document.addEventListener('keydown',event=>{if(event.key==='Escape')close();});
  document.addEventListener('pointerdown',event=>{if(!event.target.closest('[data-tip]')&&!tip.contains(event.target))close();});
  window.addEventListener('resize',close);
  document.addEventListener('scroll',close,true);
}
function clockText(seconds){const s=Math.max(0,Math.floor(seconds));return Math.floor(s/60).toString().padStart(2,'0')+':'+(s%60).toString().padStart(2,'0');}
function progressLabel(record){
  if(record.state==='queued')return '等待处理';
  if(record.state==='completed')return '处理完成';
  if(record.state==='awaiting_selection')return '识别完成';
  const message=record.message||'';
  if(/[\u4e00-\u9fff]/.test(message))return message;
  if(/guidance|advis|AI/i.test(message))return 'AI 正在分析场景';
  if(/track|plate candidates|detect/i.test(message))return '正在识别与跟踪车牌';
  if(/geometry|trajectory|plane/i.test(message))return '正在匹配透视与时序';
  if(/export|encod|audio|ready/i.test(message))return '正在整理处理结果';
  return record.phase==='prepare'?'正在读取与识别素材':'正在融合铭牌与场景';
}
function observeProgress(record){
  const active=['queued','preparing','rendering'].includes(record.state);
  const key=record.id+':'+(record.phase||(record.state==='rendering'?'render':'prepare'))+':'+(record.render_revision||0);
  if(active&&key!==progressStateKey){progressMotion.reset();progressStateKey=key;}
  progressSample=record;progressReceipt=performance.now();connectionHealthy=true;
  const panel=document.querySelector('.canvas-panel');
  if(active){progressWasActive=true;progressFinishing=false;progressFinishUntil=0;panel.classList.add('is-processing');panel.classList.remove('process-complete');$('processOverlay').hidden=false;panel.setAttribute('aria-busy','true');}
  else if(progressWasActive&&['completed','awaiting_selection'].includes(record.state)){
    progressFinishing=true;progressWasActive=false;panel.classList.add('process-complete');
  }else if(record.state==='failed'||record.state==='cancelled')resetProgressUI();
  if((active||progressFinishing)&&!progressAnimation)progressAnimation=requestAnimationFrame(tickProgress);
}
function tickProgress(now){
  progressAnimation=0;if(!progressSample)return;
  const r=progressSample,active=['queued','preparing','rendering'].includes(r.state);
  if(!active&&!progressFinishing)return;
  const elapsed=r.state==='queued'?0:(Number(r.phase_elapsed_seconds)||0)+(active?(now-progressReceipt)/1000:0);
  const value=connectionHealthy?progressMotion.step(r,elapsed,now):progressMotion.value;
  const pct=Math.floor(Math.min(active?99:100,value));
  const label=connectionHealthy?progressLabel(r):'连接恢复中，任务状态待同步';
  if($('processLabel').textContent!==label)$('processLabel').textContent=label;
  $('processPercent').textContent=pct+'%';$('progressNumber').textContent=(active?'预估 ':'')+pct+'%';
  $('processElapsed').textContent=r.state==='queued'?'排队中 · 尚未开始':(active?'预估进度 · 已用 ':'已用 ')+clockText(elapsed);
  $('progressTrack').classList.remove('indeterminate');$('progressBar').style.width=pct+'%';
  document.querySelector('.canvas-panel').classList.toggle('is-queued',r.state==='queued'||!connectionHealthy);
  if(progressFinishing&&value>=99.95){
    $('processPercent').textContent='100%';$('progressNumber').textContent='100%';$('progressBar').style.width='100%';
    if(!progressFinishUntil)progressFinishUntil=now+500;
    if(now>=progressFinishUntil){resetProgressUI();return;}
  }
  progressAnimation=requestAnimationFrame(tickProgress);
}
function resetProgressUI(){
  if(progressAnimation)cancelAnimationFrame(progressAnimation);progressAnimation=0;
  progressSample=null;progressWasActive=false;progressFinishing=false;progressFinishUntil=0;progressStateKey=null;progressMotion.reset();
  const panel=document.querySelector('.canvas-panel');panel.classList.remove('is-processing','is-queued','process-complete');panel.setAttribute('aria-busy','false');$('processOverlay').hidden=true;
}
function initInteractions(){
  initTooltips();
  $('aiModel').onchange=()=>{const effort=advisorPayload().ai_effort;setAdvisor($('aiModel').value,effort);changeAdvisor();};
  $('aiEffort').oninput=updateAdvisorLabel;$('aiEffort').onchange=changeAdvisor;
  $('resetAdvisor').onclick=()=>{setAdvisor('gpt-6.1-sol','medium');changeAdvisor();};
  $('aiAssist').addEventListener('change',renderSettingsChanged);
}
