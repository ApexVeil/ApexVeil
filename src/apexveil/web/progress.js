/* Presentation-only progress; renderer state is the sole completion authority. */
(function(root){
  'use strict';
  function target(job, elapsed){
    if(job.state==='completed') return 100;
    if(job.state==='awaiting_selection') return 100;
    if(job.state==='queued') return 1;
    if(!['preparing','rendering'].includes(job.state)) return null;
    const estimate=Math.max(2,Number(job.estimated_phase_seconds)||60);
    const timed=1+98*(1-Math.exp(-Math.max(0,elapsed)/(estimate*.67)));
    const milestone=Number.isFinite(job.progress)?Math.max(0,Math.min(.97,job.progress))*96:0;
    return Math.min(99,Math.max(1,timed,milestone));
  }
  class SmoothProgress{
    constructor(){this.reset();}
    reset(){this.key=null;this.value=1;this.lastNow=null;}
    step(job,elapsed,now){
      const key=job.id+':'+(job.phase||(job.state==='rendering'?'render':'prepare'));
      if(this.key!==key){this.key=key;this.value=1;this.lastNow=now;}
      const dt=Math.max(0,Math.min(.25,(now-this.lastNow)/1000));this.lastNow=now;
      const wanted=target(job,elapsed);
      if(wanted===null) return this.value; // Failure never turns into 100%.
      if(['completed','awaiting_selection'].includes(job.state)){
        this.value=Math.min(100,this.value+Math.max(40,(100-this.value)*8)*dt);
      }else{
        const gap=Math.max(0,wanted-this.value);
        this.value=Math.min(99,this.value+Math.min(12,gap*3)*dt);
      }
      return this.value;
    }
  }
  const api={target,SmoothProgress};
  if(typeof module==='object'&&module.exports)module.exports=api;
  else root.ApexProgress=api;
})(typeof globalThis!=='undefined'?globalThis:this);
