const test=require('node:test');const assert=require('node:assert/strict');
const {target,SmoothProgress}=require('../src/apexveil/web/progress.js');
test('estimated progress cannot complete a running or failed job',()=>{
 for(const state of ['queued','preparing','rendering'])assert.ok(target({state,progress:1,estimated_phase_seconds:1},99999)<100);
 assert.equal(target({state:'failed'},9999),null);
 assert.equal(target({state:'completed'},1),100);
});
test('progress moves smoothly, stays monotonic, survives late lower milestones',()=>{
 const p=new SmoothProgress();const job={id:'a',phase:'render',state:'rendering',estimated_phase_seconds:30,progress:0};
 let previous=1;
 for(let i=0;i<3000;i++){job.progress=i<200?0.8:0.1;const value=p.step(job,i/60,i*1000/60);assert.ok(value>=previous);assert.ok(value-previous<=.201);assert.ok(value<100);previous=value;}
 job.state='failed';assert.equal(p.step(job,100,51000),previous);
 job.state='completed';for(let t=51001;t<54000;t+=16)p.step(job,51,t);assert.equal(p.value,100);
});
test('a new processing phase starts at 1 rather than inherited detection 100',()=>{
 const p=new SmoothProgress();p.step({id:'a',phase:'prepare',state:'completed'},30,0);
 for(let t=16;t<4000;t+=16)p.step({id:'a',phase:'prepare',state:'completed'},30,t);
 assert.equal(p.value,100);
 assert.equal(p.step({id:'a',phase:'render',state:'queued'},0,4000),1);
});
