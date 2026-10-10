const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function element() {
  const attributes = new Map();
  const classes = new Set();
  return {
    children: [], dataset: {}, hidden: false, disabled: false, checked: false,
    value: '', textContent: '', style: {setProperty() {}},
    classList: {
      add(...names) { names.forEach(name => classes.add(name)); },
      remove(...names) { names.forEach(name => classes.delete(name)); },
      toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
      contains(name) { return classes.has(name); },
    },
    setAttribute(name, value) { attributes.set(name, value); },
    getAttribute(name) { return attributes.get(name) ?? null; },
    removeAttribute(name) { attributes.delete(name); },
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
    addEventListener() {}, pause() {},
  };
}

function loadStudio() {
  const nodes = new Map();
  const get = id => {
    if (!nodes.has(id)) nodes.set(id, element());
    return nodes.get(id);
  };
  get('modeButtons').children = ['replace', 'blur', 'mosaic'].map(mode => ({...element(), dataset: {mode}}));
  const scheduled = [];
  const context = vm.createContext({
    document: {
      getElementById: get, querySelector: get, querySelectorAll: () => [],
      createElement: element, createTextNode: text => text,
    },
    window: {addEventListener() {}},
    location: {pathname: '/', search: ''}, history: {replaceState() {}},
    performance: {now: () => 1000}, URLSearchParams,
    setTimeout(callback) { scheduled.push(callback); return scheduled.length; },
    clearTimeout() {}, requestAnimationFrame: () => 1, cancelAnimationFrame() {},
    Option: function(text, value) { return {...element(), textContent: text, value}; },
  });
  const web = path.join(__dirname, '../src/apexveil/web');
  for (const name of ['progress.js', 'interactions.js']) {
    vm.runInContext(fs.readFileSync(path.join(web, name), 'utf8'), context, {filename: name});
  }
  const html = fs.readFileSync(path.join(web, 'index.html'), 'utf8');
  const inline = [...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)].find(match => !match[1].includes('src='))[2];
  // Execute the shipped handlers without starting its network-backed bootstrap.
  vm.runInContext(inline.slice(0, inline.indexOf('\ninitInteractions();')), context, {filename: 'index.html'});
  vm.runInContext("config={ai_available:true,default_template:false,ai_models:[{id:'gpt-6.1-sol',efforts:['medium']}],ai_defaults:{model:'gpt-6.1-sol',effort:'medium'}};", context);
  return {context, get, scheduled, run: source => vm.runInContext(source, context)};
}

const cancelledJob = {
  id: 'a'.repeat(32), state: 'cancelled', kind: 'image', media_id: 'b'.repeat(32),
  message: '已被新设置替代', phase: 'prepare', stage1: {state: 'ready'},
  public: {width: 100, height: 100, targets: [{id: 1, bbox: [5, 5, 20, 10]}]},
};

test('a superseded job unlocks importing and detection when restored or polled', async () => {
  for (const action of ['restoreJob()', 'poll(runToken)']) {
    const ui = loadStudio();
    ui.context.savedJob = structuredClone(cancelledJob);
    ui.context.location.search = '?job=' + cancelledJob.id;
    ui.run("request=async()=>savedJob;media={asset_id:savedJob.media_id,kind:'image'};job={...savedJob,state:'preparing'};busy=true;");
    await ui.run(action);
    assert.equal(ui.run('busy'), false);
    assert.equal(ui.get('importTop').disabled, false);
    assert.equal(ui.get('replaceMedia').disabled, false);
    assert.equal(ui.get('detect').disabled, false);
    assert.equal(ui.get('render').disabled, true);
    assert.equal(ui.get('selection').hidden, true);
    assert.equal(ui.get('activity').hidden, true);
    assert.equal(ui.get('processOverlay').hidden, true);
    assert.equal(ui.get('statePill').textContent, '已被替代');
    assert.equal(ui.scheduled.length, 0);
  }
});

test('restoring a custom template reveals it after the missing default preview failed', async () => {
  const ui = loadStudio();
  ui.context.savedJob = {...structuredClone(cancelledJob), template_id: 'c'.repeat(32), template_name: 'Custom artwork.png'};
  ui.context.location.search = '?job=' + cancelledJob.id;
  ui.get('templatePreview').onerror();
  assert.equal(ui.get('templatePreview').hidden, true);
  ui.run('request=async()=>savedJob;');
  await ui.run('restoreJob()');
  assert.equal(ui.get('templatePreview').hidden, false);
  assert.equal(ui.get('templatePlaceholder').hidden, true);
  assert.equal(ui.get('templatePreview').src, '/api/assets/' + 'c'.repeat(32));
  assert.equal(ui.get('templateButton').textContent, 'Custom artwork.png');
});

function readyToRender(ui) {
  ui.context.savedJob = {...structuredClone(cancelledJob), state: 'completed', phase: 'render', render_revision: 1};
  ui.run("job={...savedJob};media={asset_id:job.media_id,kind:'image'};selected=1;mode='mosaic';$('aiEffort').value='high';calls=[];");
}

test('a lost accepted render response follows the new revision without repeating POST', async () => {
  const ui = loadStudio();
  readyToRender(ui);
  ui.run("request=async(url,options={})=>{calls.push(options.method||'GET');if(options.method==='POST'){savedJob={...savedJob,state:'rendering',render_revision:2};throw new TypeError('network');}return {...savedJob};};");
  await ui.run('render()');
  await ui.run('Promise.resolve()');
  assert.deepEqual(Array.from(ui.run('calls')), ['POST', 'GET']);
  assert.equal(ui.run('job.render_revision'), 2);
  assert.equal(ui.run('job.state'), 'rendering');
  assert.equal(ui.run('pendingRender'), null);
  assert.equal(ui.run('busy'), true);
  assert.equal(ui.get('render').disabled, true);
  assert.equal(ui.scheduled.length, 1);
});

test('uncertain submission stays locked through a failed reconciliation GET and reconnects', async () => {
  const ui = loadStudio();
  readyToRender(ui);
  ui.run("getAttempts=0;request=async(url,options={})=>{calls.push(options.method||'GET');if(options.method==='POST'){savedJob={...savedJob,state:'rendering',render_revision:2};throw new TypeError('network');}if(++getAttempts===1)throw new TypeError('network');return {...savedJob};};");
  await ui.run('render()');
  await ui.run('Promise.resolve()');
  assert.equal(ui.run('busy'), true);
  assert.match(ui.get('error').textContent, /提交结果尚未确认/);
  assert.equal(ui.scheduled.length, 1);
  await ui.scheduled[0]();
  assert.deepEqual(Array.from(ui.run('calls')), ['POST', 'GET', 'GET']);
  assert.equal(ui.run('job.render_revision'), 2);
  assert.equal(ui.run('pendingRender'), null);
  assert.equal(ui.run('busy'), true);
});

test('unaccepted or rejected submissions keep the edited settings available for correction', async () => {
  for (const status of [null, 400]) {
    const ui = loadStudio();
    readyToRender(ui);
    ui.context.rejectionStatus = status;
    ui.run("request=async(url,options={})=>{calls.push(options.method||'GET');if(options.method==='POST'){if(rejectionStatus)throw httpError('请修改设置',rejectionStatus);throw new TypeError('network');}return {...savedJob};};");
    await ui.run('render()');
    await ui.run('Promise.resolve()');
    assert.deepEqual(Array.from(ui.run('calls')), status ? ['POST'] : ['POST', 'GET']);
    assert.equal(ui.run('busy'), false);
    assert.equal(ui.run('mode'), 'mosaic');
    assert.equal(ui.get('aiEffort').value, 'high');
    assert.equal(ui.get('render').disabled, false);
    assert.equal(ui.get('error').hidden, false);
    assert.equal(ui.scheduled.length, 0);
  }
});
