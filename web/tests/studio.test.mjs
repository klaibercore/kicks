import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

function load(file, globals = {}) {
  const source = readFileSync(new URL(`../${file}`, import.meta.url), 'utf8');
  const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 } }).outputText;
  const exports = {};
  vm.runInNewContext(code, { exports, URLSearchParams, require: () => ({ config: { apiUrl: 'http://localhost:8080' } }), ...globals });
  return exports;
}

class FakeGain {
  constructor() {
    this.events = [];
    this.gain = { value: 1, cancelScheduledValues: (t) => this.events.push(['cancel', t]),
      setValueAtTime: (v,t) => this.events.push(['set',v,t]), linearRampToValueAtTime: (v,t) => this.events.push(['ramp',v,t]) };
  }
  connect() {} disconnect() { this.disconnected = true; }
}
class FakeContext {
  constructor() { this.state = 'running'; this.currentTime = 1; this.gains = []; this.sources = []; }
  createGain() { const g = new FakeGain(); this.gains.push(g); return g; }
  createBufferSource() {
    const source = { connect() {}, disconnect() { this.disconnected = true; }, start() { this.started = true; }, stop(t) { this.stoppedAt = t; } };
    this.sources.push(source); return source;
  }
}

test('choke groups release only the previous grouped voice and clean up ended nodes', () => {
  const { engine } = load('lib/audio/engine.ts', { window: { AudioContext: FakeContext } });
  const kick = engine.play({}, 1);
  const open = engine.play({}, .8, 'hihat');
  const closed = engine.play({}, 1, 'hihat');
  assert.equal(open.stoppedAt, 1.006);
  assert.equal(kick.stoppedAt, undefined);
  open.onended(); // An old completion must not remove the newer grouped voice.
  const next = engine.play({}, 1, 'hihat');
  assert.equal(closed.stoppedAt, 1.006);
  assert.equal(open.disconnected, true);
  engine.stopAll();
  assert.equal(next.stoppedAt, 1.006);
  assert.equal(kick.stoppedAt, 1.006);
});

test('preview cache stays bounded and repeated sounds retain their slot', () => {
  const { engine } = load('lib/audio/engine.ts', { window: { AudioContext: FakeContext } });
  for (let n = 0; n < 256; n++) engine.remember(String(n), { n });
  engine.remember('255', { n: 500 });
  assert.equal(engine.cached('0').n, 0);
  engine.remember('new', {});
  assert.equal(engine.cached('0'), undefined);
  assert.equal(engine.cached('255').n, 500);
});

test('render queries preserve engine identity for old VAE and new DSP presets', () => {
  const { soundQuery } = load('lib/api/client.ts');
  const base = { instrument: 'hihat', sliders: { voice_decay_ms: .4 }, effects: {} };
  assert.equal(new URLSearchParams(soundQuery(base)).get('backend'), 'vae');
  assert.equal(new URLSearchParams(soundQuery({ ...base, backend: 'dsp' })).get('backend'), 'dsp');
  assert.equal(new URLSearchParams(soundQuery({ ...base, backend: 'diffusion', steps: 25 })).get('steps'), '25');
  assert.equal(new URLSearchParams(soundQuery({ ...base, backend: 'codec', steps: 10 })).get('steps'), '10');
  for (const backend of ['parameters','ddsp','noiseband','hybrid','drumgan','wavevae']) {
    assert.equal(new URLSearchParams(soundQuery({ ...base, backend })).get('backend'), backend);
  }
});

test('kit import rejects bad seeds, control values and unsupported engines', () => {
  const { emptyKit, parseKit } = load('lib/pads.ts');
  const kit = emptyKit('Test');
  kit.pads[0].sound = { instrument: 'kick', backend: 'dsp', sliders: { voice_tune_hz: .5 }, effects: {}, seed: 3 };
  assert.equal(parseKit(kit).pads[0].sound.backend, 'dsp');
  for (const backend of ['parameters','ddsp','noiseband','hybrid','drumgan','codec','wavevae']) {
    kit.pads[0].sound.backend = backend;
    assert.equal(parseKit(kit).pads[0].sound.backend, backend);
  }
  kit.pads[0].sound.seed = -1;
  assert.throws(() => parseKit(kit));
  kit.pads[0].sound.seed = 1;
  kit.pads[0].sound.sliders.voice_tune_hz = NaN;
  assert.throws(() => parseKit(kit));
  kit.pads[0].sound.sliders.voice_tune_hz = .5;
  kit.pads[0].sound.backend = 'unavailable';
  assert.throws(() => parseKit(kit));
});
