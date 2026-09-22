/* UniTS (Gao et al., NeurIPS 2024) in the browser: the general-purpose encoder
 * of comparable size the OTIS paper uses as its real-world comparison.
 *
 * One static ONNX graph per input shape, fed the same
 * preprocessed signal OTIS receives plus the dataset's prompt and CLS tokens.
 * The paper's frozen-encoder evaluation draws those tokens at random for a
 * dataset UniTS never saw; the page reuses the seeded draw from the manifest so
 * the browser and the reference script embed with identical tokens.
 */
(function (global) {
  'use strict';

  const state = { meta: null, sessions: {}, precision: null };

  async function loadMeta() {
    if (state.meta) return state.meta;
    state.meta = await (await global.OTIS.fetchFirst('units_meta.json')).json();
    return state.meta;
  }

  /** Load the graph for one (V, T) shape; int8 first, float32 if the runtime rejects it. */
  async function loadModel(graph, onProgress, signal) {
    if (state.sessions[graph]) return state.sessions[graph];
    const meta = await loadMeta();
    const g = meta.graphs[graph];
    if (!g) throw new Error(`no UniTS graph for shape ${graph}`);
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'error';
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    let order = [['int8', 'int8'], ['float32', 'float32']];
    if (global.OTIS.state.precisionPref === 'float32' || g.default === 'float32') order = [['float32', 'float32']];
    for (const [file, label] of order) {
      try {
        onProgress && onProgress('downloading UniTS…');
        const buf = await (await global.OTIS.fetchFirst(g.files[file], signal)).arrayBuffer();
        onProgress && onProgress('starting the runtime…');
        state.sessions[graph] = await ort.InferenceSession.create(buf, opts);
        state.precision = label;
        return state.sessions[graph];
      } catch (err) {
        if (err && err.name === 'AbortError') throw err;   // stopped: do not fall back to the larger float32 graph
        console.warn(`[units] ${label} model unavailable:`, err);
        if (file === 'float32') throw err;
      }
    }
  }

  /** One vector per sample: the mean over every token of every variate (the paper's pooling).
   *  `tokens` is the (V, prompt_num + 1, d_model) draw from the manifest, CLS last. */
  async function encode(graph, signal, V, T, tokens) {
    const sess = state.sessions[graph], meta = state.meta;
    const pn = meta.prompt_num, D = meta.d_model;
    const prompt = new Float32Array(V * pn * D), cls = new Float32Array(V * D);
    for (let v = 0; v < V; v++) {
      prompt.set(tokens.subarray(v * (pn + 1) * D, (v * (pn + 1) + pn) * D), v * pn * D);
      cls.set(tokens.subarray((v * (pn + 1) + pn) * D, (v + 1) * (pn + 1) * D), v * D);
    }
    const t0 = performance.now();
    const { feat } = await sess.run({
      x: new ort.Tensor('float32', signal, [1, V, T]),
      prompt: new ort.Tensor('float32', prompt, [1, V, pn, D]),
      cls: new ort.Tensor('float32', cls, [1, V, 1, D])
    });
    return { feat: feat.data, encMs: performance.now() - t0 };
  }

  /* ---------------- generative: the pre-training path with a caller-chosen mask ---------------- */

  function rng(seed) {   // mulberry32, as in otis.js
    let a = seed >>> 0;
    return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  /** The tokens UniTS.Model.__init__ draws for a dataset it has not seen: prompt (normal, std .02),
   *  mask token (zeros), CLS (normal, std .02). Layout (V, prompt_num + 2, d_model), mask then CLS last. */
  function randomTokens(V, seed) {
    const meta = state.meta, pn = meta.prompt_num, D = meta.d_model, r = rng(seed);
    const out = new Float32Array(V * (pn + 2) * D);
    for (let v = 0; v < V; v++) for (let t = 0; t < pn + 2; t++) {
      if (t === pn) continue;                       // the mask token stays zero
      for (let j = 0; j < D; j++) out[(v * (pn + 2) + t) * D + j] = gauss(r) * meta.token_std;
    }
    return out;
  }

  /** The (V, prompt_num + 2, d_model) draw above in `encode`'s (V, prompt_num + 1, d_model) layout: the mask token dropped, CLS last. */
  function encodeTokens(tokens) {
    const meta = state.meta, pn = meta.prompt_num, D = meta.d_model, V = tokens.length / ((pn + 2) * D);
    const out = new Float32Array(V * (pn + 1) * D);
    for (let v = 0; v < V; v++) {
      out.set(tokens.subarray(v * (pn + 2) * D, (v * (pn + 2) + pn) * D), v * (pn + 1) * D);
      out.set(tokens.subarray((v * (pn + 2) + pn + 1) * D, (v + 1) * (pn + 2) * D), (v * (pn + 1) + pn) * D);
    }
    return out;
  }

  /** The tokens prompt-tuned on one source (sine, chirp or square) from 20 to 120 Hz, or null if
   *  not published: units_<name>_tokens.bin, one file per source like OTIS' fitted embeddings. */
  async function loadTokens(name = 'sine') {
    state.tokens = state.tokens || {};
    if (state.tokens[name] !== undefined) return state.tokens[name];
    try {
      const buf = await (await global.OTIS.fetchFirst(`units_${name}_tokens.bin`)).arrayBuffer();
      state.tokens[name] = new Float32Array(buf);
    } catch (err) {
      console.warn(`[units] prompt-tuned ${name} tokens unavailable:`, err);
      state.tokens[name] = null;
    }
    return state.tokens[name];
  }
  const loadSineTokens = () => loadTokens('sine');

  /** Masked modelling on UniTS' own 16-step grid: `mask` has one entry per patch, 1 = hidden.
   *  Returns the reconstructed signal (V*T) in the input's units. */
  async function generate(graph, signal, V, T, mask, tokens) {
    const sess = state.sessions[graph], meta = state.meta;
    const pn = meta.prompt_num, D = meta.d_model, L = mask.length;
    const prompt = new Float32Array(V * pn * D), maskTok = new Float32Array(V * D), cls = new Float32Array(V * D);
    for (let v = 0; v < V; v++) {
      const base = v * (pn + 2) * D;
      prompt.set(tokens.subarray(base, base + pn * D), v * pn * D);
      maskTok.set(tokens.subarray(base + pn * D, base + (pn + 1) * D), v * D);
      cls.set(tokens.subarray(base + (pn + 1) * D, base + (pn + 2) * D), v * D);
    }
    const m = new Float32Array(L);
    for (let i = 0; i < L; i++) m[i] = mask[i] ? 1 : 0;
    const t0 = performance.now();
    const { pred } = await sess.run({
      x: new ort.Tensor('float32', signal, [1, V, T]),
      mask: new ort.Tensor('float32', m, [1, L]),
      prompt: new ort.Tensor('float32', prompt, [1, V, pn, D]),
      mask_token: new ort.Tensor('float32', maskTok, [1, V, 1, D]),
      cls: new ort.Tensor('float32', cls, [1, V, 1, D])
    });
    return { pred: pred.data, encMs: performance.now() - t0 };
  }

  /** The fine-tuned forecasting graphs: the context in, the horizon out. */
  async function forecast(graph, context, V, ctx) {
    const sess = state.sessions[graph];
    const t0 = performance.now();
    const { pred } = await sess.run({ x: new ort.Tensor('float32', context, [1, V, ctx]) });
    return { pred: pred.data, encMs: performance.now() - t0 };
  }

  global.UNITS = { state, loadMeta, loadModel, encode, randomTokens, encodeTokens, loadTokens, loadSineTokens, generate, forecast };
})(window);
