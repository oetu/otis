/* The zero-shot forecasters, Chronos-2 (Amazon, 120 M parameters) and TimesFM-3
 * (Google, 330 M), in the browser: one ONNX graph each
 * that forecasts 96 steps from up to 336 context steps, every variate given at
 * once; a mask input left-pads a shorter context the way the reference
 * pipelines do. forecastTail fills a hidden tail of any length: the visible
 * steps before it are the context, a longer tail is rolled 96 steps at a time,
 * each pass appended to the context as observed. The graphs are fetched when a
 * Fill-in row is run, since they weigh 131 MB and 344 MB (int8).
 *
 * Chronos-2 doubles as a representation encoder for the Explore panel: a second
 * graph (128 MB int8) takes a whole sample, any number
 * of variates and any length, and returns its embedding -- the encoder states of
 * the context patches averaged over the patches and then over the variates.
 */
(function (global) {
  'use strict';

  const state = { meta: null, sessions: {}, precision: {}, embed: null, embedPrecision: null };

  async function loadMeta() {
    if (state.meta) return state.meta;
    state.meta = await (await global.OTIS.fetchFirst('zero_shot_meta.json')).json();
    return state.meta;
  }

  /** Fetch and start one model's int8 graph, reporting download progress. */
  async function loadModel(key, onProgress, signal) {
    if (state.sessions[key]) return state.sessions[key];
    const meta = await loadMeta();
    const g = meta.models[key];
    if (!g) throw new Error(`no zero-shot graph for ${key}`);
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'error';
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const file = g.default || 'int8';
    onProgress && onProgress(`downloading the ${file} ${g.name} model (${g.sizes_mb[file]} MB)…`);
    const res = await global.OTIS.fetchFirst(g.files[file], signal);
    const total = +res.headers.get('content-length') || 0;
    const reader = res.body && res.body.getReader ? res.body.getReader() : null;
    let buf;
    if (reader) {
      const chunks = []; let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        if (signal && signal.aborted) { reader.cancel().catch(() => {}); chunks.length = 0; throw new DOMException('stopped', 'AbortError'); }
        onProgress && total && onProgress(`downloading the ${g.name} model… ${Math.round(got / 1e6)} / ${Math.round(total / 1e6)} MB`);
      }
      buf = new Uint8Array(got); let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
    } else buf = new Uint8Array(await res.arrayBuffer());
    onProgress && onProgress(`starting the runtime (${g.params} parameters take a moment)…`);
    state.sessions[key] = await ort.InferenceSession.create(buf, opts);
    state.precision[key] = file;
    return state.sessions[key];
  }

  /** Fetch and start the Chronos-2 embedding graph (Explore). */
  async function loadEmbed(onProgress, signal) {
    if (state.embed) return state.embed;
    const meta = await loadMeta();
    const g = meta.models.chronos2, e = g && g.embed;
    if (!e) throw new Error('no Chronos-2 embedding graph');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'error';
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const file = e.default || 'int8';
    onProgress && onProgress(`downloading ${g.name} (${e.sizes_mb[file]} MB)…`);
    const res = await global.OTIS.fetchFirst(e.files[file], signal);
    const total = +res.headers.get('content-length') || 0;
    const reader = res.body && res.body.getReader ? res.body.getReader() : null;
    let buf;
    if (reader) {
      const chunks = []; let got = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        if (signal && signal.aborted) { reader.cancel().catch(() => {}); chunks.length = 0; throw new DOMException('stopped', 'AbortError'); }
        onProgress && total && onProgress(`downloading ${g.name}… ${Math.round(got / 1e6)} / ${Math.round(total / 1e6)} MB`);
      }
      buf = new Uint8Array(got); let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
    } else buf = new Uint8Array(await res.arrayBuffer());
    onProgress && onProgress(`starting the runtime (a ${g.params}-parameter encoder takes a moment)…`);
    state.embed = await ort.InferenceSession.create(buf, opts);
    state.embedPrecision = file;
    return state.embed;
  }

  /** One preprocessed sample (V x T, row-major) -> its mean-pooled patch embedding. A window whose length is not a
   *  multiple of the 16-step patch is left-padded and masked out, as the reference pipeline pads it internally. */
  async function embed(x, V, T) {
    const sess = state.embed;
    if (!sess) throw new Error('the Chronos-2 encoder is not loaded');
    const P = (state.meta.models.chronos2.embed.patch) || 16, pad = (P - T % P) % P, L = T + pad;
    const ctx = new Float32Array(V * L), mask = new Float32Array(V * L);
    for (let v = 0; v < V; v++) { ctx.set(x.subarray(v * T, (v + 1) * T), v * L + pad); mask.fill(1, v * L + pad, (v + 1) * L); }
    const t0 = performance.now();
    const { emb } = await sess.run({ context: new ort.Tensor('float32', ctx, [V, L]), mask: new ort.Tensor('float32', mask, [V, L]) }, ['emb']);
    return { feat: emb.data, encMs: performance.now() - t0 };
  }

  /** The median forecast of the graph's horizon after a (V, context) window; `context` and `mask` (1 = observed,
   *  0 = left padding) are V x context, row-major. */
  async function forecast(key, context, mask, V) {
    const sess = state.sessions[key], ctx = state.meta.context;
    if (!sess) throw new Error(`${key} is not loaded`);
    if (context.length !== V * ctx) throw new Error(`the ${state.meta.models[key].name} graph takes ${ctx} context steps per variate`);
    const feeds = { context: new ort.Tensor('float32', context, [V, ctx]) };
    if (sess.inputNames.includes('mask')) feeds.mask = new ort.Tensor('float32', mask, [V, ctx]);
    else if (mask.some(m => !m)) throw new Error(`this ${state.meta.models[key].name} graph has no mask input, so it needs ${ctx} visible steps`);
    const t0 = performance.now();
    const { pred } = await sess.run(feeds, ['pred']);
    return { pred: pred.data, ms: performance.now() - t0 };
  }

  /** Fill the tail of a (V, T) series from step `start` on: the steps before it are the context (the last `context`
   *  of them, left-padded under the mask when fewer), the horizon is rolled until the end. Returns the V x (T - start)
   *  tail, the model time over all passes and the number of passes; `onPass(i, n)` reports each. */
  /** `stop()` is asked before every pass: a rolled forecast is many passes on this thread, and the visitor
   *  may stop it (app.js). The passes already made are dropped — a half-filled tail is not a prediction. */
  async function forecastTail(key, series, V, T, start, onPass, stop) {
    const ctx = state.meta.context, hz = state.meta.horizon, len = T - start;
    if (start < 1 || len < 1) throw new Error('a tail needs at least one visible step before it');
    const hist = new Float32Array(V * T); hist.set(series);                 // the series with its tail overwritten pass by pass
    const passes = Math.ceil(len / hz);
    let ms = 0;
    for (let i = 0; i < passes; i++) {
      if (stop && stop()) throw new DOMException('stopped', 'AbortError');
      onPass && onPass(i + 1, passes);
      const at = start + i * hz, n = Math.min(ctx, at), cin = new Float32Array(V * ctx), mask = new Float32Array(V * ctx);
      for (let v = 0; v < V; v++) { cin.set(hist.subarray(v * T + at - n, v * T + at), v * ctx + ctx - n); mask.fill(1, v * ctx + ctx - n, (v + 1) * ctx); }
      const res = await forecast(key, cin, mask, V);
      ms += res.ms;
      const take = Math.min(hz, T - at);
      for (let v = 0; v < V; v++) hist.set(res.pred.subarray(v * hz, v * hz + take), v * T + at);
    }
    const tail = new Float32Array(V * len);
    for (let v = 0; v < V; v++) tail.set(hist.subarray(v * T + start, (v + 1) * T), v * len);
    return { tail, ms, passes };
  }

  global.ZEROSHOT = { state, loadMeta, loadModel, forecast, forecastTail, loadEmbed, embed };
})(window);
