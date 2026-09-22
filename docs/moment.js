/* MOMENT-large (Goswami et al., ICML 2024) in the browser: the 386 M-parameter
 * general-purpose encoder the OTIS paper compares against.
 *
 * One dynamic-shape ONNX graph serves every section: with
 * an all-ones mask its `feat` output is the paper's frozen representation (mean
 * over channels and patches), its `pred` output is MOMENT's masked
 * reconstruction for the sine, and its `emb` output (the patch embeddings, with
 * `mean` / `std` from RevIN) feeds the linear-probing head of a benchmark, one
 * small int8 graph per dataset. The mask token and the reconstruction head are
 * graph inputs (`tokens`, one float32 file: mask_token | head_w | head_b): the
 * released pair for Explore and Forecast, the pair fitted on sines, chirps or
 * squares from 20 to 120 Hz for the Impute row, on the same signals OTIS and
 * UniTS get their fitted tokens from.
 * The int8 graph is ~330 MB and a 12-lead ECG is
 * 1512 tokens through a 24-layer T5 encoder, so it runs live on purpose: the
 * page lets the user feel what 341 M parameters cost, and stop whenever they
 * have (precomputed vectors cover the samples not embedded).
 */
(function (global) {
  'use strict';

  const state = { meta: null, sessions: {}, precision: null, heads: {}, headPrecision: {} };

  /** The mask token and reconstruction head as one Float32Array (moment_meta.json graphs.any.tokens): 'released'
   *  (moment_released_tokens.bin, the checkpoint's) or one fitted on a source (moment_<sine|chirp|square>_tokens.bin);
   *  null when a fitted file is not published. */
  async function loadTokens(name = 'released') {
    state.tokens = state.tokens || {};
    if (state.tokens[name] !== undefined) return state.tokens[name];
    try {
      const buf = await (await global.OTIS.fetchFirst(`moment_${name}_tokens.bin`)).arrayBuffer();
      state.tokens[name] = new Float32Array(buf);
    } catch (err) {
      if (name === 'released') throw err;
      console.warn(`[moment] fitted ${name} tokens unavailable:`, err);
      state.tokens[name] = null;
    }
    return state.tokens[name];
  }
  /** The three token inputs of a run from one packed array (the released pair unless given). */
  function tokenFeeds(tokens) {
    const D = state.meta.d_model, pl = state.meta.patch_len, t = tokens || state.tokens.released;
    if (!t) throw new Error('MOMENT tokens are not loaded');
    return { mask_token: new ort.Tensor('float32', t.subarray(0, D), [D]),
             head_w: new ort.Tensor('float32', t.subarray(D, D + pl * D), [pl, D]),
             head_b: new ort.Tensor('float32', t.subarray(D + pl * D, D + pl * D + pl), [pl]) };
  }

  async function loadMeta() {
    if (state.meta) return state.meta;
    state.meta = await (await global.OTIS.fetchFirst('moment_meta.json')).json();
    return state.meta;
  }

  async function loadModel(graph, onProgress, signal) {
    if (state.sessions[graph]) return state.sessions[graph];
    const meta = await loadMeta();
    const g = meta.graphs[graph];
    if (!g) throw new Error(`no MOMENT graph for shape ${graph}`);
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'error';
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    const file = g.default || 'int8';
    onProgress && onProgress(`downloading MOMENT (${g.sizes_mb[file]} MB)…`);
    const res = await global.OTIS.fetchFirst(g.files[file], signal);
    // stream so the progress line can show how much of the large file has arrived
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
        onProgress && total && onProgress(`downloading MOMENT… ${Math.round(got / 1e6)} / ${Math.round(total / 1e6)} MB`);
      }
      buf = new Uint8Array(got); let o = 0;
      for (const c of chunks) { buf.set(c, o); o += c.length; }
    } else buf = new Uint8Array(await res.arrayBuffer());
    onProgress && onProgress('starting the runtime (a 341 M-parameter encoder takes a moment)…');
    await loadTokens('released');
    state.sessions[graph] = await ort.InferenceSession.create(buf, opts);
    state.precision = file;
    return state.sessions[graph];
  }

  /** Masked reconstruction on MOMENT's 8-step grid. `mask` has one entry per patch, 1 = hidden; `tokens` the mask token
   *  and head to use (a fitted pair from loadTokens), the released pair by default. */
  async function generate(graph, signal, V, T, mask, tokens) {
    const sess = state.sessions[graph], pl = state.meta.patch_len;
    const m = new Float32Array(T);                        // MOMENT's convention: 1 = observed
    for (let t = 0; t < T; t++) m[t] = mask[Math.floor(t / pl)] ? 0 : 1;
    const t0 = performance.now();
    const { pred } = await sess.run({
      x: new ort.Tensor('float32', signal, [1, V, T]),
      mask: new ort.Tensor('float32', m, [1, T]), ...tokenFeeds(tokens)
    }, ['pred']);
    return { pred: pred.data, encMs: performance.now() - t0 };
  }

  /** One dataset's linear-probing head (moment_meta.json `forecast.heads`), a small
   *  int8 graph fetched when the Benchmarks row is first run. */
  async function loadHead(key, onProgress, signal) {
    if (state.heads[key]) return state.heads[key];
    const meta = await loadMeta();
    const h = meta.forecast && meta.forecast.heads[key];
    if (!h) throw new Error(`no MOMENT forecasting head for ${key}`);
    const file = h.default || 'int8';
    onProgress && onProgress(`downloading MOMENT's ${key} head (${h.sizes_mb[file]} MB)…`);
    const buf = new Uint8Array(await (await global.OTIS.fetchFirst(h.files[file], signal)).arrayBuffer());
    state.heads[key] = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    state.headPrecision[key] = file;
    return state.heads[key];
  }

  /** MOMENT's forecasting path, linear probing: the (V, ctx) context left-padded to the 512-step input under a padding
   *  mask (-1: kept out of the attention, as momentfm.forecast does with its input_mask), the encoder's patch embeddings
   *  through the dataset's head, RevIN's statistics mapping the horizon back. Channel-independent, so the variates
   *  run in slices of `chunk` and `onPart(done, V)` reports each; `stop()` is asked between slices. Returns the
   *  (V, horizon) forecast and the model time. */
  async function forecast(graph, key, context, V, ctx, onPart, chunk = 4, stop) {
    const sess = state.sessions[graph], head = state.heads[key], meta = state.meta, T = meta.forecast.seq_len, pad = T - ctx;
    if (!sess) throw new Error('MOMENT is not loaded');
    if (!head) throw new Error(`MOMENT's ${key} head is not loaded`);
    const hz = meta.forecast.heads[key].horizon, out = new Float32Array(V * hz);
    const m = new Float32Array(T).fill(-1); m.fill(1, pad);
    let ms = 0;
    for (let v0 = 0; v0 < V; v0 += chunk) {
      if (stop && stop()) throw new DOMException('stopped', 'AbortError');   // between chunks: one chunk of a 341 M-parameter encoder cannot be interrupted
      const n = Math.min(chunk, V - v0), x = new Float32Array(n * T);
      for (let i = 0; i < n; i++) x.set(context.subarray((v0 + i) * ctx, (v0 + i + 1) * ctx), i * T + pad);
      const t0 = performance.now();
      const { emb, mean, std } = await sess.run({ x: new ort.Tensor('float32', x, [1, n, T]), mask: new ort.Tensor('float32', m, [1, T]), ...tokenFeeds() }, ['emb', 'mean', 'std']);
      const { pred } = await head.run({ emb: new ort.Tensor('float32', emb.data, emb.dims.slice(1)) }, ['pred']);
      ms += performance.now() - t0;
      for (let i = 0; i < n; i++) for (let t = 0; t < hz; t++) out[(v0 + i) * hz + t] = pred.data[i * hz + t] * std.data[i] + mean.data[i];
      onPart && onPart(v0 + n, V);
    }
    return { pred: out, encMs: ms };
  }

  /** The paper's frozen representation: every patch visible, mean over channels and patches. */
  async function encode(graph, signal, V, T) {
    const sess = state.sessions[graph];
    const t0 = performance.now();
    const { feat } = await sess.run({
      x: new ort.Tensor('float32', signal, [1, V, T]),
      mask: new ort.Tensor('float32', new Float32Array(T).fill(1), [1, T]), ...tokenFeeds()
    }, ['feat']);
    return { feat: feat.data, encMs: performance.now() - t0 };
  }

  global.MOMENT = { state, loadMeta, loadModel, loadHead, loadTokens, generate, forecast, encode };
})(window);
