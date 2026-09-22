/* OTIS in the browser.
 *
 * Mirrors the reference implementation:
 *   - preprocessing follows util/preprocess.py (robust z-score, then clamp)
 *   - variate embeddings follow the paper's methods: the learned rows are used
 *     only for a domain OTIS saw in pre-training (indexed as
 *     util/dataset.py:226 does it, arange(V) + 1 + domain_offset); an unseen
 *     domain gets randomly initialised ones, matching
 *     models_otis.initialize_weights (trunc_normal, std 0.02). The exported
 *     graph therefore takes embedding vectors rather than integer ids.
 *   - masking follows OTIS.random_masking, which is NOT uniform over patches:
 *       * default branch keeps a fixed quota per variate (stratified)
 *       * forecasting branch is deterministic and keeps a leading context window
 *     Masking lives here rather than in the graph because random_masking draws
 *     from an RNG and indexes with a boolean mask, which ONNX cannot express.
 */
(function (global) {
  'use strict';

  // a local docs/models/ first, else the Hub; on GitHub Pages (no models folder) the Hub first, sparing every 404 probe
  const HUB = 'https://huggingface.co/oetu/otis-base-onnx/resolve/main/';
  const MODEL_BASES = /\.github\.io$/.test(location.hostname) ? [HUB, './models/'] : ['./models/', HUB];

  const state = { meta: null, enc: null, dec: null, base: null, precision: null, table: null,
                  posX: null, posXDec: null };

  /* Attention is quadratic in the token count V*T', so the page caps the total
     rather than the window: the model itself has no length limit once
     pos_embed_x is interpolated (util/pos_embed.interpolate_pos_embed_x). */
  const MAX_TOKENS = 1024;

  /* ---------------- loading ---------------- */

  /** `signal` (an AbortController's) lets the visitor abandon a download: the competitors' graphs run to
   *  100+ MB and a run is one click away, so every loader passes one through (app.js `dlStop`). */
  async function fetchFirst(file, signal) {
    if (state.base) {
      const r = await fetch(state.base + file, { signal });
      if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
      return r;
    }
    let lastErr;
    for (const b of MODEL_BASES) {
      try {
        const r = await fetch(b + file, { signal });
        if (r.ok) { state.base = b; return r; }
        lastErr = new Error(`HTTP ${r.status}`);
      } catch (e) {
        if (e && e.name === 'AbortError') throw e;   // stopped, not missing: no point trying the other host
        lastErr = e;
      }
    }
    throw new Error(`could not load ${file} (${lastErr && lastErr.message})`);
  }

  async function loadMeta() {
    if (state.meta) return state.meta;
    state.meta = await (await fetchFirst('otis_meta.json')).json();
    return state.meta;
  }

  /** The learned variate-embedding table, raw little-endian float32 (rows x dim). */
  async function loadTable() {
    if (state.table) return state.table;
    await loadMeta();
    const buf = await (await fetchFirst('pos_embed_y.bin')).arrayBuffer();
    state.table = new Float32Array(buf);
    return state.table;
  }

  const embedDim = (b = state) => b.meta.embed_dim / 2;   // pos_embed_y is D/2 wide

  /** The two fixed sincos position tables, (1, 1+42, D/2) and (1, 1+42, D_dec/2).
   *  They are `requires_grad=False` in models_otis and never trained, but they
   *  are what bounds the window -- so the page carries them and resamples. */
  async function loadPosX() {
    if (state.posX) return;
    await loadMeta();
    const [a, b] = await Promise.all([
      fetchFirst('pos_embed_x.bin').then(r => r.arrayBuffer()),
      fetchFirst('decoder_pos_embed_x.bin').then(r => r.arrayBuffer())
    ]);
    state.posX = new Float32Array(a);
    state.posXDec = new Float32Array(b);
  }

  /**
   * util/pos_embed.interpolate_pos_embed_x, in the browser: keep the CLS row,
   * and resample the remaining rows onto `Tp` positions with linear
   * interpolation (align_corners=False) whenever the run is longer than the
   * pre-training window. At or below it the checkpoint rows are used as they
   * are -- the forward pass slices the leading Tp of them.
   */
  function posEmbedX(table, dim, Tp) {
    const rows = table.length / dim - 1;                 // 42, excluding CLS
    if (Tp <= rows) return { data: table, rows: rows + 1 };
    const out = new Float32Array((Tp + 1) * dim);
    out.set(table.subarray(0, dim), 0);                  // CLS row, unchanged
    const scale = rows / Tp;                             // align_corners=false
    for (let i = 0; i < Tp; i++) {
      const src = Math.max(0, (i + 0.5) * scale - 0.5);
      const lo = Math.min(Math.floor(src), rows - 1), hi = Math.min(lo + 1, rows - 1);
      const w = src - lo;
      for (let j = 0; j < dim; j++) {
        const a = table[(1 + lo) * dim + j], b = table[(1 + hi) * dim + j];
        out[(1 + i) * dim + j] = a + (b - a) * w;
      }
    }
    return { data: out, rows: Tp + 1 };
  }

  /** True when this window needs pos_embed_x resampled rather than sliced. */
  const interpolatesPosX = (Tp) => Tp > maxPatches();

  /** Seeded RNG + standard normal, so a run is reproducible. */
  function rng(seed) {
    let a = seed >>> 0;          // mulberry32 is fine with 0; `|| 1` would alias seeds 0 and 1
    return () => {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0, v = 0;
    while (u === 0) u = r();
    while (v === 0) v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Learned rows, for a domain seen during pre-training. `slots` are row ids. */
  function variateEmbeddingsLearned(slots, b = state) {
    const dim = embedDim(b);
    const out = new Float32Array(slots.length * dim);
    slots.forEach((row, i) => out.set(b.table.subarray(row * dim, (row + 1) * dim), i * dim));
    return out;
  }

  /**
   * Randomly initialised rows, for an unseen domain. models_otis.initialize_weights
   * calls trunc_normal_(std=0.02) whose default bounds are +/-2 -- at this std that
   * is +/-100 sigma, so in practice it is a plain normal.
   */
  /** The sine vector: one variate embedding fitted on clean sines from 20 to 120 Hz,
   *  with the encoder and decoder frozen. 96 float32. */
  /** A variate embedding fitted on one of the Fill-in panel's periodic sources
   *  (sine, chirp or square) from 20 to 120 Hz: docs/models/<name>_embed.bin, 96 floats. null if not shipped. */
  async function loadFittedEmbed(name = 'sine') {
    state.fitted = state.fitted || {};
    if (state.fitted[name] !== undefined) return state.fitted[name];
    try {
      const buf = await (await fetchFirst(`${name}_embed.bin`)).arrayBuffer();
      state.fitted[name] = new Float32Array(buf);
    } catch (err) {
      console.warn(`[otis] fine-tuned ${name} embedding unavailable:`, err);
      state.fitted[name] = null;
    }
    return state.fitted[name];
  }
  const loadSineEmbed = () => loadFittedEmbed('sine');

  function variateEmbeddingsRandom(V, seed) {
    const dim = embedDim(), std = state.meta.pos_embed_y_std ?? 0.02;
    const r = rng(seed);
    const out = new Float32Array(V * dim);
    for (let i = 0; i < out.length; i++) out[i] = gauss(r) * std;
    return out;
  }

  /** Drop the sessions so the next run reloads at the requested precision. */
  /** A fine-tuned model set from a sub-folder (e.g. 'forecast/etth1/'): its own meta, variate
   *  table, position tables and sessions. The base model stays in `state`; `run` and the
   *  embedding helpers take the bundle as their last argument. */
  const bundles = {};
  async function loadBundle(sub, onProgress) {
    if (bundles[sub] && bundles[sub].enc) return bundles[sub];
    const b = bundles[sub] = { sub, meta: null, table: null, posX: null, posXDec: null, enc: null, dec: null, precision: null };
    b.meta = await (await fetchFirst(sub + 'otis_meta.json')).json();
    const [t, a, c] = await Promise.all([
      fetchFirst(sub + 'pos_embed_y.bin').then(r => r.arrayBuffer()),
      fetchFirst(sub + 'pos_embed_x.bin').then(r => r.arrayBuffer()),
      fetchFirst(sub + 'decoder_pos_embed_x.bin').then(r => r.arrayBuffer())
    ]);
    b.table = new Float32Array(t); b.posX = new Float32Array(a); b.posXDec = new Float32Array(c);
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = 'error';
    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    let order = [['_int8', 'int8'], ['', 'float32']];
    if (state.precisionPref === 'float32') order = [['', 'float32']];
    for (const [suffix, label] of order) {
      try {
        onProgress && onProgress('downloading OTIS…');
        const [eb, db] = await Promise.all([
          fetchFirst(`${sub}otis_encoder${suffix}.onnx`).then(r => r.arrayBuffer()),
          fetchFirst(`${sub}otis_decoder${suffix}.onnx`).then(r => r.arrayBuffer())
        ]);
        onProgress && onProgress('starting the runtime…');
        b.enc = await ort.InferenceSession.create(eb, opts);
        b.dec = await ort.InferenceSession.create(db, opts);
        b.precision = label;
        return b;
      } catch (err) {
        console.warn(`[otis] ${label} bundle ${sub} unavailable:`, err);
        b.enc = b.dec = null;
        if (suffix === '') throw err;
      }
    }
  }

  function setPrecision(pref) {
    if (state.precisionPref === pref) return false;
    state.precisionPref = pref;
    state.enc = state.dec = null;
    state.precision = null;
    return true;
  }

  async function loadModel(onProgress) {
    if (state.enc && state.dec) return;
    await loadMeta();
    await loadTable();
    await loadPosX();
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    ort.env.wasm.numThreads = 1;         // no cross-origin isolation on GitHub Pages
    ort.env.logLevel = 'error';

    const opts = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };

    // int8 first (~7 MB); fall back to fp32 (~27 MB) if the runtime rejects it.
    // state.precisionPref pins one instead: float32 is what the PyTorch codebase runs.
    let order = [['_int8', 'int8'], ['', 'float32']];
    if (state.precisionPref === 'float32') order = [['', 'float32']];
    for (const [suffix, label] of order) {
      try {
        onProgress && onProgress('downloading OTIS…');
        const [eb, db] = await Promise.all([
          fetchFirst(`otis_encoder${suffix}.onnx`).then(r => r.arrayBuffer()),
          fetchFirst(`otis_decoder${suffix}.onnx`).then(r => r.arrayBuffer())
        ]);
        onProgress && onProgress('starting the runtime…');
        state.enc = await ort.InferenceSession.create(eb, opts);
        state.dec = await ort.InferenceSession.create(db, opts);
        state.precision = label;
        return;
      } catch (err) {
        console.warn(`[otis] ${label} model unavailable:`, err);
        state.enc = state.dec = null;
        if (suffix === '') throw err;
      }
    }
  }

  /* ---------------- preprocessing (util/preprocess.py) ---------------- */

  function robustStats(vals, threshold = 3.0) {
    let n = 0, s = 0;
    for (const v of vals) { s += v; n++; }
    const m0 = s / (n || 1);
    let q = 0;
    for (const v of vals) q += (v - m0) * (v - m0);
    const sd0 = Math.sqrt(q / (n || 1));

    let sw = 0, cw = 0;
    for (const v of vals) if (Math.abs((v - m0) / (sd0 + 1e-9)) <= threshold) { sw += v; cw++; }
    const mean = sw / (cw + 1e-9);
    let vq = 0;
    for (const v of vals) if (Math.abs((v - m0) / (sd0 + 1e-9)) <= threshold) vq += (v - mean) * (v - mean);
    return { mean, std: Math.sqrt(vq / (cw + 1e-9)) };
  }

  /** Plain mean and std over all values. */
  function plainStats(vals) {
    let s = 0; for (const v of vals) s += v; const mean = s / vals.length;
    let q = 0; for (const v of vals) q += (v - mean) * (v - mean);
    return { mean, std: Math.sqrt(q / vals.length) };
  }

  /** Preprocessing: 'variate' z-scores each variate on its inliers and clamps +/-3
   *  (util/preprocess.py); 'zscore' is a plain z-score per variate with no clamp (the ECG manifest: the
   *  robust std excludes the R peaks and the clamp flattened them); 'sample' and 'zscore_sample' take one mean
   *  and std over all variates of the sample (robust with clamp / plain without); 'none' passes data through. */
  function preprocess(data, V, T, mode) {
    const out = new Float32Array(data.length);
    if (mode === 'none') { out.set(data); return out; }
    if (mode === 'sample' || mode === 'zscore_sample') {   // one mean and std over every variate of the sample
      const { mean, std } = mode === 'sample' ? robustStats(data) : plainStats(data);
      for (let i = 0; i < data.length; i++) out[i] = (data[i] - mean) / (std + 1e-9);
      if (mode === 'zscore_sample') return out;
    } else {
      const stats = mode === 'zscore' ? plainStats : robustStats;
      for (let v = 0; v < V; v++) {
        const row = data.subarray(v * T, (v + 1) * T);
        const { mean, std } = stats(row);
        for (let t = 0; t < T; t++) out[v * T + t] = (row[t] - mean) / (std + 1e-9);
      }
    }
    if (mode === 'zscore') return out;
    for (let i = 0; i < out.length; i++) out[i] = Math.max(-3, Math.min(3, out[i]));  // clamp
    return out;
  }

  /* ---------------- patch helpers ---------------- */

  const patchW = () => state.meta.patch_width;
  /** The pre-training window, in patches (42). Not a hard limit -- see fitLength. */
  const maxPatches = () => state.meta.max_num_patches_x;
  /** How many patches this page will run for V variates, given its token budget. */
  const patchBudget = (V = 1) => Math.max(1, Math.floor(MAX_TOKENS / Math.max(1, V)));

  /** Largest usable window: whole patches, within the page's token budget.
   *  Longer than the pre-training window is allowed; pos_embed_x is
   *  interpolated for it, exactly as main_forecast.py does when time_steps
   *  exceeds the checkpoint's. */
  function fitLength(T, V = 1) {
    const pw = patchW();
    return Math.min(Math.floor(T / pw), patchBudget(V)) * pw;
  }

  /** Broadcast the per-variate embeddings across time -> (V, T', dim). */
  function posEmbedY(V, Tp, vecs, b = state) {
    const dim = embedDim(b);
    const a = new Float32Array(V * Tp * dim);
    for (let v = 0; v < V; v++) {
      const row = vecs.subarray(v * dim, (v + 1) * dim);
      for (let t = 0; t < Tp; t++) a.set(row, (v * Tp + t) * dim);
    }
    return a;
  }

  /* ---------------- masking ---------------- */

  /** argsort ascending. */
  const argsort = (arr) => Array.from(arr.keys()).sort((a, b) => arr[a] - arr[b]);

  function idsFromNoise(noise, lenKeep) {
    const idsShuffle = argsort(noise);
    const idsRestore = new BigInt64Array(noise.length);
    idsShuffle.forEach((p, i) => { idsRestore[p] = BigInt(i); });
    const idsKeep = new BigInt64Array(lenKeep);
    for (let i = 0; i < lenKeep; i++) idsKeep[i] = BigInt(idsShuffle[i]);
    return { idsKeep, idsRestore, idsShuffle };
  }

  /**
   * Stratified random masking (OTIS.random_masking, default branch).
   * Exactly `keepPerVariate` patches survive in every variate; the rest are
   * pushed to +Inf so the global argsort puts them last.
   */
  function maskRandom(V, Tp, ratio) {
    const L = V * Tp;
    const lenKeep = Math.ceil(L * (10 - 10 * ratio) / 10);
    const keepPer = Math.ceil(Tp * (10 - 10 * ratio) / 10);
    const noise = new Float64Array(L);
    for (let v = 0; v < V; v++) {
      const row = new Float64Array(Tp);
      for (let t = 0; t < Tp; t++) row[t] = Math.random();
      const order = argsort(row);
      for (let t = 0; t < Tp; t++) noise[v * Tp + t] = row[t];
      // The reference pushes these to +inf. Use a large *finite* sentinel here:
      // the comparator below subtracts, and Infinity - Infinity is NaN, which
      // makes Array.prototype.sort's behaviour undefined.
      for (let k = keepPer; k < Tp; k++) noise[v * Tp + order[k]] += 1e6;
    }
    return { ...idsFromNoise(noise, lenKeep), lenKeep };
  }

  /**
   * Forecasting mask (OTIS.random_masking, forecasting branch): deterministic,
   * noise = arange(T') + a small per-variate offset, so the leading `keepPer`
   * time patches stay visible across every variate and the tail is predicted.
   */
  function maskForecast(V, Tp, ratio) {
    const L = V * Tp;
    // len_keep uses the flattened length, exactly as random_masking does; the
    // per-variate quota only shapes the noise. The two differ for some ratios
    // (V=3, T'=42, r=0.3 -> 89 vs 90), and the reference value is the right one.
    const lenKeep = Math.ceil(L * (10 - 10 * ratio) / 10);
    const keepPer = Math.ceil(Tp * (10 - 10 * ratio) / 10);
    const noise = new Float64Array(L);
    for (let v = 0; v < V; v++) {
      const off = V > 1 ? (v / (V - 1)) * 0.5 : 0;
      for (let t = 0; t < Tp; t++) noise[v * Tp + t] = t + off + (t >= keepPer ? 1e6 : 0);
    }
    return { ...idsFromNoise(noise, lenKeep), lenKeep, contextPatches: keepPer };
  }

  /* ---------------- inference ---------------- */

  async function run(signal, V, T, vecs, masking, b = state) {
    const pw = patchW(), Tp = T / pw, N = V * Tp;
    const meta = b.meta;

    const x = new ort.Tensor('float32', signal, [1, meta.input_channels, V, T]);
    const attn = new ort.Tensor('float32', new Float32Array(N).fill(1), [1, V, Tp]);
    const pey = new ort.Tensor('float32', posEmbedY(V, Tp, vecs, b), [1, V, Tp, embedDim(b)]);
    const keep = new ort.Tensor('int64', masking.idsKeep, [1, masking.idsKeep.length]);
    const restore = new ort.Tensor('int64', masking.idsRestore, [1, N]);
    const dimDec = meta.decoder_pos_embed_x_shape[2];
    const px = posEmbedX(b.posX, embedDim(b), Tp);
    const pxd = posEmbedX(b.posXDec, dimDec, Tp);
    const pex = new ort.Tensor('float32', px.data, [1, px.rows, embedDim(b)]);
    const pexd = new ort.Tensor('float32', pxd.data, [1, pxd.rows, dimDec]);

    const t0 = performance.now();
    const { latent } = await b.enc.run({
      x, attn_mask: attn, pos_embed_y: pey, pos_embed_x: pex, ids_keep: keep });
    const encMs = performance.now() - t0;

    const t1 = performance.now();
    const { pred } = await b.dec.run({
      latent, attn_mask: attn, pos_embed_y: pey, decoder_pos_embed_x: pexd, ids_restore: restore });
    const decMs = performance.now() - t1;

    return { latent, pred, Tp, N, encMs, decMs };
  }

  /** Encoder only, every patch visible (ids_keep = arange(N)). */
  async function encodeAll(signal, V, T, vecs) {
    const pw = patchW(), Tp = T / pw, N = V * Tp;
    const ids = new BigInt64Array(N);
    for (let i = 0; i < N; i++) ids[i] = BigInt(i);
    const x = new ort.Tensor('float32', signal, [1, state.meta.input_channels, V, T]);
    const attn = new ort.Tensor('float32', new Float32Array(N).fill(1), [1, V, Tp]);
    const pey = new ort.Tensor('float32', posEmbedY(V, Tp, vecs), [1, V, Tp, embedDim()]);
    const px = posEmbedX(state.posX, embedDim(), Tp);
    const t0 = performance.now();
    const { latent } = await state.enc.run({
      x, attn_mask: attn, pos_embed_y: pey,
      pos_embed_x: new ort.Tensor('float32', px.data, [1, px.rows, embedDim()]),
      ids_keep: new ort.Tensor('int64', ids, [1, N])
    });
    return { latent, Tp, N, encMs: performance.now() - t0 };
  }

  /** One vector per sample: the mean of the N patch tokens, CLS excluded
   *  (models_vit "global_pool", which is what the paper's probes read). */
  function meanPool(latent) {
    const D = latent.dims[2], K = latent.dims[1] - 1, d = latent.data;
    const out = new Float32Array(D);
    for (let p = 1; p <= K; p++) for (let j = 0; j < D; j++) out[j] += d[p * D + j];
    for (let j = 0; j < D; j++) out[j] /= K;
    return out;
  }

  /* ---------------- reassembly ---------------- */

  /** pred (1, N, pw) -> signal (V, T), taking predicted patches only where masked. */
  function unpatchify(pred, V, Tp, pw) {
    const out = new Float32Array(V * Tp * pw);
    const d = pred.data;
    for (let v = 0; v < V; v++) for (let t = 0; t < Tp; t++) {
      const p = v * Tp + t;
      for (let k = 0; k < pw; k++) out[v * Tp * pw + t * pw + k] = d[p * pw + k];
    }
    return out;
  }

  /** mask vector (0 keep, 1 remove) in original patch order. */
  function maskVector(masking, N) {
    const m = new Uint8Array(N).fill(1);
    for (let i = 0; i < masking.idsKeep.length; i++) m[Number(masking.idsKeep[i])] = 0;
    return m;
  }

  /* ---------------- metrics ---------------- */

  /** Normalised cross-correlation over the masked patches only. */
  /** Zero-normalised cross-correlation, matching util/statistics.ncc: normalise
   *  over the whole signal axis of each variate, not within a patch. Correlating
   *  inside a 24-sample patch is a different, far harsher quantity and is not
   *  comparable to the NCC the paper reports. */
  function nccMasked(pred, truth, mask, V, Tp, pw) {
    let acc = 0, nv = 0;
    for (let v = 0; v < V; v++) {
      let n = 0, sa = 0, sb = 0;
      for (let t = 0; t < Tp; t++) {
        if (!mask[v * Tp + t]) continue;
        for (let k = 0; k < pw; k++) {
          sa += pred[v * Tp * pw + t * pw + k];
          sb += truth[v * Tp * pw + t * pw + k];
          n++;
        }
      }
      if (!n) continue;
      const ma = sa / n, mb = sb / n;
      let num = 0, da = 0, db = 0;
      for (let t = 0; t < Tp; t++) {
        if (!mask[v * Tp + t]) continue;
        for (let k = 0; k < pw; k++) {
          const a = pred[v * Tp * pw + t * pw + k] - ma;
          const b = truth[v * Tp * pw + t * pw + k] - mb;
          num += a * b; da += a * a; db += b * b;
        }
      }
      // a constant truth (say a rain counter over a dry horizon) has no shape to correlate with:
      // that variate is left out, and the result is NaN when no variate has one
      if (db <= 1e-12 * n) continue;
      acc += num / (Math.sqrt(da * db) + 1e-9); nv++;
    }
    return nv ? acc / nv : NaN;
  }

  function mae(pred, truth, mask, V, Tp, pw) {
    let s = 0, c = 0;
    for (let p = 0; p < V * Tp; p++) {
      if (!mask[p]) continue;
      const v = Math.floor(p / Tp), t = p % Tp;
      for (let k = 0; k < pw; k++) { s += Math.abs(pred[v * Tp * pw + t * pw + k] - truth[v * Tp * pw + t * pw + k]); c++; }
    }
    return c ? s / c : NaN;
  }

  function mse(pred, truth, mask, V, Tp, pw) {
    let s = 0, c = 0;
    for (let p = 0; p < V * Tp; p++) {
      if (!mask[p]) continue;
      const v = Math.floor(p / Tp), t = p % Tp;
      for (let k = 0; k < pw; k++) { const d = pred[v * Tp * pw + t * pw + k] - truth[v * Tp * pw + t * pw + k]; s += d * d; c++; }
    }
    return c ? s / c : NaN;
  }

  /* ---------------- small linear algebra ---------------- */

  /** Top-2 principal components via power iteration on the covariance. */
  function pca2(rows, dim) {
    const n = rows.length;
    const mean = new Float64Array(dim);
    for (const r of rows) for (let j = 0; j < dim; j++) mean[j] += r[j] / n;
    const X = rows.map(r => { const o = new Float64Array(dim); for (let j = 0; j < dim; j++) o[j] = r[j] - mean[j]; return o; });

    const comp = [];
    for (let c = 0; c < 2; c++) {
      let v = new Float64Array(dim);
      for (let j = 0; j < dim; j++) v[j] = Math.sin((j + 1) * (c + 1) * 0.7);   // deterministic seed
      const norm = (a) => { let s = 0; for (const x of a) s += x * x; s = Math.sqrt(s) + 1e-12; for (let j = 0; j < a.length; j++) a[j] /= s; };
      norm(v);
      for (let it = 0; it < 60; it++) {
        const nv = new Float64Array(dim);
        for (const r of X) {
          let d = 0; for (let j = 0; j < dim; j++) d += r[j] * v[j];
          for (let j = 0; j < dim; j++) nv[j] += d * r[j];
        }
        for (const p of comp) { let d = 0; for (let j = 0; j < dim; j++) d += nv[j] * p[j]; for (let j = 0; j < dim; j++) nv[j] -= d * p[j]; }
        norm(nv); v = nv;
      }
      comp.push(v);
    }
    return X.map(r => comp.map(p => { let d = 0; for (let j = 0; j < dim; j++) d += r[j] * p[j]; return d; }));
  }

  function cosine(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
  }

  global.OTIS = {
    state, bundles, loadBundle, fetchFirst, loadMeta, loadTable, loadPosX, loadModel, setPrecision, preprocess, fitLength,
    posEmbedY, posEmbedX, interpolatesPosX, patchBudget,
    variateEmbeddingsLearned, variateEmbeddingsRandom, loadSineEmbed, loadFittedEmbed, embedDim,
    maskRandom, maskForecast, maskVector, run, encodeAll, meanPool, unpatchify,
    nccMasked, mae, mse, pca2, cosine, patchW, maxPatches
  };
})(window);
