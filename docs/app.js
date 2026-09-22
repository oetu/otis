/* UI wiring for the page: three general-purpose encoders (OTIS, UniTS, MOMENT), toggled
 * in the bar and shown side by side in every panel.
 *   Explore    -- one map per encoder over the test recordings (the training vectors are the labelled reference), scored with
 *                 the paper's frozen-encoder protocols; optional live re-embedding to feel the cost
 *   Fill in    -- a 432-step signal, a hidden span chosen by dragging; each model fills it when its row is run
 *   Forecast   -- fine-tuned short-term forecasting on four public datasets (Weather, Illness, Electricity, Traffic)
 */
(function () {
  'use strict';

  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  /* Optional visit counting, defined in analytics.js and a no-op until that file is configured.
   * Guarded so the page still runs if analytics.js is ever dropped. */
  const track = (category, action, name, value) => { if (window.track) window.track(category, action, name, value); };
  const C = window.Charts, A = window.Analysis, M = window.OTIS, U = window.UNITS, MO = window.MOMENT, Z = window.ZEROSHOT;

  const ENC = ['otis', 'units', 'moment'];
  // Explore: the encoders that can fill a panel. TimesFM-3 is in the dropdown too, but its licence forbids
  // redistributing the weights, so its panel carries the same n/a note the other sections' rows carry.
  const EX_ENC = ['otis', 'units', 'moment', 'chronos2'];
  const NAME = { otis: 'OTIS', units: 'UniTS', moment: 'MOMENT', chronos2: 'Chronos-2', timesfm3: 'TimesFM-3' };
  const PARAMS = { otis: '7.1 M', units: '8.2 M', moment: '386 M', chronos2: '120 M', timesfm3: '330 M' };
  // zero-shot forecasters (Chronos-2, TimesFM-3): their int8 graphs (docs/zeroshot.js) are fetched
  // when a Fill in or Forecast row is run
  const ZS = ['chronos2', 'timesfm3'];
  const ZS_CTX = 336, ZS_HZ = 96;   // the graphs' shape: 336 context steps (fewer under the mask) -> 96 forecast steps; a longer tail is rolled
  const encColor = (e) => C.css(e.startsWith('otis') ? `--enc-${e}` : '--text-muted');   // OTIS in blue, every competitor in grey
  /** The competitor row's name: a dropdown over the encoders OTIS is compared with in that section. */
  const rivalHtml = (cur, options) => `<span class="enc-name"><select class="rival-select" aria-label="Encoder compared with OTIS">${options.map(e => `<option value="${e}"${e === cur ? ' selected' : ''}>${NAME[e]}</option>`).join('')}</select></span>`;
  const wireRival = (el, busy, onPick) => { const sel = el.querySelector('.rival-select'); if (!sel) return; sel.disabled = !!busy; sel.addEventListener('click', e => e.stopPropagation()); sel.addEventListener('change', () => { track('Compare', 'pick', sel.value); onPick(sel.value); }); };
  const RIVALS = ['units', 'moment', 'chronos2', 'timesfm3'];   // the competitors of the Fill in and Forecast rows
  const status = (sel, msg, isErr) => { const n = $(sel); n.textContent = msg || ''; n.classList.toggle('err', !!isErr); };
  // lets the page paint before a model runs on this thread (the rows' "predicting…" would otherwise show only after
  // the run): a frame, then a task after it; a hidden tab gets no frames, so it just yields
  const yieldUI = () => new Promise(r => document.hidden ? setTimeout(r, 0) : requestAnimationFrame(() => setTimeout(r, 0)));
  const fetchBuf = async (f) => { const r = await fetch(`data/${f}`); if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`); return r.arrayBuffer(); };
  const pct = v => `${Math.round(v * 100)}%`;
  const msFmt = ms => ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
  const nccFmt = (v) => Number.isNaN(v) ? 'n/a' : C.fmt(v, 2);   // NCC is undefined on a constant truth
  const secFmt = (ms) => ms < 10000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms / 1000)} s`;

  /* ---------------- stopping a download ----------------
   * A competitor's graph is 6 to 330 MB and a single click starts it, so every fetch of one runs under
   * its own AbortController: while the bytes move, the row (Impute, Forecast) or the card's message box
   * (Explore) offers a Stop, and `dlStop` abandons the fetch. OTIS' own 7 MB is not worth interrupting.
   */
  /* A phone delivers a tap as a click a moment after the finger lifts, and by then the Stop is gone —
   * replaced by whatever comes next in that spot: the empty Explore panel, or a row's Run. The click lands
   * on it and starts the very download that was just stopped. So every place a run begins ignores clicks
   * for a moment after a Stop; the pointer events that stopped it are the same ones that would restart it. */
  let stoppedAt = -1e9;
  const noteStopped = () => { stoppedAt = performance.now(); };
  const justStopped = () => performance.now() - stoppedAt < 600;

  const dls = {};                                             // encoder -> the controller of its download in flight
  const dlSignal = (enc) => (dls[enc] = new AbortController()).signal;
  const dlDone = (enc) => { delete dls[enc]; };
  const dlStop = (enc) => { noteStopped(); const c = dls[enc]; if (c) { delete dls[enc]; c.abort(); } };
  const wasStopped = (e) => !!e && (e.name === 'AbortError' || /aborted|stopped/i.test(e.message || ''));
  /** A loader's message as the phase the page shows. Three of them: the bytes are still coming (a bar, a
   *  counter and a Stop that saves them), ONNX Runtime is building the graph (a full bar, and no Stop —
   *  there is nothing left to abandon and no way to interrupt it), or neither, which the caller names. */
  const loadPhase = (m) => {
    const mb = m.match(/(\d+) \/ (\d+) MB/), starting = /starting/.test(m);
    return { label: starting ? 'starting the runtime…' : 'downloading…',
             frac: mb ? +mb[1] / Math.max(1, +mb[2]) : starting ? 1 : 0,
             mb: mb ? `${mb[1]} / ${mb[2]} MB` : '', canStop: !starting };
  };

  /** one competitor download under its own controller, so a Stop beside it can abandon the fetch */
  const withStop = async (enc, run) => { try { return await run(dlSignal(enc)); } finally { dlDone(enc); } };

  const app = { on: { otis: true, units: true, moment: true }, momentLoaded: false, momentLoading: false, zs: { loaded: {}, loading: {} }, meta: null, sync: false };
  const selected = () => ENC.filter(e => app.on[e]);

  /** A graph fetched for one panel is loaded for the other: the rows' hints ("the first run fetches…") follow. */
  const loadedElsewhere = () => { if (fill.signal) renderFill(); if (fc.ds && fc.man[fc.ds]) renderFc(); };

  /* ---------------- MOMENT-large: loaded only on request, its size shown in the chip ---------------- */
  async function loadMoment(where, onMsg) {
    if (app.momentLoaded) return true;
    if (app.momentLoading) { while (app.momentLoading) await new Promise(r => setTimeout(r, 100)); return app.momentLoaded; }
    app.momentLoading = true;
    const chip = $('.enc[data-enc="moment"]');
    const tag = document.createElement('span'); tag.className = 'enc-load'; chip.appendChild(tag);
    try {
      await MO.loadModel('any', m => { const mb = m.match(/(\d+) \/ (\d+) MB/); tag.textContent = mb ? `${mb[1]} / ${mb[2]} MB` : 'starting…'; if (onMsg) onMsg(m); else where && status(where, m); }, dlSignal('moment'));
      app.momentLoaded = true; loadedElsewhere();
    } catch (e) {
      if (wasStopped(e)) { where && status(where, 'the MOMENT download was stopped.'); return false; }   // the visitor's own doing, not an error
      console.error(e);
      where && status(where, `MOMENT could not be loaded: ${e.message}`, true);
    } finally {
      dlDone('moment'); app.momentLoading = false; tag.remove();
    }
    return app.momentLoaded;
  }

  /** Chronos-2 as an Explore encoder: its embedding graph (128 MB int8), fetched when its panel is run. */
  async function loadZsEmbed(onMsg) {
    const z = app.zs;
    if (z.embedLoaded) return true;
    if (z.embedLoading) { while (z.embedLoading) await new Promise(r => setTimeout(r, 100)); return !!z.embedLoaded; }
    z.embedLoading = true;
    try {
      await Z.loadEmbed(m => onMsg && onMsg(m), dlSignal('chronos2'));
      z.embedLoaded = true; loadedElsewhere();
    } catch (e) {
      if (wasStopped(e)) { onMsg && onMsg(`the ${NAME.chronos2} download was stopped.`); return false; }
      console.error(e);
      onMsg && onMsg(`${NAME.chronos2} could not be loaded: ${e.message || e}`);
    } finally { dlDone('chronos2'); z.embedLoading = false; }
    return !!z.embedLoaded;
  }

  /* ---------------- the zero-shot forecasters: a graph fetched on request, when a Fill-in row is run ---------------- */
  async function loadZeroShot(enc, where, onMsg) {
    const z = app.zs;
    if (z.loaded[enc]) return true;
    if (z.loading[enc]) { while (z.loading[enc]) await new Promise(r => setTimeout(r, 100)); return !!z.loaded[enc]; }
    z.loading[enc] = true;
    try {
      await Z.loadModel(enc, m => { if (onMsg) onMsg(m); else where && status(where, m); }, dlSignal(enc));
      z.loaded[enc] = true; where && status(where, ''); loadedElsewhere();
    } catch (e) {
      if (wasStopped(e)) { where && status(where, `the ${NAME[enc]} download was stopped.`); return false; }
      console.error(e);
      where && status(where, `${NAME[enc]} could not be loaded: ${e.message || e}`, true);
    } finally { dlDone(enc); z.loading[enc] = false; }
    return !!z.loaded[enc];
  }

  /* =====================================================================
   * Explore
   * ===================================================================== */

  const DS = {   // Motion first: everyone can picture accelerometer data; the clinical sets follow
    basicmotions: { file: 'basicmotions', label: 'Motion', sub: 'BasicMotions', blurb: 'Ten seconds of a smartwatch\u2019s accelerometer and gyroscope: what is the wearer doing?' },
    starlight: { file: 'starlight', label: 'Stars', sub: 'StarLightCurves', blurb: 'Brightness of a star over one period: which kind of star is it?' },
    kitchen: { file: 'kitchen', label: 'Appliances', sub: 'SmallKitchenAppliances', blurb: 'A day of an appliance\u2019s power draw: kettle, microwave or toaster?' },
    ptbxl: { file: 'ptbxl', label: 'ECG', sub: 'PTB-XL', blurb: 'Two seconds of a twelve-lead electrocardiogram: which heart rhythm is it?' },
    sleepedf: { file: 'sleepedf', label: 'EEG', sub: 'Sleep-EDF', blurb: 'Thirty seconds of brain activity from one electrode during sleep: which sleep stage is it?' },
    custom: { file: null, label: 'Your data', sub: 'CSV', blurb: 'Your own labelled series: does the encoder separate your classes?' }      // two files chosen by the visitor, embedded live (buildCustom)
  };
  const ex = {
    ds: null, data: {},
    vecs: {},          // ds -> enc -> { gallery, test, live, ms, precision, partial, stamp }
    projCache: {},
    maps: {}, cards: {}, selected: null, busy: false, stop: false, stopAfter: 0, src: {},
    rival: 'chronos2'  // the encoder shown beside OTIS, picked in the right-hand card's dropdown
  };
  const exEncs = () => ['otis', ex.rival];   // the two maps: OTIS on the left, the chosen comparison on the right

  async function loadDataset(key) {
    if (ex.data[key]) return ex.data[key];
    const r = await fetch(`data/${DS[key].file}.json`);
    if (!r.ok) throw new Error(`${key}.json: HTTP ${r.status}`);
    const man = await r.json();
    const V = man.V, T = man.T, n = man.n_test;
    const [tbuf, ubuf, vbuf] = await Promise.all([fetchBuf(man.files.test), fetchBuf(man.units.files.tokens),
      man.files.otis_vecs ? fetchBuf(man.files.otis_vecs) : null]);   // an unseen domain: the stored random variate embeddings
    // the training recordings themselves stay on the server: only their vectors are used, as the labelled reference for the scores
    const raw = new Float32Array(n * V * T);
    if (man.dtype === 'int16') { const i16 = new Int16Array(tbuf); for (let i = 0; i < raw.length; i++) raw[i] = i16[i] * man.scale; }
    else raw.set(new Float32Array(tbuf));
    const test = new Float32Array(n * V * T);
    for (let i = 0; i < n; i++) test.set(M.preprocess(raw.subarray(i * V * T, (i + 1) * V * T), V, T, man.preprocess), i * V * T);
    ex.data[key] = { man, test, unitsTokens: new Float32Array(ubuf), otisVecs: vbuf ? new Float32Array(vbuf) : null };
    // every encoder's vectors, computed offline with the same exported graphs
    const rows = (flat, D, count) => { const out = []; for (let i = 0; i < count; i++) out.push(Float64Array.from(flat.subarray(i * D, (i + 1) * D))); return out; };
    const src = {
      otis: { gallery: man.files.gallery_learned, test: man.files.test_learned, D: app.meta.embed_dim },
      units: { gallery: man.units.files.gallery, test: man.units.files.test, D: man.units.d_model },
      moment: man.moment ? { gallery: man.moment.files.gallery, test: man.moment.files.test, D: man.moment.d_model } : null,
      chronos2: man.chronos2 ? { gallery: man.chronos2.files.gallery, test: man.chronos2.files.test, D: man.chronos2.d_model } : null
    };
    // fetched only when that encoder's Run is pressed (ensureVecs), so a visit downloads nothing it does not show
    ex.src[key] = src; ex.vecs[key] = {};
    ex.projCache[key] = {};
    return ex.data[key];
  }

  const testView = (d, i, v) => d.test.subarray((i * d.man.V + v) * d.man.T, (i * d.man.V + v + 1) * d.man.T);

  function buildDatasetSeg() {
    const host = $('#dataset-seg');
    for (const key of Object.keys(DS)) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'seg-btn'; b.dataset.key = key;
      b.textContent = DS[key].label;                                    // the dataset's own name stays in the tooltip and the section's about drawer
      if (key === 'custom') b.innerHTML += `<small>${DS[key].sub}</small>`; else b.title = DS[key].sub;
      b.addEventListener('click', () => selectDataset(key));
      host.appendChild(b);
    }
  }

  async function selectDataset(key) {
    if (ex.busy) return;
    ex.ds = key; ex.selected = null;
    track('Explore', 'dataset', key);
    $$('#dataset-seg .seg-btn').forEach(b => b.classList.toggle('is-active', b.dataset.key === key));
    $('#ds-blurb').textContent = DS[key].blurb || '';
    $('#sample-view').hidden = true;
    $('#custom-form').hidden = key !== 'custom';
    if (key === 'custom' && !ex.data.custom) {           // nothing embedded yet: the form asks for the files
      $('#maps').innerHTML = ''; $('#legend-explore').innerHTML = ''; $('#ds-hint').hidden = true;
      status('#status-explore', '');
      $('#ds-blurb').innerHTML = `${DS.custom.blurb} Choose a train and a test file, then click <em>Run</em>.`;
      return;
    }
    status('#status-explore', 'loading…');
    try {
      await loadDataset(key);
      if (ex.ds !== key) return;
      status('#status-explore', '');
      renderMaps();
    } catch (e) {
      console.error(e);
      status('#status-explore', `could not load ${DS[key].sub}: ${e.message}`, true);
    }
  }

  const vecsOf = (enc) => ex.vecs[ex.ds] && ex.vecs[ex.ds][enc];
  const hasVecs = (enc) => !!vecsOf(enc) || !!(ex.src[ex.ds] && ex.src[ex.ds][enc]);   // loaded, or fetchable on Run
  /** The encoder's offline vectors for the dataset (training set for the scores, test set for a run cut short), fetched on first use. */
  async function ensureVecs(key, enc) {
    if (ex.vecs[key][enc]) return ex.vecs[key][enc];
    const s = ex.src[key][enc], man = ex.data[key].man;
    const rows = (flat, D, count) => { const out = []; for (let i = 0; i < count; i++) out.push(Float64Array.from(flat.subarray(i * D, (i + 1) * D))); return out; };
    const [g, t] = await Promise.all([fetchBuf(s.gallery), fetchBuf(s.test)]);
    if (!ex.vecs[key][enc]) ex.vecs[key][enc] = { gallery: rows(new Float32Array(g), s.D, man.n_gallery), test: rows(new Float32Array(t), s.D, man.n_test), live: 0, ms: null, stamp: 0 };
    return ex.vecs[key][enc];
  }

  /* t-SNE in a background worker per encoder, so several maps fit at once without freezing the page. */
  const workers = {}, jobs = {};
  let jobId = 0;
  function runTsne(enc, rows, opts) {
    // ?sync fits on the main thread (headless screenshots under a virtual-time budget give workers no CPU)
    if (typeof Worker === 'undefined' || app.sync) return new Promise(res => setTimeout(() => res(A.tsneAuto(rows, opts)), 0));
    if (!workers[enc]) {
      workers[enc] = new Worker('worker.js?v=102');
      workers[enc].onmessage = (e) => { const j = jobs[e.data.id]; delete jobs[e.data.id]; if (!j) return; e.data.error ? j.reject(new Error(e.data.error)) : j.resolve(e.data.res); };
    }
    return new Promise((resolve, reject) => { const id = ++jobId; jobs[id] = { resolve, reject }; workers[enc].postMessage({ id, rows, opts }); });
  }

  /** t-SNE with its perplexity picked per map by neighbourhood preservation (analysis.js); the
   *  choice is kept per dataset and encoder, so live re-fits run a single t-SNE. Returns the
   *  cached map, or null while it is being fitted (the card re-renders when it lands). */
  function projection(enc, rows, stamp) {
    const cache = ex.projCache[ex.ds][enc] || (ex.projCache[ex.ds][enc] = {});
    if (cache[stamp]) return cache[stamp];
    if (cache.pending === stamp) return null;
    cache.pending = stamp;
    const ds = ex.ds, t0 = performance.now();
    runTsne(enc, rows, cache.perplexity ? { perplexities: [cache.perplexity] } : {}).then(res => {
      res.ms = performance.now() - t0;
      cache[stamp] = res;
      if (!cache.perplexity) cache.perplexity = res.perplexity;
      if (cache.pending === stamp) cache.pending = null;
      const r = ex.ds === ds && ex.vecs[ds] && ex.vecs[ds][enc];
      if (r && r.stamp === stamp && ex.cards[enc]) renderMap(enc);
    }).catch(e => {
      console.error(e);
      if (cache.pending === stamp) cache.pending = null;
      if (ex.ds === ds && ex.cards[enc]) ex.cards[enc].chart.innerHTML = `<div class="map-wait">the map could not be fitted: ${e.message}</div>`;
    });
    return null;
  }
  const fitting = (enc) => { const c = ex.projCache[ex.ds] && ex.projCache[ex.ds][enc]; return !!(c && c.pending != null); };

  /** The panel's cards follow the encoder chips; one map per selected encoder. */

  function renderMaps() {
    if (!ex.ds) return;
    const host = $('#maps'), encs = exEncs();
    host.style.setProperty('--n', '2');
    // drop the card of a comparison no longer chosen, add the missing one, OTIS first
    for (const e of Object.keys(ex.cards)) if (!encs.includes(e)) { ex.cards[e].root.remove(); delete ex.cards[e]; delete ex.maps[e]; }
    for (const e of encs) if (!ex.cards[e]) ex.cards[e] = makeCard(e);
    encs.forEach(e => host.appendChild(ex.cards[e].root));
    renderLegend();
    for (const e of encs) renderMap(e);
  }

  function makeCard(enc) {
    const root = document.createElement('div');
    root.className = 'map-card'; root.dataset.enc = enc;
    const rivals = RIVALS;
    const name = enc === 'otis' ? `<span class="enc-name">${NAME[enc]}</span>`
      : `<span class="enc-name"><select class="rival-select" aria-label="Encoder compared with OTIS">${rivals.map(e => `<option value="${e}"${e === enc ? ' selected' : ''}>${NAME[e]}</option>`).join('')}</select></span>`;
    // no Run button: the empty panel itself starts the run (renderMap); "stop" sits beside the progress bar while it embeds
    root.innerHTML = `<div class="map-head">${name}<span class="enc-meta"></span></div><div class="map-sub"><span class="map-state"></span></div>` +
      `<div class="scores"></div><div class="map-chart"></div><div class="map-foot" hidden><div class="bar-track"><div class="bar-fill"></div></div><span class="foot-text"></span><button type="button" class="stop-btn foot-stop">Stop</button></div>`;
    root.querySelector('.foot-stop').addEventListener('click', (e) => {
      // a demo run stops from the card that runs; on your data either card stops the whole run (the encoder running and the ones queued behind it)
      if (ex.running === enc || (ex.ds === 'custom' && ex.busy)) { noteStopped(); ex.stop = true; e.target.disabled = true; if (ex.ds === 'custom' && ex.cards[ex.running]) ex.cards[ex.running].state.textContent = 'stopping after this series…'; }
    });
    const sel = root.querySelector('.rival-select');
    if (sel) sel.addEventListener('change', () => { if (ex.busy) { sel.value = enc; return; } track('Compare', 'pick', sel.value); ex.rival = sel.value; renderMaps(); });
    return { root, meta: root.querySelector('.enc-meta'), state: root.querySelector('.map-state'), scores: root.querySelector('.scores'),
             chart: root.querySelector('.map-chart'), foot: root.querySelector('.map-foot'), fill: root.querySelector('.bar-fill'), footText: root.querySelector('.foot-text') };
  }

  /** The three tiles with a dash each: shown before the panel is clicked and while it embeds, replaced by the run's numbers. */
  function placeholderScores(card) {
    card.scores.innerHTML = '';
    for (const [label, title] of [['on this CPU', "the encoder's time for the recordings on your CPU"], ['throughput', 'recordings encoded per second on your CPU'], ['accuracy', 'balanced accuracy of a weighted k-NN over the test recordings']])
      score(card.scores, label, '–', '', title);
  }
  function score(host, label, value, note, title, extra = '') {
    const d = document.createElement('div');
    d.className = 'score' + (extra ? ' ' + extra : ''); d.dataset.tip = title;   // shown by the page's own tooltip (app.css [data-tip])
    d.innerHTML = `<span class="val">${value}${note ? `<small>${note}</small>` : ''}</span><span class="cap">${label}</span>`;
    host.appendChild(d);
  }

  function renderLegend() {
    const man = ex.data[ex.ds].man, k = man.classes.length, host = $('#legend-explore');
    host.innerHTML = '';
    const names = man.classes.map(c => man.class_names[c] || c);
    const items = names.map((n, i) => ({ label: n, color: C.catColor(i), shape: 'dot', cls: i }));
    let focus = -1;
    const nodes = items.map(it => {
      const s = document.createElement('span');
      s.className = 'legend-item is-clickable';
      s.innerHTML = `<span class="swatch ${it.shape}" style="background:${it.color}"></span>${it.label}`;
      s.addEventListener('click', () => {
        focus = focus === it.cls ? -1 : it.cls;
        nodes.forEach((n, j) => { n.classList.toggle('is-focus', items[j].cls === focus); n.classList.toggle('is-dim', focus >= 0 && items[j].cls !== focus); });
        for (const e of Object.keys(ex.maps)) ex.maps[e].setFocus(focus, null);
      });
      host.appendChild(s); return s;
    });
    host.hidden = true; $('#ds-hint').hidden = true;   // the classes and the dot hint appear with the first map (renderMap); until then they have nothing to refer to
  }

  function renderMap(enc) {
    const card = ex.cards[enc], d = ex.data[ex.ds], man = d.man, r = vecsOf(enc);
    if (!card) return;
    if (enc === 'timesfm3') {                            // no graph on this page: its licence forbids redistributing the weights
      card.meta.textContent = `${PARAMS[enc]} parameters`;
      card.state.textContent = ''; card.state.classList.remove('live');
      card.scores.innerHTML = '';
      card.chart.innerHTML = `<div class="map-wait" style="height:340px"><span>n/a: ${NAME[enc]}'s license does not allow its weights to be redistributed.</span></div>`;
      delete ex.maps[enc];
      return;
    }
    if (!r && man.custom && (enc === 'otis' || enc === 'moment' || enc === 'chronos2' || (enc === 'units' && man.units))) {   // your data, not embedded yet (or stopped): the empty panel is the button, like the demo datasets
      card.meta.textContent = `${PARAMS[enc]} parameters`;
      card.state.textContent = ''; card.state.classList.remove('live');
      const size = parseFloat(PARAMS[enc]), fetch = (enc === 'moment' && !app.momentLoaded) || (enc === 'chronos2' && !app.zs.embedLoaded);
      const hint = (fetch ? `the first click fetches ${NAME[enc]} (${ASK_MB[enc]}); ` : '') + (size >= 100 ? 'due to > 100 M parameters, this will take a while :(' : `due to ${Math.round(size)} M parameters, this will be quick :)`);
      placeholderScores(card); card.chart.innerHTML = `<div class="map-wait map-idle" role="button" tabindex="0" style="height:340px"><span>click to explore your data on your CPU<br><small>${hint}</small></span></div>`;
      const idle = card.chart.querySelector('.map-idle');
      idle.addEventListener('click', () => { if (!ex.busy && !justStopped()) runCustomEnc(enc); });
      idle.addEventListener('keydown', (e) => { if ((e.key === 'Enter' || e.key === ' ') && !ex.busy) { e.preventDefault(); runCustomEnc(enc); } });
      card.ds = ex.ds; delete ex.maps[enc];
      return;
    }
    if (!r && !hasVecs(enc)) {
      card.meta.textContent = `${PARAMS[enc]} parameters`;
      card.state.innerHTML = man.custom && enc === 'units' ? `no UniTS graph for ${man.V}×${man.T}` : 'no vectors for this dataset';
      card.scores.innerHTML = ''; card.chart.innerHTML = '';
      return;
    }
    if (r && r.pending) {                                // your data: vectors still being computed
      card.meta.textContent = `${PARAMS[enc]} parameters`;
      card.state.textContent = `encoding here… ${r.live} / ${r.gallery.length + r.test.length}`;
      card.scores.innerHTML = ''; card.chart.innerHTML = '<div class="map-wait" style="height:340px">encoding your data…</div>';
      return;
    }
    if (!r || (!r.live && !r.partial && !man.custom)) {   // not yet run here: the empty panel is the button that starts the run
      card.meta.textContent = `${PARAMS[enc]} parameters`;
      card.state.textContent = ''; card.state.classList.remove('live');
      const size = parseFloat(PARAMS[enc]);                // the wait to expect: a small encoder is quick, one beyond 100 M parameters is not
      const hint = size >= 100 ? 'due to > 100 M parameters, this will take a while :(' : `due to ${Math.round(size)} M parameters, this will be quick :)`;
      placeholderScores(card); card.chart.innerHTML = `<div class="map-wait map-idle" role="button" tabindex="0" style="height:340px"><span>click to explore the data on your CPU${hint ? `<br><small>${hint}</small>` : ''}</span></div>`;
      const idle = card.chart.querySelector('.map-idle');
      idle.addEventListener('click', () => { if (!ex.busy && !justStopped()) runLive([enc]); });
      idle.addEventListener('keydown', (e) => { if ((e.key === 'Enter' || e.key === ' ') && !ex.busy) { e.preventDefault(); runLive([enc]); } });
      card.ds = ex.ds; delete ex.maps[enc];
      return;
    }
    if (r.partial) return;                               // a run in progress writes its own progress into the card (runLive)
    const k = man.classes.length, P = man.protocol, nGal = r.gallery.length, nTest = r.test.length;
    const yg = man.labels_gallery, yt = man.labels_test;
    const names = man.classes.map(c => man.class_names[c] || c);
    const knnDetail = A.weightedKnnDetail(r.gallery, yg, r.test, { k: P.knn_k, tau: P.knn_tau, nClasses: k });
    r.knn = knnDetail;
    const knn = A.balancedAccuracy(knnDetail.map(x => x.pred), yt, k);
    const proto = A.prototypical(r.gallery, yg, r.test, yt, { shots: P.shots, episodes: P.episodes, seed: P.seed, nClasses: k });
    r.scores = { knn, proto: proto.mean, protoStd: proto.std };

    card.meta.textContent = `${PARAMS[enc]} parameters`;
    card.state.classList.toggle('live', !!r.ms);
    card.state.innerHTML = r.ms && !man.custom && r.live < nTest ? `${r.live} of ${nTest} encoded here · rest precomputed` : '';
    card.scores.innerHTML = '';
    // the run's cost first: the time the encoder took and its throughput, then the scores
    const nLive = man.custom ? nGal + nTest : r.live;
    if (r.ms) {
      score(card.scores, 'on this CPU', secFmt(r.ms * nLive), '', `the encoder's time for the ${nLive} recordings on your CPU`);
      score(card.scores, 'throughput', `${(1000 / r.ms).toFixed(1)}/s`, '', 'recordings encoded per second on your CPU');
    }
    // one score on the page: the k-NN's balanced accuracy (the 5-shot score is still computed for the self-test report)
    score(card.scores, 'accuracy', pct(knn), '', `balanced accuracy of a weighted k-NN (k = ${P.knn_k}, τ = ${P.knn_tau}): each test recording takes the label of its nearest training recordings; the mean recall over the ${k} classes, so guessing at random scores ${pct(1 / k)}`);

    const tsne = projection(enc, r.test, r.stamp);   // the map shows the test recordings; the training vectors only score them
    if (!tsne) {
      // a re-fit of the same dataset keeps the current map on screen; a new dataset shows a placeholder
      if (card.ds !== ex.ds || !card.chart.querySelector('figure')) card.chart.innerHTML = `<div class="map-wait" style="height:340px">fitting the map…</div>`;
      card.ds = ex.ds; delete ex.maps[enc];
      return;
    }
    card.ds = ex.ds;
    const proj = tsne.proj;
    r.tsne = tsne;
    const points = proj.map((xy, j) => {
      const det = knnDetail[j];
      const pending = r.partial && j >= r.live;                 // not yet re-embedded in this run
      const wrong = det.pred !== yt[j];
      return {
        x: xy[0], y: xy[1], hollow: pending,
        cls: yt[j],
        label: `${man.name} ${man.ids_test[j]}`,
        sub: `${names[yt[j]]} · k-NN says ${names[det.pred]} (${Math.round(det.share[det.pred] * 100)}%${wrong ? ', wrong' : ''})`,
        values: testView(d, j, man.preview_variate),
        links: null                                             // its nearest training recordings are not on the map
      };
    });
    const sel = ex.selected ? ex.selected.i : -1;
    ex.maps[enc] = C.redraw(card.chart, (box, width) => C.scatter(box, {
      width, points, classes: names,
      height: 340, selected: sel, minWidth: 200,   // narrower than any card, so the map never overflows its panel
      onHover: (i) => { for (const e of Object.keys(ex.maps)) if (e !== enc) ex.maps[e].highlight(i); },
      onSelect: (i) => showSample(i)
    }));
    $('#legend-explore').hidden = false; $('#ds-hint').hidden = false;   // a map is on screen: the class legend and the dot hint refer to it
  }

  /** The card's chart area as a message box: the model's loading, the embedding count and the fit are reported
   *  there, so the wait is felt where the map will appear. */
  function wait(card, html) {
    let w = card.chart.querySelector('.map-wait');
    if (!w) { card.chart.innerHTML = '<div class="map-wait" style="height:340px"></div>'; w = card.chart.querySelector('.map-wait'); }
    w.innerHTML = html;
  }
  /** The loading line, written into `host`: the same phase, bar, counter and Stop as an Impute or Forecast
   *  row, so a download looks and behaves the same wherever the page fetches a graph. Rebuilt only when it
   *  is not there yet, so the button survives every progress tick (and a tap on a phone). */
  function stopLine(host, enc, msg) {
    let box = host.querySelector('.dl-line');
    if (!box) {
      host.innerHTML = '<span class="dl-line"><span class="load-text"></span><span class="load-bar"><span></span></span>' +
                       '<span class="load-mb"></span><button type="button" class="stop-btn dl-stop">Stop</button></span>';
      box = host.querySelector('.dl-line');
      box.querySelector('.dl-stop').addEventListener('click', () => dlStop(enc));
    }
    const p = loadPhase(msg);
    box.querySelector('.load-text').textContent = p.label;
    box.querySelector('.load-bar').firstElementChild.style.width = `${Math.round(100 * Math.max(0, Math.min(1, p.frac)))}%`;
    box.querySelector('.load-mb').textContent = p.mb;
    box.querySelector('.dl-stop').hidden = !p.canStop;   // gone while the runtime starts: the Stop had nothing to act on and looked broken
  }
  /** the same line inside a card's message box (Explore), where the map will appear */
  function waitLoad(card, enc, msg) {
    if (!card.chart.querySelector('.map-wait')) wait(card, '');
    stopLine(card.chart.querySelector('.map-wait'), enc, msg);
  }

  /** The progress bar and its text live inside the message box while the recordings are embedded, and go back under
   *  the chart before the map is drawn (renderMap replaces the chart's content). */
  function footInto(card, inside) {
    if (inside) { const w = card.chart.querySelector('.map-wait'); if (w) w.appendChild(card.foot); }
    else if (card.foot.parentElement !== card.root) card.root.insertBefore(card.foot, card.chart.nextSibling);
  }

  /* live re-embedding of the test set, encoder after encoder, to feel the cost; the map is fitted only once every
   * recording is embedded */
  async function runLive(encs = exEncs()) {
    const key = ex.ds; if (!key || ex.busy) return;
    const d = ex.data[key], man = d.man, n = man.n_test, V = man.V, T = man.T;
    ex.busy = true; ex.stop = false;
    track('Explore', 'run-live', encs.join('+'));
    $$('#dataset-seg .seg-btn, .enc, #maps .rival-select').forEach(b => { b.disabled = true; });
    try {
      for (const enc of encs) {
        if (ex.stop) break;
        const card = ex.cards[enc];
        if (!card || !hasVecs(enc)) continue;
        ex.running = enc;
        card.state.textContent = ''; card.state.classList.remove('live'); placeholderScores(card); delete ex.maps[enc];
        wait(card, 'loading the training vectors…');
        const r = await ensureVecs(key, enc);
        r.partial = true; r.live = 0; r.ms = 0; r.stamp++;
        wait(card, 'loading the model…');
        if (enc === 'moment' && !(await loadMoment('#status-explore', m => waitLoad(card, 'moment', m)))) { r.partial = false; renderMap(enc); continue; }   // its 330 MB graph is fetched on demand, and can be stopped
        if (enc === 'chronos2' && !(await loadZsEmbed(m => waitLoad(card, 'chronos2', m)))) { r.partial = false; renderMap(enc); continue; }                          // its 128 MB graph likewise
        const loader = enc === 'moment' || enc === 'chronos2' ? Promise.resolve()
          : enc === 'units' ? withStop('units', sig => U.loadModel(man.units.graph, m => waitLoad(card, 'units', m), sig))
          : M.loadModel(m => wait(card, m));
        await loader;
        const vecs = enc === 'otis' ? (d.otisVecs || M.variateEmbeddingsLearned(man.embed_rows)) : null;
        wait(card, ''); footInto(card, true);
        card.foot.hidden = false; card.fill.style.width = '0%'; card.footText.textContent = `encoding 0 / ${n} recordings…`;
        let ms = 0;
        const t0 = performance.now();
        for (let i = 0; i < n; i++) {
          const x = d.test.subarray(i * V * T, (i + 1) * V * T);
          let feat, t;
          if (enc === 'moment') { const res = await MO.encode('any', x, V, T); feat = res.feat; t = res.encMs; }
          else if (enc === 'chronos2') { const res = await Z.embed(x, V, T); feat = res.feat; t = res.encMs; }
          else if (enc === 'units') { const res = await U.encode(man.units.graph, x, V, T, d.unitsTokens); feat = res.feat; t = res.encMs; }
          else { const res = await M.encodeAll(x, V, T, vecs); feat = M.meanPool(res.latent); t = res.encMs; }
          ms += t; r.test[i] = Float64Array.from(feat); r.live = i + 1; r.ms = ms / r.live;
          const done = i + 1, eta = (performance.now() - t0) / done * (n - done) / 1000;
          card.fill.style.width = `${done / n * 100}%`;
          card.footText.textContent = `encoding ${done} / ${n} recordings · ${msFmt(ms / done)} each` + (done < n ? ` · ${eta < 1 ? '<1' : Math.round(eta)} s left` : '');
          if (ex.stopAfter && done >= ex.stopAfter) ex.stop = true;
          if (ex.stop) break;
          await yieldUI();
        }
        r.partial = false; r.stamp++; r.precision = enc === 'moment' ? MO.state.precision : enc === 'chronos2' ? Z.state.embedPrecision : enc === 'units' ? U.state.precision : M.state.precision;
        footInto(card, false); card.foot.hidden = true; card.root.querySelector('.foot-stop').disabled = false;
        if (ex.stop && !ex.stopAfter) r.live = 0;      // stopped by the visitor: back to the empty panel, no map of a partial embedding
        renderMap(enc);
      }
      status('#status-explore', '');
    } catch (e) {
      if (wasStopped(e)) status('#status-explore', 'the download was stopped.');
      else { console.error(e); status('#status-explore', e.message || String(e), true); }
    } finally {
      // a run stopped by the visitor goes back to the empty panel: the map is shown only for a complete embedding (the
      // self-test's stopAfter keeps the partly live map)
      for (const enc of EX_ENC) { const r = vecsOf(enc); if (r && r.partial) { r.partial = false; r.stamp++; if (!ex.stopAfter) r.live = 0; if (ex.cards[enc]) { footInto(ex.cards[enc], false); ex.cards[enc].foot.hidden = true; renderMap(enc); } } }
      ex.busy = false; ex.stop = false; ex.running = null;
      $$('#dataset-seg .seg-btn, .enc, #maps .rival-select').forEach(b => { b.disabled = false; });
    }
  }

  /* =====================================================================
   * Your data: two CSV files, embedded live by every selected encoder
   * ===================================================================== */

  /** CSV -> { labels, ids, data (n*V*T), n, T }: a header names the label / id columns; without one, column 0 is the label. */
  function parseCsv(text, V) {
    const lines = text.split(/\r?\n/).filter(l => l.trim());
    if (!lines.length) throw new Error('empty file');
    let cols = lines[0].split(/[,;\t]/).map(c => c.trim().replace(/^"|"$/g, ''));
    let li = 0, ii = -1, start = 1;
    if (cols.some(c => isNaN(+c) && c !== '')) {          // header
      li = cols.findIndex(c => /^(label|class|y|target)$/i.test(c)); if (li < 0) li = 0;
      ii = cols.findIndex(c => /^id$/i.test(c));
    } else start = 0;
    const rows = [];
    for (let i = start; i < lines.length; i++) {
      const c = lines[i].split(/[,;\t]/);
      const vals = [];
      for (let j = 0; j < c.length; j++) if (j !== li && j !== ii) vals.push(+c[j]);
      rows.push({ label: c[li].trim().replace(/^"|"$/g, ''), id: ii >= 0 ? c[ii].trim() : String(i - start), vals });
    }
    const len = rows[0].vals.length;
    if (rows.some(r => r.vals.length !== len)) throw new Error(`rows differ in length (${len} values in the first row): every row needs the label and the same number of values`);
    if (len % V) throw new Error(`${len} values per row do not split into ${V} variates`);
    const Traw = len / V, T = Math.floor(Traw / PW) * PW;
    if (T < PW) throw new Error(`fewer than ${PW} steps per variate`);
    const data = new Float32Array(rows.length * V * T);
    rows.forEach((r, i) => { for (let v = 0; v < V; v++) for (let t = 0; t < T; t++) data[(i * V + v) * T + t] = r.vals[v * Traw + t]; });
    return { labels: rows.map(r => r.label), ids: rows.map(r => r.id), data, n: rows.length, T, trimmed: Traw - T };
  }

  /** Build ex.data.custom from two CSV texts, then embed both sets with every selected encoder. */
  async function buildCustom(trainText, testText) {
    const V = Math.max(1, +$('#custom-v').value || 1), pre = 'zscore_sample';   // uploads are z-scored per sample, all variates together (M.preprocess)
    const g = parseCsv(trainText, V), t = parseCsv(testText, V);
    if (g.T !== t.T) throw new Error(`training rows have ${g.T} steps, test rows ${t.T}`);
    const T = g.T;
    const num = g.labels.concat(t.labels).every(l => l !== '' && !isNaN(+l));
    const classes = Array.from(new Set(g.labels.concat(t.labels))).sort(num ? (a, b) => +a - +b : undefined);
    const idx = (l) => classes.indexOf(l);
    await M.loadMeta(); await U.loadMeta();
    const ug = U.state.meta.graphs;
    const unitsGraph = ug[`${V}x${T}`] ? `${V}x${T}` : ug.any ? 'any' : null;   // a static graph for the shape if hosted, else the shape-free one
    const man = {
      name: 'your data', custom: true, V, T, classes: classes.map((_, i) => i), class_names: Object.fromEntries(classes.map((c, i) => [i, c])),
      n_gallery: g.n, n_test: t.n, labels_gallery: g.labels.map(idx), labels_test: t.labels.map(idx), ids_gallery: g.ids, ids_test: t.ids,
      variates: Array.from({ length: V }, (_, v) => `variate ${v + 1}`), preview_variate: 0, preprocess: pre,
      protocol: { knn_k: Math.min(20, ...classes.map(c => g.labels.filter(l => l === c).length)), knn_tau: 0.07, shots: 5, episodes: 5, seed: 0 },   // k capped at the smallest class of the training file
      units: unitsGraph ? { graph: unitsGraph, d_model: U.state.meta.d_model } : null, moment: app.momentLoaded
    };
    const prep = (src, n) => { const out = new Float32Array(n * V * T); for (let i = 0; i < n; i++) out.set(M.preprocess(src.subarray(i * V * T, (i + 1) * V * T), V, T, pre), i * V * T); return out; };
    const gal = prep(g.data, g.n), test = prep(t.data, t.n);
    const preview = new Float32Array(g.n * T);
    for (let i = 0; i < g.n; i++) preview.set(gal.subarray(i * V * T, i * V * T + T), i * T);
    const otisVecs = M.variateEmbeddingsRandom(V, 0), how = 'random variate embedding, seed 0';   // unseen data: the paper's protocol (Appendix F)
    ex.data.custom = { man, test, gal, preview, unitsTokens: U.encodeTokens(U.randomTokens(V, 0)), otisVecs, how };
    ex.vecs.custom = {}; ex.projCache.custom = {};
    for (const e of EX_ENC) ensureCustomVecs(e);
    const note = `${g.n} training and ${t.n} test series, ${V} × ${T} steps, ${classes.length} classes` + (g.trimmed ? ` (${g.trimmed} trailing steps dropped)` : '');
    status('#status-explore', note);
    renderMaps();
    await runCustom();
  }

  /** The (still empty) vector slots of one encoder for the custom data, if that encoder can embed it. */
  function ensureCustomVecs(e) {
    const d = ex.data.custom; if (!d || ex.vecs.custom[e]) return;
    const man = d.man;
    if (e === 'moment') man.moment = app.momentLoaded;
    if (e === 'chronos2') man.chronos2 = app.zs.embedLoaded;
    const ok = e === 'otis' || (e === 'units' && man.units) || (e === 'moment' && app.momentLoaded) || (e === 'chronos2' && app.zs.embedLoaded);
    if (!ok) { ex.vecs.custom[e] = null; return; }
    const D = e === 'otis' ? app.meta.embed_dim : e === 'units' ? U.state.meta.d_model
      : e === 'chronos2' ? Z.state.meta.models.chronos2.embed.d_model : MO.state.meta.d_model;
    const zeros = (n) => Array.from({ length: n }, () => new Float64Array(D));
    ex.vecs.custom[e] = { gallery: zeros(man.n_gallery), test: zeros(man.n_test), live: 0, ms: null, stamp: 0, pending: true };
  }

  /** A panel clicked on your data: MOMENT's or Chronos-2's graph is fetched first (the other sections then find it loaded), then the encoder embeds. */
  async function runCustomEnc(enc) {
    if (ex.busy || ex.ds !== 'custom' || !ex.data.custom) return;
    const card = ex.cards[enc];
    if (enc === 'moment' || enc === 'chronos2') {
      ex.busy = true; ex.running = enc; ex.stop = false;   // no other panel starts while the graph is fetched; a stop meanwhile cancels the run that would follow
      let ok = false;
      try { ok = enc === 'moment' ? await loadMoment('#status-explore', m => waitLoad(card, 'moment', m)) : await loadZsEmbed(m => waitLoad(card, 'chronos2', m)); }
      finally { ex.busy = false; ex.running = null; }
      if (!ok || ex.stop) { ex.stop = false; renderMap(enc); return; }
    }
    delete ex.vecs.custom[enc]; ensureCustomVecs(enc);
    if (ex.vecs.custom[enc] && ex.vecs.custom[enc].pending) { renderMaps(); await runCustom(); } else renderMap(enc);
  }

  /** Embed the custom gallery and test sets with every selected encoder, then fit the maps. */
  async function runCustom() {
    const d = ex.data.custom, man = d.man, V = man.V, T = man.T;
    ex.busy = true; ex.stop = false;
    $('#custom-go').disabled = true;
    $$('#dataset-seg .seg-btn, .enc, #maps .rival-select').forEach(b => { b.disabled = true; });
    try {
      for (const enc of exEncs()) {
        const r = ex.vecs.custom[enc], card = ex.cards[enc]; if (!r || !r.pending) continue;
        ex.running = enc;                                // the card's stop button acts on the encoder running (makeCard)
        if (card) { card.root.querySelector('.foot-stop').disabled = false; card.state.textContent = 'loading the model…'; }
        if (enc === 'units') await withStop('units', sig => U.loadModel(man.units.graph, m => { if (card) stopLine(card.state, 'units', m); }, sig));
        else if (enc === 'otis') await M.loadModel(m => { if (card) card.state.textContent = m; });
        const sets = [['training', d.gal, r.gallery], ['test', d.test, r.test]];
        let ms = 0, done = 0; const total = r.gallery.length + r.test.length, t0 = performance.now();
        if (card) { card.foot.hidden = false; card.fill.style.width = '0%'; card.footText.textContent = `0 / ${total}`; }   // the same bar as Time it
        for (const [which, sig, dst] of sets) {
          for (let i = 0; i < dst.length; i++) {
            const x = sig.subarray(i * V * T, (i + 1) * V * T);
            let feat, t;
            if (enc === 'moment') { const res = await MO.encode('any', x, V, T); feat = res.feat; t = res.encMs; }
            else if (enc === 'chronos2') { const res = await Z.embed(x, V, T); feat = res.feat; t = res.encMs; }
            else if (enc === 'units') { const res = await U.encode(man.units.graph, x, V, T, d.unitsTokens); feat = res.feat; t = res.encMs; }
            else { const res = await M.encodeAll(x, V, T, d.otisVecs); feat = M.meanPool(res.latent); t = res.encMs; }
            dst[i] = Float64Array.from(feat); ms += t; done++; r.live = done;
            if (card) {
              const eta = (performance.now() - t0) / done * (total - done) / 1000;
              card.fill.style.width = `${done / total * 100}%`;
              card.footText.textContent = `${done} / ${total} · ${which} set · ${msFmt(ms / done)} per series` + (done < total ? ` · ${eta < 1 ? '<1' : Math.round(eta)} s left` : '');
              card.state.textContent = `encoding here… ${done} / ${total}`;
            }
            if (ex.stop) break;
            await yieldUI();
            if (ex.stop) break;                          // the click lands in this pause, before the next series starts
          }
          if (ex.stop) break;
        }
        if (ex.stop) {                                   // stopped by the visitor: this encoder and the ones queued behind it go back to the empty panel; a click runs one again
          if (card) card.foot.hidden = true;
          for (const e of EX_ENC) { const q = ex.vecs.custom[e]; if (q && q.pending) { delete ex.vecs.custom[e]; if (ex.cards[e]) renderMap(e); } }
          break;
        }
        if (card) card.foot.hidden = true;
        r.pending = false; r.live = r.test.length; r.ms = ms / total; r.stamp++;
        r.precision = enc === 'moment' ? MO.state.precision : enc === 'chronos2' ? Z.state.embedPrecision : enc === 'units' ? U.state.precision : M.state.precision;
        renderMap(enc);
        console.log(`[custom] ${enc}: ${total} series embedded, ${msFmt(r.ms)} each; k-NN ${r.scores ? pct(r.scores.knn) : '-'}, 5-shot ${r.scores ? pct(r.scores.proto) : '-'}`);
      }
    } catch (e) {
      if (wasStopped(e)) status('#status-explore', 'the download was stopped.');
      else { console.error(e); status('#status-explore', e.message || String(e), true); }
    } finally {
      for (const e of EX_ENC) if (ex.cards[e]) ex.cards[e].foot.hidden = true;
      for (const e of EX_ENC) { const r = ex.vecs.custom[e]; if (r && r.pending && !r.live) { delete ex.vecs.custom[e]; if (ex.cards[e]) renderMap(e); } }   // the encoders the stop (or an error) left waiting: their panels are buttons again
      ex.busy = false; ex.stop = false; ex.running = null;
      $('#custom-go').disabled = false;
      $$('#dataset-seg .seg-btn, .enc, #maps .rival-select').forEach(b => { b.disabled = false; });
    }
  }

  /** The test recording i under the maps: its label, every encoder's k-NN verdict, and up to three of its variates. */
  function showSample(i) {
    const d = ex.data[ex.ds], man = d.man;
    ex.selected = { i };
    const label = man.labels_test[i];
    const id = man.ids_test[i];
    const cls = man.class_names[man.classes[label]] || man.classes[label];
    const host = $('#sample-view'); host.innerHTML = '';
    const head = document.createElement('div');
    head.className = 'sample-head';
    let verdicts = '';
    for (const enc of exEncs()) {
      const r = vecsOf(enc); if (!r || !r.knn) continue;
      const det = r.knn[i], ok = det.pred === label, pc = man.class_names[man.classes[det.pred]] || man.classes[det.pred];
      verdicts += `<span class="chip ${ok ? 'chip-ok' : 'chip-bad'}"><span class="enc-dot" style="background:${encColor(enc)}"></span>${NAME[enc]}: ${pc} · ${Math.round(det.share[det.pred] * 100)}%${ok ? '' : ' · wrong'}</span>`;
    }
    head.innerHTML = `<strong>Test · ${man.name} ${id}</strong>` +
      `<span class="chip"><span class="swatch dot" style="background:${C.catColor(label)}"></span>${cls}</span>` + verdicts +
      `<button class="link-btn" type="button" id="close-sample">close</button>`;
    host.appendChild(head);
    head.querySelector('#close-sample').addEventListener('click', () => { host.hidden = true; ex.selected = null; for (const e of Object.keys(ex.maps)) ex.maps[e].select(-1); });
    host.hidden = false;
    const pv = man.preview_variate;
    // up to three variates of a test sample: the preview one plus two spread over the rest (twelve-lead ECG: II, V1, V5)
    const pick = man.V === 1 ? [pv] : Array.from(new Set([pv, Math.floor(man.V / 2), man.V - 2 < 0 ? man.V - 1 : man.V - 2])).filter(v => v < man.V).slice(0, 3);
    const colors = [C.css('--truth'), C.css('--cat-4'), C.css('--cat-1')];
    const secs = man.fs ? 1 / man.fs : null;                     // seconds per step; null: an index axis (phase samples)
    const ax = !secs ? { xScale: 1, xTitle: man.x_title || 'time step', xFmt: (t) => String(Math.round(t)) }
      : secs >= 60 ? { xScale: secs / 3600, xTitle: 'hours', xFmt: (t) => `${(+t).toFixed(1)} h` }
      : { xScale: secs, xTitle: 'seconds', xFmt: (t) => `${(+t).toFixed(1)} s` };
    C.lines(host, {
      series: pick.map((v, j) => ({ label: man.variates[v], color: colors[j], values: Array.from(testView(d, i, v)), width: 1.6 })),
      height: 200, ...ax,
      caption: man.V > 1 ? `${pick.length} of the ${man.V} variates, as the model sees them.` : `The ${man.variates[0]} channel as the model sees it.`
    });
    for (const e of Object.keys(ex.maps)) ex.maps[e].select(i);
    host.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  /* =====================================================================
   * Fill in
   * ===================================================================== */

  const FS = 1000, T_FILL = 432, PW = 24, TP = T_FILL / PW;     // the paper's sine corpus: 18 patches of 24 steps
  const SOURCES = [['sine', 'Sine'], ['chirp', 'Chirp'], ['square', 'Square'], ['walk', 'Random walk'], ['draw', 'Draw'], ['paste', 'Paste']];
  const PERIODIC = ['sine', 'chirp', 'square'];
  const HIDES = [['beginning', 'the beginning'], ['middle', 'the middle'], ['end', 'the end'], ['random', 'random parts']];
  const fill = { rival: 'chronos2', source: 'sine', freq: 60, noise: 10, seed: 1, signal: null, mask: new Uint8Array(TP), hide: 'random', results: {}, busy: false, chart: null,
                 V: 1, T: T_FILL, fs: FS, names: null, endPatches: 4, off: new Set(), zsMeta: undefined };   // zsMeta: zero_shot_meta.json, fetched at start (null when unreachable)   // off: models hidden in the chart by a click; a multivariate source may set another shape (setFillShape)
  const fillTp = () => fill.T / PW;

  /** Switch the panel to V variates of T steps; when the grid changes the hide vector is rebuilt on it. */
  function setFillShape(V, T, fs, names, endPatches = 4) {
    const changed = V !== fill.V || T !== fill.T || endPatches !== fill.endPatches;
    fill.V = V; fill.T = T; fill.fs = fs; fill.names = names; fill.endPatches = endPatches;
    if (changed) {
      fill.signal = null; fill.results = {};
      if (!fill.hide) fill.hide = 'random';
      fill.mask = makeMask();
      $$('#hide-chips .chip-btn').forEach(x => x.classList.toggle('is-active', x.dataset.hide === fill.hide));
    }
    return changed;
  }

  function rng(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function zscore(a) {
    const s = a.slice().sort((p, q) => p - q), med = s[Math.floor(s.length / 2)];
    const mad = s.map(v => Math.abs(v - med)).sort((p, q) => p - q)[Math.floor(s.length / 2)] * 1.4826 || 1;
    return a.map(v => Math.max(-3, Math.min(3, (v - med) / mad)));
  }
  function resample(vals, n) {
    const out = new Float32Array(n), m = vals.length;
    for (let i = 0; i < n; i++) { const p = m > 1 ? i / (n - 1) * (m - 1) : 0, k = Math.floor(p), f = p - k; out[i] = k + 1 < m ? vals[k] * (1 - f) + vals[k + 1] * f : vals[k]; }
    return out;
  }

  async function makeSignal() {
    const f = fill.freq, T = T_FILL, a = new Float32Array(T), r = rng(fill.seed);
    switch (fill.source) {
      case 'sine': for (let t = 0; t < T; t++) a[t] = Math.sin(2 * Math.PI * f * t / FS); break;
      case 'chirp': { const f0 = f / 3; for (let t = 0; t < T; t++) { const s = t / FS; a[t] = Math.sin(2 * Math.PI * (f0 * s + (f - f0) * s * s / (2 * T / FS))); } break; }
      case 'square': for (let t = 0; t < T; t++) a[t] = Math.sin(2 * Math.PI * f * t / FS) >= 0 ? 1 : -1; break;
      case 'walk': { let v = 0; for (let t = 0; t < T; t++) { v += gauss(r); a[t] = v; } a.set(zscore(Array.from(a))); break; }
      case 'draw': case 'paste': if (fill.signal) return fill.signal; for (let t = 0; t < T; t++) a[t] = Math.sin(2 * Math.PI * f * t / FS); return a;
    }
    if (fill.noise > 0) { const rn = rng(fill.seed + 7); for (let t = 0; t < T; t++) a[t] += gauss(rn) * fill.noise / 100; }
    return a;
  }

  function makeMask() {
    const N = fillTp(), m = new Uint8Array(N);   // the beginning / the end: 4 patches (96 steps; a source may say more); the middle: the second third; random: half the patches
    if (fill.hide === 'beginning') for (let t = 0; t < fill.endPatches; t++) m[t] = 1;
    else if (fill.hide === 'end') for (let t = N - fill.endPatches; t < N; t++) m[t] = 1;
    else if (fill.hide === 'middle') for (let t = Math.floor(N / 3); t < Math.floor(2 * N / 3); t++) m[t] = 1;
    else if (fill.hide === 'random') { const r = rng(fill.seed + 99); const order = Array.from({ length: N }, (_, i) => i).sort((p, q) => r() - .5); for (let i = 0; i < N / 2; i++) m[order[i]] = 1; }
    return m;
  }

  /** OTIS masking (ids_keep / ids_restore) from a per-patch hide vector, keeping the visible patches in order. */
  function maskingFrom(mask) {
    const N = mask.length, keep = [], hide = [];
    for (let i = 0; i < N; i++) (mask[i] ? hide : keep).push(i);
    const shuffle = keep.concat(hide);
    const idsRestore = new BigInt64Array(N);
    shuffle.forEach((p, i) => { idsRestore[p] = BigInt(i); });
    return { idsKeep: BigInt64Array.from(keep.map(BigInt)), idsRestore, lenKeep: keep.length };
  }

  /** the hidden steps rounded to another grid: a patch is hidden if any of its steps is */
  function regrid(mask, pw) {
    const gTp = fill.T / pw, out = new Uint8Array(gTp);
    for (let p = 0; p < gTp; p++) for (let s = p * pw; s < (p + 1) * pw; s++) if (mask[Math.floor(s / PW)]) { out[p] = 1; break; }
    return out;
  }

  function buildFillControls() {
    const sc = $('#signal-chips');
    SOURCES.forEach(([k, label]) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip-btn' + (k === fill.source ? ' is-active' : ''); b.dataset.source = k; b.textContent = label;
      b.addEventListener('click', async () => {
        if (fill.source === k && k === 'walk') fill.seed++;              // re-roll
        fill.source = k;
        $$('#signal-chips .chip-btn').forEach(x => x.classList.toggle('is-active', x === b));
        $('#paste').hidden = k !== 'paste';
        $('#freq-ctl').hidden = !PERIODIC.includes(k);
        $('#noise-ctl').hidden = !(PERIODIC.includes(k) || k === 'walk');
        $('#props-row').hidden = $('#freq-ctl').hidden && $('#noise-ctl').hidden;   // the Properties row only when a slider applies
        $('#hide-hint').textContent = k === 'draw' ? 'drag across the chart to draw the time series' : 'or drag across the chart';
        setFillShape(1, T_FILL, FS, null);
        if (k !== 'draw' && k !== 'paste') fill.signal = await makeSignal();
        if ((k === 'draw' || k === 'paste') && !fill.signal) fill.signal = await makeSignal();
        fillChanged();
      });
      sc.appendChild(b);
    });
    const hc = $('#hide-chips');
    HIDES.forEach(([k, label]) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip-btn' + (k === fill.hide ? ' is-active' : ''); b.dataset.hide = k; b.textContent = label;
      b.addEventListener('click', () => {
        if (fill.hide === k && k === 'random') fill.seed++;
        fill.hide = k; fill.mask = makeMask();
        $$('#hide-chips .chip-btn').forEach(x => x.classList.toggle('is-active', x === b));
        fillChanged();
      });
      hc.appendChild(b);
    });
    $('#freq').addEventListener('input', async () => { fill.freq = +$('#freq').value; $('#freq-out').textContent = `${fill.freq} Hz`; fill.signal = await makeSignal(); fillChanged(); });
    $('#noise').addEventListener('input', async () => { fill.noise = +$('#noise').value; $('#noise-out').textContent = `${fill.noise}%`; fill.signal = await makeSignal(); fillChanged(); });
    $('#paste-go').addEventListener('click', () => {
      const vals = $('#paste-text').value.split(/[^0-9eE+\-.]+/).filter(s => s && !isNaN(+s)).map(Number);
      if (vals.length < 8) { status('#status-fill', 'paste at least 8 numbers', true); return; }
      fill.signal = resample(Float32Array.from(zscore(vals)), T_FILL);
      status('#status-fill', `${vals.length} values → ${T_FILL} steps`);
      fillChanged();
    });
  }

  /* Nothing runs by itself here: every row carries a "Run" button, so the time each model takes is
   * always felt on this CPU (and the first click fetches the model). A change to the signal or to the hidden
   * span makes the results stale, and the rows offer to run again. */
  function fillChanged() {
    holdHeight('#fill-rows');
    fill.stamp++;
    renderFill();
  }

  /** Run one model on the current state; requests queue up and run one at a time. */
  function runFillModel(enc) {
    if (justStopped() || fill.running[enc] || fill.queue.includes(enc)) return;
    track('Impute', 'run', enc);
    fill.queue.push(enc); renderFill();
    if (!fill.busy) drainFill();
  }
  async function drainFill() {
    fill.busy = true;
    try { while (fill.queue.length) await runFillOne(fill.queue.shift()); }
    finally { fill.busy = false; fill.abort = false; }
  }
  /** The text of a running row; with `frac` (0..1) its loading bar shows that much (and `mb`, "36 / 131 MB", follows
   *  the bar), without it the bar is hidden. */
  const fillNote = (enc, m, frac, mb, canStop) => {
    const row = fill.rows && fill.rows[enc]; if (!row || !row.note) return;
    row.note.textContent = m;
    if (row.bar) { row.bar.hidden = frac === undefined; if (frac !== undefined) row.bar.firstElementChild.style.width = `${Math.round(100 * Math.max(0, Math.min(1, frac)))}%`; }
    if (row.mb) row.mb.textContent = mb || '';
    if (row.stop) row.stop.hidden = !canStop;                        // gone only while the runtime starts, which cannot be interrupted
  };

  /** What the row says once Stop is pressed: a download or a run with a point to be left at ends now, a
   *  prediction already inside one uninterruptible call ends when that call returns. */
  const stopWord = (enc) => !dls[enc] && ZS.includes(enc) ? 'stopping after this prediction…' : 'stopping…';

  /* The visitor's Stop, for the whole run and not only its download: the fetch is abandoned, the models still
   * queued behind this one are dropped, and the model itself is left at the first point it can be left —
   * between the passes of a rolled forecast, else when the call it is inside returns. A prediction that was
   * stopped is discarded rather than scored: half a forecast is not a forecast. It is their CPU. */
  function stopFill(enc) {
    dlStop(enc); if (fill.busy) { fill.abort = true; fill.queue.length = 0; }
    renderFill(); fillNote(enc, stopWord(enc), undefined, '', false);   // the render rebuilds the rows, so the message is written after it
  }

  async function runFillOne(enc) {
    const sig = fill.signal, mask = fill.mask, V = fill.V, T = fill.T, Tp = T / PW, stamp = fill.stamp;
    if (!sig) return;
    fill.running[enc] = true; renderFill(); await yieldUI();   // the row says predicting… before the model blocks the thread
    const say = m => fillNote(enc, m, undefined, '', true);   // progress lives in the row only; the status line keeps errors
    // the loaders' messages, shown in this row only, reduced to a bar and the download so far
    // ("loading… [bar] 12 / 131 MB", the row already names the model); the bar is full while the runtime starts
    const load = m => { const p = loadPhase(m); fillNote(enc, p.label, p.frac, p.mb, enc !== 'otis' && p.canStop); };
    // what each model is fed: the signal with the hidden steps (on that model's grid) set to zero.
    // None of the three reads them (OTIS gathers the kept patches, UniTS and MOMENT normalise over
    // the visible steps and swap in a mask token), but the int8 graphs pick their activation scales
    // from whole tensors, so zeros keep the prediction exactly independent of what is hidden.
    const blank = (gm, gPw) => { const s = Float32Array.from(sig); for (let v = 0; v < V; v++) for (let t = 0; t < T; t++) if (gm[Math.floor(t / gPw)]) s[v * T + t] = 0; return s; };
    // the hide vector is one span of time steps shared by every variate; OTIS' masking and the metrics take it per variate
    const perVariate = (gm, gTp) => { const m = new Uint8Array(V * gTp); for (let v = 0; v < V; v++) m.set(gm, v * gTp); return m; };
    try {
      let pred, gm, gPw, gTp, ms, how;
      if (ZS.includes(enc)) {
        // a forecaster, so it fills a hidden tail only: from the visible steps before it (at most 336) to the end,
        // every variate at once, 96 steps per pass; a longer tail is rolled, each pass appended to the context
        gPw = PW; gTp = Tp; gm = mask;
        const split = fillTail();
        if (!split) throw new Error(`${NAME[enc]} forecasts, so it can only fill a hidden tail`);
        if (!(await loadZeroShot(enc, '#status-fill', load))) return;   // the reason stays in the status line
        if (stamp !== fill.stamp) return;
        say('predicting…'); await yieldUI();
        const { ctx, hz } = split;
        const res = await Z.forecastTail(enc, sig, V, T, ctx, (i, n) => n > 1 && say(`predicting… pass ${i} of ${n}`), () => fill.abort);
        pred = Float32Array.from(sig);
        for (let v = 0; v < V; v++) pred.set(res.tail.subarray(v * hz, (v + 1) * hz), v * T + ctx);
        ms = res.ms;
        how = `zero-shot, ${PARAMS[enc]} parameters · median forecast of the last ${hz} steps from the ${Math.min(ctx, ZS_CTX)} before` +
              (res.passes > 1 ? `, ${res.passes} passes of ${ZS_HZ}` : '');
      } else if (enc === 'otis') {
        if (!M.state.enc) await M.loadModel(load);
        if (stamp !== fill.stamp) return;
        gPw = PW; gTp = Tp; gm = mask;
        // one embedding per periodic source, each fitted on that source from 20 to 120 Hz; the sine's for everything else
        const fit = PERIODIC.includes(fill.source) ? fill.source : 'sine';
        const vecs = await M.loadFittedEmbed(fit) || (fit !== 'sine' ? await M.loadSineEmbed() : null);
        if (!vecs) throw new Error('the sine embedding is not available');
        how = `variate embedding fitted on ${vecs === M.state.fitted[fit] ? fit : 'sine'}s from 20 to 120 Hz`;
        say('predicting…'); await yieldUI();
        const res = await M.run(blank(gm, gPw), V, T, vecs, maskingFrom(perVariate(mask, Tp)));
        pred = M.unpatchify(res.pred, V, Tp, PW); ms = res.encMs + res.decMs;
      } else if (enc === 'units') {
        const graph = `gen_${V}x${T}`;                                // one static graph per shape
        await U.loadMeta();
        if (!U.state.meta.graphs[graph]) { fill.results[enc] = { na: true, stamp, why: `n/a: no UniTS graph for ${V} × ${T}` }; return; }
        if (!U.state.sessions[graph]) await withStop(enc, sig => U.loadModel(graph, load, sig));
        if (stamp !== fill.stamp) return;
        gPw = U.state.meta.patch_len; gTp = T / gPw; gm = regrid(mask, gPw);
        // tokens prompt-tuned on the source from 20 to 120 Hz for the periodic presets, the sine's for everything else: OTIS' rule
        const ufit = PERIODIC.includes(fill.source) ? fill.source : 'sine';
        let tokens = await U.loadTokens(ufit) || (ufit !== 'sine' ? await U.loadSineTokens() : null);
        how = tokens ? `tokens prompt-tuned on ${tokens === U.state.tokens[ufit] ? ufit : 'sine'}s from 20 to 120 Hz` : 'random dataset tokens, zero-shot';
        if (!tokens) tokens = U.randomTokens(V, 0);
        say('predicting…'); await yieldUI();
        const res = await U.generate(graph, blank(gm, gPw), V, T, gm, tokens);
        pred = res.pred; ms = res.encMs;
      } else {
        if (!(await loadMoment('#status-fill', load))) return;
        if (stamp !== fill.stamp) return;
        gPw = MO.state.meta.patch_len; gTp = T / gPw; gm = regrid(mask, gPw);
        // MOMENT's masked reconstruction (its pre-training task) with its two task parameters, the mask token and the
        // reconstruction head, fitted on the source from 20 to 120 Hz for the periodic presets and on the sine for everything
        // else: OTIS' rule, as for UniTS' prompt tokens. The encoder is frozen; the released pair is the fallback.
        const mfit = PERIODIC.includes(fill.source) ? fill.source : 'sine';
        let tokens = await MO.loadTokens(mfit) || (mfit !== 'sine' ? await MO.loadTokens('sine') : null);
        how = tokens ? `mask token and reconstruction head fitted on ${tokens === MO.state.tokens[mfit] ? mfit : 'sine'}s from 20 to 120 Hz` : 'released mask token and head, zero-shot';
        if (stamp !== fill.stamp) return;
        say('predicting…'); await yieldUI();
        const res = await MO.generate('any', sig, V, T, gm, tokens || undefined);
        pred = res.pred; ms = res.encMs;
      }
      if (fill.abort) { status('#status-fill', `the ${NAME[enc]} run was stopped.`); return; }   // said here as in the catch: a run stopped inside one call lands here
      if (stamp !== fill.stamp) return;                                 // the state changed meanwhile: the rows offer to run again
      let nHidden = 0; for (let t = 0; t < gTp; t++) nHidden += gm[t];
      const gmV = perVariate(gm, gTp);
      fill.results[enc] = { pred, mask: gm, gPw, gTp, nHidden, ms, how, ncc: M.nccMasked(pred, sig, gmV, V, gTp, gPw), mae: M.mae(pred, sig, gmV, V, gTp, gPw), mse: M.mse(pred, sig, gmV, V, gTp, gPw), stamp };
      status('#status-fill', '');
    } catch (e) {
      if (wasStopped(e)) status('#status-fill', `the ${NAME[enc]} run was stopped.`);
      else { console.error(e); status('#status-fill', `${NAME[enc]}: ${e.message || String(e)}`, true); }
    } finally {
      delete fill.running[enc];
      renderFill();
    }
  }
  fill.stamp = 0; fill.running = {}; fill.queue = []; fill.abort = false;

  const zsHosted = (enc) => !!(fill.zsMeta && fill.zsMeta.models[enc]);   // zero_shot_meta.json lists the graphs this host serves
  const zsMb = (enc) => zsHosted(enc) ? `${fill.zsMeta.models[enc].sizes_mb[fill.zsMeta.models[enc].default || 'int8']} MB` : 'a large download';
  // what the first click fetches (the int8 graphs; OTIS: encoder + decoder, UniTS: its 1 x 432 generative graph)
  const FILL_MB = { otis: '7 MB', units: '6 MB', moment: '330 MB' };
  const fillLoaded = (enc) => enc === 'otis' ? !!M.state.enc : enc === 'units' ? !!(U.state.sessions && U.state.sessions[`gen_${fill.V}x${fill.T}`]) : enc === 'moment' ? app.momentLoaded : !!app.zs.loaded[enc];
  const fillMb = (enc) => ZS.includes(enc) ? zsMb(enc) : FILL_MB[enc];
  /** hovering a model's row highlights its line in the chart (charts.js focus) */
  const linkRow = (el, chart, label) => {
    el.classList.add('is-linked');
    el.addEventListener('pointerenter', () => { const c = chart(); c && c.api && c.api.focus(label); });
    el.addEventListener('pointerleave', () => { const c = chart(); c && c.api && c.api.focus(null); });
  };
  const fillRows = () => ['otis', fill.rival];   // OTIS and the competitor picked in the second row
  /** {ctx, hz} when the hidden steps are one span reaching the end with at least one visible patch before it
   *  (what a forecaster can fill: ctx visible steps, then hz hidden ones), else null. */
  function fillTail() {
    const N = fillTp();
    let a = 0; while (a < N && !fill.mask[a]) a++;                     // the visible head
    if (a === 0 || a === N) return null;
    for (let p = a; p < N; p++) if (!fill.mask[p]) return null;        // then hidden to the end, without a gap
    return { ctx: a * PW, hz: (N - a) * PW };
  }

  function renderFill() {
    const sig = fill.signal; if (!sig) return;
    const V = fill.V, T = fill.T, N = fillTp(), encs = fillRows();
    const bands = [];
    let run = -1;
    for (let t = 0; t <= N; t++) {
      const h = t < N && fill.mask[t];
      if (h && run < 0) run = t; else if (!h && run >= 0) { bands.push({ from: run * PW, to: t * PW }); run = -1; }
    }
    // each model's prediction on the hidden steps of its own grid, NaN elsewhere (V * T, variate-major like the signal)
    // clicking a model in the legend or on its row hides its line (the row stays, dimmed); click again to bring it back
    const toggle = (enc) => { if (fill.off.has(enc)) fill.off.delete(enc); else fill.off.add(enc); renderFill(); };
    const done = encs.filter(enc => { const r = fill.results[enc]; return r && r.stamp === fill.stamp && !r.na; });
    const legendItems = [{ label: 'observed', color: C.css('--truth') }].concat(done.map(enc => ({ label: NAME[enc], color: encColor(enc), off: fill.off.has(enc), onClick: () => toggle(enc) })));
    const preds = [];
    for (const enc of done) {
      if (fill.off.has(enc)) continue;
      const r = fill.results[enc];
      const shown = new Array(V * T).fill(NaN);
      for (let v = 0; v < V; v++) for (let p = 0; p < r.gTp; p++) if (r.mask[p]) for (let k = 0; k < r.gPw; k++) { const t = p * r.gPw + k; shown[v * T + t] = r.pred[v * T + t]; }
      preds.push({ enc, shown });
    }
    const draw = fill.source === 'draw', truth = C.css('--truth');
    let series, chartRows = null, yDomain = null, height = 340;
    if (V === 1) {
      series = [{ label: 'observed', color: truth, values: Array.from(sig), width: 2 }];
      for (const p of preds) series.push({ label: NAME[p.enc], color: encColor(p.enc), values: p.shown, dashed: true, width: 2 });
      if (draw) yDomain = [-2.5, 2.5];
    } else {
      // one row per variate, the first on top, each scaled into its own band; the tooltip reports the real values
      series = []; chartRows = [];
      const band = 0.92, clamp = x => Math.max(-0.55, Math.min(0.55, x));
      for (let v = 0; v < V; v++) {
        const off = V - 1 - v;
        chartRows.push({ label: fill.names ? fill.names[v] : `variate ${v + 1}`, offset: off });
        let amp = 0; for (let t = 0; t < T; t++) amp = Math.max(amp, Math.abs(sig[v * T + t]));
        const sc = band / 2 / (amp || 1);
        const raw = Array.from(sig.subarray(v * T, (v + 1) * T));
        series.push({ label: 'observed', color: truth, values: raw.map(x => off + clamp(x * sc)), raw, width: 1.5, row: v });
        for (const p of preds) {
          const pr = p.shown.slice(v * T, (v + 1) * T);
          series.push({ label: NAME[p.enc], color: encColor(p.enc), values: pr.map(x => Number.isFinite(x) ? off + clamp(x * sc) : NaN), raw: pr, dashed: true, width: 1.5, row: v });
        }
      }
      yDomain = [-0.6, V - 0.4]; height = 56 * V + 70;
    }
    fill.chart = C.redraw($('#chart-fill'), (box, width) => C.lines(box, {
      width, series, bands, height, rows: chartRows, legendItems, yDomain, xScale: 1 / fill.fs, xTitle: 'seconds', xFmt: (t) => `${(+t).toFixed(2)} s`,
      interact: {
        mode: draw ? 'draw' : 'span', patch: PW,
        onSpan: (from, to) => { const m = new Uint8Array(N); for (let p = from / PW; p < to / PW; p++) m[p] = 1; fill.mask = m; fill.hide = null; $$('#hide-chips .chip-btn').forEach(x => x.classList.remove('is-active')); fillChanged(); },
        onDraw: (vals) => { fill.signal = Float32Array.from(vals); fillChanged(); }
      }
    }));
    // one row per model: its result on this state, or the button that runs it
    const rows = $('#fill-rows'); rows.innerHTML = ''; fill.rows = {};
    const atEnd = !!fillTail();
    for (const enc of encs) {
      const r = fill.results[enc], cur = r && r.stamp === fill.stamp;
      const el = document.createElement('div'); el.className = 'enc-row'; el.dataset.enc = enc;
      const name = enc === 'otis' ? `<span class="enc-name">${NAME[enc]}</span>` : rivalHtml(enc, RIVALS);
      if (fill.running[enc]) {
        el.innerHTML = `${name}<span class="note"><span class="load-text">starting…</span><span class="load-bar" hidden><span></span></span><span class="load-mb"></span><button type="button" class="stop-btn load-stop">Stop</button></span>`;   // the loader's progress replaces this text
        el.querySelector('.load-stop').addEventListener('click', () => stopFill(enc));
      } else if (fill.queue.includes(enc)) {
        el.innerHTML = `${name}<span class="note">queued, after the model running now…</span>`;
      } else if (cur && r.na) {
        el.innerHTML = `${name}<span class="note">${r.why}</span>`;
      } else if (cur) {
        el.innerHTML = name +
          `<span class="cell" data-tip="${ZS.includes(enc) ? 'forecast' : 'encoder and decoder'} time for this run, in your browser"><b>${msFmt(r.ms)}</b><span>on this CPU</span></span>` +
          `<span class="cell" data-tip="mean squared error on the hidden steps: like MAE, but large misses weigh more. Lower is better."><b>${C.fmt(r.mse, 2)}</b><span>MSE</span></span>` +
          `<span class="cell" data-tip="mean absolute error: the average distance between prediction and truth on the hidden steps, in the signal's units. Lower is better."><b>${C.fmt(r.mae, 2)}</b><span>MAE</span></span>` +
          `<span class="cell" data-tip="${Number.isNaN(r.ncc) ? 'not defined here: the truth is constant on the hidden steps, so it has no shape to correlate with. MAE and MSE still score the fit.' : 'normalised cross-correlation: how well the predicted shape follows the truth on the hidden steps, ignoring offset and scale. From -1 to 1: 1 is a perfect match, 0 no relation, -1 the inverted shape.'}"><b>${nccFmt(r.ncc)}</b><span>NCC</span></span>`;
      } else if (ZS.includes(enc) && !atEnd) {
        el.innerHTML = `${name}<span class="note">n/a · ${NAME[enc]} is a forecasting model, so it can unfortunately fill a hidden tail only :(</span>`;
      } else if (ZS.includes(enc) && fill.zsMeta && !zsHosted(enc) && !app.zs.loaded[enc]) {
        // TimesFM-3's license forbids redistributing the weights, so the public page has no graph for it (a local copy does)
        el.innerHTML = `${name}<span class="note">n/a: ${NAME[enc]}'s license does not allow its weights to be redistributed.</span>`;
      } else {
        const hint = fillLoaded(enc) ? `${PARAMS[enc]} parameters, loaded.` : `${PARAMS[enc]} parameters; the first run fetches ${fillMb(enc)}.`;
        el.innerHTML = `${name}<span class="note">${hint} <button type="button" class="btn tiny" data-run="${enc}">Run</button></span>`;
        el.querySelector('[data-run]').addEventListener('click', () => runFillModel(enc));
      }
      if (cur && !r.na) { el.classList.toggle('is-off', fill.off.has(enc)); const nm = el.querySelector('.enc-name'); nm.classList.add('is-toggle'); nm.title = fill.off.has(enc) ? 'click to show this model again' : 'click to hide this model in the chart'; nm.addEventListener('click', () => toggle(enc)); }
      if (cur && !r.na && !fill.off.has(enc)) linkRow(el, () => fill.chart, NAME[enc]);
      wireRival(el, Object.values(fill.running).some(Boolean) || fill.queue.length, v => { fill.rival = v; holdHeight('#fill-rows'); renderFill(); });
      rows.appendChild(el);
      fill.rows[enc] = { note: el.querySelector('.load-text') || el.querySelector('.note') || el.querySelector('.right'), bar: el.querySelector('.load-bar'), mb: el.querySelector('.load-mb'), stop: el.querySelector('.load-stop') };
    }
  }

  /* =====================================================================
   * Forecast (section id: bench)
   * ===================================================================== */

  // the four sets a reader outside the field can picture; the ETT transformer-load sets stay hosted (docs/data/forecast_ett*) but are not shown
  const FC_KEYS = ['weather', 'ili', 'electricity', 'traffic'];
  const FC_DOM = { weather: 'weather', ili: 'health', electricity: 'energy', traffic: 'traffic' };
  // one plain line under the chips, like the Explore blurbs: what the series is and what is asked of the model (336 steps in, 96 out; 48 for Illness)
  const FC_BLURB = {
    weather: 'Two days of weather readings, one every ten minutes: what does the weather do over the next sixteen hours?',
    ili: 'Six years of weekly flu case counts: how many people fall ill over the next year?',
    electricity: 'Two weeks of hourly electricity use: how much power is used over the next four days?',
    traffic: 'Two weeks of hourly traffic on one road: how busy does it get over the next four days?'
  };
  // result: key -> window -> model; running / queue: jobs keyed key/window/model, so a run finishes for the window it
  // was started on even when the reader moves on; off: models hidden in the chart by a click
  const fc = { rival: 'chronos2', man: {}, windows: {}, ds: null, win: 0, variate: 0, result: {}, running: {}, queue: [], busy: false, abort: false, off: new Set(), rows: {}, chart: null };
  const fcRes = (key, win, enc) => fc.result[key] && fc.result[key][win] && fc.result[key][win][enc];
  const fcJob = (key, win, enc) => `${key}/${win}/${enc}`;

  async function fcManifest(key) {
    if (fc.man[key]) return fc.man[key];
    const r = await fetch(`data/forecast_${key}.json`);
    if (!r.ok) throw new Error(`forecast_${key}.json: HTTP ${r.status}`);
    return (fc.man[key] = await r.json());
  }
  async function fcLoad(key) {
    const man = await fcManifest(key);
    if (!fc.windows[key]) fc.windows[key] = new Float32Array(await fetchBuf(man.files.windows));
    return man;
  }
  const fcWindow = (key, i) => { const m = fc.man[key], n = m.V * m.T; return fc.windows[key].subarray(i * n, (i + 1) * n); };

  async function buildBenchList() {
    const host = $('#bench-list');
    const mans = await Promise.all(FC_KEYS.map(k => fcManifest(k).catch(() => null)));
    mans.forEach((man, i) => {
      if (!man) return;
      const key = FC_KEYS[i];
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'chip-btn'; b.dataset.key = key; b.textContent = man.name; b.title = fcBlurb(key, man);
      b.addEventListener('click', () => selectBench(key));
      host.appendChild(b);
    });
  }
  /** one line under the chips: domain, variates, sampling and the forecasting setup of the chosen dataset */
  const fcBlurb = (key, man) => `${FC_DOM[key]} · ${man.n_series > 1 ? `${man.n_series} series` : `${man.V} variates`} · every ${man.step} · ${man.context} → ${man.horizon} steps`;

  /** Keep an element at least as tall as it has been (until the viewport is resized), so re-rendering rows or
   *  clearing them while a dataset loads never shortens the page under the reader: the cause of the page jumping. */
  const held = {};
  function holdHeight(sel) {
    const el = $(sel); held[sel] = Math.max(held[sel] || 0, el.offsetHeight); el.style.minHeight = `${held[sel]}px`;
  }
  const holdBenchHeight = () => holdHeight('#bench');
  async function selectBench(key) {
    holdBenchHeight();
    fc.ds = key; fc.win = 0;
    track('Forecast', 'benchmark', key);
    $$('#bench-list button').forEach(b => b.classList.toggle('is-active', b.dataset.key === key));
    $('#fc-blurb').textContent = FC_BLURB[key] || '';
    status('#status-fc', 'loading the test windows…');
    try {
      const man = await fcLoad(key);
      if (fc.ds !== key) return;
      const seg = $('#fc-windows');
      seg.querySelectorAll('button').forEach(b => b.remove());
      man.windows.forEach((w, i) => {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'seg-btn' + (i === 0 ? ' is-active' : ''); b.dataset.win = String(i);
        b.textContent = `${i + 1}`;   // numbered on every dataset; the series behind a multi-series window is in the tooltip
        b.title = man.n_series > 1 ? `series ${w.series}, from step ${w.offset} of the test split` : `from step ${w.offset} of the test split`;
        b.addEventListener('click', () => { fc.win = i; seg.querySelectorAll('button').forEach(x => x.classList.toggle('is-active', x === b)); fcChanged(); });
        seg.appendChild(b);
      });
      const sel = $('#fc-variate'); sel.innerHTML = '';
      const names = man.variates || ['the series'];
      names.forEach((n, i) => { const o = document.createElement('option'); o.value = String(i); o.textContent = n; sel.appendChild(o); });
      fc.variate = Math.max(0, names.indexOf('OT'));
      sel.value = String(fc.variate);
      $('#fc-variate-ctl').classList.toggle('is-void', names.length < 2);   // keeps its space: no layout shift between datasets
      status('#status-fc', '');
      fcChanged();
    } catch (e) {
      console.error(e);
      status('#status-fc', `could not load ${key}: ${e.message}`, true);
    }
  }

  /* Nothing runs by itself here either: like the Fill-in rows, every model runs when its "Run" is clicked,
   * in the browser, from its int8 graph, and the row shows how long the window took on this CPU. A window's
   * results are exact for that window, so they are kept while the reader moves between windows and datasets. */
  function fcChanged() { holdBenchHeight(); renderFc(); }
  const fcRows = () => ['otis', fc.rival];   // OTIS and the competitor picked in the second row
  const fcCurrent = (key, win) => fc.ds === key && fc.win === win;

  function runFcModel(enc) {
    const key = fc.ds, win = fc.win, id = fcJob(key, win, enc);
    if (justStopped() || !key || fc.running[id] || fc.queue.some(j => j.id === id) || fcRes(key, win, enc)) return;
    track('Forecast', 'run', enc);
    fc.queue.push({ key, win, enc, id }); renderFc();
    if (!fc.busy) drainFc();
  }
  async function drainFc() {
    fc.busy = true;
    try { while (fc.queue.length) await runFcOne(fc.queue.shift()); }
    finally { fc.busy = false; fc.abort = false; }
  }
  /** the text of a running row (see fillNote) */
  const fcNote = (enc, m, frac, mb, canStop) => {
    const row = fc.rows[enc]; if (!row || !row.note) return;
    row.note.textContent = m;
    if (row.bar) { row.bar.hidden = frac === undefined; if (frac !== undefined) row.bar.firstElementChild.style.width = `${Math.round(100 * Math.max(0, Math.min(1, frac)))}%`; }
    if (row.mb) row.mb.textContent = mb || '';
    if (row.stop) row.stop.hidden = !canStop;
  };
  // the OTIS fine-tune with RevIN: each variate z-scored on its context (std floored as in training), the
  // prediction mapped back. The manifest's `revin` block carries its bundle and scores; the plain fine-tune stays
  // in the manifest as `reference` but is not shown
  const fcOtisBlock = (man) => man.revin || man;

  /** the Stop of a Forecast row: see stopFill */
  function stopFc(enc) {
    dlStop(enc); if (fc.busy) { fc.abort = true; fc.queue.length = 0; }
    renderFc(); fcNote(enc, stopWord(enc), undefined, '', false);
  }

  async function runFcOne(job) {
    const { key, win, enc, id } = job, man = fc.man[key];
    if (!man || !fc.windows[key]) return;
    fc.running[id] = true; renderFc(); await yieldUI();
    const here = () => fcCurrent(key, win);
    const say = m => { if (here()) fcNote(enc, m, undefined, '', true); };   // progress lives in the row only; the status line keeps errors
    const load = m => { if (here()) { const p = loadPhase(m); fcNote(enc, p.label, p.frac, p.mb, enc !== 'otis' && p.canStop); } };
    const V = man.V, T = man.T, pw = man.patch, Tp = man.n_patches, N = V * Tp, ctx = man.context, hz = man.horizon;
    const x = fcWindow(key, win);
    const context = new Float32Array(V * ctx);
    for (let v = 0; v < V; v++) context.set(x.subarray(v * T, v * T + ctx), v * ctx);
    // the scored patches: the horizon on OTIS' 24-step grid, the same for every row
    const masking = M.maskForecast(V, Tp, man.mask_ratio), mask = M.maskVector(masking, N);
    const slot = () => ((fc.result[key] = fc.result[key] || {})[win] = fc.result[key][win] || {});
    try {
      let pred = new Float32Array(V * T), ms, precision, how;
      if (ZS.includes(enc)) {
        if (ctx !== ZS_CTX) throw new Error(`the ${NAME[enc]} graph takes ${ZS_CTX} context steps`);
        if (!(await loadZeroShot(enc, '#status-fc', load))) return;   // the reason stays in the status line
        say('predicting…'); await yieldUI();
        const res = await Z.forecast(enc, context, new Float32Array(V * ctx).fill(1), V);
        for (let v = 0; v < V; v++) pred.set(res.pred.subarray(v * ZS_HZ, v * ZS_HZ + hz), v * T + ctx);   // Illness: the first 48 of the 96
        ms = res.ms; precision = Z.state.precision[enc];
        const zs = man.zero_shot && man.zero_shot[enc];
        how = `zero-shot, ${zs ? zs.params : PARAMS[enc]} parameters · ${zs ? zs.point : 'median'} forecast`;
      } else if (enc === 'units') {
        if (!man.units) { slot()[enc] = { na: true, why: `no fine-tune for ${man.name}` }; return; }
        const graph = man.units.graph;
        await withStop(enc, sig => U.loadModel(graph, load, sig));
        say('predicting…'); await yieldUI();
        const res = await U.forecast(graph, context, V, ctx);
        for (let v = 0; v < V; v++) pred.set(res.pred.subarray(v * hz, (v + 1) * hz), v * T + ctx);
        ms = res.encMs; precision = U.state.precision; how = `fine-tuned on ${man.name}`;
      } else if (enc === 'moment') {
        await MO.loadMeta();
        const head = MO.state.meta.forecast && MO.state.meta.forecast.heads[key];
        if (!head) { slot()[enc] = { na: true, why: `no forecasting head for ${man.name}` }; return; }
        if (!(await loadMoment('#status-fc', load))) return;
        await withStop(enc, sig => MO.loadHead(key, load, sig));
        say('predicting…'); await yieldUI();
        const res = await MO.forecast('any', key, context, V, ctx, (d, n) => n > 1 && say(`predicting… variate ${d} of ${n}`), 4, () => fc.abort);
        for (let v = 0; v < V; v++) pred.set(res.pred.subarray(v * hz, (v + 1) * hz), v * T + ctx);
        ms = res.encMs; precision = `${MO.state.precision} encoder, ${MO.state.headPrecision[key]} head`;
        how = `linear probing, ${man.moment ? man.moment.params : PARAMS[enc]} parameters`;
      } else {
        const blk = fcOtisBlock(man), rv = blk !== man;
        const bundle = await M.loadBundle(blk.model, load);
        say('predicting…'); await yieldUI();
        const vecs = M.variateEmbeddingsLearned(blk.embed_rows, bundle);
        let xin = x; const rvMean = new Float64Array(V), rvStd = new Float64Array(V);
        if (rv) {
          xin = new Float32Array(V * T);
          for (let v = 0; v < V; v++) {
            let m = 0; for (let t = 0; t < ctx; t++) m += x[v * T + t]; m /= ctx;
            let q = 0; for (let t = 0; t < ctx; t++) q += (x[v * T + t] - m) ** 2;
            rvMean[v] = m; rvStd[v] = Math.max(Math.sqrt(q / ctx), blk.std_floor || 0.05);
            for (let t = 0; t < T; t++) xin[v * T + t] = (x[v * T + t] - m) / rvStd[v];
          }
        }
        const res = await M.run(xin, V, T, vecs, masking, bundle);
        pred = M.unpatchify(res.pred, V, Tp, pw);
        if (rv) for (let v = 0; v < V; v++) for (let t = 0; t < T; t++) pred[v * T + t] = pred[v * T + t] * rvStd[v] + rvMean[v];
        ms = res.encMs + res.decMs; precision = bundle.precision; how = `fine-tuned on ${man.name}${rv ? ', RevIN' : ''}`;
      }
      if (fc.abort) { status('#status-fc', `the ${NAME[enc]} run was stopped.`); return; }   // stopped by the visitor: nothing to score
      let sq = 0, ab = 0, n = 0;
      const perVar = [];
      for (let v = 0; v < V; v++) {
        let s2 = 0, a2 = 0, c = 0;
        for (let t = 0; t < Tp; t++) {
          if (!mask[v * Tp + t]) continue;
          for (let k = 0; k < pw; k++) { const d = pred[(v * Tp + t) * pw + k] - x[(v * Tp + t) * pw + k]; s2 += d * d; a2 += Math.abs(d); c++; }
        }
        // per variate, for the row of the plotted variate: its own NCC on the hidden steps, like the Fill-in rows
        const ncc = M.nccMasked(pred.subarray(v * T, (v + 1) * T), x.subarray(v * T, (v + 1) * T), mask.subarray(v * Tp, (v + 1) * Tp), 1, Tp, pw);
        perVar.push({ mse: s2 / c, mae: a2 / c, ncc }); sq += s2; ab += a2; n += c;
      }
      slot()[enc] = { pred, mask, mse: sq / n, mae: ab / n, ncc: M.nccMasked(pred, x, mask, V, Tp, pw), perVar, ms, precision, how };
      if (here()) status('#status-fc', '');
    } catch (e) {
      if (wasStopped(e)) status('#status-fc', `the ${NAME[enc]} run was stopped.`);
      else { console.error(e); status('#status-fc', `${NAME[enc]}: ${e.message || String(e)}`, true); }
    } finally {
      delete fc.running[id];
      if (here()) renderFc();
    }
  }

  // what the first click fetches (int8): OTIS' bundle, UniTS' forecasting graph for the dataset, MOMENT's encoder
  // (shared with Explore and Fill in) and the dataset's head, the zero-shot graphs shared with Fill in
  const FC_MB = { otis: '7 MB', units: '4 MB', moment: '330 MB and a 6 MB head' };
  const fcLoaded = (man, key, enc) => enc === 'otis' ? !!(M.bundles[fcOtisBlock(man).model] || {}).enc
    : enc === 'units' ? !!(man.units && U.state.sessions[man.units.graph])
    : enc === 'moment' ? app.momentLoaded && !!MO.state.heads[key] : !!app.zs.loaded[enc];

  function renderFc() {
    const key = fc.ds; if (!key || !fc.windows[key]) return;
    const man = fc.man[key], V = man.V, T = man.T, win = fc.win;
    const encs = fcRows();
    const x = fcWindow(key, win);
    const v = Math.min(fc.variate, V - 1);
    const names = man.variates || man.series_names || ['series'];
    const truth = Array.from(x.subarray(v * T, (v + 1) * T));
    const ctx = man.context;
    // the variate is named in the selector above the chart, so the truth line is just "observed"
    const series = [{ label: 'observed', color: C.css('--truth'), values: truth, width: 2 }];
    // clicking a model in the legend or on its row hides its line (the row stays, dimmed); click again to bring it back
    const toggle = (enc) => { if (fc.off.has(enc)) fc.off.delete(enc); else fc.off.add(enc); renderFc(); };
    const legendItems = [{ label: 'observed', color: C.css('--truth') }];
    for (const enc of encs) {
      const r = fcRes(key, win, enc); if (!r || r.na) continue;
      legendItems.push({ label: NAME[enc], color: encColor(enc), off: fc.off.has(enc), onClick: () => toggle(enc) });
      if (fc.off.has(enc)) continue;
      const shown = new Array(T).fill(NaN);
      for (let t = ctx - 1; t < T; t++) shown[t] = t < ctx ? truth[t] : r.pred[v * T + t];
      series.push({ label: NAME[enc], color: encColor(enc), values: shown, dashed: true, width: 2 });
    }
    // the legend is drawn even before the first result so the chart's height does not change while models run
    fc.chart = C.redraw($('#chart-fc'), (box, width) => C.lines(box, { width, series, bands: [{ from: ctx, to: T }], height: 320, xTitle: `steps of ${man.step}`, legendItems }));   // no caption: the window buttons and the shaded band say it

    const rows = $('#fc-rows'); rows.innerHTML = ''; fc.rows = {};
    for (const enc of encs) {
      const el = document.createElement('div'); el.className = 'enc-row bench-row'; el.dataset.enc = enc;
      const r = fcRes(key, win, enc), id = fcJob(key, win, enc);
      const name = enc === 'otis' ? `<span class="enc-name">${NAME[enc]}</span>` : rivalHtml(enc, RIVALS);
      if (fc.running[id]) {
        el.innerHTML = `${name}<span class="note"><span class="load-text">starting…</span><span class="load-bar" hidden><span></span></span><span class="load-mb"></span><button type="button" class="stop-btn load-stop">Stop</button></span>`;   // the loader's progress replaces this text
        el.querySelector('.load-stop').addEventListener('click', () => stopFc(enc));
      } else if (fc.queue.some(j => j.id === id)) {
        el.innerHTML = `${name}<span class="note">queued, after the model running now…</span>`;
      } else if (r && r.na) {
        el.innerHTML = `${name}<span class="note">n/a: ${r.why}</span>`;
      } else if (r) {
        const zs = man.zero_shot && man.zero_shot[enc], mo = enc === 'moment' ? man.moment : null;
        // the cells score the plotted variate on this window, laid out like the Fill-in rows
        // the horizon on this row's own patch grid: OTIS 24 steps; UniTS' forecast graphs tokenise the context in
        // patches of 16 and append ceil(horizon / 16) output patches (units_meta n_tokens = both + 10 prompt tokens)
        // MOMENT: 8-step patches, the horizon read off its linear-probing head
        const pl = enc === 'units' ? U.state.meta.patch_len : mo ? mo.patch : man.patch, one = V === 1;
        const nHid = Math.ceil(man.horizon / pl), Tp = Math.ceil(man.context / pl) + nHid, pv = r.perVar[v];
        el.innerHTML = name +
          `<span class="cell" data-tip="model time for this window, all ${V} variate${V > 1 ? 's' : ''}, in your browser"><b>${msFmt(r.ms)}</b><span>on this CPU</span></span>` +
          `<span class="cell" data-tip="mean squared error on the hidden steps of ${one ? 'this series' : names[v]}"><b>${C.fmt(pv.mse, 3)}</b><span>MSE</span></span>` +
          `<span class="cell" data-tip="mean absolute error between prediction and truth on the hidden steps of ${one ? 'this series' : names[v]}"><b>${C.fmt(pv.mae, 3)}</b><span>MAE</span></span>` +
          `<span class="cell" data-tip="${Number.isNaN(pv.ncc) ? `not defined here: ${one ? 'this series' : names[v]} is constant on the hidden steps, so it has no shape to correlate with. MAE and MSE still score the fit.` : `normalised cross-correlation between prediction and truth on the hidden steps of ${one ? 'this series' : names[v]}: from -1 to 1, 1 a perfect match of shape`}"><b>${nccFmt(pv.ncc)}</b><span>NCC</span></span>`;
        el.classList.toggle('is-off', fc.off.has(enc));
        const nm = el.querySelector('.enc-name'); nm.classList.add('is-toggle'); nm.title = fc.off.has(enc) ? 'click to show this model again' : 'click to hide this model in the chart'; nm.addEventListener('click', () => toggle(enc));
        if (!fc.off.has(enc)) linkRow(el, () => fc.chart, NAME[enc]);
      } else if (enc === 'units' && !man.units) {
        el.innerHTML = `${name}<span class="note">no fine-tune for ${man.name}</span>`;
      } else if (enc === 'moment' && !man.moment) {
        el.innerHTML = `${name}<span class="note">not fine-tuned for ${man.name}</span>`;
      } else if (ZS.includes(enc) && fill.zsMeta && !zsHosted(enc) && !app.zs.loaded[enc]) {
        // TimesFM-3's license forbids redistributing the weights, so the public page has no graph for it (a local copy does)
        el.innerHTML = `${name}<span class="note">n/a: ${NAME[enc]}'s license does not allow its weights to be redistributed.</span>`;
      } else {
        const mb = ZS.includes(enc) ? zsMb(enc) : enc === 'moment' && app.momentLoaded ? `a ${(MO.state.meta.forecast.heads[key] || { sizes_mb: { int8: 6 } }).sizes_mb.int8} MB head` : FC_MB[enc];
        const hint = fcLoaded(man, key, enc) ? `${PARAMS[enc]} parameters, loaded.` : `${PARAMS[enc]} parameters; the first run fetches ${mb}.`;
        el.innerHTML = `${name}<span class="note">${hint} <button type="button" class="btn tiny" data-run="${enc}">Run</button></span>`;
        el.querySelector('[data-run]').addEventListener('click', () => runFcModel(enc));
      }
      wireRival(el, Object.values(fc.running).some(Boolean) || fc.queue.length, v => { fc.rival = v; renderFc(); });
      rows.appendChild(el);
      fc.rows[enc] = { note: el.querySelector('.load-text') || el.querySelector('.note'), bar: el.querySelector('.load-bar'), mb: el.querySelector('.load-mb'), stop: el.querySelector('.load-stop') };
    }
  }

  /* =====================================================================
   * encoder chips, about drawer, init
   * ===================================================================== */

  function encodersChanged() {
    renderMaps();
    if (fill.signal) { holdHeight('#fill-rows'); renderFill(); }   // a chip adds or drops a row; results already run stay
    if (fc.ds) fcChanged();
  }

  /** The drawer shows one section at a time: the About button the overview, each panel's ? its own notes. */
  // the rail marker follows the section whose top has passed the upper part of the viewport; none above Explore
  function railSpy() {
    const rail = $('#rail'); if (!rail) return;
    const links = $$('.rail-nav a[data-sec]'), secs = links.map(a => document.getElementById(a.dataset.sec));
    let cur = null;
    function update() {
      const line = window.innerHeight * 0.4;
      let idx = -1;
      secs.forEach((sec, i) => { if (sec && sec.getBoundingClientRect().top <= line) idx = i; });
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) idx = secs.length - 1;   // the last section at the page end
      const a = idx >= 0 ? links[idx] : null;
      if (a === cur) return;
      cur = a;
      links.forEach(l => l.classList.toggle('is-active', l === a));
      rail.classList.toggle('has-active', !!a);
      if (a) rail.style.setProperty('--rail-y', `${a.offsetTop}px`);
    }
    let raf = 0;
    const tick = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; update(); }); };
    window.addEventListener('scroll', tick, { passive: true }); window.addEventListener('resize', tick);
    update();
  }

  function openAbout(anchor) {
    const t = $(`#about-${anchor}`) || $('#about-top');
    $('#about').classList.toggle('is-page', t.id === 'about-top');   // the top About fills the window; a section's notes stay a side drawer
    $$('#about .about-sec').forEach(sec => { sec.hidden = sec !== t; });
    $('#about').hidden = false; $('#about-veil').hidden = false;
    $('#about').scrollTop = 0;
    document.documentElement.classList.add('about-open');   // the page behind stops scrolling: the wheel over the drawer would otherwise move it once the drawer's own scroll ends
  }
  function closeAbout() { $('#about').hidden = true; $('#about-veil').hidden = true; document.documentElement.classList.remove('about-open'); }

  // the intro on the first visit: the tagline one line at a time, then the wordmark, which lifts as the subtitle appears,
  // then the overlay fades and the hero rises in. index.html turns it on before the first paint; a click skips it.
  // the hero's waves, after theforecastingcompany.com: a fan of thin trajectories and one bold one, each a sum of three
  // sines whose phases drift at their own pace, spreading out to the right. Drawn only while on screen; still with reduced motion.
  function heroWaves() {
    const svg = document.getElementById('hero-waves'); if (!svg) return;
    const H = 180, N = 14, PTS = 120, AMP = 1.3, NS = 'http://www.w3.org/2000/svg';   // AMP scales the swing of every line
    // the drawing box keeps the element's aspect ratio (square units), so a wavelength spans the same pixels on every
    // screen: a fixed 1000-unit box stretched over a wide page pulled the waves into long noodles. 0.6 to 2 periods per 1000 units.
    let W = 1000;
    const fit = () => { const r = svg.getBoundingClientRect(); W = r.height ? Math.max(500, Math.round(H * r.width / r.height)) : 1000; svg.setAttribute('viewBox', `0 0 ${W} ${H}`); };
    let seed = 20261; const rnd = () => { seed = (seed * 48271) % 2147483647; return seed / 2147483647; };
    const lines = [];
    for (let k = 0; k < N; k++) {
      const comps = [];
      for (let j = 0; j < 3; j++) comps.push({ a: AMP * (26 + 60 * rnd()) / (j + 1), w: (0.6 + 1.4 * rnd()) * 2 * Math.PI / 1000, p: rnd() * 2 * Math.PI, s: (0.12 + 0.3 * rnd()) * (rnd() < 0.5 ? -1 : 1) });
      const path = document.createElementNS(NS, 'path'); path.setAttribute('class', k === 0 ? 'wave-bold' : 'wave');
      if (k > 0) {   // depth: the thin lines sit behind the bold one, fading and thinning towards the back (k = 1 nearest)
        const depth = (k - 1) / (N - 2);
        const near = (1 - depth) ** 2.2;   // only the bold line is black: the thin ones start at 80 % and fade fast into the distance
        path.style.opacity = (0.08 + 0.72 * near).toFixed(2); path.style.strokeWidth = `${(0.5 + 1.0 * near).toFixed(2)}px`;
      }
      lines.push({ comps, path, off: (rnd() - 0.5) * 70 });
    }
    for (const L of lines.slice(1).reverse()) svg.appendChild(L.path); svg.appendChild(lines[0].path);   // back to front, the bold one on top
    const draw = (t) => {
      for (const L of lines) {
        let d = '';
        for (let i = 0; i <= PTS; i++) {
          const x = i / PTS * W, spread = 0.25 + 0.75 * i / PTS;        // the fan opens to the right
          let y = H / 2 + L.off * spread;
          for (const c of L.comps) y += c.a * spread * Math.sin(c.w * x + c.p + c.s * t);
          d += `${i ? 'L' : 'M'}${x.toFixed(1)} ${y.toFixed(1)}`;
        }
        L.path.setAttribute('d', d);
      }
    };
    fit();
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { draw(0); window.addEventListener('resize', () => { fit(); draw(0); }); return; }
    window.addEventListener('resize', fit);
    let onScreen = false, raf = 0;
    const tick = (ms) => { raf = 0; if (!onScreen || document.hidden) return; draw(ms / 1000); raf = requestAnimationFrame(tick); };
    const start = () => { if (!raf) raf = requestAnimationFrame(tick); };
    new IntersectionObserver(es => { onScreen = es[0].isIntersecting; if (onScreen) start(); }).observe(svg);
    document.addEventListener('visibilitychange', () => { if (!document.hidden) start(); });
    draw(0);
  }

  async function playIntro() {
    const root = document.documentElement, box = document.getElementById('intro'), line = document.getElementById('intro-line');
    if (!box) return;
    const otis = box.querySelector('.intro-otis'), sub = box.querySelector('.intro-sub'), brand = box.querySelector('.intro-brand');
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    let over = false;
    const finish = () => {
      if (over) return; over = true;
      box.classList.add('done'); document.querySelector('.hero').classList.add('reveal');
      try { localStorage.setItem('otis-intro', '1'); } catch (e) {}
      setTimeout(() => { box.remove(); root.removeAttribute('data-intro'); }, 1500);   // after the fades: the attribute's rules no longer differ from the plain ones
    };
    box.addEventListener('click', finish);
    try { await document.fonts.ready; } catch (e) {}
    await sleep(400);
    for (const t of ['No text.', 'No image.', 'Simply numbers.']) {
      if (over) return;
      line.textContent = t; line.classList.remove('hide'); void line.offsetWidth; line.classList.add('show');
      await sleep(900);
      line.classList.remove('show'); line.classList.add('hide');
      await sleep(433);   // the three lines take 4 s together
    }
    if (over) return;
    otis.classList.add('show');
    await sleep(1400);
    brand.classList.add('lift'); sub.classList.add('show');
    await sleep(2000);
    finish();
  }

  async function init() {
    // ?enc=otis,units,moment presets the chips (shareable state); ?about opens the drawer
    const q = new URLSearchParams(location.search);
    app.sync = q.has('sync');
    if (document.documentElement.dataset.intro === 'on') playIntro();
    heroWaves();
    // the headline's rotating word cycles through the three sections and then 'cook', one every 2.2 s, and pauses while the tab is hidden.
    // The word's box is as wide as the widest word and the words end where "locally, on your CPU." begins, so the rest of the line never moves.
    {
      const rot = $('.hero h1 .rot'), word = $('#rot-word'), words = ['explore', 'impute', 'forecast', 'cook'];
      let i = 0;
      const measure = () => {
        const m = document.createElement('span'); m.className = 'rot-word rot-measure'; rot.parentElement.appendChild(m);
        rot.style.width = Math.max(...words.map(t => { m.textContent = t; return m.getBoundingClientRect().width; })) + 'px';
        m.remove();
      };
      measure();
      if (document.fonts) { document.fonts.ready.then(measure); document.fonts.addEventListener('loadingdone', measure); }   // the first measurement may be in the fallback font; Instrument Sans's italic is wider
      const next = () => {
        if (document.hidden) return;
        word.classList.add('out');
        setTimeout(() => { i = (i + 1) % words.length; word.textContent = words[i]; word.classList.remove('out'); word.classList.add('pre'); void word.offsetWidth; word.classList.remove('pre'); }, 340);
      };
      setInterval(next, 2200);
      let timer = 0; window.addEventListener('resize', () => { clearTimeout(timer); timer = setTimeout(measure, 150); });   // the headline scales with the viewport
    }
    // dark mode: the page follows the clock, dark from 19:00 to 07:00 local time, re-checked every minute so it flips
    // while open. The button overrides that for the current visit (sessionStorage); index.html applies the same rule
    // before the first paint. The charts read their colours when drawn, so they are redrawn on every change.
    {
      const btn = $('#theme-btn');
      const byClock = () => { const h = new Date().getHours(); return (h >= 19 || h < 7) ? 'dark' : 'light'; };
      const chosen = () => { try { const t = sessionStorage.getItem('theme'); return t === 'dark' || t === 'light' ? t : null; } catch (e) { return null; } };
      const current = () => document.documentElement.dataset.theme || byClock();
      const label = () => { const dark = current() === 'dark'; btn.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode'); btn.title = dark ? 'Light mode' : 'Dark mode'; };
      const retheme = () => { if (ex.ds && ex.data[ex.ds]) renderMaps(); if (fill.signal) renderFill(); if (fc.ds && fc.man[fc.ds]) renderFc(); };
      const apply = t => { if (current() === t) { label(); return; } document.documentElement.dataset.theme = t; label(); retheme(); };
      btn.addEventListener('click', () => { const t = current() === 'dark' ? 'light' : 'dark'; try { sessionStorage.setItem('theme', t); } catch (e) {} apply(t); });
      const tick = () => { if (!chosen()) apply(byClock()); };
      label(); setInterval(tick, 60_000);
      document.addEventListener('visibilitychange', () => { if (!document.hidden) tick(); });   // a tab woken after hours
      try { localStorage.removeItem('theme'); } catch (e) {}                                     // the old permanent choice
    }
    // the phone menu: the hamburger opens a full-screen list of the sections with the theme toggle and the actions at the
    // bottom (bfl.ai's pattern); a link, About, Escape or a window wider than 900px closes it.
    {
      const btn = $('#menu-btn'), menu = $('#menu'), theme = $('#theme-btn'), mtheme = $('#menu-theme');
      const isOpen = () => menu.classList.contains('is-open');
      const set = open => {
        menu.classList.toggle('is-open', open); btn.classList.toggle('is-open', open);
        btn.setAttribute('aria-expanded', String(open)); btn.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
        document.documentElement.classList.toggle('menu-open', open); menu.inert = !open;
      };
      set(false);
      btn.addEventListener('click', () => set(!isOpen()));
      menu.querySelectorAll('a, button').forEach(el => { if (el !== mtheme) el.addEventListener('click', () => set(false)); });
      document.addEventListener('keydown', e => { if (e.key === 'Escape' && isOpen()) set(false); });
      window.addEventListener('resize', () => { if (window.innerWidth > 900 && isOpen()) set(false); });
      const label = () => { mtheme.textContent = theme.title; };   // 'Dark mode' / 'Light mode', kept current by the theme code above
      mtheme.addEventListener('click', () => theme.click());
      new MutationObserver(label).observe(theme, { attributes: true, attributeFilter: ['title'] }); label();
    }
    if (q.get('enc')) { const want = q.get('enc').split(','); if (ENC.some(e => want.includes(e))) for (const e of ENC) { app.on[e] = want.includes(e); $(`.enc[data-enc="${e}"]`).setAttribute('aria-pressed', String(app.on[e])); } }
    $$('.enc').forEach(b => b.addEventListener('click', async () => {
      const e = b.dataset.enc, on = !app.on[e];
      if (!on && selected().length === 1) return;                       // keep at least one
      app.on[e] = on; b.setAttribute('aria-pressed', String(on));
      holdHeight('#fill-rows'); holdHeight('#bench');                   // one row fewer must not move the page
      encodersChanged();                    // MOMENT's maps use offline vectors; the model is loaded when a panel asks for it
      if (on && ex.ds === 'custom' && ex.data.custom && !ex.busy) {   // your data: an encoder switched on later embeds it now
        ensureCustomVecs(e);
        if (ex.vecs.custom[e] && ex.vecs.custom[e].pending) { renderMaps(); await runCustom(); }
      }
    }));
    $$('[data-about]').forEach(b => b.addEventListener('click', () => openAbout(b.dataset.about)));
    $('#about-close').addEventListener('click', closeAbout);
    $('#about-veil').addEventListener('click', closeAbout);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeAbout(); });
    railSpy();

    buildDatasetSeg();
    $('#custom-go').addEventListener('click', async () => {
      const f1 = $('#custom-train').files[0], f2 = $('#custom-test').files[0];
      if (!f1 || !f2) { status('#status-explore', 'choose both files first', true); return; }
      status('#status-explore', 'reading the files…');
      try {
        for (const f of [f1, f2]) {                       // binary files (parquet, npy) cannot be read here
          const head = new Uint8Array(await f.slice(0, 8).arrayBuffer());
          const magic = String.fromCharCode(...head);
          if (magic.startsWith('PAR1') || /\.parquet$/i.test(f.name)) throw new Error(`${f.name} is a parquet file: export it as a text CSV first (a sample of the classes keeps the files small)`);
          if (magic.startsWith('\x93NUMPY') || magic.startsWith('PK') || head.some(b => b === 0)) throw new Error(`${f.name} is not a text CSV`);
          if (f.size > 200e6) throw new Error(`${f.name} is ${Math.round(f.size / 1e6)} MB; sample it first`);
        }
        await buildCustom(await f1.text(), await f2.text());
      } catch (e) { console.error(e); status('#status-explore', e.message || String(e), true); }
    });

    buildFillControls();
    $('#freq-ctl').hidden = false; $('#noise-ctl').hidden = false; $('#props-row').hidden = false;

    buildBenchList().catch(e => console.warn('[bench] no manifests:', e.message));
    $('#fc-variate').addEventListener('change', (e) => { fc.variate = +e.target.value; renderFc(); });

    try {
      app.meta = await M.loadMeta();
      M.loadSineEmbed(); M.loadFittedEmbed('chirp'); M.loadFittedEmbed('square');
      fill.mask = makeMask();
      fill.signal = await makeSignal();
      renderFill();
      Z.loadMeta().then(m => { fill.zsMeta = m; renderFill(); }, () => { fill.zsMeta = null; renderFill(); });
      // the brand always returns to the very top, even when the URL already carries #top or the page sits on another anchor
      $('.brand').addEventListener('click', (e) => { e.preventDefault(); window.scrollTo({ top: 0, behavior: 'smooth' }); if (location.hash) history.replaceState(null, '', location.pathname + location.search); });
      if (q.has('src') && $(`[data-source="${q.get('src')}"]`)) $(`[data-source="${q.get('src')}"]`).click();   // ?src=walk opens Fill in on that source
      window.addEventListener('resize', () => { for (const sel of Object.keys(held)) { held[sel] = 0; $(sel).style.minHeight = ''; } });   // held heights belong to one viewport width
      // the Explore maps are drawn at their card's width, so a change of the viewport width redraws them there (debounced;
      // a height-only change, such as a phone's address bar sliding away, leaves them alone)
      { let timer = 0, lastW = window.innerWidth; window.addEventListener('resize', () => { if (window.innerWidth === lastW) return; clearTimeout(timer); timer = setTimeout(() => { lastW = window.innerWidth; if (ex.ds && ex.data[ex.ds]) renderMaps(); }, 150); }); }
      await selectDataset(q.has('ds') && DS[q.get('ds')] ? q.get('ds') : 'basicmotions');   // ?ds=ptbxl opens Explore on that dataset (?ds=custom: the upload form)
      fillChanged();
      const q0 = new URLSearchParams(location.search).get('fc');   // ?fc=weather opens Forecast on that dataset
      const first = q0 && FC_KEYS.includes(q0) ? q0 : ($('#bench-list button') || {}).dataset?.key; if (first) selectBench(first);
    } catch (e) {
      status('#status-explore', 'could not load model metadata: ' + e.message, true);
    }
    if (q.has('about')) openAbout(q.get('about') || 'top');
    if (q.has('custom')) {                     // ?custom=path/train.csv,path/test.csv: the form's files fetched from the site (checks)
      const [a, b] = q.get('custom').split(',');
      if (q.has('v')) $('#custom-v').value = q.get('v');   // &v=3: variates per row
      const txt = async (u) => { const r = await fetch(u); if (!r.ok) throw new Error(`${u}: HTTP ${r.status}`); return r.text(); };
      (async () => { try { await selectDataset('custom'); await buildCustom(await txt(a), await txt(b)); } catch (e) { console.error(e); status('#status-explore', e.message, true); } })();
    }
    if (q.has('selftest')) selftest();
  }

  /* Headless check: maps and scores for every encoder on both datasets, the fill-in panel, the forecast panel. */
  async function selftest() {
    const only = new URLSearchParams(location.search).get('selftest');   // 'full' | 'fc' | ''
    const full = only === 'full';
    // hovering a model's row: its line thickens, the other predictions fade, its legend entry is marked; leaving restores all
    const hoverCheck = (what, rowsSel, chartSel) => {
      const row = $(`${rowsSel} .enc-row.is-linked`); if (!row) { console.log(`[selftest] ${what} hover: no linked row`); return; }
      const attrs = () => $$(`${chartSel} svg path`).map(p => `${p.getAttribute('stroke-width')}@${p.getAttribute('opacity')}`).join(' ');
      const before = attrs(); row.dispatchEvent(new Event('pointerenter')); const during = attrs(), focus = $(`${chartSel} .legend-item.is-focus`);
      row.dispatchEvent(new Event('pointerleave'));
      console.log(`[selftest] ${what} hover ${row.dataset.enc}: before ${before} | during ${during} | legend focus=${focus ? focus.textContent : 'none'} | restored=${attrs() === before}`);
    };
    const settle = async (flag) => { await new Promise(r => setTimeout(r, 300)); while (flag()) await new Promise(r => setTimeout(r, 50)); };
    const fitted = async () => { await new Promise(r => setTimeout(r, 200)); while (exEncs().some(e => fitting(e))) await new Promise(r => setTimeout(r, 100)); };
    const report = (key) => {
      for (const enc of exEncs()) {
        const r = vecsOf(enc); if (!r || !r.scores) continue;
        console.log(`[selftest] ${key} ${enc} kNN=${r.scores.knn.toFixed(3)} 5shot=${r.scores.proto.toFixed(3)}±${r.scores.protoStd.toFixed(3)} live=${r.live} ms=${r.ms ? r.ms.toFixed(0) : '-'}`);
        if (r.tsne) console.log(`[selftest] ${key} ${enc} t-SNE perplexity=${r.tsne.perplexity} kept=${r.tsne.preserved.toFixed(3)} kl=${r.tsne.kl.toFixed(3)} sweep ${r.tsne.ms.toFixed(0)} ms: ` + r.tsne.tried.map(t => `${t.perplexity}:${t.preserved.toFixed(3)}/${t.kl.toFixed(2)}`).join(' '));
      }
    };
    if (only === 'sample') {                                   // the sample view of every hosted dataset
      for (const key of Object.keys(DS).filter(k => DS[k].file)) {
        await selectDataset(key);
        for (const [set, i] of [['test', 0]]) {
          showSample(i);
          const pts = Array.from($('#sample-view').querySelectorAll('svg path, svg polyline')).map(e => (e.getAttribute('d') || e.getAttribute('points') || '').split(/[ ,LM]+/).filter(Boolean).length);
          const bad = /NaN|Infinity/.test($('#sample-view').innerHTML);
          console.log(`[selftest] sample ${key} ${set}: series=${pts.length} points=${pts.join('/')} axis=${($('#sample-view').textContent.match(/(seconds|hours|time step|phase \([^)]*\))/) || [])[1] || '?'} nan=${bad}`);
        }
      }
      const wide = Array.from(document.querySelectorAll('body *')).filter(e => e.getBoundingClientRect().right > innerWidth + 1 && e.getBoundingClientRect().width > 0);
      console.log(`[selftest] overflow at ${innerWidth}px: ${document.documentElement.scrollWidth}px wide; ` + wide.slice(0, 12).map(e => `${e.tagName.toLowerCase()}${e.id ? '#' + e.id : ''}${e.className && typeof e.className === 'string' ? '.' + e.className.split(' ').join('.') : ''}=${Math.round(e.getBoundingClientRect().right)}`).join(' '));
      console.log('[selftest] done'); return;
    }
    if (only !== 'fc' && only !== 'fill') {
      for (const e of ENC) if (!app.on[e]) $(`.enc[data-enc="${e}"]`).click();   // the check covers every encoder, whatever the default chips
      for (const key of ['basicmotions', 'starlight', 'kitchen', 'ptbxl', 'sleepedf']) {
        await selectDataset(key);
        await fitted();
        report(key + ' offline');
        if (full) await loadMoment('#status-explore');
        ex.stopAfter = full ? 3 : 0;          // MOMENT: feel three recordings, keep the rest offline
        // the full check compares with MOMENT (its 330 MB graph is fetched); the quick one keeps UniTS beside OTIS
        ex.rival = full ? 'moment' : 'units'; renderMaps();
        await runLive(exEncs());
        ex.stopAfter = 0;
        await fitted();
        report(key + ' live');
      }
      showSample(0);
      // the report from the page: a pinned test sample, then a dataset switch, must refit every map
      await selectDataset('ptbxl'); await fitted(); showSample(3);
      await selectDataset('sleepedf'); await fitted();
      for (const enc of exEncs()) {
        const note = ex.cards[enc].chart.querySelector('svg text') ? Array.from(ex.cards[enc].chart.querySelectorAll('svg text')).map(t => t.textContent).find(t => t.startsWith('t-SNE')) : null;
        console.log(`[selftest] after switch ${enc}: figure=${!!ex.cards[enc].chart.querySelector('figure')} state="${ex.cards[enc].state.textContent}" note="${note}" points=${ex.cards[enc].chart.querySelectorAll('circle').length}`);
      }
    }
    if (only !== 'fc') {
      const rep = () => console.log(`[selftest] fill ${fill.source} ${$('#fill-rows').textContent.replace(/\s+/g, ' ').trim()}`);
      // nothing has run yet: every row offers its button; clicking them runs the models one after another
      const offers = () => $$('#fill-rows [data-run]').map(b => b.dataset.run).join(' ');
      const runAll = async () => { for (const b of $$('#fill-rows [data-run]')) if (b.dataset.run !== 'moment' || app.momentLoaded) b.click(); await settle(() => fill.busy || fill.queue.length); };
      console.log(`[selftest] fill offers: ${offers()}`); rep();
      await runAll(); rep();
      {  // switching a model off and on again: a click on its legend entry hides its line, a click on its row name brings it back
        const paths = () => $$('#chart-fill svg path').length;
        const n0 = paths(); $('#chart-fill .legend-item.is-clickable').click(); const n1 = paths(); $('#fill-rows .enc-name.is-toggle').click(); const n2 = paths();
        console.log(`[selftest] fill toggle: paths ${n0} -> ${n1} -> ${n2}, off=${[...fill.off].join(',') || 'none'}`);
      }
      hoverCheck('fill', '#fill-rows', '#chart-fill');
      $('[data-source="chirp"]').click(); await settle(() => fill.busy); console.log(`[selftest] fill offers after a change: ${offers()}`); await runAll(); rep();
      $('[data-hide="random"]').click(); await settle(() => fill.busy); await runAll(); rep();
      // pointer interactions on the chart: drag to hide a span, then draw a signal by hand
      const drag = (fx0, fx1, fy) => {
        const svg = $('#chart-fill svg'), hit = svg.querySelector('svg > rect:last-of-type'), r = svg.getBoundingClientRect();
        const ev = (type, fx) => hit.dispatchEvent(new PointerEvent(type, { clientX: r.left + r.width * fx, clientY: r.top + r.height * fy, bubbles: true, pointerId: 1 }));
        ev('pointerdown', fx0); ev('pointermove', (fx0 + fx1) / 2); ev('pointermove', fx1); ev('pointerup', fx1);
      };
      drag(.3, .5, .5); await settle(() => fill.busy);
      console.log(`[selftest] drag-to-hide mask=${Array.from(fill.mask).join('')}`); await runAll(); rep();
      // a span dragged to the end is a tail: the forecasters fill it too, rolled when it is longer than 96 steps
      drag(.55, .995, .5); await settle(() => fill.busy);
      console.log(`[selftest] drag-to-end mask=${Array.from(fill.mask).join('')} tail=${JSON.stringify(fillTail())}`); await runAll(); rep();
      console.log(`[selftest] tail forecasts: ${ZS.map(e => `${e}=${fill.results[e] && fill.results[e].stamp === fill.stamp ? (fill.results[e].na ? 'n/a' : `NCC ${fill.results[e].ncc.toFixed(3)} ${fill.results[e].how}`) : 'none'}`).join(' | ')}`);
      $('[data-source="draw"]').click(); await settle(() => fill.busy);
      const before = fill.signal.slice(); drag(.1, .9, .2); await settle(() => fill.busy);
      let changed = 0; for (let t = 0; t < before.length; t++) if (Math.abs(before[t] - fill.signal[t]) > 1e-6) changed++;
      console.log(`[selftest] draw changed ${changed} of ${before.length} steps`); await runAll(); rep();
      // the zero-shot forecasters on a pasted series with the end hidden: their buttons fetch the graphs, then run
      $('[data-source="paste"]').click(); await settle(() => fill.busy);
      $('#paste-text').value = Array.from({ length: 500 }, (_, i) => (Math.sin(i / 7) + 0.4 * Math.cos(i / 2.3)).toFixed(4)).join(', ');
      $('#paste-go').click(); $('[data-hide="end"]').click(); await settle(() => fill.busy);
      console.log(`[selftest] fill offers on a pasted series: ${offers()}`); rep();
      await runAll(); rep();
      console.log(`[selftest] zero-shot loaded: ${ZS.map(e => `${e}=${!!app.zs.loaded[e]}`).join(' ')}`);
    }
    if (only !== 'fill' && $('#bench-list button')) {
      // every row offers its button; the quick check runs the small models, ?selftest=fc and full the large ones too
      const offers = () => $$('#fc-rows [data-run]').map(b => b.dataset.run).join(' ');
      const runAll = async () => { for (const b of $$('#fc-rows [data-run]')) if (full || only === 'fc' || !['moment', ...ZS].includes(b.dataset.run)) b.click(); await settle(() => fc.busy || fc.queue.length); };
      const rep = (key, win) => {
        for (const enc of ['otis', 'units', 'moment', ...ZS]) {
          const r = fcRes(key, win, enc); if (!r || r.na) continue;
          const man = fc.man[key], blk = man.zero_shot && man.zero_shot[enc] ? man.zero_shot[enc] : enc === 'moment' ? man.moment : enc === 'units' ? man.units : fcOtisBlock(man);
          const mref = blk && blk.windows ? blk.windows[win].mse : NaN;
          const vn = man.variates || [], ot = Math.max(0, vn.indexOf('OT'));
          console.log(`[selftest] forecast ${key} ${enc} window ${win + 1}: MSE=${r.mse.toFixed(4)} MAE=${r.mae.toFixed(4)} NCC=${r.ncc.toFixed(3)} ms=${r.ms.toFixed(0)} ${r.precision} manifest=${mref.toFixed(4)}; variate ${vn[ot] || 0}: MSE=${r.perVar[ot].mse.toFixed(4)} MAE=${r.perVar[ot].mae.toFixed(4)}`);
        }
      };
      for (const key of (full || only === 'fc' ? FC_KEYS : ['etth1', 'weather'])) {
        if (!$(`#bench-list button[data-key="${key}"]`)) continue;
        await selectBench(key);
        console.log(`[selftest] forecast ${key} offers: ${offers()} | rows: ${$('#fc-rows').textContent.replace(/\s+/g, ' ').trim()}`);
        await runAll(); rep(key, 0);
        if (key === 'etth1') {
          const paths = () => $$('#chart-fc svg path').length;
          const n0 = paths(); $('#chart-fc .legend-item.is-clickable').click(); const n1 = paths(); $('#fc-rows .enc-name.is-toggle').click(); const n2 = paths();
          console.log(`[selftest] forecast toggle: paths ${n0} -> ${n1} -> ${n2}, off=${[...fc.off].join(',') || 'none'}`);
          hoverCheck('forecast', '#fc-rows', '#chart-fc');
          // the second window offers the buttons again, the first keeps its results
          $$('#fc-windows button')[1].click(); console.log(`[selftest] forecast ${key} window 2 offers: ${offers()}`);
          await runAll(); rep(key, 1);
          $$('#fc-windows button')[0].click(); console.log(`[selftest] forecast ${key} back on window 1 offers: "${offers()}" results=${Object.keys(fc.result[key][0]).join(',')}`);
        }
      }
    }
    document.title = 'selftest done';
    console.log('[selftest] done');
  }

  document.addEventListener('DOMContentLoaded', init);
})();
