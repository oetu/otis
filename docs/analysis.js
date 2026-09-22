/* Representation-space analysis for the page: PCA, t-SNE (with a data-chosen perplexity),
 * k-means and the two label-aware scores. Everything is plain JavaScript on Float32/Float64 arrays so
 * the page stays dependency-free. Sizes are small (100 samples x 192 dims), so the
 * O(n^2) t-SNE is fine and runs in a few tens of milliseconds.
 *
 * All randomness is seeded, so a run is reproducible and matches the
 * offline computation up to the usual k-means restarts.
 */
(function (global) {
  'use strict';

  /* mulberry32, same generator as otis.js */
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  function l2rows(rows) {
    return rows.map(r => {
      const n = Math.sqrt(dot(r, r)) + 1e-12;
      const o = new Float64Array(r.length);
      for (let i = 0; i < r.length; i++) o[i] = r[i] / n;
      return o;
    });
  }

  /* ---------------- PCA (top-2, power iteration with deflation) ---------------- */

  function pca(rows) {
    const n = rows.length, dim = rows[0].length;
    const mean = new Float64Array(dim);
    for (const r of rows) for (let j = 0; j < dim; j++) mean[j] += r[j] / n;
    const X = rows.map(r => { const o = new Float64Array(dim); for (let j = 0; j < dim; j++) o[j] = r[j] - mean[j]; return o; });
    const norm = (a) => { const s = Math.sqrt(dot(a, a)) + 1e-12; for (let j = 0; j < a.length; j++) a[j] /= s; };
    const comps = [];
    for (let c = 0; c < 2; c++) {
      let v = new Float64Array(dim);
      for (let j = 0; j < dim; j++) v[j] = Math.sin((j + 1) * (c + 1) * 0.7);
      norm(v);
      for (let it = 0; it < 80; it++) {
        const nv = new Float64Array(dim);
        for (const r of X) { const d = dot(r, v); for (let j = 0; j < dim; j++) nv[j] += d * r[j]; }
        for (const p of comps) { const d = dot(nv, p); for (let j = 0; j < dim; j++) nv[j] -= d * p[j]; }
        norm(nv); v = nv;
      }
      comps.push(v);
    }
    return X.map(r => [dot(r, comps[0]), dot(r, comps[1])]);
  }

  /* ---------------- t-SNE (exact, PCA-initialised, deterministic) ----------------
   * Runs on L2-normalised vectors, so the map lives in the same cosine geometry the three
   * scores use. Returns the positions; `tsne.last` carries the KL divergence of the run.
   */

  function tsne(rows, { perplexity = 20, iters = 750, lr = 50, cosine = true } = {}) {
    const n = rows.length;
    if (n < 4) return pca(rows);
    if (cosine) rows = l2rows(rows);
    perplexity = Math.min(perplexity, Math.floor((n - 1) / 3));

    // squared euclidean distances
    const D2 = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      let s = 0; const a = rows[i], b = rows[j];
      for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d * d; }
      D2[i * n + j] = D2[j * n + i] = s;
    }

    // conditional P_{j|i} by binary search on the bandwidth, then symmetrise
    const P = new Float64Array(n * n);
    const logU = Math.log(perplexity);
    for (let i = 0; i < n; i++) {
      let beta = 1, lo = -Infinity, hi = Infinity;
      const row = new Float64Array(n);
      for (let t = 0; t < 60; t++) {
        let sum = 0;
        for (let j = 0; j < n; j++) { row[j] = j === i ? 0 : Math.exp(-D2[i * n + j] * beta); sum += row[j]; }
        let H = 0;
        for (let j = 0; j < n; j++) if (row[j] > 0) { const p = row[j] / sum; H -= p * Math.log(p); }
        const diff = H - logU;
        if (Math.abs(diff) < 1e-5) break;
        if (diff > 0) { lo = beta; beta = hi === Infinity ? beta * 2 : (beta + hi) / 2; }
        else { hi = beta; beta = lo === -Infinity ? beta / 2 : (beta + lo) / 2; }
      }
      let sum = 0; for (let j = 0; j < n; j++) sum += row[j];
      for (let j = 0; j < n; j++) P[i * n + j] = row[j] / sum;
    }
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const p = Math.max((P[i * n + j] + P[j * n + i]) / (2 * n), 1e-12);
      P[i * n + j] = P[j * n + i] = p;
    }

    // PCA initialisation, scaled small as in the reference implementation
    const init = pca(rows);
    let sx = 0; for (const p of init) sx += p[0] * p[0]; sx = Math.sqrt(sx / n) || 1;
    const Y = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) { Y[2 * i] = init[i][0] / sx * 1e-4; Y[2 * i + 1] = init[i][1] / sx * 1e-4; }

    const dY = new Float64Array(n * 2), iY = new Float64Array(n * 2), gains = new Float64Array(n * 2).fill(1);
    const Q = new Float64Array(n * n);
    for (let it = 0; it < iters; it++) {
      const exag = it < 100 ? 12 : 1, mom = it < 250 ? 0.5 : 0.8;
      let qs = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1];
        const q = 1 / (1 + dx * dx + dy * dy);
        Q[i * n + j] = Q[j * n + i] = q; qs += 2 * q;
      }
      dY.fill(0);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const q = Q[i * n + j];
        const m = (exag * P[i * n + j] - q / qs) * q;
        dY[2 * i] += 4 * m * (Y[2 * i] - Y[2 * j]);
        dY[2 * i + 1] += 4 * m * (Y[2 * i + 1] - Y[2 * j + 1]);
      }
      for (let k = 0; k < 2 * n; k++) {
        gains[k] = Math.max(0.01, (dY[k] > 0) !== (iY[k] > 0) ? gains[k] + 0.2 : gains[k] * 0.8);
        iY[k] = mom * iY[k] - lr * gains[k] * dY[k];
        Y[k] += iY[k];
      }
      // keep the map centred
      let mx = 0, my = 0;
      for (let i = 0; i < n; i++) { mx += Y[2 * i] / n; my += Y[2 * i + 1] / n; }
      for (let i = 0; i < n; i++) { Y[2 * i] -= mx; Y[2 * i + 1] -= my; }
    }
    // KL(P || Q) of the final map, the quantity t-SNE minimises
    let qs = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const dx = Y[2 * i] - Y[2 * j], dy = Y[2 * i + 1] - Y[2 * j + 1];
      const q = 1 / (1 + dx * dx + dy * dy); Q[i * n + j] = Q[j * n + i] = q; qs += 2 * q;
    }
    let kl = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) { const p = P[i * n + j]; kl += p * Math.log(p / (Q[i * n + j] / qs)); }
    tsne.last = { kl, perplexity };
    const out = [];
    for (let i = 0; i < n; i++) out.push([Y[2 * i], Y[2 * i + 1]]);
    return out;
  }

  /** Neighbourhood preservation: for every point, the share of its k nearest neighbours in the
   *  vector space (cosine) that are also among its k nearest in the 2-D map. 1 = the map keeps
   *  every local neighbourhood; the number a reader can trust the map by. */
  function knnPreserved(rows, Y, k = 10) {
    const Z = l2rows(rows), n = Z.length;
    const near = (dist) => {
      const out = [];
      for (let i = 0; i < n; i++) {
        const d = new Float64Array(n);
        for (let j = 0; j < n; j++) d[j] = j === i ? Infinity : dist(i, j);
        out.push(new Set(Array.from(d.keys()).sort((a, b) => d[a] - d[b]).slice(0, k)));
      }
      return out;
    };
    const hi = near((i, j) => -dot(Z[i], Z[j]));
    const lo = near((i, j) => (Y[i][0] - Y[j][0]) ** 2 + (Y[i][1] - Y[j][1]) ** 2);
    let kept = 0;
    for (let i = 0; i < n; i++) for (const j of hi[i]) if (lo[i].has(j)) kept++;
    return kept / (n * k);
  }

  /** t-SNE with the perplexity chosen by the data: every candidate is run from the same PCA
   *  initialisation and the map that preserves the most k-nearest neighbourhoods wins.
   *  Returns { proj, perplexity, preserved, kl, tried }. */
  function tsneAuto(rows, { perplexities = [10, 20, 30, 50], k = 10, iters = 750, selectIters = 300 } = {}) {
    // candidates are compared after a shorter schedule (the ranking settles early); the winner
    // is then run in full. One candidate skips the sweep, e.g. a perplexity chosen earlier.
    let best = null;
    const tried = [];
    for (const perplexity of perplexities) {
      const proj = tsne(rows, { perplexity, iters: perplexities.length > 1 ? selectIters : iters });
      const preserved = knnPreserved(rows, proj, k), kl = tsne.last.kl;
      tried.push({ perplexity, preserved, kl });
      if (!best || preserved > best.preserved + 1e-9) best = { proj, perplexity, preserved, kl };
    }
    if (perplexities.length > 1) {
      best.proj = tsne(rows, { perplexity: best.perplexity, iters });
      best.preserved = knnPreserved(rows, best.proj, k); best.kl = tsne.last.kl;
    }
    best.tried = tried;
    return best;
  }

  /* ---------------- k-means (k-means++ seeding, best of a few restarts) ---------------- */

  function kmeans(rows, k, { seed = 0, restarts = 5, iters = 100 } = {}) {
    const Z = l2rows(rows), n = Z.length, dim = Z[0].length;
    const r = rng(seed);
    const sq = (a, b) => { let s = 0; for (let i = 0; i < dim; i++) { const d = a[i] - b[i]; s += d * d; } return s; };
    let best = null;
    for (let rs = 0; rs < restarts; rs++) {
      const C = [Z[Math.floor(r() * n)].slice()];
      while (C.length < k) {
        const d = Z.map(z => Math.min(...C.map(c => sq(z, c))));
        const tot = d.reduce((a, b) => a + b, 0);
        let u = r() * tot, pick = n - 1;
        for (let i = 0; i < n; i++) { u -= d[i]; if (u <= 0) { pick = i; break; } }
        C.push(Z[pick].slice());
      }
      let assign = new Int32Array(n);
      for (let it = 0; it < iters; it++) {
        const next = new Int32Array(n);
        for (let i = 0; i < n; i++) {
          let bj = 0, bd = Infinity;
          for (let j = 0; j < k; j++) { const d = sq(Z[i], C[j]); if (d < bd) { bd = d; bj = j; } }
          next[i] = bj;
        }
        let changed = false;
        for (let i = 0; i < n; i++) if (next[i] !== assign[i]) { changed = true; break; }
        assign = next;
        for (let j = 0; j < k; j++) {
          const m = new Float64Array(dim); let c = 0;
          for (let i = 0; i < n; i++) if (assign[i] === j) { c++; for (let t = 0; t < dim; t++) m[t] += Z[i][t]; }
          if (c) for (let t = 0; t < dim; t++) C[j][t] = m[t] / c;
        }
        if (!changed && it > 0) break;
      }
      let inertia = 0;
      for (let i = 0; i < n; i++) inertia += sq(Z[i], C[assign[i]]);
      if (!best || inertia < best.inertia) best = { assign, inertia };
    }
    return best.assign;
  }

  /* ---------------- label-aware scores ---------------- */

  /** Share of samples whose cluster's majority label is their own. */
  function purity(assign, labels, k, nClasses) {
    const M = contingency(assign, labels, k, nClasses);
    let s = 0;
    for (const row of M) s += Math.max(...row);
    return s / labels.length;
  }

  /** rows = clusters, columns = classes. */
  function contingency(assign, labels, k, nClasses) {
    const M = Array.from({ length: k }, () => new Array(nClasses).fill(0));
    for (let i = 0; i < labels.length; i++) M[assign[i]][labels[i]]++;
    return M;
  }

  /* ---------------- the paper's frozen-encoder protocols (Appendix F) ---------------- */

  /** Balanced accuracy: mean per-class recall. */
  function balancedAccuracy(pred, labels, nClasses) {
    let sum = 0, cnt = 0;
    for (let c = 0; c < nClasses; c++) {
      let n = 0, ok = 0;
      for (let i = 0; i < labels.length; i++) if (labels[i] === c) { n++; if (pred[i] === c) ok++; }
      if (n) { sum += ok / n; cnt++; }
    }
    return cnt ? sum / cnt : NaN;
  }

  /**
   * Weighted k-NN, the DINO protocol the paper follows: for each test vector the top-k
   * cosine neighbours in the gallery vote with weight exp(cos / tau); argmax over classes.
   * Returns one predicted class per test vector.
   */
  function weightedKnn(gallery, galleryLabels, test, opts = {}) {
    return weightedKnnDetail(gallery, galleryLabels, test, opts).map(d => d.pred);
  }

  /** Same as weightedKnn but also returns the neighbours and the vote shares behind each prediction. */
  function weightedKnnDetail(gallery, galleryLabels, test, { k = 20, tau = 0.07, nClasses } = {}) {
    const G = l2rows(gallery), T = l2rows(test);
    nClasses = nClasses || Math.max(...galleryLabels) + 1;
    return T.map(z => {
      const sims = G.map((g, j) => [dot(z, g), j]).sort((a, b) => b[0] - a[0]).slice(0, k);
      const votes = new Array(nClasses).fill(0);
      for (const [s, j] of sims) votes[galleryLabels[j]] += Math.exp(s / tau);
      let pred = 0;
      for (let c = 1; c < nClasses; c++) if (votes[c] > votes[pred]) pred = c;
      const total = votes.reduce((a, b) => a + b, 0) || 1;
      return { pred, neighbors: sims.map(x => x[1]), share: votes.map(v => v / total) };
    });
  }

  /**
   * Prototypical k-shot: per episode draw `shots` gallery samples per class, average them
   * into a prototype, assign each test vector to the nearest prototype by cosine similarity.
   * Balanced accuracy, mean and std over `episodes` seeded episodes.
   */
  function prototypical(gallery, galleryLabels, test, testLabels,
                         { shots = 5, episodes = 5, seed = 0, nClasses } = {}) {
    const G = l2rows(gallery), T = l2rows(test), dim = G[0].length;
    nClasses = nClasses || Math.max(...galleryLabels) + 1;
    const r = rng(seed);
    const byClass = Array.from({ length: nClasses }, (_, c) => galleryLabels.map((l, i) => l === c ? i : -1).filter(i => i >= 0));
    const accs = [];
    for (let e = 0; e < episodes; e++) {
      const protos = byClass.map(idx => {
        const pool = idx.slice();
        const p = new Float64Array(dim);
        for (let s = 0; s < Math.min(shots, pool.length); s++) {
          const pick = pool.splice(Math.floor(r() * pool.length), 1)[0];
          for (let d = 0; d < dim; d++) p[d] += G[pick][d];
        }
        return l2rows([p])[0];
      });
      const pred = T.map(z => {
        let best = 0, bs = -Infinity;
        protos.forEach((p, c) => { const s = dot(z, p); if (s > bs) { bs = s; best = c; } });
        return best;
      });
      accs.push(balancedAccuracy(pred, testLabels, nClasses));
    }
    const mean = accs.reduce((a, b) => a + b, 0) / accs.length;
    const std = Math.sqrt(accs.reduce((a, b) => a + (b - mean) ** 2, 0) / accs.length);
    return { mean, std, accs };
  }

  global.Analysis = { pca, tsne, tsneAuto, knnPreserved, kmeans, purity, contingency, balancedAccuracy, weightedKnn, weightedKnnDetail, prototypical };
})(typeof self !== 'undefined' ? self : window);
