/* Chart primitives for the page: an SVG line chart with hidden spans, a crosshair and
 * two pointer modes (drag to choose a span, drag to draw a signal); a categorical
 * scatter of a 2-D projection with linked highlighting across several maps; small
 * preview strips; a table. Hand-rolled so the page stays dependency-free. Colours
 * come from the CSS custom properties in app.css, so light and dark swap for free.
 */
(function (global) {
  'use strict';

  const SVGNS = 'http://www.w3.org/2000/svg';
  const el = (n, a) => { const e = document.createElementNS(SVGNS, n); for (const k in (a || {})) e.setAttribute(k, a[k]); return e; };
  const css = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
  const fmt = (v, d = 3) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 1e-3)) ? v.toExponential(1) : v.toFixed(d);
  const catColor = (i) => css(`--cat-${(i % 6) + 1}`);

  function tooltip() {
    let node = document.querySelector('.viz-tip');
    if (!node) {
      node = document.createElement('div');
      node.className = 'viz-tip';
      document.body.appendChild(node);
    }
    return {
      show(html, x, y) {
        node.innerHTML = html; node.style.opacity = '1';
        const r = node.getBoundingClientRect();
        let left = x + 14, top = y - r.height - 10;
        if (left + r.width > innerWidth - 8) left = x - r.width - 14;
        if (top < 8) top = y + 16;
        node.style.left = left + 'px'; node.style.top = top + 'px';
      },
      hide() { node.style.opacity = '0'; }
    };
  }
  const TIP = tooltip();

  function frame(host, { width, height = 260, pad = { t: 14, r: 16, b: 30, l: 46 }, minWidth = 860 }) {
    // draw at the host's width so charts fill their box at their designed height
    if (width) width = Math.max(minWidth, Math.round(width));   // a width measured by the caller (redraw) counts like a measured one
    if (!width) {
      const main = document.querySelector('main');
      const measured = (host.getBoundingClientRect && host.getBoundingClientRect().width) ||
                       (host.parentElement && host.parentElement.getBoundingClientRect().width) || 0;
      width = measured ? Math.max(minWidth, Math.round(measured)) : Math.max(minWidth, (main && main.clientWidth - 80) || 0);
    }
    const scroll = document.createElement('div');
    scroll.className = 'viz-scroll';
    const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, width: '100%', preserveAspectRatio: 'xMidYMid meet', role: 'img' });
    svg.style.minWidth = Math.min(width, 320) + 'px';
    svg.style.display = 'block';
    scroll.appendChild(svg); host.appendChild(scroll);
    return { svg, width, height, pad, iw: width - pad.l - pad.r, ih: height - pad.t - pad.b };
  }

  function ticks(lo, hi, n) {
    const span = hi - lo || 1;
    const step = Math.pow(10, Math.floor(Math.log10(span / n)));
    const err = span / n / step;
    const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    const s = step * mult, out = [];
    for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9; v += s) out.push(+v.toFixed(10));
    return out;
  }

  function axes(f, xDom, yDom, { xTitle, xFmt = (t) => String(Math.round(t)), showX = true, yTicks = null } = {}) {
    const { svg, pad, iw, ih } = f;
    const gridC = css('--border'), textC = css('--text-muted');
    const sx = v => pad.l + (v - xDom[0]) / ((xDom[1] - xDom[0]) || 1) * iw;
    const sy = v => pad.t + ih - (v - yDom[0]) / ((yDom[1] - yDom[0]) || 1) * ih;
    // yTicks: [{ v, label }] replaces the numeric ticks (stacked rows label each baseline by name)
    for (const t of (yTicks || ticks(yDom[0], yDom[1], 4).map(v => ({ v, label: fmt(v, 2) })))) {
      const y = sy(t.v);
      svg.appendChild(el('line', { x1: pad.l, x2: pad.l + iw, y1: y, y2: y, stroke: gridC, 'stroke-width': 1 }));
      const lab = el('text', { x: pad.l - 8, y: y + 4, 'text-anchor': 'end', fill: textC, 'font-size': 11 });
      lab.textContent = t.label; svg.appendChild(lab);
    }
    if (showX) for (const t of ticks(xDom[0], xDom[1], 6)) {
      const lab = el('text', { x: sx(t), y: pad.t + ih + 19, 'text-anchor': 'middle', fill: textC, 'font-size': 11 });
      lab.textContent = xFmt(t); svg.appendChild(lab);
    }
    if (xTitle) {
      const t = el('text', { x: pad.l + iw / 2, y: f.height - 3, 'text-anchor': 'middle', fill: textC, 'font-size': 11 });
      t.textContent = xTitle; svg.appendChild(t);
    }
    return { sx, sy };
  }

  function legend(host, items) {
    const box = document.createElement('div');
    box.className = 'legend';
    for (const it of items) {
      const s = document.createElement('span');
      s.className = 'legend-item' + (it.onClick ? ' is-clickable' : '') + (it.off ? ' is-off' : '');
      s.dataset.label = it.label;
      s.innerHTML = `<span class="swatch ${it.shape || 'line'}" style="background:${it.color}"></span>${it.label}`;
      if (it.onClick) { s.title = it.off ? 'click to show this model again' : 'click to hide this model'; s.addEventListener('click', it.onClick); }
      box.appendChild(s);
    }
    host.appendChild(box);
    return box;
  }

  /* ---------- line chart with hidden spans, crosshair and pointer modes ----------
   * series: [{ label, color, values, dashed, width, row, raw }]; bands: [{ from, to }] in steps.
   * interact: { mode: 'span' | 'draw' | null, patch, onSpan(from, to), onDraw(values) }
   *   'span': drag to choose the steps to hide, snapped to `patch`;
   *   'draw': drag to draw series[0] by hand; the path follows the pointer, onDraw gets the result.
   * rows: [{ label, offset }] stacks several variates in one chart: each series carries the index of
   *   its row (`row`), its `values` are already shifted to the row's offset and `raw` holds the real
   *   values for the tooltip, which then names the row nearest the pointer and lists only its series.
   * legendItems: [{ label, color }] replaces the one-entry-per-series legend.
   */
  function lines(host, { series, bands = [], caption, xTitle = 'time step', height = 250, xScale = 1, xFmt,
                         showLegend = true, interact = null, yDomain = null, rows = null, legendItems = null, width = 0, padLeft = 46 } = {}) {
    const fig = document.createElement('figure');
    host.appendChild(fig);
    if (showLegend && (legendItems || series.length >= 2)) legend(fig, legendItems || series.map(s => ({ label: s.label, color: s.color })));

    const f = frame(fig, { width, height, pad: { t: 14, r: 16, b: xTitle ? 44 : 30, l: padLeft }, minWidth: 600 });   // padLeft: room for long row labels
    const n = Math.max(...series.map(s => s.values.length));
    let lo = Infinity, hi = -Infinity;
    for (const s of series) for (const v of s.values) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!Number.isFinite(lo)) { lo = -1; hi = 1; }
    if (yDomain) { lo = Math.min(lo, yDomain[0]); hi = Math.max(hi, yDomain[1]); }
    const padY = (hi - lo) * 0.08 || 0.5;
    const yDom = [lo - padY, hi + padY];
    const { sx, sy } = axes(f, [0, (n - 1) * xScale], yDom, { xTitle, xFmt, yTicks: rows ? rows.map(r => ({ v: r.offset, label: r.label })) : null });
    const X = i => sx(i * xScale);

    const bandLayer = el('g'); f.svg.appendChild(bandLayer);
    const drawBands = (bs) => {
      bandLayer.innerHTML = '';
      for (const b of bs) bandLayer.appendChild(el('rect', { x: X(b.from), y: f.pad.t, width: Math.max(1, X(b.to) - X(b.from)), height: f.ih, fill: css('--mask-band') }));
    };
    drawBands(bands);
    const pathOf = (vals) => {
      let d = '', pen = false;
      vals.forEach((v, i) => {
        if (!Number.isFinite(v)) { pen = false; return; }
        d += (pen ? 'L' : 'M') + X(i).toFixed(2) + ' ' + sy(v).toFixed(2) + ' ';
        pen = true;
      });
      return d;
    };
    const paths = series.map(s => {
      const p = el('path', { d: pathOf(s.values), fill: 'none', stroke: s.color, 'stroke-width': s.width || 2,
        'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': s.dashed ? '5 4' : '', opacity: s.opacity == null ? 1 : s.opacity });
      f.svg.appendChild(p); return p;
    });

    const rule = el('line', { y1: f.pad.t, y2: f.pad.t + f.ih, stroke: css('--border-strong'), 'stroke-width': 1, opacity: 0 });
    f.svg.appendChild(rule);
    const dots = series.map(s => { const c = el('circle', { r: 4, fill: s.color, stroke: css('--surface-0'), 'stroke-width': 2, opacity: 0 }); f.svg.appendChild(c); return c; });
    const selRect = el('rect', { y: f.pad.t, height: f.ih, fill: css('--accent'), opacity: 0, rx: 2 });
    f.svg.appendChild(selRect);
    const hit = el('rect', { x: f.pad.l, y: f.pad.t, width: f.iw, height: f.ih, fill: 'transparent' });
    f.svg.appendChild(hit);
    const mode = interact && interact.mode;
    if (mode) hit.style.cursor = mode === 'draw' ? 'crosshair' : 'col-resize';
    const toIndex = (ev) => {
      const r = f.svg.getBoundingClientRect();
      const px = (ev.clientX - r.left) / r.width * f.width;
      return Math.max(0, Math.min(n - 1, Math.round((px - f.pad.l) / f.iw * (n - 1))));
    };
    const toValue = (ev) => {
      const r = f.svg.getBoundingClientRect();
      const py = (ev.clientY - r.top) / r.height * f.height;
      const v = yDom[0] + (f.pad.t + f.ih - py) / f.ih * (yDom[1] - yDom[0]);
      return Math.max(yDom[0], Math.min(yDom[1], v));
    };
    let drag = null;
    hit.addEventListener('pointermove', ev => {
      const i = toIndex(ev);
      if (drag && mode === 'span') {
        const a = Math.min(drag.start, i), b = Math.max(drag.start, i);
        selRect.setAttribute('x', X(a)); selRect.setAttribute('width', Math.max(1, X(b) - X(a))); selRect.setAttribute('opacity', .18);
        TIP.show(`hide steps ${a}–${b}`, ev.clientX, ev.clientY);
        return;
      }
      if (drag && mode === 'draw') {
        const v = toValue(ev), vals = drag.values;
        const a = Math.min(drag.last, i), b = Math.max(drag.last, i);
        for (let k = a; k <= b; k++) vals[k] = a === b ? v : drag.lastV + (v - drag.lastV) * (k - drag.last) / (i - drag.last);
        drag.last = i; drag.lastV = v;
        paths[0].setAttribute('d', pathOf(vals));
        return;
      }
      rule.setAttribute('x1', X(i)); rule.setAttribute('x2', X(i)); rule.setAttribute('opacity', 1);
      let html = `<strong>${xFmt ? xFmt(i * xScale) : 't = ' + i}</strong>`;
      let rowSel = -1;
      if (rows) {                                             // the row whose baseline is nearest the pointer
        const r = f.svg.getBoundingClientRect(), py = (ev.clientY - r.top) / r.height * f.height;
        rows.forEach((rw, k) => { if (rowSel < 0 || Math.abs(sy(rw.offset) - py) < Math.abs(sy(rows[rowSel].offset) - py)) rowSel = k; });
        html += ` · ${rows[rowSel].label}`;
      }
      series.forEach((s, k) => {
        const v = s.values[i];
        if (Number.isFinite(v) && (!rows || s.row === undefined || s.row === rowSel)) {
          dots[k].setAttribute('cx', X(i)); dots[k].setAttribute('cy', sy(v)); dots[k].setAttribute('opacity', 1);
          html += `<br><span style="color:${s.color}">&#9679;</span> ${s.label}: ${fmt(s.raw ? s.raw[i] : v)}`;
        } else dots[k].setAttribute('opacity', 0);
      });
      if (mode === 'span') html += `<br><span class="tip-sub">drag to choose what to hide</span>`;
      if (mode === 'draw') html += `<br><span class="tip-sub">drag to draw the signal</span>`;
      TIP.show(html, ev.clientX, ev.clientY);
    });
    const clear = () => { rule.setAttribute('opacity', 0); dots.forEach(d => d.setAttribute('opacity', 0)); TIP.hide(); };
    hit.addEventListener('pointerleave', () => { if (!drag) clear(); });
    if (mode) {
      hit.addEventListener('pointerdown', ev => {
        ev.preventDefault();
        try { hit.setPointerCapture(ev.pointerId); } catch (_) { /* synthetic events carry no live pointer */ }
        const i = toIndex(ev);
        clear();
        drag = mode === 'draw' ? { last: i, lastV: toValue(ev), values: Array.from(series[0].values) } : { start: i };
        if (mode === 'draw') { drag.values[i] = drag.lastV; paths[0].setAttribute('d', pathOf(drag.values)); }
      });
      const finish = ev => {
        if (!drag) return;
        const d = drag; drag = null;
        selRect.setAttribute('opacity', 0); TIP.hide();
        if (mode === 'span') {
          const i = toIndex(ev), p = interact.patch || 1;
          let a = Math.floor(Math.min(d.start, i) / p), b = Math.floor(Math.max(d.start, i) / p) + 1;
          if (b - a >= Math.ceil(n / p)) b = a + Math.ceil(n / p) - 1;    // never hide everything
          interact.onSpan && interact.onSpan(a * p, Math.min(n, b * p));
        } else interact.onDraw && interact.onDraw(d.values);
      };
      hit.addEventListener('pointerup', finish);
      hit.addEventListener('pointercancel', finish);
    }

    // focus one model (a hover on its row or legend entry): its line thickens, the other predictions fade,
    // the observed series stays; focus(null) restores everything
    const legendBox = fig.querySelector('.legend');
    const focus = (label) => {
      const on = label != null && series.some(s => s.label === label);
      paths.forEach((p, k) => {
        const s = series[k], w = s.width || 2, base = s.opacity == null ? 1 : s.opacity, mine = s.label === label;
        p.setAttribute('stroke-width', on && mine ? w + 1.5 : w);
        p.setAttribute('opacity', !on || mine || s.label === 'observed' ? base : Math.min(base, 0.2));
      });
      if (legendBox) legendBox.querySelectorAll('.legend-item').forEach(li => li.classList.toggle('is-focus', on && li.dataset.label === label));
    };
    if (legendBox) legendBox.querySelectorAll('.legend-item').forEach(li => {
      if (li.dataset.label === 'observed') return;
      li.addEventListener('pointerenter', () => focus(li.dataset.label));
      li.addEventListener('pointerleave', () => focus(null));
    });
    if (caption) { const c = document.createElement('figcaption'); c.textContent = caption; fig.appendChild(c); }
    fig.api = { setBands: drawBands, setValues: (k, vals) => paths[k].setAttribute('d', pathOf(vals)), focus };
    return fig;
  }

  /* ---------- sparkline as an HTML string (for tooltips and preview strips) ---------- */
  function sparkPath(vals, w, h, p, dom) {
    const n = vals.length, lo = dom[0], span = (dom[1] - dom[0]) || 1;
    const px = i => p + (n > 1 ? i / (n - 1) : .5) * (w - 2 * p);
    const py = v => p + (1 - (v - lo) / span) * (h - 2 * p);
    let d = '';
    for (let i = 0; i < n; i++) d += (i ? 'L' : 'M') + px(i).toFixed(1) + ' ' + py(vals[i]).toFixed(1);
    return d;
  }
  function spark(vals, dom, stroke, w = 150, h = 40) {
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" style="display:block;margin-top:5px">` +
           `<path d="${sparkPath(vals, w, h, 3, dom)}" fill="none" stroke="${stroke}" stroke-width="1.3" ` +
           `stroke-linejoin="round" stroke-linecap="round"/></svg>`;
  }

  /* ---------- categorical scatter of a 2-D projection ----------
   * points: { x, y, cls, label, sub, values, hollow, links (point indices) }
   * The axes of a projection carry no units, so the map is drawn bare: a frame, the class
   * names at each class's median, and the neighbours of a test sample on hover. Returns an
   * api so several maps can share one legend and highlight the same sample together.
   */
  function scatter(host, { points, classes, colors, caption, note = '', height = 420,
                           valueDomain = [-3, 3], valueLabel = '', onSelect = null, onHover = null, selected = -1,
                           linkLabel = 'nearest training samples', minWidth = 320, width = 0 }) {
    const fig = document.createElement('figure');
    host.appendChild(fig);
    const cols = colors || classes.map((_, i) => catColor(i));
    const pad = { t: 8, r: 8, b: 8, l: 8 };
    const f = frame(fig, { width, height, pad, minWidth });
    const surface = css('--surface-0'), border = css('--border'), textC = css('--text-muted');
    f.svg.appendChild(el('rect', { x: pad.l, y: pad.t, width: f.iw, height: f.ih, fill: css('--surface-1'), stroke: border, rx: 8 }));
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const ex = [Math.min(...xs), Math.max(...xs)], ey = [Math.min(...ys), Math.max(...ys)];
    const mx = (ex[1] - ex[0]) * .08 || 1, my = (ey[1] - ey[0]) * .10 || 1;
    const sx = v => pad.l + (v - (ex[0] - mx)) / ((ex[1] - ex[0]) + 2 * mx) * f.iw;
    const sy = v => pad.t + f.ih - (v - (ey[0] - my)) / ((ey[1] - ey[0]) + 2 * my) * f.ih;
    if (note) {
      const t = el('text', { x: pad.l + f.iw - 10, y: pad.t + f.ih - 8, 'text-anchor': 'end', fill: textC, 'font-size': 11 });
      t.textContent = note; f.svg.appendChild(t);
    }

    const linkLayer = el('g'), pointLayer = el('g'), labelLayer = el('g'), ringLayer = el('g');
    f.svg.appendChild(linkLayer); f.svg.appendChild(pointLayer); f.svg.appendChild(labelLayer); f.svg.appendChild(ringLayer);

    const median = a => { const b = a.slice().sort((p, q) => p - q); return b.length ? b[Math.floor(b.length / 2)] : 0; };
    const labelNodes = classes.map((name, c) => {
      const pts = points.filter(p => p.cls === c);
      if (!pts.length) return null;
      const x = sx(median(pts.map(p => p.x))), y = sy(median(pts.map(p => p.y)));
      const t = el('text', { x, y: y + 4, 'text-anchor': 'middle', fill: cols[c], 'font-size': 12.5, 'font-weight': 650,
        stroke: surface, 'stroke-width': 4, 'paint-order': 'stroke', 'stroke-linejoin': 'round' });
      t.textContent = name; t.style.pointerEvents = 'none';
      labelLayer.appendChild(t); return t;
    });

    const R = 5;
    const circles = [];
    points.forEach((p, i) => {
      const fill = cols[p.cls];
      const c = p.hollow
        ? el('circle', { cx: sx(p.x), cy: sy(p.y), r: R, fill, 'fill-opacity': .22 })
        : el('circle', { cx: sx(p.x), cy: sy(p.y), r: R, fill, stroke: surface, 'stroke-width': 1.2 });
      c.style.cursor = 'pointer';
      c.addEventListener('mouseenter', ev => {
        if (p.values && p._spark === undefined) {
          p._spark = (valueLabel ? `<span class="tip-sub">${valueLabel}</span>` : '') + spark(p.values, valueDomain, fill);
        }
        TIP.show(`<strong>${p.label}</strong>${p.sub ? `<br><span class="tip-sub">${p.sub}</span>` : ''}` + (p._spark || ''), ev.clientX, ev.clientY);
        drawLinks(i);
        onHover && onHover(i);
      });
      c.addEventListener('mouseleave', () => { TIP.hide(); drawLinks(selected); onHover && onHover(-1); });
      if (onSelect) c.addEventListener('click', () => { TIP.hide(); onSelect(i); });
      pointLayer.appendChild(c); circles.push(c);
    });

    /* neighbour lines for a test sample */
    let linkTitle = null;
    function drawLinks(i) {
      linkLayer.innerHTML = ''; if (linkTitle) { linkTitle.remove(); linkTitle = null; }
      circles.forEach((c, j) => { if (points[j].hollow) { c.setAttribute('fill-opacity', .22); c.setAttribute('stroke-width', 0); } });
      const p = points[i];
      if (i < 0 || !p || !p.links || !p.links.length) return;
      const x0 = sx(p.x), y0 = sy(p.y);
      for (const j of p.links) {
        const q = points[j];
        linkLayer.appendChild(el('line', { x1: x0, y1: y0, x2: sx(q.x), y2: sy(q.y), stroke: cols[q.cls], 'stroke-width': 1.2, opacity: .55 }));
        circles[j].setAttribute('fill-opacity', 1);
        circles[j].setAttribute('stroke', surface); circles[j].setAttribute('stroke-width', 1.2);
      }
      linkTitle = el('text', { x: pad.l + 10, y: pad.t + f.ih - 8, fill: textC, 'font-size': 11 });
      linkTitle.textContent = `lines: the ${p.links.length} ${linkLabel}`;
      f.svg.appendChild(linkTitle);
    }

    /* a ring around one point: the sample hovered or selected in any map */
    let ring = null, ringSel = null;
    function ringAt(i, kind) {
      const cur = kind === 'sel' ? ringSel : ring;
      if (cur) cur.remove();
      let node = null;
      if (i >= 0 && points[i]) {
        node = el('circle', { cx: sx(points[i].x), cy: sy(points[i].y), r: R + 3.5, fill: 'none',
          stroke: css('--text-primary'), 'stroke-width': kind === 'sel' ? 2.2 : 1.6, opacity: kind === 'sel' ? 1 : .8 });
        node.style.pointerEvents = 'none';
        ringLayer.appendChild(node);
      }
      if (kind === 'sel') ringSel = node; else ring = node;
    }
    ringAt(selected, 'sel');
    drawLinks(selected);

    let focus = -1, setFocus = null;
    const visible = (p) => (focus < 0 || p.cls === focus) && (!setFocus || (p.hollow ? 'gallery' : 'test') === setFocus);
    function applyFocus() {
      circles.forEach((c, i) => { c.setAttribute('opacity', visible(points[i]) ? 1 : .08); });
      labelNodes.forEach((t, c) => { if (t) t.setAttribute('opacity', focus < 0 || c === focus ? 1 : .25); });
    }

    if (caption) { const cap = document.createElement('figcaption'); cap.textContent = caption; fig.appendChild(cap); }
    return {
      fig,
      highlight: (i) => { ringAt(i, 'hover'); drawLinks(i >= 0 ? i : selected); },
      select: (i) => { selected = i; ringAt(i, 'sel'); drawLinks(i); },
      setFocus: (cls, set) => { focus = cls; setFocus = set; applyFocus(); }
    };
  }

  function minis(host, items, { height = 54, valueDomain = [-3, 3] } = {}) {
    const row = document.createElement('div');
    row.className = 'minis';
    for (const it of items) {
      const card = document.createElement('div');
      card.className = 'mini';
      card.innerHTML =
        `<svg viewBox="0 0 220 ${height}" preserveAspectRatio="none" aria-hidden="true">` +
        `<path d="${sparkPath(it.values, 220, height, 3, valueDomain)}" fill="none" stroke="${it.color}" ` +
        `stroke-width="1.2" vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>` +
        `<span class="mini-label"><span class="swatch dot" style="background:${it.color}"></span>${it.label}</span>`;
      if (it.onClick) { card.style.cursor = 'pointer'; card.addEventListener('click', it.onClick); card.title = 'show this sample'; }
      row.appendChild(card);
    }
    host.appendChild(row);
    return row;
  }

  function table(host, { columns, rows, maxRows = 200, rowClass = null }) {
    host.innerHTML = '';
    const box = document.createElement('div');
    box.className = 'table-scroll';
    const t = document.createElement('table');
    const thead = document.createElement('thead');
    thead.innerHTML = '<tr>' + columns.map(c => `<th>${c}</th>`).join('') + '</tr>';
    t.appendChild(thead);
    const tb = document.createElement('tbody');
    rows.slice(0, maxRows).forEach((r, i) => {
      const tr = document.createElement('tr');
      if (rowClass) tr.className = rowClass(i);
      tr.innerHTML = r.map(v => `<td>${typeof v === 'number' ? fmt(v) : v}</td>`).join('');
      tb.appendChild(tr);
    });
    t.appendChild(tb); box.appendChild(t); host.appendChild(box);
    return box;
  }

  /** Redraw a chart without ever leaving its box empty for a layout pass: measure the box, build the new
   *  figure detached (`build(box, width)`), then swap it in with one DOM change. Clearing the box first and
   *  measuring it empty made Firefox re-anchor or clamp the scroll position, so the page jumped on every redraw. */
  function redraw(host, build) {
    const width = host.getBoundingClientRect().width;
    const box = document.createElement('div');
    const out = build(box, width);
    host.replaceChildren(...box.childNodes);
    return out;
  }

  global.Charts = { lines, scatter, minis, table, legend, css, fmt, catColor, spark, redraw };
})(window);
