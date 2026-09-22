/* Background t-SNE for the maps: the perplexity sweep runs off the main thread so the page
 * stays responsive while several maps fit at once. Messages: { id, rows, opts } -> { id, res }. */
importScripts('analysis.js?v=49');
onmessage = (e) => {
  const { id, rows, opts } = e.data;
  try { postMessage({ id, res: Analysis.tsneAuto(rows, opts) }); }
  catch (err) { postMessage({ id, error: String(err && err.message || err) }); }
};
