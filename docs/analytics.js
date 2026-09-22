/* Self-hosted visit counting. Inert unless MATOMO_URL below is filled in.
 *
 * The rest of the page never mentions Matomo: app.js only calls track(category, action, name),
 * which this file defines. With MATOMO_URL empty that call does nothing at all -- no script is
 * fetched, no request is sent, no cookie is set, nothing leaves the browser -- so the page behaves
 * exactly as it would without this file, and there is nothing to disclose in a privacy note.
 *
 * Counting runs cookie-free (disableCookies), with the visitor's address truncated server-side and
 * Do-Not-Track honoured, which is the configuration that needs no consent banner.
 *
 * The server MUST send `Cross-Origin-Resource-Policy: cross-origin` on the two tracker endpoints.
 * This page is cross-origin isolated, and without that header the browser refuses both the script
 * and the tracking request -- silently, so nothing is counted and nothing looks wrong. index.html
 * does ask coi-serviceworker.js for COEP `credentialless`, which would waive the requirement, but
 * that setting lives in a variable inside the service worker, arrives by postMessage and defaults
 * to false, so it is restored to `require-corp` whenever the browser restarts the worker. Measured
 * on the live page: without the header the script load returns BLOCKED; with it, the script loads
 * and the tracking beacon is accepted.
 *
 * Only pages served from *.github.io are counted, so the local preview and the headless self-test
 * stay out of the numbers.
 */
(function () {
  'use strict';

  const MATOMO_URL = 'https://ip31-70-152-239.pbiaas.com/';
  const MATOMO_SITE_ID = '1';

  // Not matomo.js / matomo.php: blocklists match those two by path and would drop a large
  // share of the counting before it happens. The server aliases these to the real files.
  const TRACKER_JS = 'c.js';
  const TRACKER_PHP = 'c.php';

  const q = new URLSearchParams(location.search);
  const off = !MATOMO_URL || !MATOMO_SITE_ID          // not configured
    || !/\.github\.io$/.test(location.hostname)        // local preview
    || q.has('selftest');                              // the headless check

  if (off) { window.track = function () {}; return; }

  try {
    const _paq = window._paq = window._paq || [];
    _paq.push(['disableCookies']);                     // no terminal-device storage, so no consent banner
    // setDoNotTrack, not setDoNotTrackEnabled: an unknown method name makes matomo.js
    // throw while draining this queue, so every command after it -- trackPageView
    // included -- is silently dropped and nothing is ever counted.
    _paq.push(['setDoNotTrack', true]);
    _paq.push(['setTrackerUrl', MATOMO_URL + TRACKER_PHP]);
    _paq.push(['setSiteId', MATOMO_SITE_ID]);
    _paq.push(['trackPageView']);
    _paq.push(['enableLinkTracking']);

    const s = document.createElement('script');
    s.async = true; s.src = MATOMO_URL + TRACKER_JS;
    document.head.appendChild(s);

    /** One interaction. Never throws: a blocked or missing tracker must not break the page. */
    window.track = function (category, action, name, value) {
      try { window._paq.push(['trackEvent', category, action, name, value]); } catch (e) {}
    };

    /* Which sections a visitor actually reaches, counted once each. The whole demo is one page, so a
     * page view alone says only that somebody arrived. The margins fire when a section overlaps the
     * middle half of the viewport, which a section taller than the screen would never do at a
     * threshold on its own area. */
    const seen = new Set();
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting || seen.has(e.target.id)) continue;
        seen.add(e.target.id);
        window.track('Section', 'reached', e.target.id);
      }
    }, { rootMargin: '-25% 0px -25% 0px', threshold: 0 });
    document.addEventListener('DOMContentLoaded', () => {
      for (const id of ['explore', 'fill', 'bench', 'cook']) {
        const el = document.getElementById(id);
        if (el) io.observe(el);
      }
    });
  } catch (e) {
    window.track = function () {};
  }
})();
