const BUILD = '4.2.2-sovereign';
const CACHE = 'never-ai-monolith-pwa-' + BUILD;
const SHELL = ['./index.html', './manifest.webmanifest', './icon-192.png', './icon-512.png', './favicon.ico'];

self.addEventListener('message', (e) => {
  if (e.data && e.data.type === 'SKIP_WAITING') self.skipWaiting();
  if (e.data && e.data.type === 'GET_BUILD' && e.ports && e.ports[0]) {
    e.ports[0].postMessage({ build: BUILD });
  }
});

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).catch(() => {}));
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
      .then(() => self.clients.matchAll({ type: 'window', includeUncontrolled: true }))
      .then((cs) => cs.forEach((c) => c.postMessage({ type: 'NEVER_AI_SW_UPDATED', build: BUILD })))
  );
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  const u = new URL(e.request.url);
  if (u.pathname.includes('/api/') || u.pathname.includes('/download/')) return;
  const isDoc =
    e.request.mode === 'navigate' ||
    u.pathname.endsWith('/') ||
    u.pathname.endsWith('.html') ||
    u.pathname.endsWith('.webmanifest');

  if (isDoc) {
    const netUrl = u.hostname.endsWith('githack.com')
      ? (u.origin + u.pathname + (u.search ? u.search + '&' : '?') + '_b=' + Math.floor(Date.now() / 15000))
      : e.request;
    e.respondWith(
      fetch(netUrl, { cache: 'no-store' })
        .then((r) => {
          if (r && r.status === 200) {
            const cl = r.clone();
            caches.open(CACHE).then((c) => c.put(e.request, cl)).catch(() => {});
          }
          return r;
        })
        .catch(() => caches.match(e.request).then((m) => m || caches.match('./index.html')))
    );
    return;
  }

  e.respondWith(
    caches.match(e.request).then((cached) => {
      const net = fetch(e.request)
        .then((r) => {
          if (r && r.status === 200) {
            const cl = r.clone();
            caches.open(CACHE).then((c) => c.put(e.request, cl)).catch(() => {});
          }
          return r;
        })
        .catch(() => cached);
      return cached || net;
    })
  );
});
