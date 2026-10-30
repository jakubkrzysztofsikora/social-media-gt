/// <reference types="@cloudflare/workers-types" />
// share.jakub.team: stores shared analysis HTML under random ids.
// Writes require the SHARE_TOKEN secret; reads are public by design.

export interface Env {
  SHARE: KVNamespace;
  SHARE_TOKEN: string;
}

const ID_ALPHABET = "abcdefghijkmnopqrstuvwxyz23456789"; // no ambiguous chars

function randomId(length = 10): string {
  const bytes = new Uint8Array(length);
  crypto.getRandomValues(bytes);
  let id = "";
  for (const byte of bytes) {
    id += ID_ALPHABET[byte % ID_ALPHABET.length];
  }
  return id;
}

function htmlResponse(html: string): Response {
  return new Response(html, {
    headers: {
      "content-type": "text/html; charset=utf-8",
      "x-robots-tag": "noindex, nofollow",
      "referrer-policy": "no-referrer",
      "x-content-type-options": "nosniff",
      // KaTeX CSS/JS come from the CDN; everything else is locked down.
      "content-security-policy":
        "default-src 'none'; style-src 'unsafe-inline' https://cdn.jsdelivr.net; " +
        "script-src https://cdn.jsdelivr.net; font-src https://cdn.jsdelivr.net; " +
        "img-src data:; connect-src 'none'; frame-ancestors 'none'; base-uri 'none'",
    },
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    if (request.method === "POST" && url.pathname === "/api/share") {
      const token = request.headers.get("x-share-token") ?? "";
      if (!env.SHARE_TOKEN || token !== env.SHARE_TOKEN) {
        return Response.json({ error: "unauthorized" }, { status: 401 });
      }
      const originUrl = new URL(request.headers.get("origin") ?? url.origin);
      if (originUrl.protocol !== "https:") {
        return Response.json({ error: "bad origin" }, { status: 400 });
      }
      let body: { html?: unknown };
      try {
        body = (await request.json()) as { html?: unknown };
      } catch {
        return Response.json({ error: "bad json" }, { status: 400 });
      }
      const html = typeof body.html === "string" ? body.html : "";
      if (!html || html.length > 5 * 1024 * 1024) {
        return Response.json({ error: "html missing or too large" }, { status: 400 });
      }
      const id = randomId();
      const ttl = 90 * 24 * 3600; // 90 days, then the link dies
      await env.SHARE.put(`s:${id}`, html, { expirationTtl: ttl });
      return Response.json({
        id,
        url: `${url.origin}/${id}`,
        expires_in_days: 90,
      });
    }

    if (request.method === "GET" || request.method === "HEAD") {
      if (url.pathname === "/healthz" && request.method === "GET") {
        return Response.json({ ok: true });
      }
      const match = /^\/([a-z0-9]{6,20})$/.exec(url.pathname);
      if (match) {
        const html = await env.SHARE.get(`s:${match[1]}`);
        if (html === null) {
          return new Response("Not found", { status: 404 });
        }
        return htmlResponse(html);
      }
    }

    return new Response("Not found", { status: 404 });
  },
};
