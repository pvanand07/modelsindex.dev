export const prerender = false;

import type { APIRoute } from 'astro';
import { env } from 'cloudflare:workers';
import { validateCommentInput, hashIp } from '../../../lib/comments';
import { verifyTurnstile } from '../../../lib/turnstile';

export const POST: APIRoute = async ({ request }) => {
  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return new Response('invalid json', { status: 400 });
  }

  const input = validateCommentInput(raw);
  if (!input) return new Response('invalid input', { status: 400 });

  const turnstileToken = typeof (raw as Record<string, unknown>).turnstileToken === 'string'
    ? ((raw as Record<string, unknown>).turnstileToken as string)
    : undefined;
  const remoteIp = request.headers.get('cf-connecting-ip') ?? undefined;
  const turnstileOk = await verifyTurnstile(turnstileToken, env.TURNSTILE_SECRET_KEY, remoteIp);
  if (!turnstileOk) return new Response('bot check failed', { status: 403 });

  const ipHash = await hashIp(remoteIp ?? 'unknown', env.IP_HASH_PEPPER ?? 'dev-pepper');

  await env.DB.prepare(
    `INSERT INTO comments (entity_type, entity_slug, parent_id, author_name, body, status, ip_hash, created_at)
     VALUES ('blog', ?1, ?2, ?3, ?4, 'approved', ?5, ?6)`,
  )
    .bind(input.entity_slug, input.parent_id, input.author_name, input.body, ipHash, Date.now())
    .run();

  return new Response(null, { status: 201 });
};
