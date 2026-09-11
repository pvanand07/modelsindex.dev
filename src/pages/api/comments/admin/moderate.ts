export const prerender = false;

import type { APIRoute } from 'astro';
import { env } from 'cloudflare:workers';
import { checkAdminAuth } from '../../../../lib/comments';

export const POST: APIRoute = async ({ request }) => {
  if (!checkAdminAuth(request, env.ADMIN_SECRET)) {
    return new Response('unauthorized', { status: 401 });
  }

  let raw: unknown;
  try {
    raw = await request.json();
  } catch {
    return new Response('invalid json', { status: 400 });
  }

  const { id, action, reason } = (raw ?? {}) as Record<string, unknown>;
  if (typeof id !== 'number' || (action !== 'hide' && action !== 'unhide')) {
    return new Response('invalid input', { status: 400 });
  }

  const status = action === 'hide' ? 'rejected' : 'approved';
  await env.DB.prepare(
    `UPDATE comments SET status = ?1, moderated_by = 'human', moderated_reason = ?2 WHERE id = ?3`,
  )
    .bind(status, typeof reason === 'string' ? reason : null, id)
    .run();

  return new Response(null, { status: 200 });
};
