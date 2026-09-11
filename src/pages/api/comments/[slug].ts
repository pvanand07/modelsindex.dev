export const prerender = false;

import type { APIRoute } from 'astro';
import { env } from 'cloudflare:workers';
import type { CommentRow } from '../../../lib/comments';

export const GET: APIRoute = async ({ params, url }) => {
  const slug = params.slug;
  if (!slug) return new Response('missing slug', { status: 400 });
  const entityType = url.searchParams.get('entity_type') ?? 'blog';

  const { results } = await env.DB.prepare(
    `SELECT id, parent_id, author_name, body, created_at FROM comments
     WHERE entity_type = ?1 AND entity_slug = ?2 AND status = 'approved'
     ORDER BY created_at ASC`,
  )
    .bind(entityType, slug)
    .all<CommentRow>();

  return new Response(JSON.stringify(results), {
    headers: { 'content-type': 'application/json' },
  });
};
