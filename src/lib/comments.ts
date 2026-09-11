const MAX_NAME_LEN = 80;
const MAX_BODY_LEN = 4000;
const MAX_SLUG_LEN = 200;

export interface CommentInput {
  entity_slug: string;
  author_name: string;
  body: string;
  parent_id: number | null;
}

export interface CommentRow {
  id: number;
  parent_id: number | null;
  author_name: string;
  body: string;
  created_at: number;
}

export function validateCommentInput(raw: unknown): CommentInput | null {
  if (typeof raw !== 'object' || raw === null) return null;
  const { entity_slug, author_name, body, parent_id } = raw as Record<string, unknown>;

  if (typeof entity_slug !== 'string' || entity_slug.length === 0 || entity_slug.length > MAX_SLUG_LEN) {
    return null;
  }
  if (typeof author_name !== 'string' || author_name.trim().length === 0 || author_name.length > MAX_NAME_LEN) {
    return null;
  }
  if (typeof body !== 'string' || body.trim().length === 0 || body.length > MAX_BODY_LEN) {
    return null;
  }
  let parsedParentId: number | null = null;
  if (parent_id !== undefined && parent_id !== null) {
    if (typeof parent_id !== 'number' || !Number.isInteger(parent_id) || parent_id <= 0) return null;
    parsedParentId = parent_id;
  }

  return {
    entity_slug,
    author_name: author_name.trim(),
    body: body.trim(),
    parent_id: parsedParentId,
  };
}

export async function hashIp(ip: string, pepper: string): Promise<string> {
  const data = new TextEncoder().encode(`${pepper}:${ip}`);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

export function buildCommentTree(rows: CommentRow[]): (CommentRow & { replies: CommentRow[] })[] {
  const byId = new Map<number, CommentRow & { replies: CommentRow[] }>();
  const roots: (CommentRow & { replies: CommentRow[] })[] = [];

  for (const row of rows) byId.set(row.id, { ...row, replies: [] });
  for (const row of rows) {
    const node = byId.get(row.id)!;
    if (row.parent_id !== null && byId.has(row.parent_id)) {
      byId.get(row.parent_id)!.replies.push(node);
    } else {
      roots.push(node);
    }
  }
  return roots;
}

export function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let result = 0;
  for (let i = 0; i < a.length; i++) result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return result === 0;
}

export function checkAdminAuth(request: Request, adminSecret: string | undefined): boolean {
  if (!adminSecret) return false;
  const header = request.headers.get('authorization') ?? '';
  const match = header.match(/^Bearer\s+(.+)$/i);
  if (!match) return false;
  return timingSafeEqual(match[1], adminSecret);
}
