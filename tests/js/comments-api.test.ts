import { describe, it, expect, vi, afterEach } from 'vitest';
import {
  validateCommentInput,
  hashIp,
  buildCommentTree,
  checkAdminAuth,
  timingSafeEqual,
  type CommentRow,
} from '../../src/lib/comments';
import { verifyTurnstile } from '../../src/lib/turnstile';

describe('validateCommentInput', () => {
  it('accepts a well-formed top-level comment', () => {
    const result = validateCommentInput({
      entity_slug: 'my-post',
      author_name: '  Ada  ',
      body: '  Great post!  ',
    });
    expect(result).toEqual({
      entity_slug: 'my-post',
      author_name: 'Ada',
      body: 'Great post!',
      parent_id: null,
    });
  });

  it('accepts a reply with a valid parent_id', () => {
    const result = validateCommentInput({
      entity_slug: 'my-post',
      author_name: 'Ada',
      body: 'Agreed.',
      parent_id: 3,
    });
    expect(result?.parent_id).toBe(3);
  });

  it.each([
    [{}],
    [{ entity_slug: '', author_name: 'Ada', body: 'hi' }],
    [{ entity_slug: 'p', author_name: '   ', body: 'hi' }],
    [{ entity_slug: 'p', author_name: 'Ada', body: '   ' }],
    [{ entity_slug: 'p', author_name: 'Ada', body: 'hi', parent_id: -1 }],
    [{ entity_slug: 'p', author_name: 'Ada', body: 'hi', parent_id: 1.5 }],
    [{ entity_slug: 'p', author_name: 'a'.repeat(81), body: 'hi' }],
    [{ entity_slug: 'p', author_name: 'Ada', body: 'a'.repeat(4001) }],
    [null],
    ['not an object'],
  ])('rejects invalid input %#', (input) => {
    expect(validateCommentInput(input)).toBeNull();
  });
});

describe('hashIp', () => {
  it('is deterministic for the same ip and pepper', async () => {
    const a = await hashIp('1.2.3.4', 'pepper');
    const b = await hashIp('1.2.3.4', 'pepper');
    expect(a).toBe(b);
  });

  it('differs for different ips or peppers', async () => {
    const base = await hashIp('1.2.3.4', 'pepper');
    expect(await hashIp('5.6.7.8', 'pepper')).not.toBe(base);
    expect(await hashIp('1.2.3.4', 'other-pepper')).not.toBe(base);
  });

  it('never contains the raw ip', async () => {
    const hash = await hashIp('1.2.3.4', 'pepper');
    expect(hash).not.toContain('1.2.3.4');
    expect(hash).toMatch(/^[0-9a-f]{64}$/);
  });
});

describe('buildCommentTree', () => {
  it('nests replies under their parent and keeps top-level order', () => {
    const rows: CommentRow[] = [
      { id: 1, parent_id: null, author_name: 'A', body: 'root 1', created_at: 1 },
      { id: 2, parent_id: 1, author_name: 'B', body: 'reply to 1', created_at: 2 },
      { id: 3, parent_id: null, author_name: 'C', body: 'root 2', created_at: 3 },
    ];
    const tree = buildCommentTree(rows);
    expect(tree.map((n) => n.id)).toEqual([1, 3]);
    expect(tree[0].replies.map((n) => n.id)).toEqual([2]);
    expect(tree[1].replies).toEqual([]);
  });

  it('falls back to a root node when parent_id references a missing comment', () => {
    const rows: CommentRow[] = [
      { id: 5, parent_id: 999, author_name: 'A', body: 'orphan', created_at: 1 },
    ];
    const tree = buildCommentTree(rows);
    expect(tree.map((n) => n.id)).toEqual([5]);
  });
});

describe('timingSafeEqual', () => {
  it('returns true only for exact matches', () => {
    expect(timingSafeEqual('secret', 'secret')).toBe(true);
    expect(timingSafeEqual('secret', 'wrong')).toBe(false);
    expect(timingSafeEqual('secret', 'secre')).toBe(false);
  });
});

describe('checkAdminAuth', () => {
  function requestWithAuth(header: string | null) {
    const headers = new Headers();
    if (header !== null) headers.set('authorization', header);
    return new Request('https://example.com', { headers });
  }

  it('rejects when no admin secret is configured', () => {
    expect(checkAdminAuth(requestWithAuth('Bearer anything'), undefined)).toBe(false);
  });

  it('rejects a missing or malformed header', () => {
    expect(checkAdminAuth(requestWithAuth(null), 'secret')).toBe(false);
    expect(checkAdminAuth(requestWithAuth('Basic secret'), 'secret')).toBe(false);
  });

  it('rejects a wrong token and accepts the right one', () => {
    expect(checkAdminAuth(requestWithAuth('Bearer wrong'), 'secret')).toBe(false);
    expect(checkAdminAuth(requestWithAuth('Bearer secret'), 'secret')).toBe(true);
  });
});

describe('verifyTurnstile', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('passes through when no secret is configured (pre-Phase-5 stub)', async () => {
    expect(await verifyTurnstile(undefined, undefined)).toBe(true);
  });

  it('rejects when a secret is configured but no token was supplied', async () => {
    expect(await verifyTurnstile(undefined, 'a-secret')).toBe(false);
  });

  it('calls the Cloudflare siteverify endpoint and returns its outcome', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ success: true }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    const ok = await verifyTurnstile('tok', 'a-secret', '1.2.3.4');

    expect(ok).toBe(true);
    expect(fetchMock).toHaveBeenCalledOnce();
    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe('https://challenges.cloudflare.com/turnstile/v0/siteverify');
  });

  it('returns false when Cloudflare reports failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ success: false }), { status: 200 })));
    expect(await verifyTurnstile('tok', 'a-secret')).toBe(false);
  });
});
