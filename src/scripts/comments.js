function buildTree(rows) {
  const byId = new Map();
  const roots = [];
  for (const row of rows) byId.set(row.id, { ...row, replies: [] });
  for (const row of rows) {
    const node = byId.get(row.id);
    if (row.parent_id && byId.has(row.parent_id)) {
      byId.get(row.parent_id).replies.push(node);
    } else {
      roots.push(node);
    }
  }
  return roots;
}

function renderNode(node) {
  const li = document.createElement('li');
  li.className = 'comment';

  const meta = document.createElement('p');
  meta.className = 'comment-meta';
  meta.textContent = `${node.author_name} · ${new Date(node.created_at).toLocaleDateString()}`;

  const body = document.createElement('p');
  body.className = 'comment-body';
  body.textContent = node.body;

  li.append(meta, body);

  if (node.replies.length) {
    const replyList = document.createElement('ul');
    replyList.className = 'comment-list comment-reply';
    for (const reply of node.replies) replyList.appendChild(renderNode(reply));
    li.appendChild(replyList);
  }

  return li;
}

export function initComments(root) {
  const slug = root.dataset.slug;
  const list = root.querySelector('#comment-list');
  const empty = root.querySelector('#comment-empty');
  const form = root.querySelector('#comment-form');
  const status = root.querySelector('#comment-status');

  async function load() {
    const res = await fetch(`/api/comments/${encodeURIComponent(slug)}`);
    if (!res.ok) return;
    const rows = await res.json();
    render(rows);
  }

  function render(rows) {
    const tree = buildTree(rows);
    list.innerHTML = '';
    if (tree.length === 0) {
      empty.hidden = false;
      return;
    }
    empty.hidden = true;
    for (const node of tree) list.appendChild(renderNode(node));
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const submitBtn = form.querySelector('button');
    submitBtn.disabled = true;
    status.textContent = '';
    delete status.dataset.state;

    const authorName = form.author_name.value.trim();
    const bodyText = form.body.value.trim();
    if (!authorName || !bodyText) {
      status.textContent = 'Name and comment are required.';
      status.dataset.state = 'error';
      submitBtn.disabled = false;
      return;
    }

    const turnstileInput = form.querySelector('[name="cf-turnstile-response"]');
    const turnstileToken = turnstileInput ? turnstileInput.value : undefined;

    try {
      const res = await fetch('/api/comments/submit', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ entity_slug: slug, author_name: authorName, body: bodyText, turnstileToken }),
      });
      if (!res.ok) throw new Error('submit failed');
      form.reset();
      status.textContent = 'Comment posted.';
      status.dataset.state = 'success';
      await load();
    } catch {
      status.textContent = 'Something went wrong. Try again.';
      status.dataset.state = 'error';
    } finally {
      submitBtn.disabled = false;
      if (window.turnstile) window.turnstile.reset();
    }
  });

  load();
}
