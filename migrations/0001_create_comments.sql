CREATE TABLE comments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_type TEXT NOT NULL DEFAULT 'blog',
  entity_slug TEXT NOT NULL,
  parent_id INTEGER REFERENCES comments(id),
  author_name TEXT NOT NULL,
  body TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'approved',
  ip_hash TEXT,
  created_at INTEGER NOT NULL,
  moderated_by TEXT,
  moderated_reason TEXT
);

CREATE INDEX idx_comments_entity ON comments(entity_type, entity_slug, status);
CREATE INDEX idx_comments_parent ON comments(parent_id);
