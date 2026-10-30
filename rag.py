"""
RAG over per-organization policy documents (corrected version).

INDEXING (when a finance admin uploads a policy):
    text -> split into passages (one clause per passage when the text is numbered)
         -> embed each passage locally with fastembed (384 numbers per passage)
         -> store text + vector in policy_chunks, tagged with org_id

RETRIEVAL (when someone asks a question):
    question -> embed -> pgvector finds the closest passages of THIS org only
    If semantic search is impossible (no pgvector, no embedding model, or the
    org's passages were stored without vectors) it falls back to Postgres
    full-text search, so the assistant is never silently blind.

Every function returns {"ok": True, "data": ...} or {"ok": False, "error": ...}.
search_policy's data is ALWAYS a plain list of dicts.

Setup: pip install fastembed
"""
import logging
import re
import threading
import time

import expense_db as db

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # 384 dimensions
EMBEDDING_DIM = 384
MAX_CHUNKS_PER_DOC = 200
MAX_CHUNK_CHARS = 1200
TOP_K = 5
# Cosine similarity below this is treated as "not relevant". It depends on the
# model: tune it with the not-covered questions in the test guide.
MIN_SIMILARITY = 0.40
MAX_QUERY_CHARS = 500
RETRY_MODEL_AFTER_SECONDS = 300   # a failed model download is retried, not given up on

# ---------------------------------------------------------------
# Embedding model (lazy singleton)
# ---------------------------------------------------------------
_embed_model = None
_embed_failed_at = None
_embed_lock = threading.Lock()


def _get_embed_model():
    """Load the fastembed model once. Returns None if unavailable right now."""
    global _embed_model, _embed_failed_at
    if _embed_model is not None:
        return _embed_model
    if _embed_failed_at is not None and time.monotonic() - _embed_failed_at < RETRY_MODEL_AFTER_SECONDS:
        return None
    with _embed_lock:
        if _embed_model is None:
            try:
                from fastembed import TextEmbedding
                _embed_model = TextEmbedding(model_name=EMBEDDING_MODEL)
                _embed_failed_at = None
                logger.info("fastembed model loaded.")
            except Exception as exc:
                logger.warning("fastembed unavailable (%s); using keyword search.", exc)
                _embed_failed_at = time.monotonic()
    return _embed_model


def embed_text(texts):
    """Embed a list of texts. Returns a list of lists of Python floats, or None."""
    model = _get_embed_model()
    if model is None:
        return None
    try:
        # float(x): fastembed yields numpy float32, which psycopg cannot adapt.
        return [[float(x) for x in vec] for vec in model.embed(texts)]
    except Exception as exc:
        logger.warning("Embedding failed (%s); using keyword search.", exc)
        return None


def _vector_literal(vec):
    """pgvector's text format: '[0.1,0.2,...]'. Cast with ::vector in SQL."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


# ---------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------
# A clause starts a line: "4.1 ...", "4.1: ..." or "Clause 4.1: ...".
_CLAUSE_START = re.compile(r"(?mi)^[ \t]*(?:clause[ \t]+)?\d{1,2}\.\d{1,2}(?![\d.])")
_HEADING = re.compile(r"^\s*(?:section\s+\d+\.|\d{1,2}\.\s+\S).{0,90}$", re.IGNORECASE)


def _sentence_chunks(text, max_chars=MAX_CHUNK_CHARS):
    """Pack sentences into chunks of at most max_chars (long sentences are cut)."""
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    chunks, current = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        while len(s) > max_chars:                 # a single huge "sentence"
            if current:
                chunks.append(current)
                current = ""
            chunks.append(s[:max_chars])
            s = s[max_chars:]
        if current and len(current) + len(s) + 1 > max_chars:
            chunks.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks


def _strip_trailing_headings(unit):
    lines = unit.splitlines()
    while lines and (not lines[-1].strip() or _HEADING.match(lines[-1])):
        lines.pop()
    return "\n".join(lines).strip()


def _chunk_text(text):
    """
    Returns (chunks, truncated). Numbered policies become one passage per clause,
    which retrieves precisely and lets answers cite the exact clause. Other text
    falls back to sentence packing.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return [], False

    starts = [m.start() for m in _CLAUSE_START.finditer(text)]
    if len(starts) >= 3:
        units = []
        head = text[:starts[0]].strip()
        if head:
            units.append(head)
        for i, s in enumerate(starts):
            end = starts[i + 1] if i + 1 < len(starts) else len(text)
            units.append(_strip_trailing_headings(text[s:end]))
    else:
        units = [text]

    chunks = []
    for unit in units:
        unit = re.sub(r"[ \t]+", " ", unit).strip()
        if not unit:
            continue
        chunks.extend([unit] if len(unit) <= MAX_CHUNK_CHARS else _sentence_chunks(unit))

    truncated = len(chunks) > MAX_CHUNKS_PER_DOC
    return chunks[:MAX_CHUNKS_PER_DOC], truncated


# ---------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------
def _vector_ready(conn):
    """True if the pgvector extension is installed in the connected database."""
    try:
        return conn.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'").fetchone() is not None
    except Exception:
        return False


def add_policy_document(document_id, org_id, content, sensitive_categories=None):
    """
    Chunk, embed and index a policy document.
    Returns {"ok": True, "data": {"chunks": n, "embedded": bool, "truncated": bool}}.

    sensitive_categories is accepted for compatibility but NOT used to drop policy
    text: the policy is not sensitive, individual medical-type CLAIMS are. Dropping
    a whole passage because it mentions "medical" would also delete the unrelated
    clauses merged into it, and the assistant would then claim the policy is silent.
    """
    error = db._check_org(org_id)
    if error:
        return db._fail(error)
    chunks, truncated = _chunk_text(content)
    if not chunks:
        return db._fail("The document has no text to index.")

    try:
        with db._conn() as conn:
            vector_ok = _vector_ready(conn)
    except db._DB_ERRORS as exc:
        return db._database_error("checking the search index", exc)

    embeddings = embed_text(chunks) if vector_ok else None
    use_vectors = embeddings is not None

    entries = []
    for i, chunk in enumerate(chunks):
        entry = {"chunk_index": i, "content": chunk}
        if use_vectors:
            entry["embedding"] = _vector_literal(embeddings[i])
        entries.append(entry)

    result = db.add_policy_chunks(document_id, org_id, entries)
    if result["ok"]:
        result["data"] = {"chunks": len(entries), "embedded": use_vectors, "truncated": truncated}
    return result


def index_policy_document(org_id, document_id, content, sensitive_categories=None):
    """Convenience wrapper (argument order differs from add_policy_document)."""
    return add_policy_document(document_id, org_id, content, sensitive_categories)


# ---------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------
def _semantic_rows(conn, org_id, query, limit):
    """Closest passages by meaning, or None if semantic search is not possible."""
    if not _vector_ready(conn):
        return None
    vectors = embed_text([query])
    if vectors is None:
        return None
    literal = _vector_literal(vectors[0])
    rows = conn.execute(
        "SELECT content, 1 - (embedding <=> %s::vector) AS similarity "
        "FROM policy_chunks WHERE org_id = %s AND embedding IS NOT NULL "
        "ORDER BY embedding <=> %s::vector LIMIT %s",
        (literal, org_id, literal, limit),
    ).fetchall()
    return [{"content": r["content"], "similarity": round(float(r["similarity"]), 3),
             "method": "semantic"} for r in rows]


def _keyword_rows(conn, org_id, query, limit):
    """Full-text search with OR between the words, ranked. Numbers count as words."""
    terms = [t for t in re.findall(r"[A-Za-z0-9]{2,}", query) if t.lower() != "or"][:12]
    if not terms:
        return []
    tsquery = " or ".join(terms)      # alphanumeric only: nothing to inject
    rows = conn.execute(
        "SELECT content, ts_rank_cd(to_tsvector('english', content), "
        "websearch_to_tsquery('english', %s)) AS score "
        "FROM policy_chunks WHERE org_id = %s "
        "AND to_tsvector('english', content) @@ websearch_to_tsquery('english', %s) "
        "ORDER BY score DESC LIMIT %s",
        (tsquery, org_id, tsquery, limit),
    ).fetchall()
    return [{"content": r["content"], "similarity": None, "method": "keyword"} for r in rows]


def search_policy(org_id, query, limit=TOP_K):
    """
    Passages of THIS organization's policy relevant to the query.
    data is always a list of {"content", "similarity", "method"}.
    """
    error = db._check_org(org_id)
    if error:
        return db._fail(error)
    query = (query or "").strip()[:MAX_QUERY_CHARS]
    if not query:
        return db._ok([])
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        limit = TOP_K

    try:
        with db._conn() as conn:
            rows = _semantic_rows(conn, org_id, query, limit)
            if rows is None or not rows:
                # Semantic search impossible, or this org has no embedded passages:
                # never return nothing just because vectors are missing.
                return db._ok(_keyword_rows(conn, org_id, query, limit))
            # Semantic search ran: low scores genuinely mean "not covered".
            return db._ok([r for r in rows if r["similarity"] >= MIN_SIMILARITY])
    except db._DB_ERRORS as exc:
        return db._database_error("searching the policy", exc)
    except Exception as exc:
        logger.exception("Policy search failed")
        return db._database_error("searching the policy", exc)


def policy_index_status(org_id):
    """Diagnostics: is this org's policy indexed, and how?"""
    error = db._check_org(org_id)
    if error:
        return db._fail(error)
    try:
        with db._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS chunks, COUNT(embedding) AS embedded "
                "FROM policy_chunks WHERE org_id = %s", (org_id,)).fetchone()
            return db._ok({"chunks": row["chunks"], "embedded": row["embedded"],
                           "pgvector_installed": _vector_ready(conn),
                           "embedding_model_loaded": _get_embed_model() is not None})
    except db._DB_ERRORS as exc:
        return db._database_error("checking the policy index", exc)