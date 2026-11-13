"""Sub-document retrieval units, cut on the fly -- the same units and ids as retrieval-unit-coverage/run_bm25.py.

Used by `EncodeDataset` when `--chunk_field` is set (passages only). Units of a document, ids "{docid}#{k}":
  --chunk_field text    sentences of the document text (fewer than --chunk_min_words words dropped),
                        grouped into windows of --chunk_size sliding by --chunk_stride
  --chunk_field claims  the document's list of claims (claimgen output, {"docid", "claims": [...]}),
                        grouped the same way (no word filter)
With --chunk_size 0, all sentences/claims of a document are joined into one unit that keeps the original docid.
As in BM25, the title (if any) is prepended to every unit; documents without sentences/claims give no units.
"""
import re

# split after .!? when followed by whitespace and an uppercase letter/digit/opening quote or bracket
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(\[A-Z0-9])")

# fallback fields, as get_text / get_units in run_bm25.py
FIELD_FALLBACKS = {"text": ["text", "segment"], "claims": ["claims", "statements"]}


def split_sentences(text):
    sents = []
    for para in text.splitlines():
        sents.extend(s.strip() for s in SENT_RE.split(para))
    return [s for s in sents if s]


def windows(items, size, stride):
    # slide until a window reaches the end; the last one may be shorter, every item is covered
    out = []
    for i in range(0, len(items), stride):
        out.append(items[i: i + size])
        if i + size >= len(items):
            break
    return out


def get_items(doc, field, min_words=3):
    """The sentences (string field) or list items (list field) a document is cut into."""
    value = None
    for f in FIELD_FALLBACKS.get(field, [field]):
        value = doc.get(f)
        if value:
            break
    if not value:
        return []
    if isinstance(value, str):
        # drop web boilerplate fragments ("Advertisement", "- Compass", ...) before windowing
        return [s for s in split_sentences(value) if len(s.split()) >= min_words]
    return list(value)


def get_units(doc, field, size=1, stride=None, min_words=3):
    """One document -> {unit id: unit text}.
    size (of the chunk). strid (between chunks)"""
    docid = doc.get("docid") or doc.get("id") or doc.get("_id")
    items = get_items(doc, field, min_words)
    title = doc.get("title") or ""
    if size <= 0:
        # whole document as one unit, keeping the original docid
        return {docid: f"{title} {' '.join(items)}".strip()} if items else {}
    return {f"{docid}#{k}": f"{title} {' '.join(w)}".strip()
            for k, w in enumerate(windows(items, size, stride or size))}


def to_units(batch, field, size, stride, min_words):
    """Batched `datasets.map` function: rows of documents -> rows of units {docid, text}."""
    keys = list(batch.keys())
    out = {"docid": [], "text": []}
    for i in range(len(batch[keys[0]])):
        doc = {k: batch[k][i] for k in keys}
        for uid, text in get_units(doc, field, size, stride, min_words).items():
            out["docid"].append(uid)
            out["text"].append(text)
    return out
