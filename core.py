# -*- coding: utf-8 -*-
"""Browser core for the rizzo-pii live demo (runs in Pyodide).

Port of Rizzo-AI-Academy/rizzo-pii src/app/app.py (MIT, (c) Rizzo AI Academy):
chunking, model+regex merge, reversible placeholders and restore. The regex/checksum
net is `detectors.py`, copied verbatim from the repo. The neural model runs in the
browser via transformers.js/onnxruntime-web; this module receives its logits.
"""
import bisect
import json
import math
import re

from detectors import SOFT_REGEX_LABELS, complete_time, detect_regex

MAX_WORDS, OVERLAP = 120, 20
ID2LABEL = {}


def set_labels(id2label_json):
    global ID2LABEL
    ID2LABEL = {int(k): v for k, v in json.loads(id2label_json).items()}


def chunk_text(text):
    words = list(re.finditer(r"\S+", text))
    chunks, i, step = [], 0, max(1, MAX_WORDS - OVERLAP)
    while i < len(words):
        block = words[i:i + MAX_WORDS]
        s, e = block[0].start(), block[-1].end()
        chunks.append((text[s:e], s))
        if i + MAX_WORDS >= len(words):
            break
        i += step
    return chunks


def chunks_json(text):
    return json.dumps([c for c, _ in chunk_text(text)], ensure_ascii=False)


_BYTE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def align(text, tokens):
    """Character offsets for SentencePiece/Metaspace tokens (prepend_scheme=always,
    byte fallback), same convention as the HF fast tokenizer's offset_mapping."""
    offs, p, n = [], 0, len(text)
    pend = b""
    pend_start = None
    for tok in tokens:
        m = _BYTE.match(tok)
        if m:
            if pend_start is None:
                pend_start = p
            pend += bytes([int(m.group(1), 16)])
            try:
                ch = pend.decode("utf-8")
                pend = b""
                if p < n and text[p] == ch:
                    p += 1
                offs.append((pend_start, p))
                pend_start = None
            except UnicodeDecodeError:
                offs.append((pend_start, p))
            continue
        start = p
        for c in tok:
            if c == "\u2581":
                if p < n and text[p] == " ":
                    p += 1
            elif p < n and text[p] == c:
                p += 1
            else:
                j = text.find(c, p)          # resync on any normalisation drift
                if j != -1 and j - p < 8:
                    p = j + 1
        offs.append((start, p))
    return offs


def _softmax(row):
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return [v / s for v in ex]


def _tag(name):
    if name.startswith("B-"):
        return "B", name[2:]
    if name.startswith("I-"):
        return "I", name[2:]
    return "I", name


def aggregate_simple(text, tokens, logits):
    """HF TokenClassificationPipeline aggregation_strategy='simple'.
    tokens: tokens without specials; logits: rows for <bos> + tokens + <eos>."""
    offs = align(text, tokens)
    rows = logits[1:1 + len(tokens)]
    pre = []
    for (s, e), row in zip(offs, rows):
        pr = _softmax(row)
        k = max(range(len(pr)), key=pr.__getitem__)
        pre.append({"entity": ID2LABEL[k], "score": pr[k], "start": s, "end": e})
    groups, cur = [], []

    def flush():
        if cur:
            bi, tag = _tag(cur[0]["entity"])
            groups.append({"entity_group": tag,
                           "score": sum(t["score"] for t in cur) / len(cur),
                           "start": cur[0]["start"], "end": cur[-1]["end"]})

    for t in pre:
        if not cur:
            cur = [t]
            continue
        bi, tag = _tag(t["entity"])
        _, last_tag = _tag(cur[-1]["entity"])
        if tag == last_tag and bi != "B":
            cur.append(t)
        else:
            flush()
            cur = [t]
    flush()
    return [g for g in groups if g["entity_group"] != "O"]


def _is_word(ch):
    return ch.isalnum() or ch == "_"


def _merge(cands, text):
    order = sorted(cands, key=lambda e: (
        1 if e["validated"] else 0,
        1 if (e["source"] == "regex" and e["label"] not in SOFT_REGEX_LABELS) else 0,
        e["score"], e["end"] - e["start"]), reverse=True)
    kept = []
    for e in order:
        i = bisect.bisect_right([k["start"] for k in kept], e["start"])
        if (i and kept[i - 1]["end"] > e["start"]) or (i < len(kept) and kept[i]["start"] < e["end"]):
            continue
        kept.insert(i, dict(e))
    for e in kept:
        while e["start"] < e["end"] and text[e["start"]].isspace():
            e["start"] += 1
        while e["end"] > e["start"] and text[e["end"] - 1].isspace():
            e["end"] -= 1
    kept = [e for e in kept if e["end"] > e["start"]]
    for e in kept:
        while e["start"] > 0 and _is_word(text[e["start"] - 1]) and _is_word(text[e["start"]]):
            e["start"] -= 1
        while e["end"] < len(text) and _is_word(text[e["end"]]) and _is_word(text[e["end"] - 1]):
            e["end"] += 1
    kept.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    merged = []
    for e in kept:
        if merged and e["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
            continue
        if merged and e["start"] == merged[-1]["end"] and e["label"] == merged[-1]["label"]:
            merged[-1]["end"] = e["end"]
            continue
        merged.append(e)
    return merged


def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).casefold()


def analyze(text, chunk_results_json, excluded_json="[]"):
    """chunk_results_json: [{"tokens": [...], "logits": [[...], ...]}, ...] in chunk order."""
    excluded = set(json.loads(excluded_json))
    results = json.loads(chunk_results_json)
    model_ents = []
    for (chunk, off), res in zip(chunk_text(text), results):
        for g in aggregate_simple(chunk, res["tokens"], res["logits"]):
            model_ents.append({"label": g["entity_group"], "start": g["start"] + off,
                               "end": g["end"] + off, "score": float(g["score"]),
                               "validated": False, "source": "model"})
    complete_time(model_ents, text)
    cands = [e for e in model_ents + detect_regex(text) if e["label"] not in excluded]
    kept = _merge(cands, text)

    counters, seen, mapping = {}, {}, {}
    for e in kept:
        val = text[e["start"]:e["end"]]
        key = (e["label"], _norm(val))
        if key not in seen:
            counters[e["label"]] = counters.get(e["label"], 0) + 1
            seen[key] = f"[{e['label']}_{counters[e['label']]}]"
            mapping[seen[key]] = val
        e["ph"] = seen[key]

    segments, anon, pos, by_label = [], [], 0, {}
    for e in kept:
        if e["start"] > pos:
            segments.append({"t": text[pos:e["start"]]})
            anon.append(text[pos:e["start"]])
        segments.append({"t": text[e["start"]:e["end"]], "label": e["label"], "ph": e["ph"],
                         "src": e["source"], "validated": e["validated"]})
        anon.append(e["ph"])
        by_label[e["label"]] = by_label.get(e["label"], 0) + 1
        pos = e["end"]
    if pos < len(text):
        segments.append({"t": text[pos:]})
        anon.append(text[pos:])
    return json.dumps({
        "segments": segments, "anonymized_text": "".join(anon), "mapping": mapping,
        "n_entities": len(kept), "n_unique": len(seen),
        "n_validated": sum(1 for e in kept if e["validated"]),
        "by_label": dict(sorted(by_label.items(), key=lambda x: -x[1])),
    }, ensure_ascii=False)


def restore(reply, mapping_json):
    mapping = json.loads(mapping_json or "{}")
    if not mapping:
        return reply
    by_inner = {ph.strip("[]"): v for ph, v in mapping.items()}
    alts = []
    for inner in sorted(by_inner, key=len, reverse=True):
        esc = re.escape(inner)
        alts.append(r"\[\s*" + esc + r"\s*\]|(?<!\w)" + esc + r"(?!\w)")
    rx = re.compile(r"(\*{0,2})(?:" + "|".join(alts) + r")\1")

    def sub(m):
        core = m.group(0).strip("*")
        br = re.match(r"^\[\s*([\s\S]*?)\s*\]$", core)
        return by_inner.get(br.group(1) if br else core, m.group(0))

    return rx.sub(sub, reply)
