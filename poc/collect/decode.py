"""Decode a saved browser-tool result (JSON array of {type,text}) carrying base64 thumbnails."""
import sys, json, base64, os, re
src, label = sys.argv[1], sys.argv[2]
outdir = os.environ.get("RAW_DIR", "raw")
meta_path = os.environ.get("META_PATH", "meta.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
blocks = json.load(open(src))
txt = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
body = txt[: txt.index("\n\n(captured")] if "(captured" in txt else txt
v = json.loads(body.strip())
items = json.loads(v) if isinstance(v, str) else v
n = 0
for it in items:
    if "b64" not in it:
        continue
    key = it["id"]
    if key in meta:
        continue
    fn = f"{label}__{len(meta):04d}.jpg"
    open(os.path.join(outdir, fn), "wb").write(base64.b64decode(it["b64"]))
    meta[key] = {"file": fn, "query_label": label, "title": it.get("t"), "page": it.get("purl"), "orig": it.get("murl")}
    n += 1
json.dump(meta, open(meta_path, "w"), ensure_ascii=False, indent=1)
print("saved", n, "total", len(meta))
