import json, sys
gt = json.load(open("gt.json")); pr = json.load(open(sys.argv[1]))
FIELDS = ["machine", "mode", "total", "date", "pitch", "stability", "dynamics", "longtone", "technique",
          "bonus", "kobushi", "shakuri", "vibrato", "national_avg", "prev", "best", "rank", "rank_of"]
stat = {}
rows = []
for k, g in gt.items():
    p = pr.get(k, {})
    marks = []
    for f in FIELDS:
        if f not in g:
            continue
        ok = p.get(f) == g[f]
        stat.setdefault(f, [0, 0]); stat[f][1] += 1; stat[f][0] += ok
        if not ok:
            marks.append(f"{f}: 正解={g[f]} 読取={p.get(f)}")
    rows.append((k, g["machine"], p.get("check"), marks, p.get("title_artist_raw")))
for k, m, chk, marks, ta in rows:
    print(f"{k} {m:8s} check={chk}")
    print("   曲名/歌手(生):", ta)
    for x in marks:
        print("   ✗", x)
print()
tot = [0, 0]
for f, (a, n) in stat.items():
    print(f"{f:14s} {a}/{n}"); tot[0] += a; tot[1] += n
print(f"ALL {tot[0]}/{tot[1]} = {tot[0]/tot[1]:.1%}")
