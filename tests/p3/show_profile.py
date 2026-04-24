import csv, sys
rows = list(csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else "taichi_compile_profile.csv")))
rows.sort(key=lambda r: -float(r["total_s"]))
for r in rows[:35]:
    # only show leaf (strip shared prefix to readable form)
    p = r["path"]
    # take last 2 segments
    segs = p.split("/")
    tail = "/".join(segs[-2:]) if len(segs) > 2 else p
    print(f"{float(r['total_s']):7.4f}  calls={int(r['calls']):4d}  {tail[:110]}")
