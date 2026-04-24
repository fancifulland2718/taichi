"""P3.c — verify early-exit path actually fires.

Uses TI_COMPILE_PROFILE=1 and compares 'scalarize' pass time between a pure
scalar kernel (should early-exit, tiny time) and a matrix-heavy kernel
(should still run the full pipeline).
"""
import os
import subprocess
import sys
import tempfile
import csv
import glob
import shutil


RUNNER = r'''
import taichi as ti
ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)

{FIELDS}

@ti.kernel
def run():
    {BODY}

run()
ti.sync()
'''


SCALAR = {
    "FIELDS": """N = 1024
a = ti.field(ti.f32, shape=N)
b = ti.field(ti.f32, shape=N)
c = ti.field(ti.f32, shape=N)""",
    "BODY": """for i in a:
        x = a[i]
        y = b[i]
        z = ti.f32(0.0)
        # 400 scalar stmts but no matrices — forces 5-pass walks to be
        # relatively expensive if the early-exit is missing.
        for k in ti.static(range(400)):
            z += x + float(k) * y
        c[i] = z""",
}

MATRIX = {
    "FIELDS": """N = 1024
mat_a = ti.Matrix.field(8, 8, ti.f32, shape=N)
mat_b = ti.Matrix.field(8, 8, ti.f32, shape=N)
mat_c = ti.Matrix.field(8, 8, ti.f32, shape=N)""",
    "BODY": """for i in mat_a:
        mat_c[i] = mat_a[i] @ mat_b[i]""",
}


def run_and_collect(label, spec):
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "k.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(RUNNER.format(**spec))
        # Auto-flush writes to cwd when TI_COMPILE_PROFILE is set to a path
        # prefix. We set cwd=tmp to contain the CSV output.
        env = dict(os.environ)
        env["TI_COMPILE_PROFILE"] = os.path.join(tmp, "prof")
        subprocess.run([sys.executable, "-W", "ignore", script],
                       env=env, cwd=tmp, capture_output=True, text=True)
        csvs = glob.glob(os.path.join(tmp, "*.csv"))
        if not csvs:
            print(f"{label:10s}  <no profile CSV>")
            return
        # Dump rows containing 'scalar' for inspection
        print(f"--- {label} scalarize rows ---")
        for path in csvs:
            with open(path, newline="") as fh:
                for row in csv.reader(fh):
                    if row and any("scalar" in c.lower() for c in row):
                        print(" ", " | ".join(row))
        total_scalarize = 0.0
        total_all = 0.0
        rows_scalarize = 0
        for path in csvs:
            with open(path, newline="") as fh:
                for row in csv.reader(fh):
                    # CSV columns: thread,path,calls,total_s,avg_s,tpe_s
                    if len(row) < 4:
                        continue
                    name = row[1]
                    try:
                        total = float(row[3])
                    except ValueError:
                        continue
                    total_all += total
                    if "scalarize" in name.lower():
                        total_scalarize += total
                        rows_scalarize += 1
        print(f"{label:10s}  scalarize entries = {rows_scalarize:3d}  "
              f"scalarize total = {total_scalarize*1000:.3f} ms  "
              f"(all = {total_all*1000:.1f} ms)")


def main():
    run_and_collect("scalar", SCALAR)
    run_and_collect("matrix", MATRIX)


if __name__ == "__main__":
    main()
