#!/usr/bin/env python3
"""Parametric OpenFOAM sweep: angle of attack x freestream speed.

Prepares one case per (alpha, U) from cases/airfoil_template, runs them in the
OpenFOAM Docker image (several in parallel) and collects Cl, Cd, Cm into
data/sweep_results.csv.

Usage (from the project folder, on the Mac):
    python3 scripts/sweep.py                       # default grid, prepare + run + collect
    python3 scripts/sweep.py --alphas -4 0 4 8 --speeds 20 30 --jobs 4
    python3 scripts/sweep.py --collect-only        # re-read results without running
"""
import argparse, csv, math, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "cases" / "airfoil_template"
SWEEP = ROOT / "runs" / "sweep"
OUT = ROOT / "data" / "sweep_results.csv"

# Geometry of the airFoil2D tutorial mesh, measured from constant/polyMesh (wall patch):
# leading edge x = -17.549, trailing edge x = 17.500, span (z) = 0.05
X_LE, X_TE, SPAN = -17.549, 17.500, 0.05
CHORD = X_TE - X_LE
AREF = CHORD * SPAN
NU = 1e-5  # constant/transportProperties


def case_name(alpha, speed):
    return f"a{alpha:+06.2f}_U{speed:05.1f}".replace("+", "p").replace("-", "m")


def prepare(alpha, speed):
    case = SWEEP / case_name(alpha, speed)
    if case.exists():
        shutil.rmtree(case)
    shutil.copytree(TEMPLATE, case)
    shutil.copytree(case / "0.orig", case / "0")
    shutil.copytree(case / "constant" / "polyMesh.orig", case / "constant" / "polyMesh")

    a = math.radians(alpha)
    ux, uy = speed * math.cos(a), speed * math.sin(a)

    u = case / "0" / "U"
    txt = u.read_text()
    txt = re.sub(r"internalField\s+uniform\s+\([^)]*\);",
                 f"internalField   uniform ({ux:.6f} {uy:.6f} 0);", txt, count=1)
    u.write_text(txt)

    cd = case / "system" / "controlDict"
    txt = cd.read_text()
    txt = re.sub(r"endTime\s+\d+;", "endTime         1000;", txt, count=1)
    txt = re.sub(r"writeInterval\s+\d+;", "writeInterval   1000;", txt, count=1)
    txt = txt[: txt.index("\nfunctions")] + '\nfunctions\n{\n    #include "forceCoeffs"\n}\n'
    cd.write_text(txt)

    (case / "system" / "forceCoeffs").write_text(f"""forceCoeffs1
{{
    type            forceCoeffs;
    libs            (forces);
    writeControl    timeStep;
    writeInterval   1;
    patches         (walls);
    rho             rhoInf;
    rhoInf          1;
    CofR            ({X_LE + 0.25 * CHORD:.4f} 0 0);
    liftDir         ({-math.sin(a):.6f} {math.cos(a):.6f} 0);
    dragDir         ({math.cos(a):.6f} {math.sin(a):.6f} 0);
    pitchAxis       (0 0 1);
    magUInf         {speed};
    lRef            {CHORD:.4f};
    Aref            {AREF:.6f};
}}
""")
    return case


def run_all(jobs):
    script = SWEEP / "run_all.sh"
    script.write_text(
        "#!/bin/bash\n"
        f"ls -d a*_U* | xargs -P {jobs} -I{{}} bash -c "
        "'cd {} && simpleFoam > log.simpleFoam 2>&1 && echo done {} || echo FAILED {}'\n")
    script.chmod(0o755)
    subprocess.run([str(ROOT / "scripts" / "foam.sh"), str(SWEEP), "bash run_all.sh"], check=True)


def read_case(case):
    files = sorted(case.glob("postProcessing/forceCoeffs1/*/coefficient.dat"))
    if not files:
        return None
    header, rows = None, []
    for line in files[-1].read_text().splitlines():
        if line.startswith("#"):
            if "Time" in line:
                header = line.lstrip("#").split()
            continue
        if line.strip():
            rows.append([float(v) for v in line.split()])
    if not header or not rows:
        return None
    col = {name: i for i, name in enumerate(header)}
    last = rows[-1]
    tail = rows[-50:]
    cl_tail = [r[col["Cl"]] for r in tail]
    log = (case / "log.simpleFoam").read_text(errors="ignore") if (case / "log.simpleFoam").exists() else ""
    return {
        "Cl": last[col["Cl"]],
        "Cd": last[col["Cd"]],
        "Cm": last[col["CmPitch"]] if "CmPitch" in col else float("nan"),
        "Cl_range_last50": max(cl_tail) - min(cl_tail),
        "iterations": int(last[col["Time"]]),
        "converged": "SIMPLE solution converged" in log,
    }


def collect():
    rows = []
    for case in sorted(SWEEP.glob("a*_U*")):
        m = re.match(r"a([pm][\d.]+)_U([\d.]+)", case.name)
        alpha = float(m.group(1).replace("p", "+").replace("m", "-"))
        speed = float(m.group(2))
        res = read_case(case)
        if res is None:
            print(f"  no results: {case.name}")
            continue
        rows.append({"alpha_deg": alpha, "U_inf": speed, "Re": speed * CHORD / NU, **res})
    rows.sort(key=lambda r: (r["U_inf"], r["alpha_deg"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} cases -> {OUT.relative_to(ROOT)}")
    print(f"{'alpha':>6} {'U':>6} {'Cl':>8} {'Cd':>8} {'iter':>5} conv")
    for r in rows:
        print(f"{r['alpha_deg']:6.1f} {r['U_inf']:6.1f} {r['Cl']:8.4f} {r['Cd']:8.5f} {r['iterations']:5d} {r['converged']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="+", default=[-4, -2, 0, 2, 4, 6, 8, 10, 12, 14])
    ap.add_argument("--speeds", type=float, nargs="+", default=[10, 20, 30, 40])
    ap.add_argument("--jobs", type=int, default=4, help="cases run at the same time")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    if not args.collect_only:
        if not TEMPLATE.exists():
            sys.exit("Run scripts/fetch_template.sh first")
        SWEEP.mkdir(parents=True, exist_ok=True)
        cases = [prepare(a, u) for u in args.speeds for a in args.alphas]
        print(f"Prepared {len(cases)} cases in {SWEEP.relative_to(ROOT)} (chord {CHORD:.3f}, Aref {AREF:.4f})")
        if args.prepare_only:
            return
        run_all(args.jobs)
    collect()


if __name__ == "__main__":
    main()
