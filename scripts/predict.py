#!/usr/bin/env python3
"""Instant Cl / Cd prediction from the trained GPR surrogate.
Usage: python3 scripts/predict.py --alpha 7.5 --speed 25
"""
import argparse, time
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHORD, NU = 35.049, 1e-5
ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, required=True, help="angle of attack [deg]")
ap.add_argument("--speed", type=float, required=True, help="freestream speed [m/s]")
args = ap.parse_args()
x = np.array([[args.alpha, np.log10(args.speed * CHORD / NU)]])
m_cl = joblib.load(ROOT / "models" / "Cl_GPR.joblib")
m_cd = joblib.load(ROOT / "models" / "Cd_GPR.joblib")
t0 = time.perf_counter()
cl, cd = m_cl.predict(x)[0], m_cd.predict(x)[0]
dt = (time.perf_counter() - t0) * 1e3
print(f"alpha = {args.alpha} deg, U = {args.speed} m/s  ->  Cl = {cl:.4f}, Cd = {cd:.5f}, L/D = {cl/cd:.1f}  ({dt:.2f} ms)")
