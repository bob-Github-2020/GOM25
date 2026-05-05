#!/usr/bin/python3

## 8-16-2025

"""
transform_xyz_gom25.py

One-set 7-parameter (no scale) transformation between IGS20 and GOM25
for ECEF XYZ coordinate time series (with optional covariance).

Direction is chosen explicitly via --transform:
  - "IGS20->GOM25"  (forward)
  - "GOM25->IGS20"  (inverse; first-order exact and sufficient for geodetic use)

Hard-coded parameters (rates):
  - Translation rates (m/yr): T' = [dTx, dTy, dTz]
  - Rotation rates   (rad/yr): R' = [dRx, dRy, dRz]
  - Reference epoch t0 (year): 2020.0

Forward (IGS20 -> GOM25):
  P_GOM25(t) = P_IGS20(t) + Δt * T' + Δt * [R']_× * P_IGS20(t)
  A_fwd      = I + Δt * [R']_×
  C_GOM25    = A_fwd * C_IGS20 * A_fwd^T

Inverse (GOM25 -> IGS20), first-order:
  P_IGS20(t) = P_GOM25(t) - Δt * T' - Δt * [R']_× * P_GOM25(t)
  A_inv      = I - Δt * [R']_×
  C_IGS20    = A_inv * C_GOM25 * A_inv^T

Input XYZ file columns (whitespace-separated):
  decyear  X(m)  Y(m)  Z(m)  SigX(m)  SigY(m)  SigZ(m)  CovYX  CovZX  CovZY
  - CovYX = Cov(Y,X) = Cov(X,Y), CovZX = Cov(Z,X), CovZY = Cov(Z,Y)
  - If only 7 columns are present (no covariances), covariance is assumed diagonal.

Comments/blank lines are copied to output.

Output filename is derived from input and transform direction:
  SITE_IGS20.XYZ -> SITE_GOM25.XYZ (for forward)
  SITE_GOM25.XYZ -> SITE_IGS20.XYZ (for inverse)
  If suffix not present, appropriate suffix is appended before .XYZ.

Usage examples:
  python3 transform_xyz_gom25.py --transform 'IGS20->GOM25' *_IGS20.XYZ
  python3 transform_xyz_gom25.py --transform 'GOM25->IGS20' *_GOM25.XYZ
"""

from __future__ import annotations
import argparse
import glob
import math
import os
from typing import List, Optional, Tuple

import numpy as np

# ------------------------- Hard-coded 7 parameters (rates) -------------------------
# Units: translation rates in m/yr; rotation rates in rad/yr
D_T = np.array([
    2.2547388651559148e-03,   # dTx (m/yr)
   -9.5000861794855779e-05,   # dTy (m/yr)
    9.1476512873410660e-04    # dTz (m/yr)
], dtype=float)

D_R = np.array([
   -4.9614442150267694e-11,   # dRx (rad/yr)
   -3.0265313090239882e-09,   # dRy (rad/yr)
   -1.0944450082929799e-10    # dRz (rad/yr)
], dtype=float)

T0 = 2025.0  # reference epoch (year)

# ------------------------------- Math utilities -----------------------------------

def skew(rx: float, ry: float, rz: float) -> np.ndarray:
    """Return the 3x3 cross-product (skew-symmetric) matrix [R]_x for vector R = (rx, ry, rz)."""
    return np.array([[ 0.0,  rz,  -ry],
                     [-rz,  0.0,  rx],
                     [ ry, -rx,  0.0]], dtype=float)

def parse_numeric_fields(tokens: List[str]) -> Optional[Tuple[float, np.ndarray, Optional[np.ndarray]]]:
    """
    Parse a data line tokens into (t, P(3,), C(3,3)|None).
    Accepts 7-column (diagonal cov only) or 10-column (full cov) formats.
    Returns None if parsing fails.
    """
    try:
        t = float(tokens[0])
        X, Y, Z = float(tokens[1]), float(tokens[2]), float(tokens[3])
        sigX, sigY, sigZ = float(tokens[4]), float(tokens[5]), float(tokens[6])
    except Exception:
        return None

    P = np.array([X, Y, Z], dtype=float)
    # Covariance
    if len(tokens) >= 10:
        covYX, covZX, covZY = float(tokens[7]), float(tokens[8]), float(tokens[9])
        C = np.array([[sigX**2,      covYX,      covZX],
                      [covYX,        sigY**2,    covZY],
                      [covZX,        covZY,      sigZ**2]], dtype=float)
    else:
        C = np.diag(np.array([sigX**2, sigY**2, sigZ**2], dtype=float))
    return t, P, C

def format_line(t: float, P: np.ndarray, C: Optional[np.ndarray]) -> str:
    """
    Format one output record matching the expected XYZ format.
    We use scientific notation for vectors and 7 decimal places for decyear.
    """
    x, y, z = P.tolist()
    if C is None:
        sigx = sigy = sigz = float("nan")
        cxy = cxz = cyz = float("nan")
    else:
        sigx = math.sqrt(max(C[0,0], 0.0))
        sigy = math.sqrt(max(C[1,1], 0.0))
        sigz = math.sqrt(max(C[2,2], 0.0))
        cxy, cxz, cyz = C[0,1], C[0,2], C[1,2]

    return (
        f"{t:.7f} "
        f"{x:.16e} {y:.16e} {z:.16e} "
        f"{sigx:.16e} {sigy:.16e} {sigz:.16e} "
        f"{cxy:.16e} {cxz:.16e} {cyz:.16e}"
    )

# ----------------------------- Core transformations -------------------------------

def forward_igs20_to_gom25(t: float, P_igs: np.ndarray, C_igs: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply forward transform (IGS20 -> GOM25) to position and covariance."""
    dt = t - T0
    S = skew(D_R[0], D_R[1], D_R[2])        # [R']_x
    A = np.eye(3) + dt * S                  # linear operator on coordinates
    P_gom = A @ P_igs + dt * D_T            # translation rates contribution
    C_gom = None if C_igs is None else A @ C_igs @ A.T
    return P_gom, C_gom

def inverse_gom25_to_igs20(t: float, P_gom: np.ndarray, C_gom: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply inverse transform (GOM25 -> IGS20) to position and covariance (first-order)."""
    dt = t - T0
    S = skew(D_R[0], D_R[1], D_R[2])        # [R']_x
    A = np.eye(3) - dt * S                  # first-order inverse operator
    P_igs = A @ (P_gom - dt * D_T)
    C_igs = None if C_gom is None else A @ C_gom @ A.T
    return P_igs, C_igs

# ------------------------------- File processing ----------------------------------

def decide_out_name(in_path: str, direction: str) -> str:
    """Derive an output filename based on direction and common suffixes."""
    base = os.path.basename(in_path)
    root, ext = os.path.splitext(base)
    ext = ext if ext else ".XYZ"

    if direction == "IGS20->GOM25":
        if root.endswith("_IGS20"):
            root = root[:-6] + "_GOM25"
        elif root.endswith("_GOM25"):
            # user claims forward but file has _GOM25; still honor direction
            root = root  # keep and add suffix
            root += "_toGOM25"
        else:
            root = root + "_GOM25"
    else:  # GOM25->IGS20
        if root.endswith("_GOM25"):
            root = root[:-6] + "_IGS20"
        elif root.endswith("_IGS20"):
            root = root
            root += "_toIGS20"
        else:
            root = root + "_IGS20"

    return os.path.join(os.path.dirname(in_path), root + ext)

def transform_file(path: str, direction: str) -> str:
    """Transform a single XYZ file and return the output path."""
    out_path = decide_out_name(path, direction)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        # Header
        #fout.write(f"# Transformed by transform_xyz_gom25.py\n")
        #fout.write(f"# Direction: {direction}\n")
        #fout.write(f"# Parameters (rates): dTx={D_T[0]:.16e} m/yr, dTy={D_T[1]:.16e} m/yr, dTz={D_T[2]:.16e} m/yr\n")
        #fout.write(f"#                     dRx={D_R[0]:.16e} rad/yr, dRy={D_R[1]:.16e} rad/yr, dRz={D_R[2]:.16e} rad/yr\n")
        #fout.write(f"# Reference epoch t0={T0:.1f}\n")
        fout.write(f"decyear X Y Z SigX SigY SigZ CovYX CovZX CovZY\n")

        for line in fin:
            ls = line.strip()
            if not ls:
                fout.write("\n")
                continue
            if ls.startswith("#"):
                # preserve original comments but note they are from input
                fout.write("# " + ls.lstrip("#").strip() + "\n")
                continue

            tokens = ls.split()
            parsed = parse_numeric_fields(tokens)
            if parsed is None:
                # Not a data line — copy as comment to avoid data corruption
                fout.write("# (unparsed) " + ls + "\n")
                continue

            t, P_in, C_in = parsed
            if direction == "IGS20->GOM25":
                P_out, C_out = forward_igs20_to_gom25(t, P_in, C_in)
            else:
                P_out, C_out = inverse_gom25_to_igs20(t, P_in, C_in)

            fout.write(format_line(t, P_out, C_out) + "\n")

    return out_path

# ----------------------------------- CLI -----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Transform XYZ time series between IGS20 and GOM25 using one-set 7-parameter (no scale) method.")
    ap.add_argument("--transform", required=True, choices=["IGS20->GOM25", "GOM25->IGS20"],
                    help="Direction of transformation.")
    ap.add_argument("files", nargs="*", help="Input XYZ files or globs. If empty, defaults to *.XYZ")
    args = ap.parse_args()

    # Expand globs
    files: List[str] = []
    if args.files:
        for pat in args.files:
            expanded = glob.glob(pat)
            if expanded:
                files.extend(expanded)
            else:
                # If a literal file path that doesn't glob-expand
                if os.path.isfile(pat):
                    files.append(pat)
    else:
        files = glob.glob("*.XYZ")

    if not files:
        raise SystemExit("No input files found. Provide paths or run in a folder with *.XYZ")

    print(f"Transform: {args.transform}")
    print(f"Reference epoch t0 = {T0:.1f}")
    print(f"dT (m/yr): {D_T}")
    print(f"dR (rad/yr): {D_R}")
    print(f"Files: {len(files)}")

    out_paths = []
    for path in files:
        try:
            outp = transform_file(path, args.transform)
            out_paths.append(outp)
            print(f"  wrote: {outp}")
        except Exception as e:
            print(f"  ERROR on {path}: {e}")

    print("Done.")

if __name__ == "__main__":
    main()
