#!/usr/bin/env python3

## 8-16-2025

# -*- coding: utf-8 -*-
"""
Convert_XYZ2ENU.py
------------------
Convert GNSS ECEF XYZ time series to local-topocentric ENU (East, North, Up).
- Accepts files that may or may not contain a header line.
- Skips lines containing "STOP" / "END" and comment lines beginning with '#', '%', or ';'.
- Uses the median of the first N samples (default 30) of X,Y,Z to define the reference
  position and rotation (lon/lat), and also subtracts the median of the first N ENU
  samples from each component (per your request).

Input (.XYZ) expected columns (whitespace-delimited):
  1) decyear
  2) X (m)
  3) Y (m)
  4) Z (m)
  5) SigX (m)        [optional]
  6) SigY (m)        [optional]
  7) SigZ (m)        [optional]
  8) CovYX or RhoYX  [optional: covariance (m^2) or correlation]
  9) CovZX or RhoZX  [optional]
 10) CovZY or RhoZY  [optional]

If 10 columns are present, uncertainties are propagated:
  Cov_ENU = R * Cov_XYZ * R^T
where R is the ECEF→ENU rotation matrix at the reference location.
If the last three inputs have |value| ≤ ~1.5, they are interpreted as correlation
coefficients; otherwise as covariances in m^2.

Output (.enu): tab-delimited columns
  decyear  EW_cm  NS_cm  UD_cm  SigE_cm  SigN_cm  SigU_cm
If input file lacks uncertainty columns, Sig* fields are left blank.

Usage
-----
python3 Convert_XYZ2ENU.py *.XYZ
python3 Convert_XYZ2ENU.py *.XYZ --nref 40

"""

import argparse
import math
import os
from typing import List, Tuple, Optional

import numpy as np


def is_comment_or_stop(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    head = s.split()[0].upper()
    if head in ("STOP", "END"):
        return True
    if s[0] in ("#", "%", ";"):
        return True
    return False


def try_parse_row(tokens: List[str]) -> Optional[Tuple[float, ...]]:
    """
    Try to parse a numeric row. Returns tuple of floats if successful.
    Accepts at least 4 tokens; will fill missing uncertainty fields with None.
    """
    # Hard skip if first token can't be a float (this also skips header lines like "decyear ...")
    try:
        t = float(tokens[0])
    except Exception:
        return None

    if len(tokens) < 4:
        return None

    try:
        x = float(tokens[1]); y = float(tokens[2]); z = float(tokens[3])
    except Exception:
        return None

    # Optional uncertainties
    sigx = sigy = sigz = cov_yx = cov_zx = cov_zy = None
    if len(tokens) >= 10:
        try:
            sigx = float(tokens[4]); sigy = float(tokens[5]); sigz = float(tokens[6])
            cov_yx = float(tokens[7]); cov_zx = float(tokens[8]); cov_zy = float(tokens[9])
        except Exception:
            # If any fails, drop uncertainties for this row
            sigx = sigy = sigz = cov_yx = cov_zx = cov_zy = None

    return (t, x, y, z, sigx, sigy, sigz, cov_yx, cov_zx, cov_zy)


# WGS84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)  # first eccentricity squared


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert ECEF XYZ (m) to geodetic (lat, lon in radians; h in m) using Bowring's method.
    """
    lon = math.atan2(y, x)
    r = math.hypot(x, y)
    if r < 1e-12:
        # Poles: lon arbitrary
        lon = 0.0

    # Bowring’s formula for initial phi
    E2P = _E2 / (1.0 - _E2)
    u = math.atan2(z * math.sqrt(1.0 - _E2), r)
    sin_u = math.sin(u); cos_u = math.cos(u)
    lat = math.atan2(z + E2P * (6378137.0 * (1.0 - _F)) * sin_u**3,
                     r - _E2 * _A * cos_u**3)

    sin_lat = math.sin(lat)
    N = _A / math.sqrt(1.0 - _E2 * sin_lat * sin_lat)
    h = r / math.cos(lat) - N
    return lat, lon, h


def rot_ecef_to_enu(lat_rad: float, lon_rad: float) -> np.ndarray:
    """
    Rotation matrix R such that ENU = R * dXYZ (ECEF).
    """
    sl = math.sin(lat_rad); cl = math.cos(lat_rad)
    slon = math.sin(lon_rad); clon = math.cos(lon_rad)
    R = np.array([
        [-slon,       clon,       0.0],
        [-sl*clon, -sl*slon,      cl ],
        [ cl*clon,  cl*slon,      sl ]
    ], dtype=float)
    return R


def build_cov_xyz(sigx: float, sigy: float, sigz: float,
                  a: float, b: float, c: float) -> np.ndarray:
    """
    Build 3x3 covariance in XYZ. The last three inputs (a,b,c) are interpreted as either
    covariance (m^2) or correlation (unitless) depending on magnitude.
    Here we assume they were provided as (CovYX, CovZX, CovZY) or (RhoYX, RhoZX, RhoZY).
    """
    # Diagonal
    C = np.array([[sigx**2, 0.0, 0.0],
                  [0.0, sigy**2, 0.0],
                  [0.0, 0.0, sigz**2]], dtype=float)

    # Decide correlation vs covariance
    vals = [abs(a), abs(b), abs(c)]
    if max(vals) <= 1.5:  # treat as correlations
        cov_yx = a * sigy * sigx
        cov_zx = b * sigz * sigx
        cov_zy = c * sigz * sigy
    else:  # already covariances
        cov_yx, cov_zx, cov_zy = a, b, c

    # Fill symmetric off-diagonals: XY, XZ, YZ from the given ordering
    C[0,1] = C[1,0] = cov_yx  # XY = YX
    C[0,2] = C[2,0] = cov_zx  # XZ = ZX
    C[1,2] = C[2,1] = cov_zy  # YZ = ZY
    return C


def process_file(path: str, nref: int) -> Optional[str]:
    base = os.path.basename(path)
    root, ext = os.path.splitext(base)
    if ext.lower() != ".xyz":
        print(f"[WARN] Skipped {base}: not .XYZ")
        return None

    # Read & parse numeric rows only
    rows: List[Tuple[float, ...]] = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if is_comment_or_stop(line):
                continue
            toks = line.strip().split()
            if not toks:
                continue
            rec = try_parse_row(toks)
            if rec is None:
                continue
            rows.append(rec)

    if len(rows) < 2:
        print(f"[WARN] Skipped {base}: no numeric data rows found.")
        return None

    arr = np.array(rows, dtype=object)  # object to allow None in uncertainty cols
    # Core columns
    t = arr[:, 0].astype(float)
    X = arr[:, 1].astype(float)
    Y = arr[:, 2].astype(float)
    Z = arr[:, 3].astype(float)

    # Reference from median of first nref samples
    k = min(nref, len(X))
    X0 = float(np.median(X[:k]))
    Y0 = float(np.median(Y[:k]))
    Z0 = float(np.median(Z[:k]))

    lat0, lon0, _ = ecef_to_geodetic(X0, Y0, Z0)
    R = rot_ecef_to_enu(lat0, lon0)

    dXYZ = np.vstack((X - X0, Y - Y0, Z - Z0))  # 3 x N
    ENU = (R @ dXYZ).T  # N x 3
    E = ENU[:, 0].copy()
    Nn = ENU[:, 1].copy()
    U = ENU[:, 2].copy()

    # Standardize by subtracting median of first nref for each component
    E0 = float(np.median(E[:k])); N0 = float(np.median(Nn[:k])); U0 = float(np.median(U[:k]))
    E -= E0; Nn -= N0; U -= U0

    # Unit: m -> cm
    E_cm = 100.0 * E
    N_cm = 100.0 * Nn
    U_cm = 100.0 * U

    # Uncertainty propagation if we have all 10 columns
    has_unc = all(val is not None for val in arr[0, 4:10]) and np.all([all(v is not None for v in arr[i, 4:10]) for i in range(len(arr))])
    SigE_cm = SigN_cm = SigU_cm = None

    if has_unc:
        SigE = np.empty(len(t), dtype=float)
        SigN = np.empty(len(t), dtype=float)
        SigU = np.empty(len(t), dtype=float)
        for i in range(len(t)):
            sigx = float(arr[i, 4]); sigy = float(arr[i, 5]); sigz = float(arr[i, 6])
            a = float(arr[i, 7]); b = float(arr[i, 8]); c = float(arr[i, 9])
            Cxyz = build_cov_xyz(sigx, sigy, sigz, a, b, c)
            Cenu = R @ Cxyz @ R.T
            # std devs
            SigE[i] = math.sqrt(max(Cenu[0, 0], 0.0))
            SigN[i] = math.sqrt(max(Cenu[1, 1], 0.0))
            SigU[i] = math.sqrt(max(Cenu[2, 2], 0.0))
        # to cm
        SigE_cm = 100.0 * SigE
        SigN_cm = 100.0 * SigN
        SigU_cm = 100.0 * SigU

    # Write output
    out_path = os.path.join(os.path.dirname(path), f"{root}.enu")
    with open(out_path, "w") as fo:
        if has_unc:
            fo.write("decyear\tEW_cm\tNS_cm\tUD_cm\tSigE_cm\tSigN_cm\tSigU_cm\n")
            for i in range(len(t)):
                fo.write(f"{t[i]:.8f}\t{E_cm[i]:.3f}\t{N_cm[i]:.3f}\t{U_cm[i]:.3f}\t{SigE_cm[i]:.3f}\t{SigN_cm[i]:.3f}\t{SigU_cm[i]:.3f}\n")
        else:
            fo.write("decyear\tEW_cm\tNS_cm\tUD_cm\n")
            for i in range(len(t)):
                fo.write(f"{t[i]:.8f}\t{E_cm[i]:.3f}\t{N_cm[i]:.3f}\t{U_cm[i]:.3f}\n")

    return out_path


def main():
    ap = argparse.ArgumentParser(description="Convert ECEF XYZ time series to ENU, with robust header/STOP handling.")
    ap.add_argument("files", nargs="+", help="Input *.XYZ files (whitespace-delimited).")
    ap.add_argument("--nref", type=int, default=30, help="Number of initial samples to define reference & standardization (median). Default=30.")
    args = ap.parse_args()

    for path in args.files:
        try:
            outp = process_file(path, args.nref)
            if outp:
                print(f"[OK] Wrote {outp}")
        except Exception as e:
            print(f"[ERROR] {os.path.basename(path)}: {e}")


if __name__ == "__main__":
    main()
