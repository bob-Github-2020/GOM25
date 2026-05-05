#!/usr/bin/env python3

## 10-13-2025, Final GOM25, 150 RF
#./do_cal_7P_Helmert.py
# copy this to the processing_xyz.f
# dtttt=2025.0-2020.0
#        dtx= 1.6609172494671445E-03
#        dty= -1.9733409497285680E-04
#        dtz= 1.0001046009052589E-03
#        drx= -5.7773420253637979E-11
#        dry= -3.0509821952902144E-09
#        drz= -2.0985853928509080E-10
#        drs= 0.0000000000000000


"""
euler_pole_from_helmert.py

Compute the Euler pole (longitude, latitude) and rotation rate from 3 rotation-rate
components (rx, ry, rz) given in an Earth-Centered, Earth-Fixed (ECEF) frame.

- Inputs are rotation *rates* about the x, y, z axes.
- Default units are radians per year (rad/yr).
- Output includes longitude (deg), latitude (deg), and rotation-rate magnitude in
  rad/yr, deg/yr, deg/Myr, and mas/yr.

This script ships with GOM25's IGS20→GOM25 rotation-rate defaults embedded.
You can override them via CLI flags.

Usage examples
--------------
# 1) Use the built-in GOM25 numbers and write a text file
python3 Calculate_EulerPole_GOM25.py -o GOM25_EulerPole.txt

# 2) Provide your own rates in radians/year
python3 Calculate_EulerPole_GOM25.py --rx 3.8e-11 --ry -3.0e-09 --rz -1.22e-10

# 3) Provide your rates in milliarcseconds/year
python3 Calculate_EulerPole_GOM25.py --unit masyr --rx 0.0079 --ry -0.624 --rz -0.025

Notes on convention
-------------------
- The rotation vector R' = (rx, ry, rz) points toward the Euler pole.
- Positive rotation rate corresponds to counterclockwise rotation about R' (right-hand rule).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Tuple

# -----------------------------
# Unit helpers
# -----------------------------

ARCSEC_PER_RAD = 206264.80624709636
MAS_PER_RAD = ARCSEC_PER_RAD * 1000.0  # milliarcseconds per radian
DEG_PER_RAD = 180.0 / math.pi

def to_rad_per_year(value: float, unit: str) -> float:
    """
    Convert a rotation rate to radians/year.
    unit ∈ {"radyr", "degyr", "degMyr", "masyr"}
    """
    unit = unit.lower()
    if unit == "radyr":
        return value
    if unit == "degyr":
        return value / DEG_PER_RAD
    if unit == "degmyr":
        return (value / 1.0e6) / DEG_PER_RAD
    if unit == "masyr":
        return value / MAS_PER_RAD
    raise ValueError(f"Unsupported unit: {unit}")

@dataclass
class EulerPole:
    lon_deg: float   # longitude (deg, east positive, range [-180,180))
    lat_deg: float   # latitude (deg, north positive)
    omega_radyr: float  # rotation-rate magnitude (rad/yr)

    @property
    def omega_degyr(self) -> float:
        return self.omega_radyr * DEG_PER_RAD

    @property
    def omega_degMyr(self) -> float:
        return self.omega_degyr * 1.0e6

    @property
    def omega_masyr(self) -> float:
        return self.omega_radyr * MAS_PER_RAD

def compute_euler_pole(rx: float, ry: float, rz: float) -> EulerPole:
    """
    Given rotation-rate components (rx, ry, rz) in rad/yr,
    return the Euler pole and rotation-rate magnitude.
    """
    # Longitude: atan2(ry, rx) ∈ (-180, 180] degrees
    lon_deg = math.degrees(math.atan2(ry, rx))

    # Latitude: atan2(rz, sqrt(rx^2 + ry^2))
    lat_deg = math.degrees(math.atan2(rz, math.hypot(rx, ry)))

    # Rotation-rate magnitude
    omega_radyr = math.sqrt(rx*rx + ry*ry + rz*rz)

    return EulerPole(lon_deg=lon_deg, lat_deg=lat_deg, omega_radyr=omega_radyr)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Euler pole and rotation rate from rotation-rate components."
    )
    parser.add_argument("--rx", type=float, default=None, help="Rotation rate about X-axis")
    parser.add_argument("--ry", type=float, default=None, help="Rotation rate about Y-axis")
    parser.add_argument("--rz", type=float, default=None, help="Rotation rate about Z-axis")
    parser.add_argument(
        "--unit",
        type=str,
        default="radyr",
        choices=["radyr", "degyr", "degMyr", "masyr"],
        help="Units for rx, ry, rz (default: radyr)",
    )
    parser.add_argument(
        "-o", "--out", type=str, default="GOM25_EulerPole.txt",
        help="Output text file (default: GOM25_EulerPole.txt)"
    )
    parser.add_argument(
        "-n", "--name", type=str, default="IGS20_to_GOM25",
        help="Label/name written to the output file header (default: IGS20_to_GOM25)"
    )
    args = parser.parse_args()

    # -----------------------------
    # Defaults: IGS20 → GOM25 rotation-rate parameters (rad/yr)
    # -----------------------------
    gom25_defaults = dict(
        rx=-5.7773420253637979E-11,
        ry=-3.0509821952902144E-09,
        rz=-2.0985853928509080E-10,
    )

    # If user provided any of rx/ry/rz, require all three. Otherwise, use defaults.
    if args.rx is None and args.ry is None and args.rz is None:
        rx_radyr = gom25_defaults["rx"]
        ry_radyr = gom25_defaults["ry"]
        rz_radyr = gom25_defaults["rz"]
        src_name = f"{args.name} (defaults)"
    else:
        if None in (args.rx, args.ry, args.rz):
            raise SystemExit("Error: please provide all of --rx, --ry, and --rz, or none (to use defaults).")
        rx_radyr = to_rad_per_year(args.rx, args.unit)
        ry_radyr = to_rad_per_year(args.ry, args.unit)
        rz_radyr = to_rad_per_year(args.rz, args.unit)
        src_name = f"{args.name} (user-supplied)"

    pole = compute_euler_pole(rx_radyr, ry_radyr, rz_radyr)

    # -----------------------------
    # Print a concise report
    # -----------------------------
    print(f"\nEuler pole for {src_name}")
    print("-" * 54)
    print(f"Input rotation rates (rad/yr):")
    print(f"  rx = {rx_radyr:.6e}")
    print(f"  ry = {ry_radyr:.6e}")
    print(f"  rz = {rz_radyr:.6e}")
    print("\nEuler pole (deg) and rotation magnitude:")
    ew = "E" if pole.lon_deg >= 0 else "W"
    ns = "N" if pole.lat_deg >= 0 else "S"
    print(f"  Lon = {abs(pole.lon_deg):.3f}° {ew}")
    print(f"  Lat = {abs(pole.lat_deg):.3f}° {ns}")
    print(f"  |ω| = {pole.omega_radyr:.6e} rad/yr  "
          f"= {pole.omega_degyr:.6e} deg/yr  "
          f"= {pole.omega_degMyr:.6f} deg/Myr  "
          f"= {pole.omega_masyr:.3f} mas/yr")

    # -----------------------------
    # Write a simple text file
    # -----------------------------
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Euler pole report: {src_name}\n")
        f.write("# Right-hand rule: positive ω is counterclockwise about the pole.\n")
        f.write("# Units: lon/lat in degrees, rates as noted.\n\n")
        f.write("Input rotation rates (rad/yr):\n")
        f.write(f"rx = {rx_radyr:.16e}\n")
        f.write(f"ry = {ry_radyr:.16e}\n")
        f.write(f"rz = {rz_radyr:.16e}\n\n")
        f.write("Euler pole:\n")
        f.write(f"Longitude_deg = {pole.lon_deg:.9f}\n")
        f.write(f"Latitude_deg  = {pole.lat_deg:.9f}\n")
        f.write("\nRotation magnitude:\n")
        f.write(f"omega_rad_per_year  = {pole.omega_radyr:.16e}\n")
        f.write(f"omega_deg_per_year  = {pole.omega_degyr:.16e}\n")
        f.write(f"omega_deg_per_Myr   = {pole.omega_degMyr:.9f}\n")
        f.write(f"omega_mas_per_year  = {pole.omega_masyr:.9f}\n")

if __name__ == "__main__":
    main()
