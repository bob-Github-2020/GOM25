#!/usr/bin/python3
## 2-4-2025
## This program works to convert GOM20 ENU velocities to IGS14 velocities
## I tested with TXLI, it works fine.
   ## Reverse the transformation: v_igs14 = v_local - T + ω×X - SX
   # v_ecef_igs14 = v_ecef_local - trans_vec + cross_rot - scale_contrib
 

import math
import numpy as np

def transform_enu_velocity_local_to_IGS14(
    vE_local, vN_local, vU_local, 
    lat_deg, lon_deg, h_m,
    Tx_rate, Ty_rate, Tz_rate,  # m/yr, published from IGS to local
    Rx_rate, Ry_rate, Rz_rate,  # rad/yr, published from IGS to local
    scale_rate=0.0              # 1/yr
):
    """
    Converts ENU velocities from a local frame to IGS14 ENU velocities.
    
    INPUTS:
      vE_local, vN_local, vU_local : Velocity in local ENU (mm/yr)
      lat_deg, lon_deg, h_m        : Station coordinates (deg, deg, meters)
      Tx_rate, Ty_rate, Tz_rate    : Translation rates (m/yr)
      Rx_rate, Ry_rate, Rz_rate    : Rotation rates (rad/yr)
      scale_rate                   : Scale rate (1/yr)
      
    RETURNS:
      (vE_igs14, vN_igs14, vU_igs14) in mm/yr
    """
    # Convert coordinates to radians
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    
    # WGS84 ellipsoid constants
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2*f - f**2
    N = a / math.sqrt(1 - e2 * (math.sin(lat_rad))**2)
    
    # Compute ECEF coordinates (meters)
    X0 = (N + h_m) * math.cos(lat_rad) * math.cos(lon_rad)
    Y0 = (N + h_m) * math.cos(lat_rad) * math.sin(lon_rad)
    Z0 = (N*(1 - e2) + h_m) * math.sin(lat_rad)
    
    # ENU to ECEF rotation matrix
    R_enu = np.array([
        [-math.sin(lon_rad), -math.sin(lat_rad)*math.cos(lon_rad), math.cos(lat_rad)*math.cos(lon_rad)],
        [math.cos(lon_rad), -math.sin(lat_rad)*math.sin(lon_rad), math.cos(lat_rad)*math.sin(lon_rad)],
        [0, math.cos(lat_rad), math.sin(lat_rad)]
    ])
    
    # Convert local ENU velocity to ECEF (m/yr)
    v_enu_local_m = np.array([vE_local, vN_local, vU_local]) * 1e-3
    v_ecef_local = R_enu.dot(v_enu_local_m)
    
    # Compute inverse Helmert transformation
    trans_vec = np.array([Tx_rate, Ty_rate, Tz_rate])
    rot_vec = np.array([Rx_rate, Ry_rate, Rz_rate])
    cross_rot = np.cross(rot_vec, [X0, Y0, Z0])
    scale_contrib = scale_rate * np.array([X0, Y0, Z0])
    
    # Reverse the transformation: v_igs14 = v_local - T + ω×X - SX
    v_ecef_igs14 = v_ecef_local - trans_vec + cross_rot - scale_contrib
    
    # Convert back to ENU (mm/yr)
    v_enu_igs14_m = R_enu.T.dot(v_ecef_igs14)
    return (v_enu_igs14_m[0]*1e3, v_enu_igs14_m[1]*1e3, v_enu_igs14_m[2]*1e3)

# Example usage (reverse transformation)
if __name__ == "__main__":
    # Station TXLI in Houston20 (should have near-zero velocity)
    lat = 30.056   # deg
    lon = -94.771  # deg
    h = 11.0       # meters
    
    # Velocity in Houston20 (near-zero for stable station)
    vE_loc, vN_loc, vU_loc = -0.2444, -0.0609, 0.7708  # mm/yr
    
    # Houston20 parameters (same as previous)
    Tx=   7.1281610281319764E-004
    Ty=   5.6136740734561222E-004
    Tz=   2.9287337527455419E-003
    Rx=  -4.0941604346875451E-010
    Ry=  -3.1975595938303966E-009
    Rz=  -2.3610546814299809E-010
    
    # Transform back to IGS14
    vE_igs14, vN_igs14, vU_igs14 = transform_enu_velocity_local_to_IGS14(
        vE_loc, vN_loc, vU_loc, lat, lon, h, Tx, Ty, Tz, Rx, Ry, Rz
    )
    
    print(f"Houston20 Velocity  : {vE_loc:.3f}, {vN_loc:.3f}, {vU_loc:.3f} mm/yr")
    print(f"IGS14 Velocity (ENU): {vE_igs14:.3f}, {vN_igs14:.3f}, {vU_igs14:.3f} mm/yr")
