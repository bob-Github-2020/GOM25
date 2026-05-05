#!/usr/bin/python3
## 2-5-2025
## This program converts IGS14 ENU velocities to GOM20 ENU velocities
## I tested with TXLI, it works fine.
## Spent some time to figure out this---- helmert_shift = trans_vec - cross_rot + scale_contrib

import math
import numpy as np

def transform_enu_velocity_IGS14_to_local(
    vE_igs14, vN_igs14, vU_igs14, 
    lat_deg, lon_deg, h_m,
    Tx_rate, Ty_rate, Tz_rate,  # translation rates (m/yr)
    Rx_rate, Ry_rate, Rz_rate,  # rotation rates (rad/yr)
    scale_rate=0.0              # optional scale rate (1/yr), often very small
):
    """
    Transforms a station velocity from the IGS14 frame to a "local" frame
    via a Helmert-like velocity transformation in 3D (ECEF).
    
    INPUTS:
      vE_igs14, vN_igs14, vU_igs14 : Station velocity in ENU (mm/yr) wrt IGS14
      lat_deg, lon_deg, h_m       : Station geodetic coordinates (deg, deg, meters), h is the hight above the land surface
      Tx_rate, Ty_rate, Tz_rate   : Translation rates (m/yr)
      Rx_rate, Ry_rate, Rz_rate   : Rotation rates (rad/yr)
      scale_rate                  : Scale rate (1/yr); often near 0, optional
      
    RETURNS:
      (vE_local, vN_local, vU_local) in mm/yr wrt the local (new) frame.
      
    NOTES:
    - Sign conventions can differ between geodetic software packages.
      Adjust +/– if needed to match your official frames.
    - This snippet assumes the local frame is defined so that removing 
      (Tx_rate, Rx_rate, etc.) from IGS14 yields "Local". 
      Double-check your official transformation signs.
    """

    # ---------------------
    # 1) Convert station lat/lon to radians
    # ---------------------
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    
    # WGS84 ellipsoid constants
    a = 6378137.0                  # semi-major axis in meters
    f = 1.0 / 298.257223563
    e2 = 2*f - f**2
    
    # Radius of curvature in the prime vertical
    N = a / math.sqrt(1 - e2 * (math.sin(lat_rad))**2)
    
    # ---------------------
    # 2) Compute station position in ECEF (X0, Y0, Z0)
    #    (assuming height = h_m)
    # ---------------------
    X0 = (N + h_m) * math.cos(lat_rad) * math.cos(lon_rad)
    Y0 = (N + h_m) * math.cos(lat_rad) * math.sin(lon_rad)
    Z0 = (N*(1 - e2) + h_m) * math.sin(lat_rad)
    print(X0,Y0,Z0)
    # ---------------------
    # 3) Build rotation matrix R_enu to go from ENU -> ECEF
    #    This is the same logic you used in your time-series code.
    # ---------------------
    R_enu = np.array([
        [-math.sin(lon_rad), 
         -math.sin(lat_rad)*math.cos(lon_rad), 
          math.cos(lat_rad)*math.cos(lon_rad)],
        [ math.cos(lon_rad),
         -math.sin(lat_rad)*math.sin(lon_rad),
          math.cos(lat_rad)*math.sin(lon_rad)],
        [0,
          math.cos(lat_rad),
          math.sin(lat_rad)]
    ])
    
    # ---------------------
    # 4) Convert the ENU velocity (IGS14) [mm/yr] -> [m/yr], then to ECEF
    # ---------------------
    v_enu_igs14_m = np.array([vE_igs14, vN_igs14, vU_igs14]) * 1e-3  # mm/yr -> m/yr
    v_ecef_igs14 = R_enu.dot(v_enu_igs14_m)  # shape (3,)

    # ---------------------
    # 5) Apply the 7-parameter velocity transform in ECEF
    #    Typically: v_local = v_igs14 - [T_rate + rotation x position + scale * X0]
    #
    #    We'll assume the local frame is the "target" of the transform, so we 
    #    SUBTRACT the Helmert velocity from v_igs14. 
    #
    #    CAUTION: The exact sign depends on your definition of the parameters.
    # ---------------------
    
    # (A) translation rates = (Tx_rate, Ty_rate, Tz_rate)
    #     in m/yr
    trans_vec = np.array([Tx_rate, Ty_rate, Tz_rate])
    
    # (B) rotation rates = (Rx_rate, Ry_rate, Rz_rate)
    #     apply cross product: w x X0  (where w=rotation vector, X0=station ECEF)
    rot_vec = np.array([Rx_rate, Ry_rate, Rz_rate])
    pos_vec = np.array([X0, Y0, Z0])
    cross_rot = np.cross(rot_vec, pos_vec)  # m/yr
    
    # (C) scale rate = scale_rate * (X0, Y0, Z0)
    #     If scale_rate = 0, this part vanishes.
    scale_contrib = scale_rate * pos_vec
    
    # Combine them: helmert_shift = T_rate - cross_rot + scale_contrib
    helmert_shift = trans_vec - cross_rot + scale_contrib
    
    # v_ecef_local = v_ecef_igs14 + helmert_shift
    
    ## from global to local, use "+ helmert_shift"
    v_ecef_local = v_ecef_igs14 + helmert_shift

    # ---------------------
    # 6) Finally, convert v_ecef_local back to ENU in the local frame
    # ---------------------
    v_enu_local_m = R_enu.T.dot(v_ecef_local)
    # Convert to mm/yr
    vE_local = v_enu_local_m[0] * 1e3
    vN_local = v_enu_local_m[1] * 1e3
    vU_local = v_enu_local_m[2] * 1e3
    
    return (vE_local, vN_local, vU_local)


# -----------------------------------------------------------------------
# EXAMPLE USAGE
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # Suppose we have a station at lat=25.0°, lon=121.0°, height=20 m, TXLI
    lat_station = 30.056  # Latitude of the station in degrees
    lon_station = -94.771  # Longitude of the station in degrees
    height_station = 11.0  # meters above the ellipisolid
    
    # The station velocity in IGS14 (ENU), say 5 mm/yr east, -3 mm/yr north, 1 mm/yr up
    # TXLI
    vE_igs14 = -12.599
    vN_igs14 = -2.049
    vU_igs14 = -1.00
    
    # Helmert velocity parameters (example, IGS14 to GOM20)
    
    scale_rate = 0.0   # 1/yr (often negligible)
    
    Tx_rate = 1.4040400186633741E-002  # Rate of Tx in m/year
    Ty_rate = 9.6139040019832154E-004  # Rate of Ty in m/year
    Tz_rate = 7.2404861999567256E-003  # Rate of Tz in m/year
    Rx_rate = -9.8590126255842714E-010  # Rate of Rx in radians/year
    Ry_rate = -1.7311089298738490E-009  # Rate of Ry in radians/year
    Rz_rate = 1.3205310572876208E-009  # Rate of Rz in radians/year

## IGS14 to GOM20
         #dtx=   7.1281610281319764E-004
         #dty=   5.6136740734561222E-004
         #dtz=   2.9287337527455419E-003
         #drx=  -4.0941604346875451E-010
         #dry=  -3.1975595938303966E-009
         #drz=  -2.3610546814299809E-010
         #drs=   0.0000000000000000 
## IGS14 to Houston20         
         #dtx=   1.4040400186633741E-002
         #dty=   9.6139040019832154E-004
         #dtz=   7.2404861999567256E-003
         #drx=  -9.8590126255842714E-010
         #dry=  -1.7311089298738490E-009
         #drz=   1.3205310572876208E-009
         #drs=   0.0000000000000000     
    # Transform
    (vE_local, vN_local, vU_local) = transform_enu_velocity_IGS14_to_local(
        vE_igs14, vN_igs14, vU_igs14,
        lat_station, lon_station, height_station,
        Tx_rate, Ty_rate, Tz_rate,
        Rx_rate, Ry_rate, Rz_rate,
        scale_rate
    )
    
    print("Velocity in IGS14 (mm/yr): ", (vE_igs14, vN_igs14, vU_igs14))
    print("Velocity in Local Frame   : ", (vE_local, vN_local, vU_local))

