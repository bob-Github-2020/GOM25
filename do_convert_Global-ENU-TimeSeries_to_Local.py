#!/usr/bin/python3
## 11-27-2024. 
## This is designed to convert IGS14 ENU time series to Local ENU time series
## The input includes the (1) Long, Lat. of the station; (2) 7 parameters, (3) and the IGS14 ENU time series
## I did comparisons with the TXAG ENU time series from IGS14 to GOM20, it works perfectly.

import math
import numpy as np
import pandas as pd

# Function to convert ENU time series from IGS14 to GOM20 reference frame
def enu_igs14_to_gom20(input_file, output_file, lat, lon, t0, Tx_rate, Ty_rate, Tz_rate, Rx_rate, Ry_rate, Rz_rate):
    # Read input file into a pandas DataFrame
    df = pd.read_csv(input_file, delim_whitespace=True, header=0, names=["Decimal_Year", "NS", "EW", "UD", "Sigma_NS", "Sigma_EW", "Sigma_UD"])

    # Ensure the DataFrame is sorted by Decimal_Year
    df = df.sort_values(by="Decimal_Year").reset_index(drop=True)

    # Convert latitude and longitude to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563
    e2 = 2 * f - f ** 2

    # Calculate the radius of curvature in the prime vertical
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)

    # Calculate the reference point ECEF coordinates (assuming ellipsoidal height h = 0)
    X0 = (N) * math.cos(lat_rad) * math.cos(lon_rad)
    Y0 = (N) * math.cos(lat_rad) * math.sin(lon_rad)
    Z0 = (N * (1 - e2)) * math.sin(lat_rad)

    # Corrected rotation matrix from ENU to ECEF
    R_enu = np.array([
        [-math.sin(lon_rad), -math.sin(lat_rad) * math.cos(lon_rad), math.cos(lat_rad) * math.cos(lon_rad)],
        [math.cos(lon_rad), -math.sin(lat_rad) * math.sin(lon_rad), math.cos(lat_rad) * math.sin(lon_rad)],
        [0, math.cos(lat_rad), math.sin(lat_rad)]
    ])

    # Prepare lists to hold transformed ENU coordinates
    gom20_ns = []
    gom20_ew = []
    gom20_ud = []

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Extract time and ENU coordinates in IGS14
        t = row["Decimal_Year"]
        E = row["EW"] / 100.0  # Convert from cm to m
        N = row["NS"] / 100.0  # Convert from cm to m
        U = row["UD"] / 100.0  # Convert from cm to m
        
        # Convert ENU to ECEF displacement in IGS14
        enu_vector = np.array([E, N, U])
        ecef_displacement_igs14 = R_enu.dot(enu_vector)

        # Get the full ECEF coordinates by adding the reference point
        X_igs14 = X0 + ecef_displacement_igs14[0]
        Y_igs14 = Y0 + ecef_displacement_igs14[1]
        Z_igs14 = Z0 + ecef_displacement_igs14[2]

        # Apply the rate-based transformation to GOM20 (translation + rotation)
        delta_t = t - t0

        # Translation component
        dX = Tx_rate * delta_t
        dY = Ty_rate * delta_t
        dZ = Tz_rate * delta_t

        # Rotation component
        dX_rot = Rz_rate * delta_t * Y_igs14 - Ry_rate * delta_t * Z_igs14
        dY_rot = -Rz_rate * delta_t * X_igs14 + Rx_rate * delta_t * Z_igs14
        dZ_rot = Ry_rate * delta_t * X_igs14 - Rx_rate * delta_t * Y_igs14

        # Total change in ECEF coordinates
        X_gom20 = X_igs14 + dX + dX_rot
        Y_gom20 = Y_igs14 + dY + dY_rot
        Z_gom20 = Z_igs14 + dZ + dZ_rot

        # Convert transformed ECEF displacement back to ENU (relative to original point)
        ecef_displacement_gom20 = np.array([X_gom20 - X0, Y_gom20 - Y0, Z_gom20 - Z0])
        enu_displacement_gom20 = R_enu.T.dot(ecef_displacement_gom20)

        # Append the transformed ENU coordinates to the respective lists (convert back to cm)
        gom20_ew.append(enu_displacement_gom20[0] * 100.0)
        gom20_ns.append(enu_displacement_gom20[1] * 100.0)
        gom20_ud.append(enu_displacement_gom20[2] * 100.0)

    # Add the transformed ENU coordinates to the DataFrame
    df["NS_GOM20"] = gom20_ns
    df["EW_GOM20"] = gom20_ew
    df["UD_GOM20"] = gom20_ud

    # Standardize the transformed time series by removing the mean of the first 30 days
    first_30_days = df[df['Decimal_Year'] <= df['Decimal_Year'][0] + (30 / 365.25)]
    ns_mean = first_30_days['NS_GOM20'].mean()
    ew_mean = first_30_days['EW_GOM20'].mean()
    ud_mean = first_30_days['UD_GOM20'].mean()

    df['NS_GOM20'] = df['NS_GOM20'] - ns_mean
    df['EW_GOM20'] = df['EW_GOM20'] - ew_mean
    df['UD_GOM20'] = df['UD_GOM20'] - ud_mean

    # Write the output DataFrame to a new file with the same format as the input
    df = df[["Decimal_Year", "NS_GOM20", "EW_GOM20", "UD_GOM20", "Sigma_NS", "Sigma_EW", "Sigma_UD"]]
    df.columns = ["Decimal_Year", "NS", "EW", "UD", "Sigma_NS", "Sigma_EW", "Sigma_UD"]
    df.to_csv(output_file, index=False, sep=' ', float_format='%.4f')
    print(f"Transformed ENU time series saved to '{output_file}'")


# Example usage
input_file = "TXAG_IGS14_neu_cm.col"
output_file = "TTTT_GOM20_neu_cm.col"
lat = 29.164128  # Latitude of the station in degrees
lon = -95.41906  # Longitude of the station in degrees

## Seven parameters for IGS14 to GOM20
t0 = 2015.0  # Reference epoch
Tx_rate = 7.1281610E-004  # Rate of Tx in m/year
Ty_rate = 5.6136741E-004  # Rate of Ty in m/year
Tz_rate = 2.9287337E-003  # Rate of Tz in m/year
Rx_rate = -4.0941604E-010  # Rate of Rx in radians/year
Ry_rate = -3.1975595E-009  # Rate of Ry in radians/year
Rz_rate = -2.3610546E-010  # Rate of Rz in radians/year

enu_igs14_to_gom20(input_file, output_file, lat, lon, t0, Tx_rate, Ty_rate, Tz_rate, Rx_rate, Ry_rate, Rz_rate)

