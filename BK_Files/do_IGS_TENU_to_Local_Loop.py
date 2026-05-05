#!/usr/bin/python3
## 8-24-2025
## Transform ENU time series (TENU) from the IGS references (global) to local reference frame

## The input includes the (1) Long, Lat. of the station; (2) 7 parameters, (3) and the IGS20 ENU time series
## I did comparisons with the TXAG ENU time series from IGS14 to GOM20, it works perfectly.

import math
import numpy as np
import pandas as pd
import os

# Reference frames and their parameters
# 8-25-2025, 150 RF, updated, tried weight (1,1.5,2) in calculating Helmert Parameters
## Used wights: 1,1,2
#reference_frames = {
#    "GOM25": {
#        "t0": 2025.0,
#        "dtx": 2.8228642614369945E-03,
#        "dty": -8.9268242896202682E-04,
#        "dtz": -4.4254849042304399E-04,
#        "drx": 1.8843412843535334E-10,
#        "dry": -2.9693313938881471E-09,
#        "drz": -4.3671695803867793E-11,
#        "drs": 0.0000000000000000
#    },
#}

## 8-26-2025, Used weights: 1, 1.5, 2
#reference_frames = {
#    "GOM25": {
#        "t0": 2025.0,
#        "dtx": 2.8237060358249030E-03,
#        "dty": -8.8558519064015659E-04,
#        "dtz": -4.3142038726734662E-04,
#        "drx": 1.8640289448161500E-10,
#        "dry": -2.9613418703847352E-09,
#        "drz": -4.8425660045019844E-11,
#        "drs": 0.0000000000000000
#    },
#}

## 8-27-2025, Used weights: 1, 1, 1 in generating 7P, for 140 RF
reference_frames = {
    "GOM25": {
        "t0": 2025.0,
        "dtx": 1.6145982168775275E-03,
        "dty": -1.7402971842949246E-04,
        "dtz": 1.0730283771551150E-03,
        "drx": -7.1744611910711865E-11,
        "dry": -3.0585299865660676E-09,
        "drz": -2.1516868048702702E-10,
        "drs": 0.0000000000000000
    },
}
# Function to load station coordinates from llh file and convert longitude
def load_station_coords(llh_file, station):
    with open(llh_file, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == station:
                lat = float(parts[1])
                lon = float(parts[2])
                # Convert longitude from 0 to -360 to -180 to 180 range
                if lon < -180:
                    lon += 360
                return lat, lon
    raise ValueError(f"Station {station} not found in {llh_file}")

# Function to convert ENU time series from IGS to local reference frame
def enu_IGS_to_Local(input_file, output_file, lat, lon, t0, dtx, dty, dtz, drx, dry, drz, drs):
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
    Local_ns = []
    Local_ew = []
    Local_ud = []

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Extract time and ENU coordinates in IGS14
        t = row["Decimal_Year"]
        E = row["EW"] / 100.0  # Convert from cm to m
        N = row["NS"] / 100.0  # Convert from cm to m
        U = row["UD"] / 100.0  # Convert from cm to m
        
        # Convert ENU to ECEF displacement in IGS14
        enu_vector = np.array([E, N, U])
        ecef_displacement_IGS = R_enu.dot(enu_vector)

        # Get the full ECEF coordinates by adding the reference point
        X = X0 + ecef_displacement_IGS[0]
        Y = Y0 + ecef_displacement_IGS[1]
        Z = Z0 + ecef_displacement_IGS[2]

        # Apply the rate-based transformation to GOM20 (translation + rotation)
        delta_t = t - t0

        # Translation component
        dX = dtx * delta_t
        dY = dty * delta_t
        dZ = dtz * delta_t

        # Rotation component
        dX_rot = drz * delta_t * Y - dry * delta_t * Z
        dY_rot = -drz * delta_t * X + drx * delta_t * Z
        dZ_rot = dry * delta_t * X - drx * delta_t * Y

        # Scale component (drs is typically 0, so no effect here)
        # dX_scale = drs * X * delta_t  # Not implemented as drs=0
        # dY_scale = drs * Y * delta_t
        # dZ_scale = drs * Z * delta_t

        # Total change in ECEF coordinates
        X_Local = X + dX + dX_rot
        Y_Local = Y + dY + dY_rot
        Z_Local = Z + dZ + dZ_rot

        # Convert transformed ECEF displacement back to ENU (relative to original point)
        ecef_displacement_Local = np.array([X_Local - X0, Y_Local - Y0, Z_Local - Z0])
        enu_displacement_Local = R_enu.T.dot(ecef_displacement_Local)

        # Append the transformed ENU coordinates to the respective lists (convert back to cm)
        Local_ew.append(enu_displacement_Local[0] * 100.0)
        Local_ns.append(enu_displacement_Local[1] * 100.0)
        Local_ud.append(enu_displacement_Local[2] * 100.0)

    # Add the transformed ENU coordinates to the DataFrame
    df["NS_Local"] = Local_ns
    df["EW_Local"] = Local_ew
    df["UD_Local"] = Local_ud

    # Standardize the transformed time series by removing the mean of the first 30 days
    first_30_days = df[df['Decimal_Year'] <= df['Decimal_Year'][0] + (30 / 365.25)]
    ns_mean = first_30_days['NS_Local'].mean()
    ew_mean = first_30_days['EW_Local'].mean()
    ud_mean = first_30_days['UD_Local'].mean()

    df['NS_Local'] = df['NS_Local'] - ns_mean
    df['EW_Local'] = df['EW_Local'] - ew_mean
    df['UD_Local'] = df['UD_Local'] - ud_mean

    # Write the output DataFrame to a new file with the same format as the input
    df = df[["Decimal_Year", "NS_Local", "EW_Local", "UD_Local", "Sigma_NS", "Sigma_EW", "Sigma_UD"]]
    df.columns = ["Decimal_Year", "NS", "EW", "UD", "Sigma_NS", "Sigma_EW", "Sigma_UD"]
    df.to_csv(output_file, index=False, sep=' ', float_format='%.4f')
    print(f"Transformed ENU time series saved to '{output_file}'")

# Main function to process multiple files
def main(llh_file="UNV_all_GPS_02052025.llh", local_frame="GOM25"):
    # Get all .col files in the current directory
    input_files = [f for f in os.listdir('.') if f.endswith('_cm.col')]
    
    # Get transformation parameters for the selected local frame
    params = reference_frames.get(local_frame)
    if not params:
        raise ValueError(f"Unknown local reference frame: {local_frame}")

    for input_file in input_files:
        # Extract station name from filename (assuming format like "OKCB_IGS14_neu_cm.col")
        station = input_file.split('_')[0]
        
        # Load station coordinates
        try:
            lat, lon = load_station_coords(llh_file, station)
        except ValueError as e:
            print(e)
            continue

        # Set input and output filenames, replace "IGS14" with local_frame
        output_file = input_file.replace("IGS20", local_frame)

        # Transform the time series
        enu_IGS_to_Local(input_file, output_file, lat, lon, **params)

if __name__ == "__main__":
    main()
