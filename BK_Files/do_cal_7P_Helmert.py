#!/usr/bin/python3
## 7-31-2025
## Tested with 199 TW stations
## The Python results are almost identical to the Fortran results now.

import numpy as np

MAX_SITES = 2048
T1 = 2020.0
T2 = 2025.0

def read_arp_file(filename):
    names = []
    coords = []
    velocities = []
    epochs = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 8:
                    names.append(parts[0])
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    velocities.append([float(parts[4]), float(parts[5]), float(parts[6])])
                    epochs.append(float(parts[7]))
    return (np.array(names), np.array(coords), np.array(velocities), np.array(epochs))

def adjust_to_epoch(coords, velocities, epochs, target_epoch):
    adjusted_coords = coords.copy()
    for i in range(len(coords)):
        dt = target_epoch - epochs[i]
        adjusted_coords[i] += velocities[i] * dt
    return adjusted_coords

## Ellipsoidal Rotation Matrix, WGS84   
def rotation_matrix_to_neu(xyz):
    x, y, z = xyz
    # WGS84 parameters
    a = 6378137.0  # semi-major axis (m)
    f = 1 / 298.257223563  # flattening
    b = a * (1 - f)  # semi-minor axis
    e2 = 1 - (b / a) ** 2  # eccentricity squared
    
    p = np.sqrt(x**2 + y**2)
    theta = np.arctan2(z * a, p * b)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    
    lat = np.arctan2(z + e2 * b * sin_theta**3, p - e2 * a * cos_theta**3)
    lon = np.arctan2(y, x)
    
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    
    rot_matrix = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    return rot_matrix
    
# uses a spherical approximation, not used in the program  
def simple_rotation_matrix_to_neu(xyz):
    x, y, z = xyz
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    rot_matrix = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
    ])
    return rot_matrix

def compute_helmert_parameters(sys1_coords, sys2_coords, num_sites):
    num_parn = 6
    norm_eq = np.zeros((num_parn, num_parn))
    bvec = np.zeros(num_parn)
    
    for i in range(num_sites):
        dx = sys2_coords[i] - sys1_coords[i]
        rot_matrix = rotation_matrix_to_neu(sys1_coords[i])
        dn = rot_matrix @ dx
        
        xyz_part = np.zeros((3, num_parn))
        x, y, z = sys1_coords[i]
        
        # Corrected partial derivatives for rotations
        # Ry (rotation about Y-axis)
        xyz_part[0, 0] = -z  # X component
        xyz_part[2, 0] = x    # Z component
        
        # Rx (rotation about X-axis)
        xyz_part[1, 1] = z    # Y component
        xyz_part[2, 1] = -y   # Z component
        
        # Rz (rotation about Z-axis)
        xyz_part[0, 2] = y    # X component
        xyz_part[1, 2] = -x   # Y component
        
        # Translation parameters
        xyz_part[0, 3] = 1.0  # Tx
        xyz_part[1, 4] = 1.0  # Ty
        xyz_part[2, 5] = 1.0  # Tz
        
        neu_part = rot_matrix @ xyz_part
     ## those wights are for ENU, for GOM25RF, RMS ENU velocities in IGS20:  12.941991     2.823748     0.788152  
     ## 8-25-2025, try weights 1,1.5,2, the GOM25RF Vel statistics show no considerable with weight:1,1,2
     ## 8-27-2025, try weights: 1,1,1
     ## Conclusion: There no conisderable differences on the final GOM25 velocities, individual and statics
     ## So, it is better just use weight1-1-1 for the dense GOM25 RF. 
     ## Need further tests for othr regional reference frames.
      
        weights = np.array([1.0, 1.0, 1.0])
        
        for j in range(3):
            if weights[j] == 0:
                continue
            bvec += neu_part[j] * dn[j] * weights[j]
            norm_eq += np.outer(neu_part[j] * weights[j], neu_part[j])
    
    trans_parm = np.linalg.solve(norm_eq, bvec)
    return trans_parm

def main():
    arp_file = 'igs20.arp'
    names, coords, velocities, epochs = read_arp_file(arp_file)
    num_sites = len(names)
    print(f"There are {num_sites} sites in sys file {arp_file}")
    
    ## sys1 is the IGS20, sys2 is the local reference frame????
    
    sys1_coords_orig = coords.copy()
    sys1_velocities = velocities.copy()
    sys2_coords_t1 = adjust_to_epoch(coords, velocities, epochs, T1)
    
    avx, avy, avz = np.mean(velocities, axis=0)
    print(f"avx,avy,avz: {avx:.16E} {avy:.16E} {avz:.16E}")
    
    ## The mean-remvoved velicities were not used indeed.
    # removing the average velocity for each station, to remove the overal translational movement
    # original sys1_velocities have been replaced by mean-removed
    sys2_velocities = velocities - [avx, avy, avz]
    
    # the coordinates at T2 for sys1 
    sys1_coords_t2 = adjust_to_epoch(sys1_coords_orig, sys1_velocities, epochs, T2)
    #sys1_coords_t2 = adjust_to_epoch(sys1_coords_orig, sys1_velocities, epochs, T2)
    
    # the coordinates at T2 for sys2 
    #sys2_coords_t2 = adjust_to_epoch(sys2_coords_t1, sys2_velocities, np.full(num_sites, T1), T2)
    sys2_coords_t2 = sys2_coords_t1
    
    # so, the two systems aligned at T2, which is the t0 for the 7parameters
    
    np.savetxt('XYZ_2020.txt', sys2_coords_t1, fmt='%.5f')
    np.savetxt('XYZ_2025.txt', sys1_coords_t2, fmt='%.5f')
    
    trans_parm = compute_helmert_parameters(sys1_coords_t2, sys2_coords_t2, num_sites)
    dt = T2 - T1
    rates = trans_parm / dt
    
    print("\nHelmert Parameters (Ry, Rx, Rz, Tx, Ty, Tz):")
    print(f"Ry, Rx, Rz (rad): {trans_parm[0:3]}")
    print(f"Tx, Ty, Tz (m): {trans_parm[3:6]}")
    print("\nRates (per year):")
    print(f"dRy, dRx, dRz (rad/yr): {rates[0:3]}")
    print(f"dTx, dTy, dTz (m/yr): {rates[3:6]}")
    
    print("\nFormatted for processing_xyz.f:")
    print("!! copy this to the processing_xyz.f")
    print(f"!! dtttt={T2}-{T1}")
    print(f"        dtx= {rates[3]:.16E}")
    print(f"        dty= {rates[4]:.16E}")
    print(f"        dtz= {rates[5]:.16E}")
    print(f"        drx= {rates[1]:.16E}")
    print(f"        dry= {rates[0]:.16E}")
    print(f"        drz= {rates[2]:.16E}")
    print(f"        drs= 0.0000000000000000")

if __name__ == "__main__":
    main()
