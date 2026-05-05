#!/usr/bin/python3
## 7-31-2025. Process XYZ file for calculating vel, GOM25 paper
## 3-5-2025. abrupt-step and slow-step detection, add loop for find best wight for the cubic fit
## 3-3-2025. Ferine figures for AI paper Fig2
## 2-18-2025. Integrated CNN training model to the program.
## 2-14-2025. for GOM25. Output parameters for AI training
## plot ENU time series and now use a trained CNN model to decide small-step detection parameters.
## 2-2-2025. Use rpt.KernelCPD. It is much faster.
## 11-24-2024. Updated thoroughly for XYZ dataset processing.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import os
import glob
import ruptures as rpt
import joblib  # for loading XGBoost, if needed
import tensorflow as tf
import gc
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image  # Added for Image.open()
from matplotlib import font_manager

# Rebuild font cache to ensure DejaVu Sans is recognized
#font_manager._rebuild()

# Set global font and PDF options for all plots, for PDF-JGR publication
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['pdf.fonttype'] = 42

# ============================================================================
# 1. Data Loading Function (for XYZ data in meters)
# ============================================================================
def load_data_xyz(file_path):
    # Define column names based on XYZ format
    columns = ["Decimal_Year", "X_m", "Y_m", "Z_m", "Sigma_X_m", "Sigma_Y_m", "Sigma_Z_m"]
    # Read the data, assuming remaining columns exist but we only need first 4
    data = pd.read_csv(file_path, delim_whitespace=True, names=columns + ['extra1', 'extra2', 'extra3', 'extra4'], 
                      comment='#')
    # Select only the columns we need
    data = data[["Decimal_Year", "X_m", "Y_m", "Z_m"]]
    
    # Convert to numeric, handling any potential errors
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
    # Convert meters to centimeters before processing
    data['N_cm'] = data['X_m'] * 100  # X -> N
    data['E_cm'] = data['Y_m'] * 100  # Y -> E
    data['U_cm'] = data['Z_m'] * 100  # Z -> U
    
    # Rename columns to NEU for consistency with the rest of the program
    data = data.rename(columns={"N_cm": "N", "E_cm": "E", "U_cm": "U"})
    
    # Store original data
    original_data = data.copy()
    
    # Calculate medians of Original data
    median_N = original_data["N"].median()
    median_E = original_data["E"].median()
    median_U = original_data["U"].median()
    
    print("Median_N, E, U (cm):", median_N, median_E, median_U)  # Corrected units to cm
    
    # Remove medians from each time series
    data["N"] = data["N"] - median_N
    data["E"] = data["E"] - median_E
    data["U"] = data["U"] - median_U
    
    # Remove outliers, detrend from each component
    data = remove_outliers_sigma_detrend(data, "N")
    data = remove_outliers_sigma_detrend(data, "E")
    data = remove_outliers_sigma_detrend(data, "U")
    
    # Reset index after cleaning
    data = data.reset_index(drop=True)
    
    return original_data, data, median_N, median_E, median_U
# ============================================================================
# 2. Outlier Removal Function (Using Detrending) - Modified for meters
# ============================================================================
def remove_outliers_sigma_detrend(data, column, n_sigma=2.0, window=300):
    """
    Remove outliers from a time series column by first applying an absolute threshold,
    then using iterative detrending and sigma-based filtering.
    
    Parameters:
    - data: DataFrame with 'Decimal_Year' and the target column
    - column: Column name to clean (e.g., 'X', 'Y', 'Z')
    - n_sigma: Number of standard deviations for outlier threshold (default 2.5)
    - window: Rolling window size for local stats (default 350)
    
    Returns:
    - DataFrame with outliers set to NaN
    """
    max_iterations = 10  # 6
    # 5m absolute threshold for XYZ coordinates in meters
    abs_threshold = 500.0
    
    data_cleaned = data.copy()
    t = data_cleaned["Decimal_Year"].values
    y = data_cleaned[column].values
    
    # Step 1: Apply absolute threshold first to remove extreme outliers
    abs_outlier_mask = (y < -abs_threshold) | (y > abs_threshold)
    data_cleaned.loc[abs_outlier_mask, column] = np.nan
    y = data_cleaned[column].values  # Update y after absolute thresholding
    
    # Step 2: Iterative detrending and sigma-based outlier removal
    for iteration in range(max_iterations):
        valid = ~np.isnan(y)
        if valid.sum() < 2:  # Stop if too few points remain
            break
        
        # Linear detrending
        slope, intercept, _, _, _ = linregress(t[valid], y[valid])
        trend = slope * t + intercept
        detrended = y - trend
        detrended_series = pd.Series(detrended, index=data_cleaned.index)
        
        # Rolling mean and std with specified window
        mean = detrended_series.rolling(window=window, center=True, min_periods=1).mean()
        std = detrended_series.rolling(window=window, center=True, min_periods=1).std()
        
        # Sigma-based outlier bounds
        lower_bound = mean - n_sigma * std
        upper_bound = mean + n_sigma * std
        
        # Identify outliers based on sigma threshold
        sigma_outlier_mask = (detrended_series < lower_bound) | (detrended_series > upper_bound)
        
        # Remove outliers
        data_cleaned.loc[sigma_outlier_mask, column] = np.nan
        y = data_cleaned[column].values  # Update y for next iteration
    
    return data_cleaned


# ============================================================================
# 3. Step Detection (Sliding Window)
# ============================================================================

def detect_change_points_sliding_window(
    data, 
    window_size=30, 
    N_threshold=0.4, 
    E_threshold=0.4, 
    U_threshold=1.0, 
    min_distance=200  # min_distance is in days
):
    """
    Detect change points using a sliding window approach with different thresholds for N, E, and U.
    Before detection, the function removes the overall trend (via linear regression) and centers the data.
    
    Then:
      Step 2: Candidate CPs are detected using a sliding window.
      Step 3: These CPs are filtered using differences in Decimal_Year (converted to years).
      
    min_distance is interpreted in days.
    """
    thresholds = {"N": N_threshold, "E": E_threshold, "U": U_threshold}
    change_points = {}
    
    data_detrended = data.copy()

    # --- Step 1: Remove overall trend & mean for each direction.
    for direction in ["N", "E", "U"]:
        x = data["Decimal_Year"]
        y = data[direction]
        valid_mask = y.notna()
        if valid_mask.sum() < 2:
            continue
        slope, intercept, _, _, _ = linregress(x[valid_mask], y[valid_mask])
        detrended = y - (slope * x + intercept)
        detrended -= detrended.mean()
        data_detrended[direction] = detrended

    # --- Step 2: Perform sliding-window detection on detrended data.
    candidate_cp = {}  # Candidate CP indices for each direction.
    for direction in ["N", "E", "U"]:
        threshold = thresholds[direction]
        valid_mask = data_detrended[direction].notna()
        valid_indices = data_detrended.loc[valid_mask].index
        ts = data_detrended.loc[valid_mask, direction].values
        n = len(ts)
        if n < 2 * window_size:
            candidate_cp[direction] = [valid_indices[0], valid_indices[-1]] if len(valid_indices) > 1 else []
            continue

        raw_cps_local = [0]
        i = window_size
        while i < n - window_size:
            window_before = ts[i - window_size : i]
            window_after  = ts[i : i + window_size]
            mean_before = window_before.mean()
            mean_after = window_after.mean()
            if abs(mean_after - mean_before) > threshold:
                raw_cps_local.append(i + window_size)
                i += window_size  # Jump ahead.
            else:
                i += 1
        raw_cps_local.append(n - 1)
        # Convert local indices to global indices.
        cp_global = [valid_indices[loc_idx] for loc_idx in raw_cps_local if loc_idx < n]
        candidate_cp[direction] = cp_global


    # --- Step 3: Filter out change points that are too close together (using Decimal_Year differences).
    filtered_cp = {}
    min_gap_years = min_distance / 365.25  # convert days to years
    for direction in ["N", "E", "U"]:
        cp_list = candidate_cp.get(direction, [])
        if not cp_list:
            filtered_cp[direction] = []
            continue
        cp_years = [data.loc[idx, "Decimal_Year"] for idx in cp_list]
        filtered = [cp_list[0]]
        last_year = data.loc[cp_list[0], "Decimal_Year"]
        for idx in cp_list[1:]:
            year_val = data.loc[idx, "Decimal_Year"]
            if (year_val - last_year) >= min_gap_years:
                filtered.append(idx)
                last_year = year_val
        filtered_cp[direction] = filtered
    
    change_points = filtered_cp
    return change_points
#=============================================================================    

def linear_func(x, a, b):
    """Linear model: y = a*x + b."""
    return a * x + b

def quadratic_func(x, a, b, c):
    """Quadratic model: y = a*x^2 + b*x + c."""
    return a * x**2 + b * x + c

def exp_func(t, u0, v, delta, tau):
    """
    Exponential post-seismic model:
      f(t) = u0 + v*t + delta * exp(-t/tau)
    """
    return u0 + v * t + delta * np.exp(-t / tau)
    

def cubic_func(x, a, b, c, d):
    """Cubic model: f(x) = a*x^3 + b*x^2 + c*x + d."""
    return a * x**3 + b * x**2 + c * x + d
import numpy as np
from scipy.optimize import curve_fit

def cubic_func(x, a, b, c, d):
    """Cubic model: f(x) = a*x^3 + b*x^2 + c*x + d."""
    return a * x**3 + b * x**2 + c * x + d


def optimal_cubic_fit(x_seg, y_seg, magnitude=0.2, p0=None):
    """
    Perform a standard cubic fit (without weights) over the segment, then calculate the MSE in the
    first and last one-third of the segment. If the error in the first portion is larger, select a 
    positive weight coefficient; if the error in the last portion is larger, select a negative weight.
    Re-fit the cubic model with the chosen weight and return the best weight, the best-fit parameters,
    and the corresponding overall MSE.
    
    Parameters:
      x_seg (array-like): Independent variable data for the segment (e.g., Decimal_Year).
      y_seg (array-like): Dependent variable data for the segment.
      magnitude (float): Absolute value of the weight coefficient to test (default=0.2).
      p0 (list/array, optional): Initial guess for cubic parameters [a, b, c, d].
                                  If None, defaults to [0, 0, 0, np.mean(y_seg)].
    
    Returns:
      best_weight (float): The chosen weight coefficient (positive or negative).
      best_cubic_params (ndarray): The cubic model parameters [a, b, c, d] from the weighted fit.
      best_mse (float): The mean squared error over the entire segment for the weighted fit.
    """
    # Set default initial guess if none provided.
    if p0 is None:
        #p0 = [0, 0, 0, np.mean(y_seg)]
        p0 = [0, 0, 0, np.mean(y_seg)] if not np.isnan(np.mean(y_seg)) else [0, 0, 0, 0]
    # Ensure inputs are numpy arrays.
    x_seg = np.asarray(x_seg)
    y_seg = np.asarray(y_seg)
    if len(x_seg) < 4 or np.any(np.isnan(x_seg)) or np.any(np.isnan(y_seg)):
        print(f"Skipping fit: too few points ({len(x_seg)}) or NaN values")
        return 0, p0, np.inf
    
    # --- Step 1: Standard cubic fit (no weighting).
    try:
        cubic_params_std, _ = curve_fit(cubic_func, x_seg, y_seg, p0=p0, maxfev=20000)
        cubic_pred_std = cubic_func(x_seg, *cubic_params_std)
        mse_std = np.mean((y_seg - cubic_pred_std)**2)
    #except Exception as e:
    except RuntimeError:
        print("curve_fit failed to converge")
        mse_std = np.inf
        cubic_params_std = p0

    # --- Step 2: Divide the segment into first and last thirds.
    x_min, x_max = x_seg.min(), x_seg.max()
    time_span = x_max - x_min
    t_first_cut = x_min + time_span / 3.0
    t_last_cut  = x_min + 2.0 * time_span / 3.0
    first_mask = (x_seg <= t_first_cut)
    last_mask  = (x_seg >= t_last_cut)
    
    mse_first = np.mean((y_seg[first_mask] - cubic_pred_std[first_mask])**2) if np.any(first_mask) else mse_std
    mse_last  = np.mean((y_seg[last_mask]  - cubic_pred_std[last_mask])**2) if np.any(last_mask) else mse_std
    
    # --- Step 3: Decide which region is worse.
    # If the first third has a higher error, choose a positive weight to emphasize the early portion.
    # Otherwise, choose a negative weight to emphasize the later portion.
    chosen_sign = 1 if mse_first > mse_last else -1
    #print ("Comparing mse_fit:", mse_first, mse_last, chosen_sign)
    # --- Step 4: Re-fit with the chosen weight.
    x_min_val = x_min  # For weight computation
    coeff = chosen_sign * magnitude
    weights = np.exp(-coeff * (x_seg - x_min_val))
    
    try:
        best_cubic_params, _ = curve_fit(
            cubic_func, 
            x_seg, 
            y_seg, 
            p0=p0, 
            sigma=1/weights, 
            absolute_sigma=True, 
            maxfev=20000
        )
        cubic_pred_weighted = cubic_func(x_seg, *best_cubic_params)
        best_mse = np.mean((y_seg - cubic_pred_weighted)**2)
    except Exception as e:
        best_mse = np.inf
        best_cubic_params = cubic_params_std  # fallback

    best_weight = chosen_sign * magnitude
    #print(f"Chosen weight: {best_weight} with mse: {best_mse:.6f} (mse_first: {mse_first:.6f}, mse_last: {mse_last:.6f})")
    return best_weight, best_cubic_params, best_mse

  
#=============================================================================
# Using a cubic fit for the longest segment
#=============================================================================
##     curve_threshold=0.5, and  improvement_ratio=0.3  are set in this function, not anyway else

def detect_abrupt_slow_steps(
    data, 
    window_size=30, 
    N_threshold=0.4, 
    E_threshold=0.4, 
    U_threshold=1.0, 
    min_distance=100,  # in days (assume daily sampling, so also in points)
    N_curve_threshold=0.3,  # minimum amplitude (in data units) to accept a slow step
    E_curve_threshold=0.3,
    U_curve_threshold=0.5,
    improvement_ratio=0.2  # fractional improvement required (cubic vs. linear)
):
    """
    Hybrid method to detect both abrupt and slow (curve) steps in a GNSS time series using a cubic fit.
    """
    # Convert min_distance (in days) to years:
    min_gap_years = min_distance / 365.25

    directions = ["N", "E", "U"]
    thresholds = {"N": N_threshold, "E": E_threshold, "U": U_threshold}
    curve_thresholds = {"N": N_curve_threshold, "E": E_curve_threshold, "U": U_curve_threshold}  # Direction-specific curve thresholds
    final_cp = {}  # final change points per direction

    for direction in directions:
        # --- Step 1: Detrend and center the data.
        valid_mask = data[direction].notna()
        if valid_mask.sum() < 2:
            final_cp[direction] = []
            continue

        x_valid = data.loc[valid_mask, "Decimal_Year"].values
        y_valid = data.loc[valid_mask, direction].values
        slope, intercept, _, _, _ = linregress(x_valid, y_valid)
        detrended = y_valid - (slope * x_valid + intercept)
        detrended -= detrended.mean()

        # --- Step 2: Detect abrupt steps using a sliding-window on detrended data.
        valid_indices = data.loc[valid_mask].index
        n = len(detrended)
        if n < 2 * window_size:
            candidate = [valid_indices[0], valid_indices[-1]] if len(valid_indices) > 1 else []
        else:
            raw_candidates = [0]
            i = window_size
            while i < n - window_size:
                window_before = detrended[i - window_size:i]
                window_after = detrended[i:i+window_size]
                if abs(window_after.mean() - window_before.mean()) > thresholds[direction]:
                    raw_candidates.append(i + window_size)
                    i += window_size
                else:
                    i += 1
            raw_candidates.append(n - 1)
            candidate = [valid_indices[idx] for idx in raw_candidates if idx < n]

        # --- Step 3: Filter candidate steps by time difference.
        filtered = []
        if candidate:
            filtered.append(candidate[0])
            last_year = data.loc[candidate[0], "Decimal_Year"]
            for idx in candidate[1:]:
                current_year = data.loc[idx, "Decimal_Year"]
                if (current_year - last_year) >= min_gap_years:
                    filtered.append(idx)
                    last_year = current_year

        # --- Step 4: Add additional change points for large gaps in the original series.
        valid_years = data.loc[valid_indices, "Decimal_Year"].values
        additional = []
        for j in range(len(valid_years) - 1):
            if valid_years[j+1] - valid_years[j] >= 1.5:           ## minimum gap 1.5 years
                additional.append(valid_indices[j])
                additional.append(valid_indices[j+1])
        candidate_cp = sorted(set(filtered + additional))

        # --- Step 5: Identify the longest step-free segment.
        boundaries = sorted(set([valid_indices[0]] + candidate_cp + [valid_indices[-1]]))
        longest_duration = 0
        seg_start, seg_end = boundaries[0], boundaries[-1]
        for i in range(len(boundaries) - 1):
            t1 = data.loc[boundaries[i], "Decimal_Year"]
            t2 = data.loc[boundaries[i+1], "Decimal_Year"]
            duration = t2 - t1
            if duration > longest_duration:
                longest_duration = duration
                seg_start, seg_end = boundaries[i], boundaries[i+1]

        # --- Step 6: Slow-step detection in the longest segment using a cubic fit.
        candidate_updated = candidate_cp.copy()
        if seg_end - seg_start >= 1100:         # only try qubic fit if it > 4 years
            # Use detrended data for the longest segment
            seg_mask = (data.index >= seg_start) & (data.index <= seg_end) & valid_mask
            seg_data = data.loc[seg_mask]
            x_seg = data.loc[seg_mask, "Decimal_Year"].values
            y_seg = detrended[seg_mask[valid_mask].values]  # Use detrended data

            if len(x_seg) >= 5:
                try:
                    lin_params, _ = curve_fit(linear_func, x_seg, y_seg, p0=[0, np.mean(y_seg)])
                    lin_pred = linear_func(x_seg, *lin_params)
                    lin_mse = np.mean((y_seg - lin_pred)**2)
                except:
                    lin_mse = np.inf
                
                best_weight_coef, cubic_params, cubic_mse = optimal_cubic_fit(x_seg, y_seg, magnitude=0.15, p0=None)
                
                cubic_pred = cubic_func(x_seg, *cubic_params)  
                #print ("best cubic fit weight:", best_weight)
                
                              
                if not np.isfinite(cubic_mse) or lin_mse < 1e-8:
                    mse_ratio = 0
                else:
                    mse_ratio = abs((lin_mse - cubic_mse) / lin_mse)
                    #print("mse_ratio:", mse_ratio, direction)
                
                if mse_ratio >= improvement_ratio:
                    a, b, c, d = cubic_params
                    # Calculate turning points by solving the derivative: 3ax^2 + 2bx + c = 0
                    discriminant = (2 * b)**2 - 4 * 3 * a * c
                    if discriminant >= 0:  # Real turning points exist
                        turning_x1 = (-2 * b + np.sqrt(discriminant)) / (2 * 3 * a)
                        turning_x2 = (-2 * b - np.sqrt(discriminant)) / (2 * 3 * a)
                        turning_points = [turning_x1, turning_x2]
                    else:
                        turning_points = []  # No real turning points

                    # Filter turning points to ensure they lie within the data range
                    valid_turning_points = [x for x in turning_points if x_seg.min() <= x <= x_seg.max()]

                    baseline = np.mean(y_seg)
                    for turning_x in valid_turning_points:
                        turning_y = cubic_func(turning_x, *cubic_params)
                        amplitude = abs(turning_y - baseline)
                        #print("Curve Amp:", amplitude, direction)
                        if amplitude >= curve_thresholds[direction]:
                            # Convert turning_x to a global index: find the nearest Decimal_Year in seg_data.
                            seg_years = seg_data["Decimal_Year"].values
                            idx_nearest = np.argmin(np.abs(seg_years - turning_x))
                            slow_step_idx = seg_data.index[idx_nearest]
                            candidate_updated.append(slow_step_idx)
                            
                            ## Plotting the longest segment, cubic fit, and detected slow-step
                            #plt.figure(figsize=(10, 6))
                            #plt.plot(x_seg, y_seg, 'bo', label='Detrended Data')
                            #plt.plot(x_seg, cubic_pred, 'r-', label='Cubic Fit')
                            #plt.axvline(x=turning_x, color='g', linestyle='--', label='Detected Slow-Step')
                            #plt.axhline(y=baseline, color='k', linestyle='-', label='Baseline')  # Added baseline
                            #plt.xlabel('Decimal Year')
                            #plt.ylabel(f'Detrended {direction}')
                            #plt.title(f'Cubic Fit and Detected Slow-Step for {direction}')
                            #plt.legend()
                            #plt.savefig(f"Detecting_slow_step_{direction}.png")
                            #plt.show()

        # --- Step 7: Final filtering: ensure candidate steps are separated by at least min_gap_years.
        candidate_updated = sorted(set(candidate_updated))
        #final_filtered = []
        #f candidate_updated:
        #   final_filtered.append(candidate_updated[0])
        #   last_year = data.loc[candidate_updated[0], "Decimal_Year"]
        #   for idx in candidate_updated[1:]:
        #       current_year = data.loc[idx, "Decimal_Year"]
        #       if (current_year - last_year) >= min_gap_years:
        #           final_filtered.append(idx)
        #           last_year = current_year
        #final_cp[direction] = final_filtered
        ## no curve step
        #final_cp[direction]= candidate_cp 
        
        final_cp[direction] = candidate_updated
        
    return final_cp


#=============================================================================

# ============================================================================
# 4. Correct Steps
# ============================================================================
def correct_step(data, station_name, change_points, step_threshold=None):
    corrected = data.copy()
    original_data = data.copy()
    corrected_times = {"N": [], "E": [], "U": []}
    for direction in ["N", "E", "U"]:
        cplist = change_points.get(direction, [])
        if len(cplist) <= 2:
            continue
        valid_mask = corrected[direction].notna()
        valid_indices = corrected.loc[valid_mask].index
        ts = corrected.loc[valid_mask, direction].values
        for local_cp_global in cplist:
            if local_cp_global == valid_indices[0] or local_cp_global == valid_indices[-1]:
                continueN
            if step_threshold is not None:
                local_idx = np.where(valid_indices == local_cp_global)[0]
                if len(local_idx) == 0:
                    continue
                local_idx = local_idx[0]
                window_size = 20
                start_local = max(local_idx - window_size, 0)
                end_local   = min(local_idx + window_size, len(ts))
                if (local_idx - start_local) < 3 or (end_local - local_idx) < 3:
                    continue
                left_slice = ts[start_local : local_idx-1]
                right_slice = ts[local_idx+1 : end_local]    
                if len(left_slice) == 0 or len(right_slice) == 0:
                    continue                          
                left_mean  = np.mean(left_slice)
                right_mean = np.mean(right_slice)
                step_val = right_mean - left_mean
                if abs(step_val) < step_threshold:
                    continue
            else:
                local_idx = np.where(valid_indices == local_cp_global)[0]
                if len(local_idx) == 0:
                    continue
                local_idx = local_idx[0]
                if local_idx == 0 or local_idx >= len(ts):
                    continue
                step_val = ts[local_idx] - ts[local_idx - 1]
            mask_after = corrected.index >= local_cp_global
            corrected.loc[mask_after, direction] -= step_val
            corrected_times[direction].append(corrected.loc[local_cp_global, "Decimal_Year"])
            
    #fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    #for i, direction in enumerate(["N", "E", "U"]):
    #    ax = axes[i]
    #    ax.plot(original_data["Decimal_Year"], original_data[direction], 'bo', markersize=3, label="Original")
    #    ax.plot(corrected["Decimal_Year"], corrected[direction], 'r.', markersize=2, label="Corrected")
    #    for j, t_val in enumerate(corrected_times[direction]):
    #        ax.axvline(t_val, color='g', linestyle='--', linewidth=3, label="Corrected Step" if j == 0 else "")
    #    ax.set_ylabel(f"{direction} (cm)")
    #    ax.legend()
    #axes[-1].set_xlabel("Decimal Year")
    #plt.suptitle(f"Step Correction for {station_name}", y=1.05)
    #plt.savefig(f"{station_name}_step_corrected.png", bbox_inches='tight', pad_inches=0.1)
    #plt.tight_layout()
    #plt.close()
    return corrected

# ============================================================================
# 5. CPD using RBF (optional)
# ============================================================================
def detect_change_points(data, model="rbf", penalty=400, method="pelt", jump=5):
    change_points = {}
    for direction in ["N", "E", "U"]:
        valid_mask = data[direction].notna()
        valid_indices = data.loc[valid_mask].index
        ts = data.loc[valid_mask, direction].values.reshape(-1, 1)
        if method == "binseg":
            algo = rpt.Binseg(model=model).fit(ts)
        elif method == "window":
            algo = rpt.Window(model=model).fit(ts)
        else:
            algo = rpt.KernelCPD(kernel="rbf", min_size=200).fit(ts)
        cp_indices = algo.predict(pen=penalty)
        cp_indices = [0] + cp_indices + [len(ts) - 1]
        original_cps = [valid_indices[i] for i in cp_indices if i < len(valid_indices)]
        change_points[direction] = sorted(set(original_cps))
    return change_points

# ============================================================================
# 6. Calculate Velocities, Plot Time Series
# ============================================================================
def calculate_segment_velocity(data, change_points, min_years=1.0, min_samples=200):
    velocities = {}
    intercepts = {}
    longest_segments = {}
    for direction in ["N", "E", "U"]:
        segments = []
        indices = sorted(set([0] + change_points[direction] + [len(data)-1]))
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            if start == end:
                continue
            segment = data.iloc[start:end+1]
            seg_valid = segment.dropna(subset=["Decimal_Year", direction])
            if len(seg_valid) < 3:
                continue
            duration = seg_valid["Decimal_Year"].iloc[-1] - seg_valid["Decimal_Year"].iloc[0]
            if duration >= min_years and len(seg_valid) >= min_samples:
                segments.append((start, end, duration))
        if segments:
            longest_segment = max(segments, key=lambda x: x[2])
        else:
            longest_segment = (0, len(data)-1, data["Decimal_Year"].iloc[-1] - data["Decimal_Year"].iloc[0])
        segment_data = data.iloc[longest_segment[0]:longest_segment[1]+1].dropna(subset=["Decimal_Year", direction])
        if len(segment_data) < 30:
            slope, intercept = 0, 0
        else:
            slope, intercept, _, _, _ = linregress(segment_data["Decimal_Year"], segment_data[direction])
        velocities[direction] = slope
        intercepts[direction] = intercept
        longest_segments[direction] = longest_segment
    return velocities, intercepts, longest_segments
#---------------------------------------------------------------------------------    
###----plot time seris IGS with the red trend of the longest segment
#---------------------------------------------------------------------------------
def plot_time_series_with_longest_segment(data, velocities, intercepts, longest_segments, file_path):
        
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    plt.subplots_adjust(hspace=0.5, top=0.92)
    directions = ["N", "E", "U"]
    direction_labels = {"N": "NS", "E": "EW", "U": "UD"}
    y_ranges = {"N": (-2, 2), "E": (-2, 2), "U": (-3, 3)}
    mean_shifts = {direction: data[direction].mean() for direction in directions}
    data_mean_removed = data.copy()
    for direction in directions:
        data_mean_removed[direction] -= mean_shifts[direction]
    for i, direction in enumerate(directions):
        ax = axes[i]
        valid = data_mean_removed[direction].notna()
        ax.plot(data_mean_removed.loc[valid, "Decimal_Year"], data_mean_removed.loc[valid, direction], 'bo', markersize=2,
                label=f"{direction_labels[direction]} displacement")
        longest_segment = longest_segments.get(direction)
        if longest_segment is not None:
            seg_start, seg_end, _ = longest_segment
            segment_data = data.iloc[seg_start:seg_end+1].dropna(subset=["Decimal_Year", direction])
            if len(segment_data) > 1:
                slope = velocities.get(direction, 0)
                intercept = intercepts.get(direction, 0)
                trend_line = slope * segment_data["Decimal_Year"] + intercept - mean_shifts[direction]
                ax.plot(segment_data["Decimal_Year"], trend_line, "r--", linewidth=3,
                        label=f"Velocity: {slope*10:.1f} mm/yr")
        ax.set_ylabel(f"{direction_labels[direction]} (cm)")
        ymin, ymax = y_ranges.get(direction, (-3, 3))
        #ax.set_ylim(ymin, ymax)
        #ax.set_xlim(2000,2025)
        ax.legend()
    axes[-1].set_xlabel("Decimal Year")
    station_name = os.path.splitext(os.path.basename(file_path))[0]
    plt.suptitle(f"{station_name}", fontsize=12, y=0.935)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{station_name}.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    #plt.savefig(f"{station_name}.jpg", dpi=300, bbox_inches='tight', pad_inches=0.1)
    #plt.savefig(f"{station_name}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    #plt.savefig(f"Fig2a.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    del fig, axes
    gc.collect()
##-----------------------------------------------------------------------------------    
## plot the detrended time series, overlap with the longest-segment, and detected steps--vertical red line
#------------------------------------------------------------------------------------
def plot_time_series_with_velocity(data, change_points, velocities, intercepts, longest_segments, file_path):
        
    fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
    plt.subplots_adjust(hspace=0.5, top=0.92)
    directions = ["N", "E", "U"]
    direction_labels = {"N": "NS", "E": "EW", "U": "UD"}
    y_ranges = {"N": (-3, 2), "E": (-2.5, 2.5), "U": (-3, 3)}
    
    for i, direction in enumerate(directions):
        ax = axes[i]
        slope = velocities.get(direction, 0)
        intercept = intercepts.get(direction, 0)
        de_trended = data[direction] - (slope * data["Decimal_Year"] + intercept)
        valid = de_trended.notna()
        ax.plot(data.loc[valid, "Decimal_Year"], de_trended[valid], 'bo', markersize=2,
                label=f"Detrended {direction_labels[direction]} displacement")
                
        # Recalculate the longest step-free segment using change_points
        cp_indices = sorted(set([0] + change_points[direction] + [len(data)-1]))  
        segments = []      
        for j in range(len(cp_indices) - 1):
            start, end = cp_indices[j], cp_indices[j + 1]
            if start == end:
                continue
            segment_data = data.iloc[start:end+1].dropna(subset=["Decimal_Year", direction])
            if len(segment_data) < 30:  # Match min samples from calculate_segment_velocity
                continue
            duration = segment_data["Decimal_Year"].iloc[-1] - segment_data["Decimal_Year"].iloc[0]
            if duration >= 1.0:  # Match min_years from calculate_segment_velocity
                segments.append((start, end, duration))
        
        # Select the longest segment
        if segments:
            longest_segment = max(segments, key=lambda x: x[2])
            seg_start, seg_end = longest_segment[0], longest_segment[1]
        else:
            seg_start, seg_end = 0, len(data) - 1  # Fallback to full series
        
        # Plot the detrended longest segment
        segment_data = data.iloc[seg_start:seg_end+1]
        segment_de_trended = de_trended.iloc[seg_start:seg_end+1]
        valid_seg = segment_de_trended.notna()
        if valid_seg.sum() >= 2:
            ax.plot(segment_data.loc[valid_seg, "Decimal_Year"], segment_de_trended[valid_seg],
                    'r.', markersize=1, label=f"Segment for calculating velocity")
        # Plot all change points
        for j, cp_idx in enumerate(change_points.get(direction, [])):
            if cp_idx in data.index:
                ax.axvline(data.loc[cp_idx, "Decimal_Year"], color='r', linestyle='--',
                           label="Detected steps" if j == 0 else "")            
   
        #ax.set_ylabel(f"{direction} (cm)")
        ax.set_ylabel(f"{direction_labels[direction]} (cm)")
        #ymin, ymax = y_ranges.get(direction, ())
        #ax.set_ylim(ymin, ymax)
        
        # Determine y-axis limits.
        y_vals = de_trended[valid]
        ymin = np.min(y_vals)
        ymax = np.max(y_vals)                
        if direction in ["N", "E"]:
        # Use actual range if it exceeds (-2,2)
           if ymin < -2 or ymax > 2:
               ax.set_ylim(ymin, ymax)
           else:
               ax.set_ylim(-2, 2)
        elif direction == "U":
           if ymin < -3 or ymax > 3:
               ax.set_ylim(ymin, ymax)
           else:
               ax.set_ylim(-3, 3)
             
    axes[-1].set_xlabel("Decimal Year")
    ax.legend()
    ##station_name = os.path.splitext(os.path.basename(file_path))[0]
    station_name = os.path.basename(file_path).split('_')[0] 
    #axes[0].set_title(f"Detrended {station_name}-NEU: Step Detection")
    plt.suptitle(f"Detrended {station_name}-NEU: Step Dection", fontsize=12, y=0.935)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{station_name}_detrended.png", bbox_inches='tight', pad_inches=0.1)
    #plt.savefig(f"{station_name}_detrended.jpg", dpi=300, bbox_inches='tight', pad_inches=0.1)
    #plt.savefig(f"{station_name}_detrended.pdf",  dpi=300, bbox_inches='tight', pad_inches=0.1)
    #plt.savefig(f"Fig2b.pdf",  dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    del fig, axes  # Remove references
    gc.collect()
#==================================================================================================
def save_velocities(file_path, velocities, longest_segments, data):
    station_name = os.path.splitext(os.path.basename(file_path))[0]
    station = os.path.basename(file_path)[:4]
    output_file = f"{station_name}_Velocity.txt"
    
    # Get whole time series bounds
    valid_times = data["Decimal_Year"].dropna()
    whole_begin = valid_times.iloc[0] if not valid_times.empty else 'NA'
    whole_end = valid_times.iloc[-1] if not valid_times.empty else 'NA'
    whole_duration = (whole_end - whole_begin) if whole_begin != 'NA' and whole_end != 'NA' else 'NA'
    
    with open(output_file, 'w') as f:
        # Updated header with new fields
        f.write("Station Vel_N(mm/yr) Vel_E(mm/yr) Vel_U(mm/yr) Dur_N Dur_E Dur_U S_begin_N S_end_N S_begin_E S_end_E S_begin_U S_end_U  W_begin W_end W_Du\n")
        
        # Velocities and durations (as before)
        velN = velocities.get("N", 'NA')*10
        velE = velocities.get("E", 'NA')*10
        velU = velocities.get("U", 'NA')*10
        durN = (longest_segments["N"][2] if "N" in longest_segments else 'NA')
        durE = (longest_segments["E"][2] if "E" in longest_segments else 'NA')
        durU = (longest_segments["U"][2] if "U" in longest_segments else 'NA')
        
        # Segment begin and end times for each direction
        seg_begin_N = (data["Decimal_Year"].iloc[longest_segments["N"][0]] if "N" in longest_segments else 'NA')
        seg_end_N = (data["Decimal_Year"].iloc[longest_segments["N"][1]] if "N" in longest_segments else 'NA')
        seg_begin_E = (data["Decimal_Year"].iloc[longest_segments["E"][0]] if "E" in longest_segments else 'NA')
        seg_end_E = (data["Decimal_Year"].iloc[longest_segments["E"][1]] if "E" in longest_segments else 'NA')
        seg_begin_U = (data["Decimal_Year"].iloc[longest_segments["U"][0]] if "U" in longest_segments else 'NA')
        seg_end_U = (data["Decimal_Year"].iloc[longest_segments["U"][1]] if "U" in longest_segments else 'NA')
        
        # Write the formatted line with all fields
        f.write(f"{station} "
                f"{velN:.1f} {velE:.1f} {velU:.1f} "
                f"{durN:.2f} {durE:.2f} {durU:.2f} "
                f"{seg_begin_N:.4f} {seg_end_N:.3f} "
                f"{seg_begin_E:.4f} {seg_end_E:.3f} "
                f"{seg_begin_U:.4f} {seg_end_U:.3f} "
                f"{whole_begin:.4f} {whole_end:.3f} {whole_duration:.2f}\n")
                
                
def save_velocities_short(file_path, velocities, longest_segments):
    station_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"{station_name}_Velocity.txt"
    with open(output_file, 'w') as f:
        f.write("Station Vel_N(cm/yr) Vel_E(cm/yr) Vel_U(cm/yr) Duration_N(years) Duration_E(years) Duration_U(years)\n")
        velN = velocities.get("N", 'NA')
        velE = velocities.get("E", 'NA')
        velU = velocities.get("U", 'NA')
        durN = (longest_segments["N"][2] if "N" in longest_segments else 'NA')
        durE = (longest_segments["E"][2] if "E" in longest_segments else 'NA')
        durU = (longest_segments["U"][2] if "U" in longest_segments else 'NA')
        #f.write(f"{station_name} {velN} {velE} {velU} {durN} {durE} {durU}\n")
        f.write(f"{station_name} {velN:.2f} {velE:.2f} {velU:.2f} {durN:.2f} {durE:.2f} {durU:.2f}\n")
        
#===================================================================================================

def plot_original_vs_cleaned(original_data, cleaned_data, file_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    directions = ["N", "E", "U"]
    for i, direction in enumerate(directions):
        ax = axes[i]
        ax.plot(original_data["Decimal_Year"], original_data[direction], 'r.', alpha=0.5,
                label="Original Data (Outliers Included)")
        outlier_mask = original_data[direction].notna() & cleaned_data[direction].isna()
        ax.plot(original_data["Decimal_Year"][outlier_mask], original_data[direction][outlier_mask],
                'kx', markersize=6, label="Removed Outliers")
        ax.set_ylabel(f"{direction} (cm)")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Decimal Year")
    station_name = os.path.basename(file_path).split('_')[0]
    plt.suptitle(f"Original vs. Cleaned Time Series: {station_name}", y=0.96)
    plt.tight_layout()
    plt.savefig(f"{station_name}_NEU_Outliers.png", bbox_inches='tight', pad_inches=0.1)
    plt.close()

#=======================================================
# output change_points_year for AI_parameters training
#=======================================================
  
def get_change_points_year(data, change_points):
    """
    Convert the change point indices in the change_points dictionary
    to the corresponding decimal years from the data.
    """
    cp_years = {}
    for direction, cp_ids in change_points.items():
        # For each change point index, retrieve the corresponding decimal year.
        cp_years[direction] = []
        for cp_id in cp_ids:
            try:
                year = data.loc[cp_id, "Decimal_Year"]
                cp_years[direction].append(year)
            except KeyError:
                print(f"Warning: Index {cp_id} not found in data for direction {direction}.")
        # Optionally, sort the resulting years.
        cp_years[direction] = sorted(cp_years[direction])
    return cp_years

# ============================================================================
# 7. Write AI Parameters File
# ============================================================================
def write_ai_parameters(station_name, input_file,
                        big_window_size, big_N_threshold, big_E_threshold, big_U_threshold, big_min_distance,
                        step_threshold,
                        small_window_size, small_N_threshold, small_E_threshold, small_U_threshold, small_min_distance,
                        N_curve_threshold, E_curve_threshold, U_curve_threshold,improvement_ratio,
                        small_change_points_year, scores,
                        label="good"):
    cp_years_N = ",".join([str(val) for val in small_change_points_year.get("N", [])])
    cp_years_E = ",".join([str(val) for val in small_change_points_year.get("E", [])])
    cp_years_U = ",".join([str(val) for val in small_change_points_year.get("U", [])])
    
    params = {
       "input_file": input_file,
       #"big_window_size": big_window_size,
       #"big_N_threshold": big_N_threshold,
       #"big_E_threshold": big_E_threshold,
       #"big_U_threshold": big_U_threshold,
       #"big_min_distance": big_min_distance,
       #"step_threshold": step_threshold,
       "small_window_size": small_window_size,
       "small_N_threshold": small_N_threshold,
       "small_E_threshold": small_E_threshold,
       "small_U_threshold": small_U_threshold,
       "small_min_distance": small_min_distance,
       "N_curve_threshold": N_curve_threshold,
       "E_curve_threshold": E_curve_threshold,
       "U_curve_threshold": U_curve_threshold,
       "small_change_points_year_N": cp_years_N,
       "small_change_points_year_E": cp_years_E,
       "small_change_points_year_U": cp_years_U,
       "Scores": scores,
       "label": label
    }
    df_params = pd.DataFrame([params])
    output_filename = f"{station_name}_AI_parameters.csv"
    df_params.to_csv(output_filename, index=False, header=False)
    print(f"AI parameters saved to {output_filename}")

# ============================================================================
# 8. Helper: Use CNN to Predict Plot Quality
# ============================================================================
def cnn_predict_plot(cnn_model, plot_path, target_size=(224, 224)):
    """
    Predict the quality ("good" vs. "bad") of a step detection plot using a trained CNN.
    Returns the probability of being "good" (0 to 1).
    
    Args:
        cnn_model: Trained TensorFlow/Keras model
        plot_path: Path to the plot image (PNG)
        target_size: Tuple of (height, width) for resizing (default: 224, 224,or,512, 512)
    
    Returns:
        float: Probability of "good" classification (0 to 1), or None if error occurs
    """
    try:
        # Load and convert image to RGB, ensuring 3 channels
        img = Image.open(plot_path).convert('RGB')
        img = img.resize(target_size)
        # Convert to array and normalize
        img_array = img_to_array(img) / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        # Predict with suppressed output
        prediction = cnn_model.predict(img_array, verbose=0)
        #print(f"Prediction for {plot_path}: {prediction[0][0]}")  # Debug
        # Return probability of "good" (assuming sigmoid output)
        return prediction[0][0]
    except Exception as e:
        print(f"Error processing {plot_path}: {e}")
        return None
    finally:
        # Clean up (PIL objects don’t need explicit del in most cases, but kept for consistency)
        try:
            del img, img_array
        except NameError:
            pass
        gc.collect()

# ============================================================================
# 9. Helper: Select Optimal Small Parameters Using the CNN Model
# ============================================================================
## Helper: Generate Candidate Plot for CNN, exactly with the one used in CNN training

def plot_candidate_for_direction(data, candidate_change_points, temp_plot, file_path, direction):
    """
    Generate a candidate plot for a single direction (N, E, or U) for CNN evaluation.
    This function calculates the overall linear trend for the given direction, removes it,
    and plots the detrended time series with detected change points.
    """


    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a temporary DataFrame with the relevant columns, dropping any rows that have NaN.
    df_valid = pd.DataFrame({
        'x': data["Decimal_Year"],
        'y': data[direction]
    }).dropna()
    
    # If we don't have enough valid points, we cannot plot.
    if len(df_valid) < 2:
        print(f"[DEBUG] Not enough valid data for direction={direction}. Aborting plot.")
        plt.close()
        return
        
    x_valid = df_valid['x'].astype(float)
    y_valid = df_valid['y'].astype(float)
           
    
    # Compute overall linear trend using the cleaned data.
    slope, intercept, _, _, _ = linregress(x_valid, y_valid)
    
    # Compute the detrended data for the entire dataset, not just df_valid,
    # but keep in mind some might be NaN.
    x_full = data["Decimal_Year"]
    y_full = data[direction]
    
    # We can linearly detrend the entire array, but it might have NaNs.
    de_trended = y_full - (slope * x_full + intercept)
    
    # Optionally, subtract overall mean if desired:
    # de_trended = de_trended - de_trended.mean()
    
    valid = de_trended.notna()
    if valid.sum() == 0:
        print(f"No valid detrended data for direction {direction}.")
        plt.close()
        return
    
    ## Print out the min/max of the detrended data to see if it falls within any fixed limits
    # print(f"[DEBUG] Detrended Y range: {de_trended[valid].min()} to {de_trended[valid].max()}")
    
    # Plot the detrended time series.
    ax.plot(x_full[valid], de_trended[valid], 'bo', markersize=2,
            label=f"{direction} Detrended")
    #print("CPs before plot:", candidate_change_points)
    
    # Plot detected change points as vertical red dashed lines.
    for j, cp_idx in enumerate(candidate_change_points.get(direction, [])):
        if cp_idx in data.index:
            cp_year = data.loc[cp_idx, "Decimal_Year"]
            ax.axvline(cp_year, color='r', linestyle='--', linewidth=4,
                       label="Detected Change" if j == 0 else "")
    
    ax.set_ylabel(f"{direction} (cm)")
    
    # Determine y-axis limits.
    y_vals = de_trended[valid]
    y_min = np.min(y_vals)
    y_max = np.max(y_vals)
    if direction in ["N", "E"]:
        # Use actual range if it exceeds (-2,2)
        if y_min < -2 or y_max > 2:
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(-2, 2)
    elif direction == "U":
        if y_min < -3 or y_max > 3:
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(-3, 3)
    
    
    FDir = {"N": "NS", "E": "EW", "U": "UD"}.get(direction, direction)
       
    # Auto-scale the x-axis to the actual data range.
    ax.set_xlim(x_valid.min(), x_valid.max())
    ax.set_xlabel("Decimal Year", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    station_name = os.path.splitext(os.path.basename(file_path))[0]
    ax.set_title(f"Step Detection: {FDir} Direction", fontsize=14)
    
       
    plt.tight_layout()
    plt.savefig(temp_plot, bbox_inches='tight', pad_inches=0.1)
    #plt.savefig(f"Fig2b.pdf",  dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.clf()  # Clear current figure
    plt.close('all')
    gc.collect()
##=========================================================================================================
def select_optimal_small_params_sequential(step_corrected_data, cnn_model, station_name, threshold_good=0.9):
    """
    Sequentially select optimal small-step detection parameters for each direction using the CNN.
    
    For each direction (NS, EW, UD), loop over candidate values for curve amplitude (C) and thresholds (N, E, U),
    exiting the inner loop early if a candidate's CNN probability meets or exceeds threshold_good.
    Regardless of early exits, generate final plots for all directions (N, E, U) before returning.
    
    Returns a dictionary of parameters, scores, and final change points.
    """
    
    # Define candidate lists for each parameter
    candidate_C_N = [0.4]  # Curve amplitude candidates
    candidate_C_E = [0.4] 
    candidate_C_U = [0.5] 
    candidate_N = [0.3, 0.4, 0.25, 0.2, 0.15, 0.5, 0.7, 1.2, 1.6, 2.5, 5, 8, 10, 15, 20, 50 ]
    candidate_E = [0.3, 0.4, 0.25, 0.2, 0.15, 0.5, 0.7, 1.2, 1.6, 2.5, 5, 8, 10, 15, 20, 50, 200]

    candidate_U = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.5, 5, 10, 20, 50]
    #candidate_U = [0.5]
    
    # Fixed values
    small_window_size = 30
    small_min_distance = 20
    improvement_ratio = 0.2
    
    best_params = {}
    scores = {}

    ### NS Direction: Sequential search for C and N thresholds
    best_score_N = -1
    best_N = None
    best_C_N = None
    for c_val in candidate_C_N:
        for n_val in candidate_N:
            candidate_change_points = detect_abrupt_slow_steps(
                step_corrected_data,
                window_size=small_window_size,
                N_threshold=n_val,
                E_threshold=candidate_E[0],
                U_threshold=candidate_U[0],
                min_distance=small_min_distance,
                N_curve_threshold=c_val,
                E_curve_threshold=candidate_C_E[0],
                U_curve_threshold=candidate_C_U[0],
                improvement_ratio=improvement_ratio
            )
            temp_plot = f"temp_{station_name}_candidate_N.png"
            plot_candidate_for_direction(step_corrected_data, candidate_change_points, temp_plot, station_name, "N")
            prob_good = cnn_predict_plot(cnn_model, temp_plot)
            # os.remove(temp_plot)  # Uncomment if you want to delete temp files
            #print("N: c_val, n_val, score:", c_val, n_val, prob_good)
            tf.keras.backend.clear_session()  # Reset after N predictions
            
            if prob_good > best_score_N:
                best_score_N = prob_good
                best_N = n_val
                best_C_N = c_val
            
            if prob_good >= threshold_good:
                best_params["small_N_threshold"] = best_N
                best_params["small_C_N_threshold"] = best_C_N
                scores["small_N_threshold"] = best_score_N
                break  # Exit inner loop only
        else:
            continue
        break  # Exit outer loop if threshold met
    
    # Set NS parameters if no early exit
    if "small_N_threshold" not in best_params:
        best_params["small_N_threshold"] = best_N
        best_params["small_C_N_threshold"] = best_C_N
        scores["small_N_threshold"] = best_score_N

    ### EW Direction: Sequential search for C and E thresholds
    best_score_E = -1
    best_E = None
    best_C_E = None
    for c_val in candidate_C_E:
        for e_val in candidate_E:
            candidate_change_points = detect_abrupt_slow_steps(
                step_corrected_data,
                window_size=small_window_size,
                N_threshold=best_N,
                E_threshold=e_val,
                U_threshold=candidate_U[0],
                min_distance=small_min_distance,
                N_curve_threshold=best_C_N,
                E_curve_threshold=c_val,
                U_curve_threshold=candidate_C_U[0],
                improvement_ratio=improvement_ratio
            )
            temp_plot = f"temp_{station_name}_candidate_E.png"
            plot_candidate_for_direction(step_corrected_data, candidate_change_points, temp_plot, station_name, "E")
            prob_good = cnn_predict_plot(cnn_model, temp_plot)
            # os.remove(temp_plot)
            #print("E: c_val, e_val, score:", c_val, e_val, prob_good)
            tf.keras.backend.clear_session()  # Reset after N predictions
            
            if prob_good > best_score_E:
                best_score_E = prob_good
                best_E = e_val
                best_C_E = c_val
            
            if prob_good >= threshold_good:
                best_params["small_E_threshold"] = best_E
                best_params["small_C_E_threshold"] = best_C_E
                scores["small_E_threshold"] = best_score_E
                break
        else:
            continue
        break
    
    # Set EW parameters if no early exit
    if "small_E_threshold" not in best_params:
        best_params["small_E_threshold"] = best_E
        best_params["small_C_E_threshold"] = best_C_E
        scores["small_E_threshold"] = best_score_E
        
    ### UD Direction: Sequential search for C and U thresholds
    best_score_U = -1
    best_U = None
    best_C_U = None
    for c_val in candidate_C_U:
        for u_val in candidate_U:
            candidate_change_points = detect_abrupt_slow_steps(
                step_corrected_data,
                window_size=small_window_size,
                N_threshold=best_N,
                E_threshold=best_E,
                U_threshold=u_val,
                min_distance=small_min_distance,
                N_curve_threshold=best_C_N,
                E_curve_threshold=best_C_E,
                U_curve_threshold=c_val,
                improvement_ratio=improvement_ratio
            )
            temp_plot = f"temp_{station_name}_candidate_U.png"
            plot_candidate_for_direction(step_corrected_data, candidate_change_points, temp_plot, station_name, "U")
            prob_good = cnn_predict_plot(cnn_model, temp_plot)
            # os.remove(temp_plot)
            #print("U: c_val, u_val, score:", c_val, u_val, prob_good)
            tf.keras.backend.clear_session()  # Reset after N predictions
            
            if prob_good > best_score_U:
                best_score_U = prob_good
                best_U = u_val
                best_C_U = c_val
            
            if prob_good >= threshold_good:
                best_params["small_U_threshold"] = best_U
                best_params["small_C_U_threshold"] = best_C_U
                scores["small_U_threshold"] = best_score_U
                break
        else:
            continue
        break
    
    # Set UD parameters and generate final change points
    best_params["small_U_threshold"] = best_U 
    best_params["small_C_U_threshold"] = best_C_U
    scores["small_U_threshold"] = best_score_U
    
    best_params["small_window_size"] = small_window_size
    best_params["small_min_distance"] = small_min_distance
    
    # Generate final change points with best parameters for all directions
    final_change_points = detect_abrupt_slow_steps(
        step_corrected_data,
        window_size=small_window_size,
        N_threshold=best_N,
        E_threshold=best_E,
        U_threshold=best_U,
        min_distance=small_min_distance,
        N_curve_threshold=best_C_N,
        E_curve_threshold=best_C_E,
        U_curve_threshold=best_C_U,
        improvement_ratio=improvement_ratio
    )
    
    # Generate final plots for all directions
    final_plot_n = f"final1_{station_name}_candidate_N.png"
    final_plot_e = f"final1_{station_name}_candidate_E.png"
    final_plot_u = f"final1_{station_name}_candidate_U.png"
    plot_candidate_for_direction(step_corrected_data, final_change_points, final_plot_n, station_name, "N")
    plot_candidate_for_direction(step_corrected_data, final_change_points, final_plot_e, station_name, "E")
    plot_candidate_for_direction(step_corrected_data, final_change_points, final_plot_u, station_name, "U")
    
    # Calculate average score and finalize
    best_params["score"] = np.mean([best_score_N, best_score_E, best_score_U])
    #print("Optimal small parameters selected (sequential):", best_params)
    #print("Individual scores:", scores)
    #print("Final change points:", final_change_points)
    gc.collect()
    return best_params, scores, final_change_points
    
##=========================================================================================================
def save_apr_file(file_path, original_data, velocities, longest_segments):
    """
    Save station data in APR format with t0 as the median of times corresponding to median positions.
    
    Parameters:
    - file_path: Path to the input file (used to extract station name)
    - original_data: DataFrame with original XYZ data in meters (X_m, Y_m, Z_m) and Decimal_Year
    - velocities: Dictionary with velocities for N, E, U in cm/year
    - longest_segments: Dictionary with longest segment info for each direction
    
    Output:
    - Writes a file named <Station>_IGS20.apr with one row in the specified format
    """
    # Extract station name
    station_name = os.path.basename(file_path).split('.')[0]
    output_file = f"{station_name}_IGS20.apr"
    
    # Get original XYZ coordinates in meters and their medians
    ax = original_data["X_m"].median()  # X coordinate in meters
    ay = original_data["Y_m"].median()  # Y coordinate in meters
    az = original_data["Z_m"].median()  # Z coordinate in meters
    
    # Convert velocities from cm/year to m/year
    b_ns = velocities.get("N", 0.0) / 100.0  # N velocity in m/year
    b_ew = velocities.get("E", 0.0) / 100.0  # E velocity in m/year
    b_ud = velocities.get("U", 0.0) / 100.0  # U velocity in m/year
    
    # Set sigma values to 0.00001 as specified
    sigb_ns = 0.00001
    sigb_ew = 0.00001
    sigb_ud = 0.00001
    
    # Find the time corresponding to each median (t_ax, t_ay, t_az)
    def get_time_for_median(df, column, median_value):
        # Drop NaN values and find the index of the value closest to the median
        valid_data = df[[column, "Decimal_Year"]].dropna()
        if valid_data.empty:
            return None
        # Find the index where the absolute difference from the median is minimized
        idx = (valid_data[column] - median_value).abs().idxmin()
        return valid_data.loc[idx, "Decimal_Year"]
    
    t_ax = get_time_for_median(original_data, "X_m", ax)
    t_ay = get_time_for_median(original_data, "Y_m", ay)
    t_az = get_time_for_median(original_data, "Z_m", az)
    
    # Calculate t0 as the median of t_ax, t_ay, t_az
    valid_times = [t for t in [t_ax, t_ay, t_az] if t is not None]
    if valid_times:
        t0 = np.median(valid_times)
    else:
        # Fallback to midpoint of entire time series if no valid times are found
        valid_times = original_data["Decimal_Year"].dropna()
        t0 = (valid_times.iloc[0] + valid_times.iloc[-1]) / 2.0 if not valid_times.empty else 2014.29710
    
    # Format the output string with Fortran-like f15.5 precision
    with open(output_file, 'w') as f:
        f.write(f" {station_name}_GPS {ax:15.5f} {ay:15.5f} {az:15.5f} "
                f"{b_ns:15.5f} {b_ew:15.5f} {b_ud:15.5f} {t0:15.5f} "
                f"{sigb_ns:15.5f} {sigb_ew:15.5f} {sigb_ud:15.5f}\n")
    
    print(f"APR file saved: {output_file}")
##=========================================================================================================
# **********Main Function***************************
# ============================================================================
# ============================================================================
def process_file(file_path, cnn_model, big_params, step_threshold):
    # 1) Load data and remove outliers
    original_data, cleaned_data, median_N, median_E, median_U = load_data_xyz(file_path)
    station_name = os.path.basename(file_path).split('.')[0]
    print("GPS:", station_name)
    
    # 2) Use the trained CNN model to select optimal small-step detection parameters
    optimal_small, cnn_scores, change_points = select_optimal_small_params_sequential(
        cleaned_data, cnn_model, station_name, threshold_good=0.88
    )
    
    # Extract direction-specific curve thresholds
    N_curve_threshold = optimal_small["small_C_N_threshold"]
    E_curve_threshold = optimal_small["small_C_E_threshold"]
    U_curve_threshold = optimal_small["small_C_U_threshold"]
    
    # 3) Convert change point indices to decimal years
    small_change_points_year = get_change_points_year(cleaned_data, change_points)
    
    # 4) Write out the AI parameters file
    write_ai_parameters(
        station_name=station_name,
        input_file=file_path,
        big_window_size=big_params["window_size"],
        big_N_threshold=big_params["N_threshold"],
        big_E_threshold=big_params["E_threshold"],
        big_U_threshold=big_params["U_threshold"],
        big_min_distance=big_params["min_distance"],
        step_threshold=step_threshold,
        small_window_size=optimal_small["small_window_size"],
        small_N_threshold=optimal_small["small_N_threshold"],
        small_E_threshold=optimal_small["small_E_threshold"],
        small_U_threshold=optimal_small["small_U_threshold"],
        small_min_distance=optimal_small["small_min_distance"],
        N_curve_threshold=N_curve_threshold,
        E_curve_threshold=E_curve_threshold,
        U_curve_threshold=U_curve_threshold,
        improvement_ratio=0.2,
        small_change_points_year=small_change_points_year,
        scores=cnn_scores,
        label="good"
    )
    
    # 5) Calculate velocities and generate plots
    velocities, intercepts, longest_segments = calculate_segment_velocity(cleaned_data, change_points)
    plot_time_series_with_velocity(cleaned_data, change_points, velocities, intercepts, longest_segments, file_path)
    plot_time_series_with_longest_segment(cleaned_data, velocities, intercepts, longest_segments, file_path)
    save_velocities(file_path, velocities, longest_segments, original_data)
    
    # 6) Save APR file
    save_apr_file(file_path, original_data, velocities, longest_segments)
    
    # 6) Clean up
    os.remove(file_path)
    
    del original_data, cleaned_data, velocities, intercepts, longest_segments, change_points
    del optimal_small, cnn_scores, small_change_points_year
    plt.close('all')  # Clear all Matplotlib figures
    tf.keras.backend.clear_session()  # Reset TensorFlow state
    gc.collect()
# ============================================================================
# Main Block
# ============================================================================
if __name__ == "__main__":
    # Load the CNN model once
    print("Loading CNN model...")
    cnn_model = tf.keras.models.load_model("ChangePointCNN-GNSS_VGG_V7.keras")
    
    # Define fixed parameters for large step detection (used in output, even if not computed)
    big_params = {
        "window_size": 20,
        "N_threshold": 0.5,
        "E_threshold": 0.5,
        "U_threshold": 2.0,
        "min_distance": 30
    }
    step_threshold = 50

    # Get all files
    file_paths = glob.glob("*.XYZ")
    if not file_paths:
        print("No NEU .col files found in the current directory.")
        exit(1)
    
    # Process files sequentially
    total_files = len(file_paths)
    for i, file_path in enumerate(file_paths, 1):
        print(f"Processing {file_path} ({i}/{total_files})...")
        process_file(file_path, cnn_model, big_params, step_threshold)
        gc.collect()  # Cleanup after each file
    
    # Final cleanup
    del cnn_model
    gc.collect()
    print("Processing complete.")

