"""
Economic modeling utilities for New Keynesian DSGE models.

This module provides common functions and constants used across multiple notebooks
for economic modeling, particularly focusing on wage rigidity and productivity shocks.

Author: Economic Analysis Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Union, List
import itertools


# Economic model constants
INVENTORY_TO_COGS_SCALING = 36.5
DAYS_PER_MONTH = 30
TAX_RATE = 0.1
HISTOGRAM_BINS = 55
EPSILON = 1e-10  # Small value to prevent division by zero

# Economic model parameters
RISK_AVERSION = 2  # sigma parameter
NOMINAL_INTEREST_RATE = 0.01  # 1% per period
INVENTORY_DURATION = 30  # days
ADJUSTMENT_RATE = 0.9  # proportion of firms adjusting


def load_and_process_data(filepath: str) -> np.ndarray:
    """
    Load and process economic data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV data file
        
    Returns:
    --------
    numpy.ndarray
        Processed tau data (inventory adjustment times)
        
    Raises:
    -------
    FileNotFoundError
        If the data file cannot be found
    """
    try:
        data = pd.read_csv(filepath, index_col=0, parse_dates=True, infer_datetime_format=True)
        data_tau = (data['invtCogsRatio'] * INVENTORY_TO_COGS_SCALING / (DAYS_PER_MONTH * TAX_RATE))
        return data_tau.values
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {filepath}")
    except KeyError:
        raise KeyError("Required column 'invtCogsRatio' not found in data")


def calculate_discount_factors() -> Tuple[float, float, float]:
    """
    Calculate discount factors for economic model.
    
    Returns:
    --------
    tuple
        beta, beta1, beta_hh discount factors
    """
    beta = 1 / (1.0 + NOMINAL_INTEREST_RATE)
    beta1 = 1 / (1.0 + NOMINAL_INTEREST_RATE * np.exp(INVENTORY_DURATION / (INVENTORY_TO_COGS_SCALING * ADJUSTMENT_RATE))) 
    beta_hh = beta
    return beta, beta1, beta_hh


def find_optimal_c_parameter(data_tau: np.ndarray, beta: float, beta1: float, 
                           num_points: int = 1000) -> float:
    """
    Find optimal c parameter by minimizing tau differences.
    
    Parameters:
    -----------
    data_tau : numpy.ndarray
        Inventory adjustment times data
    beta : float
        Primary discount factor
    beta1 : float
        Alternative discount factor
    num_points : int
        Number of grid points to search over
        
    Returns:
    --------
    float
        Optimal c parameter value
        
    Raises:
    -------
    ValueError
        If optimization fails or data is invalid
    """
    if len(data_tau) == 0:
        raise ValueError("data_tau cannot be empty")
    
    tau_diff = np.zeros((num_points, 1))
    
    for i, c in enumerate(np.linspace(0, 0.99, num_points)): 
        # Prevent division by zero in logarithm calculations
        if beta * c <= EPSILON or beta1 * c <= EPSILON:
            tau_diff[i] = np.inf  # Skip problematic values
            continue
            
        # Safe logarithm calculations
        log_beta_c = np.log(beta * c)
        log_beta1_c = np.log(beta1 * c)
        
        N = -(data_tau) - 1 / log_beta_c 
        tau1 = -1 / log_beta_c - N
        tau1[tau1 < 0] = 0     
        
        tau2 = -1 / log_beta1_c - N
        tau2[tau2 < 0] = 0    
        
        tau_diff[i] = np.mean(np.abs(tau1 - tau2))
    
    min_val = np.min(np.abs(tau_diff - 1))
    argmins = np.where(np.abs(tau_diff - 1) == min_val)[0]
    c_list = np.linspace(0, 0.99, num_points)
    
    return c_list[argmins][0]


def euler_path_fixed_wages(i_path: np.ndarray, p_level: np.ndarray, 
                          theta: float, beta_hh: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the Euler equation for consumption path with fixed wages.
    
    This function implements the New Keynesian Euler equation:
    c_t = c_{t+1} - (1/sigma) * (i_t + Δabar_{t+1})
    
    Parameters:
    -----------
    i_path : numpy.ndarray
        Nominal interest rate path over time
    p_level : numpy.ndarray
        Log productivity level path
    theta : float
        Intertemporal elasticity of substitution parameter
    beta_hh : float
        Household discount factor
        
    Returns:
    --------
    tuple of numpy.ndarray
        c : Consumption path (equals output y_t)
        pi : Price inflation path
        
    Raises:
    -------
    ValueError
        If inputs have incompatible shapes or invalid parameter values
    """
    # Input validation
    i_path = np.asarray(i_path)
    p_level = np.asarray(p_level)
    
    if i_path.size == 0 or p_level.size == 0:
        raise ValueError("Input arrays cannot be empty")
    if i_path.shape != p_level.shape:
        raise ValueError("i_path and p_level must have the same shape")
    if not np.isfinite(i_path).all() or not np.isfinite(p_level).all():
        raise ValueError("Input arrays must contain only finite values")
    if theta <= 0:
        raise ValueError("theta (elasticity parameter) must be positive")
    
    T = len(i_path)
    
    # Compute productivity growth (inflation proxy)
    dA = np.zeros(T)
    dA[1:] = p_level[1:] - p_level[:-1]
    pi = dA  # Price inflation equals productivity growth
    
    # Initialize consumption path with terminal condition
    c = np.zeros(T + 1)  # Extra slot for terminal condition c_T = 0
    
    # Backward iteration using Euler equation
    for t in range(T - 1, -1, -1):
        if t + 1 < T:
            # Standard Euler equation with expected inflation
            c[t] = c[t + 1] - (1.0/theta) * (i_path[t] - dA[t + 1] - (1/beta_hh - 1))
        else:
            # Terminal period adjustment
            c[t] = c[t + 1] - (1.0/theta) * (i_path[t] - (1/beta_hh - 1))

    return c[:T], pi


def find_duplicate_indices_per_row(arr: np.ndarray, tol: float = 0.0) -> List[List[int]]:
    """
    Find indices of duplicate values within each row of a 2D array.
    
    This function identifies which column indices contain duplicate values
    within each row, useful for handling timing conflicts in economic models
    where multiple firms might adjust prices simultaneously.
    
    Parameters:
    -----------
    arr : numpy.ndarray
        2D array of shape (R, C) to search for duplicates
    tol : float, optional (default=0.0)
        Tolerance for considering values equal
        - If 0.0: exact match required
        - If > 0.0: values within tolerance are considered duplicates
        
    Returns:
    --------
    list of lists
        For each row, returns list of column indices that contain duplicates
        Length equals number of rows in input array
        
    Examples:
    ---------
    >>> arr = np.array([[1, 2, 1], [3, 3, 4]])
    >>> find_duplicate_indices_per_row(arr)
    [[0, 2], [0, 1]]  # Row 0: cols 0,2 both have value 1
                      # Row 1: cols 0,1 both have value 3
    """
    dup_indices = []
    
    for row in arr:
        if tol > 0:
            # Tolerance-based matching for floating point values
            idxs = []
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    if abs(row[i] - row[j]) <= tol:
                        idxs.extend([i, j])
            dup_indices.append(sorted(set(idxs)))
        else:
            # Exact matching using numpy unique
            vals, inv, counts = np.unique(row, return_inverse=True, return_counts=True)
            duplicate_vals = np.where(counts > 1)[0]  # Values that appear more than once
            duplicate_cols = np.where(np.isin(inv, duplicate_vals))[0]  # Column indices of duplicates
            dup_indices.append(duplicate_cols.tolist())
    
    return dup_indices


def generate_pattern_array_with_extra_rows(tau: np.ndarray) -> np.ndarray:
    """
    Generate a pattern array for productivity shock modeling.
    
    This function creates a matrix pattern used in economic modeling where firms
    face different adjustment costs over time periods.
    
    Parameters:
    -----------
    tau : numpy.ndarray
        Array of time periods/adjustment delays for different firms
        
    Returns:
    --------
    numpy.ndarray
        Extended pattern matrix with shock adjustment factors
        
    Raises:
    -------
    ValueError
        If tau is empty, contains negative values, or has invalid dimensions
    """
    # Input validation
    tau = np.asarray(tau)
    if tau.size == 0:
        raise ValueError("tau array cannot be empty")
    if np.any(tau < 0):
        raise ValueError("tau values must be non-negative")
    if not np.isfinite(tau).all():
        raise ValueError("tau must contain only finite values")
        
    # Calculate matrix dimensions based on tau statistics
    n = len(tau) + int(np.min(tau))
    m = int(np.max(tau)) + int(np.min(tau))
    
    max_tau = int(max(tau))
    missing_count = len(set(range(max_tau + 1)) - set(tau))

    # Initialize base matrix with ones (no shock)
    A = np.ones((m + 2, n + missing_count), dtype=float)
    count = 0
    
    # Fill shock adjustment factors (0.99) for active adjustment periods
    for i in range(1, m + 2):
        firms_adjusting = tau[tau == i - 1]
        for j in range(len(firms_adjusting)):
            A[i, n - i - j - count] = 0.99  # Shock adjustment factor
        if len(firms_adjusting) > 1:
            count += len(firms_adjusting) - 1
    
    # Add boundary condition row (all ones)
    row_all_ones = np.ones(n + missing_count, dtype=float)
    A_extended = np.vstack([A, row_all_ones])
    
    # Trim to final dimensions
    A_extended = A_extended[:, 0:n - int(np.min(tau))]

    return A_extended


def generate_rate_combinations(sequence_length: int, min_rate: float = 0.0, 
                             max_rate: float = 0.0125, num_points: int = 6) -> np.ndarray:
    """
    Generate all combinations of interest rate sequences for optimization.
    
    Parameters:
    -----------
    sequence_length : int
        Length of each rate sequence
    min_rate : float
        Minimum interest rate value
    max_rate : float
        Maximum interest rate value
    num_points : int
        Number of rate values to consider
        
    Returns:
    --------
    numpy.ndarray
        Array of all possible rate combinations, shape (num_points^sequence_length, sequence_length)
    """
    r_choice = np.linspace(min_rate, max_rate, num_points)
    all_combinations = list(itertools.product(r_choice, repeat=sequence_length))
    return np.array(all_combinations)


def plot_comparison_results(C_consumption: np.ndarray, pi_inflation: np.ndarray,
                          optimal_indices: np.ndarray, baseline_indices: np.ndarray,
                          title: str = "Economic Model Comparison", 
                          time_periods: int = 20) -> None:
    """
    Plot comparison of consumption and inflation paths.
    
    Parameters:
    -----------
    C_consumption : numpy.ndarray
        Consumption paths for all scenarios
    pi_inflation : numpy.ndarray
        Inflation paths for all scenarios
    optimal_indices : numpy.ndarray
        Indices of optimal policy scenarios
    baseline_indices : numpy.ndarray
        Indices of baseline policy scenarios
    title : str
        Plot title
    time_periods : int
        Number of time periods to plot
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(C_consumption[optimal_indices, :time_periods].T, label='Optimal Policy', linewidth=2)
    plt.plot(C_consumption[baseline_indices, :time_periods].T, label='Baseline Policy', linestyle='--', linewidth=2)
    plt.ylabel('Consumption')
    plt.title(f'{title} - Consumption Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(pi_inflation[optimal_indices, :time_periods].T, label='Optimal Policy', linewidth=2)
    plt.plot(pi_inflation[baseline_indices, :time_periods].T, label='Baseline Policy', linestyle='--', linewidth=2)
    plt.ylabel('Inflation')
    plt.xlabel('Time Period')
    plt.title(f'{title} - Inflation Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()