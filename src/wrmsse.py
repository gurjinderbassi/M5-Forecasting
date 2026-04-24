import numpy as np
import pandas as pd
from typing import Union, Tuple


def compute_rmsse(y_true: np.ndarray, y_pred: np.ndarray, 
                  scale: np.ndarray) -> np.ndarray:
    """
    Compute Root Mean Squared Scaled Error (RMSSE) for each series.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values, shape (n_series, n_timesteps)
    y_pred : np.ndarray
        Predicted values, shape (n_series, n_timesteps)
    scale : np.ndarray
        Scaling factor for each series, shape (n_series,)
    
    Returns
    -------
    np.ndarray
        RMSSE for each series, shape (n_series,)
    """
    squared_errors = (y_true - y_pred) ** 2
    mse = np.mean(squared_errors, axis=1)
    rmsse = np.sqrt(mse / scale)
    return rmsse


def compute_scale(y_train: np.ndarray) -> np.ndarray:
    """
    Compute scaling factors using naive (seasonal) forecast errors.
    
    Uses the mean squared error of the naive forecast (lag-1 difference).
    
    Parameters
    ----------
    y_train : np.ndarray
        Training data, shape (n_series, n_timesteps)
    
    Returns
    -------
    np.ndarray
        Scaling factors for each series, shape (n_series,)
    """
    # Compute differences (naive forecast error)
    # For each series, compute MSE of predictions vs actual shifted by 1
    differences = np.diff(y_train, axis=1)
    squared_diffs = differences ** 2
    scale = np.mean(squared_diffs, axis=1)
    
    # Avoid division by zero
    scale = np.where(scale == 0, 1, scale)
    
    return scale


def compute_weights(y_train: np.ndarray) -> np.ndarray:
    """
    Compute weights for each series based on total sales volume.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training data, shape (n_series, n_timesteps)
    
    Returns
    -------
    np.ndarray
        Normalized weights for each series, shape (n_series,)
    """
    # Sum of sales for each series
    total_sales = np.sum(y_train, axis=1)
    
    # Normalize weights
    weights = total_sales / np.sum(total_sales)
    
    return weights


def wrmsse(y_true: np.ndarray, y_pred: np.ndarray, 
           y_train: np.ndarray) -> float:
    """
    Compute Weighted Root Mean Squared Scaled Error (WRMSSE).
    
    WRMSSE = sqrt(sum(weights * rmsse^2) / sum(weights))
    
    where:
    - rmsse is the root mean squared scaled error for each series
    - weights are based on total sales volume in training data
    - scale is computed from naive forecast errors on training data
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values, shape (n_series, n_timesteps)
    y_pred : np.ndarray
        Predicted values, shape (n_series, n_timesteps)
    y_train : np.ndarray
        Training data used to compute scaling factors and weights,
        shape (n_series, n_timesteps_train)
    
    Returns
    -------
    float
        WRMSSE score (lower is better)
    """
    # Compute scaling factors
    scale = compute_scale(y_train)
    
    # Compute RMSSE for each series
    rmsse_values = compute_rmsse(y_true, y_pred, scale)
    
    # Compute weights
    weights = compute_weights(y_train)
    
    # Compute weighted RMSSE
    weighted_rmsse_squared = weights * (rmsse_values ** 2)
    final_wrmsse = np.sqrt(np.sum(weighted_rmsse_squared) / np.sum(weights))
    
    return final_wrmsse


def wrmsse_by_level(y_true: np.ndarray, y_pred: np.ndarray, 
                    y_train: np.ndarray, 
                    level_mapping: np.ndarray) -> dict:
    """
    Compute WRMSSE for different hierarchical levels.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values, shape (n_series, n_timesteps)
    y_pred : np.ndarray
        Predicted values, shape (n_series, n_timesteps)
    y_train : np.ndarray
        Training data, shape (n_series, n_timesteps_train)
    level_mapping : np.ndarray
        Level assignment for each series (e.g., 1 for store-dept, 2 for store, etc.)
    
    Returns
    -------
    dict
        Dictionary with WRMSSE scores for each level
    """
    results = {}
    
    for level in np.unique(level_mapping):
        mask = level_mapping == level
        level_wrmsse = wrmsse(y_true[mask], y_pred[mask], y_train[mask])
        results[f'level_{level}'] = level_wrmsse
    
    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n_series = 100
    n_train_steps = 365
    n_test_steps = 30
    
    y_train = np.random.poisson(10, size=(n_series, n_train_steps)).astype(float)
    y_test = np.random.poisson(10, size=(n_series, n_test_steps)).astype(float)
    y_pred_test = y_test + np.random.normal(0, 1, size=(n_series, n_test_steps))
    y_pred_test = np.maximum(y_pred_test, 0)  # Ensure non-negative
    
    # Compute WRMSSE
    score = wrmsse(y_test, y_pred_test, y_train)
    print(f"WRMSSE Score: {score:.6f}")
