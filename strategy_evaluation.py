import logging.config
import logging
import os
import numpy as np
import pandas as pd
import datetime as dt
import math
import simulation as sm
import matplotlib.pyplot as plt

# Configure logger
logging.config.fileConfig("log.conf")
main_logger = logging.getLogger("main")

# Directories and paths
TRADES_FILES_FOLDER = "evaluation_files"
SIMULATION_FILES_FOLDER = "simulation_files"
RESULTS_FOLDER = "results"

TRADES_FNAME = "trades.csv"  # Evaluation filename
SIMULATION_FNAME = "nifty_50_1_min.csv"  # Simulation filename

TRADES_FPATH = os.path.join(TRADES_FILES_FOLDER, TRADES_FNAME)
SIMULATION_FPATH = os.path.join(SIMULATION_FILES_FOLDER, SIMULATION_FNAME)

# Account parameters
INITIAL_CAPITAL = 300000.00
NUM_OF_UNITS = 75  # Number of units per trade

# risk-reward and stop-loss combinations to evaluate
SL_PERCENTAGES = [0.1, 0.2, 0.3, 0.4, 0.5]
REWARD_TO_RISK_RATIOS = [1, 1.5, 2, 2.5, 3, 3.5, 4]

os.makedirs(RESULTS_FOLDER, exist_ok=True)

def read_trades_and_sim_files(trades_fpath: str, simulation_fpath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads trade and simulation data from CSV files.
    
    Args:
        trades_fpath (str): Path to the trades CSV file.
        simulation_fpath (str): Path to the simulation CSV file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing trades and simulation data.
    """
    try:
        trades_df = pd.read_csv(trades_fpath)
        trades_df["entry_date_time"] = pd.to_datetime(trades_df["entry_date_time"], format="%Y-%m-%d %H:%M:%S")
        trades_df["max_hold_date_time"] = pd.to_datetime(trades_df["max_hold_date_time"], format="%Y-%m-%d %H:%M:%S")
        simulation_df = pd.read_csv(simulation_fpath)
        simulation_df["date_time"] = pd.to_datetime(simulation_df["date_time"], format="%Y-%m-%d %H:%M:%S")
        main_logger.info("Successfully read trades and simulation files.")
        return trades_df, simulation_df
    except Exception as e:
        main_logger.error(f"Error reading files: {e}")
        raise

def check_if_redundant_parameter(parameter_a=None, parameter_b=None):
    """Ensures only one of the parameters is provided.
    
    Args:
        parameter_a (any, optional): First parameter.
        parameter_b (any, optional): Second parameter.

    Raises:
        ValueError: If both parameters are provided simultaneously.
    """
    if parameter_a and parameter_b:
        raise ValueError("Only one of the parameters should be provided.")

def evaluate_trades(trades_df: pd.DataFrame, simulation_df: pd.DataFrame, 
                     reward_to_risk_ratio: float, stop_loss_percentage: float = None, stop_loss_points: float = None) -> pd.DataFrame:
    """Evaluates trades against simulation data and calculates trade outcomes.
    
    Args:
        trades_df (pd.DataFrame): DataFrame with trades.
        simulation_df (pd.DataFrame): DataFrame with simulation data.
        reward_to_risk_ratio (float): Reward-to-risk ratio.
        stop_loss_percentage (float, optional): Stop loss as a percentage. Optional only if stop_loss_points is provided instead.
        stop_loss_points (float, optional): Stop loss as absolute points. Optional only if stop_loss_percentage is provided instead.

    Returns:
        pd.DataFrame: DataFrame with trade evaluation results.
    """
    check_if_redundant_parameter(stop_loss_percentage, stop_loss_points)
    
    result_columns = ["entry_date_time", "trade_decision", "entry_price", "stop_loss_price", "target_price", 
                      "exit_date_time", "exit_price", "points_achieved", "gain_loss", "exit_reason"]
    all_results = []
    
    for _, row in trades_df.iterrows():
        entry_date_time, entry_price, trade_decision = row["entry_date_time"], row["entry_price"], int(row["trade_decision"])
        max_hold_date_time = row["max_hold_date_time"]
        simulation_sub_df = simulation_df[(simulation_df["date_time"] > entry_date_time) & (simulation_df["date_time"] <= max_hold_date_time)]
        stop_loss_price, target_price = sm.get_stop_loss_target(entry_price, trade_decision, reward_to_risk_ratio, 
                                                                stop_loss_percentage=stop_loss_percentage, stop_loss_points=stop_loss_points)
        
        if trade_decision == 0:
            exit_date_time, exit_price, points_achieved, exit_reason = sm.simulate_long_trade(entry_price, stop_loss_price, target_price, simulation_sub_df)
        else:
            exit_date_time, exit_price, points_achieved, exit_reason = sm.simulate_short_trade(entry_price, stop_loss_price, target_price, simulation_sub_df)
        
        all_results.append([entry_date_time, "long" if trade_decision == 0 else "short", entry_price, stop_loss_price, target_price,
                            exit_date_time, exit_price, points_achieved, points_achieved * NUM_OF_UNITS, exit_reason])
    
    if not all_results:
        raise ValueError("No trades were processed.")
    return pd.DataFrame(all_results, columns=result_columns)

def calculate_CAGR(gain_loss_amount: float, start_date: dt.datetime, end_date: dt.datetime) -> float:
    """Calculates Compound Annual Growth Rate (CAGR).
    
    Args:
        gain_loss_amount (float): Total gain or loss.
        start_date (dt.datetime): Start date of the evaluation.
        end_date (dt.datetime): End date of the evaluation.

    Returns:
        float: CAGR percentage.
    """
    num_of_days = (end_date - start_date).days
    final_amount = gain_loss_amount + INITIAL_CAPITAL
    return ((math.pow(final_amount / INITIAL_CAPITAL, 365 / num_of_days) - 1) * 100) if final_amount > 0 else -100.0

def evaluate_parameters_grid(trades_fpath: str, simulation_fpath: str, reward_to_risk_ratios: list, stop_loss_percentages: list, save: bool = True) -> pd.DataFrame:
    """Evaluates multiple combinations of stop-loss and reward-to-risk ratios.
    
    Args:
        trades_fpath (str): Path to trades file.
        simulation_fpath (str): Path to simulation file.
        reward_to_risk_ratios (list): List of reward-to-risk ratios.
        stop_loss_percentages (list): List of stop-loss percentages.
        save (bool, optional): Whether to save results. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing evaluation results.
    """
    trades_df, simulation_df = read_trades_and_sim_files(trades_fpath, simulation_fpath)
    results = []
    
    for sl, rr in [(sl, rr) for sl in stop_loss_percentages for rr in reward_to_risk_ratios]:
        trade_results = evaluate_trades(trades_df, simulation_df, reward_to_risk_ratio=rr, stop_loss_percentage=sl)
        points_achieved = np.sum(trade_results["points_achieved"].values)
        gain_loss = np.sum(trade_results["gain_loss"].values)
        cagr = calculate_CAGR(gain_loss, trade_results.iloc[0]["entry_date_time"], trade_results.iloc[-1]["exit_date_time"])
        results.append([sl, rr, points_achieved, gain_loss, cagr])
        main_logger.info(f"For stop loss: {sl}%, reward to risk ratio: {rr:0>5.2f}x, Total points achieved: {points_achieved:.2f}, Total gain/loss: {gain_loss:,.2f} Rs., CAGR: {cagr:.2f}%")
    results_df = pd.DataFrame(results, columns=["stop_loss_percentage", "reward_to_risk_ratio", "points_achieved", "gain_loss", "CAGR"])
    if save:
        file_name = f"parameters_evaluation_results_{dt.datetime.now().strftime('%d_%m_%Y_%H%M%S')}.csv"
        save_file_path = os.path.join(RESULTS_FOLDER, file_name)
        results_df.to_csv(save_file_path, index=False)
    return results_df

def evaluate_for_parameters(trades_fpath: str, simulation_fpath: str, reward_to_risk_ratio: float, stop_loss_percentage: float=None, stop_loss_points: float=None, save: bool=True, plot: bool=True) -> pd.DataFrame:
    """Evaluates trade performance for a given reward-to-risk ratio and stop-loss setting.

    This function reads trade and simulation data from the specified file paths, evaluates 
    trade performance based on the given reward-to-risk ratio and stop-loss parameters, 
    and optionally saves the results to a CSV file and plots account value curve achieved.

    Args:
        trades_fpath (str): Path to the CSV file containing trade data.
        simulation_fpath (str): Path to the CSV file containing simulation data.
        reward_to_risk_ratio (float): The reward-to-risk ratio to be tested.
        stop_loss_percentage (float, optional): The stop-loss percentage to be applied. 
            If None, stop_loss_points must be provided. Defaults to None.
        stop_loss_points (float, optional): The stop-loss in absolute points.
            If None, stop_loss_percentage must be provided. Defaults to None.
        save (bool, optional): If True, saves the trade evaluation results to a CSV file. Defaults to True.
        plot (bool, optional): If True, generates and displays a plot of cumulative points achieved. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing trade-wise results.
    """
    trades_df, simulation_df = read_trades_and_sim_files(trades_fpath, simulation_fpath)
    check_if_redundant_parameter(stop_loss_percentage, stop_loss_points)
    trades_result_df = evaluate_trades(trades_df, simulation_df, reward_to_risk_ratio, stop_loss_percentage)
    points_achieved = np.sum(trades_result_df["points_achieved"].values)
    total_gain_loss = np.sum(trades_result_df["gain_loss"].values)
    cagr = calculate_CAGR(total_gain_loss, trades_result_df.iloc[0]["entry_date_time"], trades_result_df.iloc[-1]["exit_date_time"])
    
    main_logger.info(f"Stop Loss: {stop_loss_percentage}%, Reward to Risk: {reward_to_risk_ratio:.2f}x, Total Points: {points_achieved:.2f}, Total P/L: {total_gain_loss:,.2f} Rs., CAGR: {cagr:.2f}%")
    
    if plot:
        cum_account_value = np.cumsum(trades_result_df["gain_loss"].values) + INITIAL_CAPITAL
        plt.plot(np.arange(len(cum_account_value)), cum_account_value)
        plt.title("Account Value Curve")
        plt.xlabel("Days")
        plt.ylabel("Account Value (Rs.)")
        file_name = f"account_value_curve_{dt.datetime.now().strftime('%d_%m_%Y_%H%M%S')}.png"
        save_file_path = os.path.join(RESULTS_FOLDER, file_name)
        plt.savefig(save_file_path)
        plt.show()

    if save:
        file_name = f"trade_results_{dt.datetime.now().strftime('%d_%m_%Y_%H%M%S')}.csv"
        save_file_path = os.path.join(RESULTS_FOLDER, file_name)
        trades_result_df.to_csv(save_file_path, index=False)
        main_logger.info(f"Results saved to {save_file_path}")
    
    return trades_result_df

if __name__ == "__main__":
    parameter_gird_eval_result_df = evaluate_parameters_grid(TRADES_FPATH, SIMULATION_FPATH, REWARD_TO_RISK_RATIOS, SL_PERCENTAGES)
    
    # use the best parameter and get the trade-wise detailed report:
    best_parameters = dict(rr = 3.5, sl_perc = 0.5)
    trade_results_for_parameter_df = evaluate_for_parameters(TRADES_FPATH, SIMULATION_FPATH, reward_to_risk_ratio = best_parameters["rr"], stop_loss_percentage = best_parameters["sl_perc"])