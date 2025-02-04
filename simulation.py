import numpy as np
import pandas as pd

def get_stop_loss_target(entry_price: float, trade_decision: int, reward_to_risk_ratio: float, stop_loss_percentage: float=None, stop_loss_points: float=None):
    """A method to get stop loss and target price when either of stop-loss points or stop-loss-percentage is provided along with reward-to-risk ratio.
    
    Args:
        entry_price (float): The entry price of the trade.
        trade_decision (int): For long trade pass 0 and for short trade pass 1.
        reward_to_risk_ratio (float): The desired reward to risk ratio for the trade.
        stop_loss_percentage (float, optional): The risk in terms of percentage of entry price.
        stop_loss_points (float, optional): USe this parameter instead of risk percentage if risk is to be in terms of absolute points.
        
    Returns:
        Tuple[(float, float)]: The stop-loss and target-price as tuple.
    """
    if (stop_loss_percentage) and (stop_loss_points):
        raise ValueError(f"Only one of {stop_loss_percentage} or {stop_loss_points} should be provided, not both.")
    if stop_loss_percentage:
        target_percentage = stop_loss_percentage * reward_to_risk_ratio
        stop_loss_price = entry_price*(1-stop_loss_percentage/100) if trade_decision == 0 else entry_price*(1+stop_loss_percentage/100)
        target_price = entry_price*(1+target_percentage/100) if trade_decision == 0 else entry_price*(1-target_percentage/100)
        return stop_loss_price, target_price
    if stop_loss_points:
        target_points = stop_loss_points * reward_to_risk_ratio
        stop_loss_price = entry_price-stop_loss_points if trade_decision == 0 else entry_price+stop_loss_points
        target_price = entry_price+target_points if trade_decision == 0 else entry_price-target_points
        return stop_loss_price, target_price

def simulate_long_trade(entry_price: float, stop_loss_price: float, target_price: float, simulation_ohlc_df: pd.Series) -> float:
    """
    Calculate the profit/loss points with the provided entry, stop-loss, and target price using the OHLC data following the long-trade entry
    used for simulation. The length of this dataframe can be as long as you want to run the trade when stop-loss or target is not hit. The resolution
    of the OHLC data influences the accuracy of the simulation results. 1 sec OHLC data can provide real trade like simulations, but atleast 
    use 1 min OHLC data for meaningful results. 
    
    Args:
        entry_price (float): Previous day's closing price.
        stop_loss_price (float): Stop-loss price.
        target_price (float): Target price.
        simulation_ohlc_df (pd.Series): The OHLC data following the trade entry time and upto the max trade hold date-time.

    Returns:
        Tuple[dt.datetime, float, float, str]: The exit date-time, exit-price, resulting-points, and the exit-reason for the trade exit.
    """
    for _, row_ohlc in simulation_ohlc_df.iterrows():
        if row_ohlc["low"] <= stop_loss_price:
            exit_price = stop_loss_price if row_ohlc["open"] > stop_loss_price else float(row_ohlc["low"])
            result_points = exit_price - entry_price
            return row_ohlc["date_time"], exit_price, result_points, "stop-loss-hit"
        if row_ohlc["high"] >= target_price:
            exit_price = target_price if row_ohlc["open"] < target_price else float(row_ohlc["low"])
            result_points = exit_price - entry_price
            return row_ohlc["date_time"], exit_price, result_points, "target-hit"
    exit_price = simulation_ohlc_df.iloc[-1]["open"] #if stop-loss or target is not hit, then exit at open of the last available candle.
    result_points = exit_price - entry_price
    return row_ohlc["date_time"], exit_price, result_points, "squared-off-at-end"

def simulate_short_trade(entry_price: float, stop_loss_price: float, target_price: float, simulation_ohlc_df: pd.Series) -> float:
    """
    Calculate the profit/loss points with the provided entry, stop-loss, and target price using the OHLC data following the short-trade entry
    used for simulation. The length of this dataframe can be as long as you want to run the trade when stop-loss or target is not hit. The resolution
    of the OHLC data influences the accuracy of the simulation results. 1 sec OHLC data can provide real trade like simulations, but atleast 
    use 1 min OHLC data for meaningful results.
    
    Args:
        entry_price (float): Previous day's closing price.
        stop_loss_price (float): Stop-loss price.
        target_price (float): Target price.
        simulation_ohlc_df (pd.Series): The OHLC data following the trade entry time and upto the max trade hold date-time.

    Returns:
        Tuple[dt.datetime, float, float, str]: The exit date-time, exit-price, resulting-points, and the exit-reason for the trade exit.
    """
    for _, row_ohlc in simulation_ohlc_df.iterrows():
        if row_ohlc["high"] >= stop_loss_price:
            exit_price = stop_loss_price if row_ohlc["open"] < stop_loss_price else float(row_ohlc["high"])
            result_points = entry_price - exit_price
            return row_ohlc["date_time"], exit_price, result_points, "stop-loss-hit"
        if row_ohlc["low"] <= target_price:
            exit_price = target_price if row_ohlc["open"] > target_price else float(row_ohlc["high"])
            result_points = entry_price - exit_price
            return row_ohlc["date_time"], exit_price, result_points, "target-hit"
    exit_price = simulation_ohlc_df.iloc[-1]["open"]
    result_points = entry_price - exit_price
    return row_ohlc["date_time"], exit_price, result_points, "squared-off-at-end"
