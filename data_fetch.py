"""
Module: data_fetch.py
Description: 
    This module provides functionality for fetching and processing equity data and option chains 
    for the Fixing Volatile Volatility project. It leverages yfinance to retrieve market data, 
    computes key metrics such as the spot price, annualized volatility, dividend yield, time to maturity,
    and risk-free rate, and processes option chain data with additional calculated columns.
"""

from datetime import datetime
import pytz
import yfinance as yf
import pandas as pd
import numpy as np

class DataPrep:
    """
    Class for preparing equity and options data.

    Attributes:
        ticker (str): The ticker symbol of the equity.
        zero_df (pd.DataFrame): DataFrame containing the zero curve data with columns 'Maturity' and 'Yield'.
        spot (float): The current spot price of the equity.
        sigma (float): The annualized volatility of the equity.
        div_yield (float): The dividend yield of the equity (as a decimal).
    """

    def __init__(self, ticker, zeros):
        """
        Initialize the DataPrep instance.

        Args:
            ticker (str): The ticker symbol of the equity.
            zeros (pd.DataFrame): DataFrame containing the zero curve data.
        """
        self.ticker = ticker
        self.zero_df = zeros
        self.spot, self.sigma, self.div_yield = self._fetch_equity_stats()

    def _fetch_equity_stats(self):
        """
        Fetch equity-specific statistics such as spot price, annualized volatility, and dividend yield.

        Returns:
            tuple: A tuple containing:
                - spot_price (float): The current spot price of the equity.
                - annualized_volatility (float): The annualized volatility of the equity.
                - dividend_yield (float): The dividend yield of the equity (as a decimal).
        """
        # Fetch the ticker object
        ticker = yf.Ticker(self.ticker)

        # Fetch dividend yield (handle cases where it's not available)
        try:
            div_yield = ticker.info.get('dividendYield', 0) / 100  # Convert percentage to decimal
        except Exception as e:
            print(f"Error fetching dividend yield: {e}. Defaulting to 0.")
            div_yield = 0

        # Fetch historical price data for the last 5 years
        price_data = ticker.history(period='5y')

        # Calculate the spot price (most recent closing price)
        spot_price = price_data['Close'].iloc[-1]

        # Calculate annualized volatility based on daily returns
        daily_returns = price_data['Close'].pct_change().dropna()
        annualized_volatility = daily_returns.std() * np.sqrt(252)

        return spot_price, annualized_volatility, div_yield

    def _ttm(self, expiry, timezone="America/Chicago") -> float:
        """
        Calculate the time to maturity (T) in years, with time zone support.

        Args:
            expiry (str): The expiry date of the option in the format "YYYY-MM-DD".
            timezone (str, optional): The time zone to use for the calculation. Defaults to "America/Chicago".

        Returns:
            float: The time to maturity in years.
        """
        # Localize the current time to the specified timezone
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)

        # Parse the expiry date and set the time to 16:00:00 (4 PM) in the specified time zone
        expire_naive = datetime.strptime(expiry, "%Y-%m-%d").replace(hour=16, minute=0, second=0)
        expire = tz.localize(expire_naive)  # Make it timezone-aware

        # Calculate the difference in seconds and convert to years
        time_diff_seconds = (expire - current_time).total_seconds()
        time_to_maturity_years = time_diff_seconds / (365 * 24 * 3600)

        return time_to_maturity_years

    def _get_risk_free_rate(self, option_maturity):
        """
        Fetch the risk-free rate corresponding to the option's expiry date from the zero curve DataFrame.

        Args:
            option_maturity (str or datetime): The maturity date of the option, either as a string 
                (e.g., "YYYY-MM-DD") or a datetime object.

        Returns:
            float: The risk-free rate corresponding to the option's maturity, expressed as a decimal 
                (e.g., 0.05 for 5%).
        """
        # Convert option_maturity to datetime if it's a string
        if isinstance(option_maturity, str):
            option_maturity = pd.to_datetime(option_maturity)

        # Find the closest maturity date in the zero curve DataFrame
        maturity_differences = (self.zero_df['Maturity'] - option_maturity).abs()
        closest_index = maturity_differences.idxmin()

        # Fetch the corresponding yield and convert from percentage to decimal
        risk_free_rate = self.zero_df['Yield'].iloc[closest_index] / 100

        return risk_free_rate

    def option_chain(self):
        """
        Fetch and process the option chain for the given ticker, ensuring only one expiry per month.

        Returns:
            pd.DataFrame: A DataFrame containing the option chain data with additional columns:
                - expiryDate: The expiry date of the option.
                - Ttm: Time to maturity in years.
                - rf: Risk-free rate corresponding to the option's expiry.
                - Moneyness: The ratio of strike price to spot price.
                - Price: The mid-price of the option (average of bid and ask).
        """
        tic_object = yf.Ticker(self.ticker)
        options = tic_object.options  # Get all available expiry dates

        # Initialize an empty DataFrame to store all option chain data
        all_data = pd.DataFrame()

        # Track months already processed to ensure only one expiry per month
        processed_months = set()

        for expiry in options:
            # Extract the year and month from the expiry date
            expiry_date = pd.to_datetime(expiry)
            year_month = (expiry_date.year, expiry_date.month)

            # Skip if an expiry for this month has already been processed
            if year_month in processed_months:
                continue

            # Mark this month as processed
            processed_months.add(year_month)

            # Fetch option chain data for the current expiry date
            data = tic_object.option_chain(expiry)
            option_chain = data.calls  # Use calls for now (can be extended to puts)

            # Add additional columns to the option chain data
            option_chain['expiryDate'] = expiry
            option_chain['Ttm'] = self._ttm(expiry)  # Time to maturity in years
            option_chain['rf'] = self._get_risk_free_rate(expiry)  # Risk-free rate

            # Append the current option chain to the main DataFrame
            all_data = pd.concat([all_data, option_chain], axis=0, ignore_index=True)

        # Calculate moneyness and mid-price for each option
        all_data['Moneyness'] = all_data['strike'] / self.spot
        all_data['Price'] = (all_data['bid'] + all_data['ask']) / 2

        # Drop contracts with a mid-price of zero
        all_data = all_data[all_data['Price'] > 0]

        return all_data
