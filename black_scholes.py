"""
Module: black_scholes.py
Description:
    This module provides utilities for pricing options and computing implied volatility 
    using the Black-Scholes model. It includes functions for calculating option prices, 
    Vega (sensitivity to volatility), implied volatility via the Newton-Raphson method, 
    and plotting functions for comparing market implied volatilities against model 
    generated ones and for visualizing the volatility surface.
    
Dependencies:
    - scipy.stats for the normal distribution functions.
    - scipy.interpolate for grid data interpolation.
    - numpy for numerical calculations.
    - plotly.graph_objects for interactive 3D plots.
    - matplotlib.pyplot for static plotting.
    
Usage:
    The BlackScholes class methods can be called directly as static methods. For example,
    to price an option or compute its implied volatility, simply use:
        price = BlackScholes.price(spot, strike, rf, div_yield, sigma, ttm, opt_type)
        implied_vol = BlackScholes.iv(spot, market_price, strike, rf, ttm, div_yield, opt_type)
"""

from scipy.stats import norm
from scipy.interpolate import griddata
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt


class BlackScholes:
    """Black-Scholes utilities for option pricing and volatility analysis."""

    @staticmethod
    def price(spot, strike, rf, div_yield, sigma, ttm, opt_type):
        """
        Calculate the option price using the Black-Scholes model.

        Args:
            spot (float): The current spot price of the underlying asset.
            strike (float): The strike price of the option.
            rf (float): The risk-free rate (as a decimal).
            div_yield (float): The dividend yield of the underlying asset (as a decimal).
            sigma (float): The annualized volatility of the underlying asset (as a decimal).
            ttm (float): The time to maturity in years.
            opt_type (str): The type of option, either "C" for call or "P" for put.

        Returns:
            float: The Black-Scholes price of the option.
        """
        # Calculate d1 and d2
        d1 = (np.log(spot / strike) + (rf - div_yield + sigma**2 / 2) * ttm) / (sigma * np.sqrt(ttm))
        d2 = d1 - sigma * np.sqrt(ttm)

        # Calculate the option price based on the type
        if opt_type[0].upper() == "C":  # Call option
            price = (spot * np.exp(-div_yield * ttm) * norm.cdf(d1) -
                     strike * np.exp(-rf * ttm) * norm.cdf(d2))
        elif opt_type[0].upper() == "P":  # Put option
            price = (strike * np.exp(-rf * ttm) * norm.cdf(-d2) -
                     spot * np.exp(-div_yield * ttm) * norm.cdf(-d1))
        else:
            raise ValueError(f"Invalid option type: {opt_type}. Use 'C' for call or 'P' for put.")

        return price

    @staticmethod
    def vega(spot, strike, rf, sigma, capt):
        """
        Calculate the Vega of the Black-Scholes model.

        Vega measures the sensitivity of the option price to changes in volatility.

        Args:
            spot (float): The current spot price of the underlying asset.
            strike (float): The strike price of the option.
            rf (float): The risk-free rate (as a decimal).
            sigma (float): The annualized volatility of the underlying asset (as a decimal).
            capt (float): The time to maturity in years.

        Returns:
            float: The Vega of the option.
        """
        # Calculate d1
        d1 = (np.log(spot / strike) + (rf + sigma**2 / 2) * capt) / (sigma * np.sqrt(capt))
        # Calculate the probability density function (PDF) of d1
        d1_pdf = norm.pdf(d1)
        # Calculate Vega
        vega = spot * np.sqrt(capt) * d1_pdf
        return vega

    @staticmethod
    def iv(spot, market_price, strike, rf, ttm, div_yield=0, opt_type="C", n_iter=400, tol=0.01):
        """
        Compute the implied volatility using the Black-Scholes model via the Newton-Raphson method.

        Args:
            spot (float): The current spot price of the underlying asset.
            market_price (float): The observed market price of the option.
            strike (float): The strike price of the option.
            rf (float): The risk-free rate (as a decimal).
            ttm (float): The time to maturity in years.
            div_yield (float, optional): The dividend yield of the underlying asset (as a decimal). Defaults to 0.
            opt_type (str, optional): The type of option, either "C" for call or "P" for put. Defaults to "C".
            n_iter (int, optional): The maximum number of iterations. Defaults to 400.
            tol (float, optional): The tolerance for convergence. Defaults to 0.01.

        Returns:
            float: The implied volatility (as a decimal) or np.nan if convergence fails.
        """
        sigma = 0.5  # Initial guess for volatility

        for _ in range(n_iter):
            # Calculate the Black-Scholes price and Vega
            cbs = BlackScholes.price(spot, strike, rf, div_yield, sigma, ttm, opt_type)
            vega_val = BlackScholes.vega(spot, strike, rf, sigma, ttm)
            # Calculate the error (difference between model price and market price)
            error = cbs - market_price
            # Check for convergence
            if abs(error) <= tol:
                return sigma
            # Update volatility using Newton-Raphson
            sigma = sigma - (error / vega_val)

        # If the loop ends without convergence, return np.nan
        return np.nan

    @staticmethod
    def plot_iv(ticker, df, model_iv, expiration, title='Black and Scholes', save_path=None):
        """
        Plot the market implied volatility (IV) and the model-fitted IV for a given DataFrame.

        Args:
            ticker (str): The ticker symbol of the underlying asset.
            df (pd.DataFrame): The DataFrame containing the data to plot. Must include 'Moneyness' and 'impliedVolatility' columns.
            model_iv (str): The column name for the model-implied volatility (e.g., 'BS_IV').
            expiration (str): The expiration date of the options being plotted.
            title (str, optional): The title of the plot (e.g., "Black-Scholes"). Defaults to 'Black and Scholes'.
            save_path (str, optional): Path to save the plot. If None, the plot is only displayed.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(df['Moneyness'], df['impliedVolatility'],
                 color='deepskyblue', linewidth=2, label='Market IV')
        plt.plot(df['Moneyness'], df[model_iv],
                 linestyle='--', color='orange', linewidth=2, label=f'{title} IV')
        plt.title(f"Market IV vs {title} Fitted IV \nTicker: {ticker}, Expiration: {expiration}",
                  loc='left', pad=15)
        plt.xlabel("Moneyness")
        plt.ylabel("IV")
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.legend(loc='upper right', frameon=False)
        # Light border style
        plt.gca().spines[['top', 'right', 'left']].set_visible(False)
        plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        plt.show()

    @staticmethod
    def plot_volatility_surface(df, iv_column, title, save_path=None):
        """
        Plot a 3D volatility surface for the given DataFrame and implied volatility column.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to plot. Must include 'Ttm' and 'strike' columns.
            iv_column (str): The column name for the implied volatility (e.g., 'BS_IV' or 'F_IV').
            title (str): The title of the plot (e.g., "Black-Scholes IV Surface").
            save_path (str, optional): Path to save the interactive HTML file. If None, the plot is displayed without saving.
        """
        # Extract data
        X = df['Ttm'].values  # Time to maturity
        Y = df['strike'].values  # Strike prices
        Z = df[iv_column].values * 100  # Implied volatility (scaled to percentage)

        # Create a grid for interpolation
        xi = np.linspace(X.min(), X.max(), 50)
        yi = np.linspace(Y.min(), Y.max(), 50)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((X, Y), Z, (xi, yi), method='linear')

        # Create the 3D surface plot
        fig = go.Figure(data=[go.Surface(z=zi, x=xi, y=yi, colorscale='Viridis')])

        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Time to Expiry',
                yaxis_title='Strike Price',
                zaxis_title='Implied Volatility (%)'
            ),
            autosize=False,
            width=800,
            height=600,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        # Save the plot as an HTML file if a path is provided
        if save_path:
            fig.write_html(save_path)
            print(f"Plot saved to {save_path}")

        fig.show()
