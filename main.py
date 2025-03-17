"""
Volatility Surface and Pricing Method Analysis
------------------------------------------------
This script:
1. Collects option chain data for a specified ticker.
2. Fits the Black-Scholes implied volatility.
3. Computes option prices using Black-Scholes.
4. Calibrates the Heston model parameters using parallel processing.
5. Computes Heston model prices and implied volatilities.
6. Plots and compares volatility surfaces.
7. Computes and prints RMSE for both models.
"""

# =============================================================================
# Imports
# =============================================================================
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from black_scholes import BlackScholes
from heston import HestonModel
from yield_curve import zeros
from data_fetch import DataPrep


# =============================================================================
# Main Workflow
# =============================================================================
def main():
    """
    Main function for executing the volatility surface and pricing analysis.

    This function performs the following tasks:
      - Prepares option chain data for the specified ticker.
      - Calibrates the Black-Scholes implied volatility.
      - Prices options using both the Black-Scholes and Heston models.
      - Cleans the data by filtering out out-of-range moneyness and IV values.
      - Calibrates Heston model parameters using parallel processing.
      - Plots model fits and volatility surfaces.
      - Computes and prints RMSE for both models.
    """
    # Data Preparation
    TICKER = 'SPY'  # Change this ticker as needed.
    ticker_data = DataPrep(TICKER, zeros)
    options_df = ticker_data.option_chain()

    # =============================================================================
    # Black-Scholes Model Calibration
    # =============================================================================
    # Compute Black-Scholes implied volatility for each option
    options_df['bs_iv'] = options_df.apply(
        lambda x: BlackScholes.iv(
            ticker_data.spot,
            x['Price'],
            x['strike'],
            x['rf'],
            x['Ttm']
        ),
        axis=1
    )

    # Compute Black-Scholes option price using the calibrated implied volatility
    options_df['bs_price'] = options_df.apply(
        lambda x: BlackScholes.price(
            ticker_data.spot,
            x['strike'],
            x['rf'],
            ticker_data.div_yield,
            x['bs_iv'],
            x['Ttm'],
            opt_type='C'
        ),
        axis=1
    )

    # =============================================================================
    # Data Cleaning
    # =============================================================================
    # Remove options with moneyness outside [0.9, 1.1]
    options_df = options_df[(options_df['Moneyness'] >= 0.9) & (options_df['Moneyness'] <= 1.1)]
    # Remove options with unrealistic Black-Scholes IV values (<= 0 or > 3)
    options_df = options_df[(options_df['bs_iv'] >= 0.0) & (options_df['bs_iv'] <= 3)]

    # =============================================================================
    # Plotting Black-Scholes Implied Volatility Fit
    # =============================================================================
    # Select one expiry date for plotting (6th unique expiry date)
    expiry_to_plot = options_df['expiryDate'].unique()[6]
    df_to_plot = options_df[options_df['expiryDate'] == expiry_to_plot]

    # Plot and save the Black-Scholes IV fit
    BlackScholes.plot_iv(TICKER, df_to_plot, 'bs_iv', expiry_to_plot, save_path='plots/bs_iv_fit')

    # =============================================================================
    # Heston Model Calibration Function
    # =============================================================================
    def parameter_calibration(df, spot):
        """
        Calibrate Heston model parameters using parallel processing.
        
        Parameters:
            df (pd.DataFrame): DataFrame containing option data.
            spot (float): The current spot price of the underlying asset.
        
        Returns:
            pd.DataFrame: Merged DataFrame with calibrated Heston parameters.
        """

        def calibrate_group(ttm, group, spot):
            """
            Calibrate parameters for a single maturity group.

            Args:
                ttm (float): Time to maturity.
                group (pd.DataFrame): Data for options with the same maturity.
                spot (float): Current spot price.
            
            Returns:
                list: A list containing Ttm, kappa, theta, sigma, rho, and V0.
            """
            # Ensure group is a DataFrame
            if not isinstance(group, pd.DataFrame):
                raise TypeError(f"Expected a DataFrame, got {type(group)}")

            market_prices = group["Price"].values
            strikes = group["strike"].values
            bs_iv = group["bs_iv"].values ** 2  # Convert IV to variance
            rf_rate = group["rf"].values[0]  # Assuming constant risk-free rate per expiry

            # Maximum IV used for long-term volatility assumption
            max_iv = group["bs_iv"].max() ** 2

            # Calibrate Heston parameters for this maturity
            calibrated_params = HestonModel.minimize_function(
                market_prices, strikes, max_iv, bs_iv.mean() ** 2, spot, ttm, rf_rate
            )

            return [ttm] + list(calibrated_params)

        # Group options by Time-to-Maturity (Ttm)
        grouped = df.groupby('Ttm')
        # Parallel calibration of parameters for each group
        params_list = Parallel(n_jobs=-1)(
            delayed(calibrate_group)(ttm, group, spot) for ttm, group in grouped
        )
        # Create a DataFrame from calibration results
        calib_df = pd.DataFrame(params_list, columns=["Ttm", "kappa", "theta", "sigma", "rho", "V0"])
        # Merge the calibrated parameters back with the original options DataFrame
        options = df.merge(calib_df, on="Ttm", how="left")
        return options

    # Calibrate Heston parameters and update the options DataFrame
    options_df = parameter_calibration(options_df, ticker_data.spot)

    # =============================================================================
    # Heston Model Pricing and IV Calculation
    # =============================================================================
    # Compute Heston model option price using the calibrated parameters
    options_df['heston_price'] = options_df.apply(
        lambda x: HestonModel.heston_call_price(
            ticker_data.spot,
            x['strike'],
            x['rf'],
            x['Ttm'],
            x['kappa'],
            x['theta'],
            x['sigma'],
            x['rho'],
            x['V0']
        ),
        axis=1
    )

    # Derive Heston implied volatility by inverting the Black-Scholes formula
    options_df['heston_iv'] = options_df.apply(
        lambda x: BlackScholes.iv(
            ticker_data.spot,
            x['heston_price'],
            x['strike'],
            x['rf'],
            x['Ttm']
        ),
        axis=1
    )

    # =============================================================================
    # Plotting Heston Implied Volatility Fit
    # =============================================================================
    # Use the same expiry date as before for consistency in plotting
    df_to_plot = options_df[options_df['expiryDate'] == expiry_to_plot]
    BlackScholes.plot_iv(TICKER, df_to_plot, 'heston_iv', expiry_to_plot, 'Heston', 'plots/heston_iv_fit')

    # =============================================================================
    # Model Performance Metrics
    # =============================================================================
    # Calculate Root Mean Squared Error (RMSE) for both models
    bs_rmse = np.sqrt(((options_df['bs_iv'] - options_df['impliedVolatility']) ** 2).mean())
    h_rmse = np.sqrt(((options_df['heston_iv'] - options_df['impliedVolatility']) ** 2).mean())

    # Print RMSE results in a formatted table
    print("Model Comparison (RMSE):")
    print(f"{'Model':<15} {'RMSE':<10}")
    print("-" * 25)
    print(f"{'Black-Scholes':<15} {bs_rmse:.6f}")
    print(f"{'Heston':<15} {h_rmse:.6f}")

    # =============================================================================
    # Plotting Volatility Surfaces
    # =============================================================================
    # Market Implied Volatility Surface
    BlackScholes.plot_volatility_surface(options_df, 'impliedVolatility', 'Market Implied Volatility Surface', 'plots/market_iv_surface.html')

    # Black-Scholes Implied Volatility Surface
    BlackScholes.plot_volatility_surface(options_df, 'bs_iv', 'Black and Scholes Implied Volatility Surface', 'plots/bs_iv_surface.html')

    # Heston Model Implied Volatility Surface
    BlackScholes.plot_volatility_surface(options_df, 'heston_iv', 'Heston Model Implied Volatility Surface', 'plots/heston_iv_surface.html')


if __name__ == '__main__':
    main()
