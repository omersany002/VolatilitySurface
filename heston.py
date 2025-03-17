"""
Module: heston.py
Description:
    This module provides functions to compute option prices using the Heston model. 
    It implements the characteristic function, a numerical integration routine for 
    the pricing formula, and a calibration routine to minimize the pricing error between 
    market prices and model prices. The module is designed to calibrate Heston model parameters 
    given market data and then compute option prices for calls and puts.

Dependencies:
    - scipy.integrate.quad for numerical integration.
    - scipy.optimize.minimize for parameter calibration.
    - numpy for numerical calculations.
    - pandas for data manipulation.

Attribution:
The Heston model pricing implementation is adapted from the discussion on Quant StackExchange 
(https://quant.stackexchange.com/questions/18684/heston-model-option-price-formula) by the user Appliqué.
"""

from scipy.integrate import quad
from scipy.optimize import minimize
import numpy as np


class HestonModel:
    """Heston Model Utilities for Option Pricing and Parameter Calibration."""

    @staticmethod
    def phi(u, T, kappa, theta, sigma, rho, v0):
        """
        Compute the characteristic function component for the Heston model.

        This function calculates the component φ(u) corresponding to the characteristic function 
        of the log-price process under the Heston model.

        Args:
            u (complex or float): The integration variable (can be complex).
            T (float): Time to maturity in years.
            kappa (float): Mean-reversion speed of the variance process.
            theta (float): Long-run average variance.
            sigma (float): Volatility of the variance process.
            rho (float): Correlation between the asset price and its variance.
            v0 (float): Initial variance.

        Returns:
            complex: The computed value of the characteristic function component.
        """
        alpha_hat = -0.5 * u * (u + 1j)
        beta = kappa - 1j * u * sigma * rho
        gamma = 0.5 * sigma**2
        d = np.sqrt(beta**2 - 4 * alpha_hat * gamma)
        g = (beta - d) / (beta + d)
        h = np.exp(-d * T)
        A_ = (beta - d) * T - 2 * np.log((g * h - 1) / (g - 1))
        A = kappa * theta / (sigma**2) * A_
        B = (beta - d) / (sigma**2) * ((1 - h) / (1 - g * h))
        return np.exp(A + B * v0)

    @staticmethod
    def integral(a, T, kappa, theta, sigma, rho, v0):
        """
        Compute the integral part of the Heston pricing formula.

        The integral I is defined as:
            I = ∫₀∞ { Re[ exp((i*u + 0.5)*a) · φ(u - 0.5i, T, kappa, theta, sigma, rho, v0) ]
                        / (u² + 0.25) } du
        where a = ln(S0/K) + (r - q)*T (with q assumed to be 0).

        Args:
            a (float): Parameter defined as ln(S0/K) + (r - q)*T.
            T (float): Time to maturity in years.
            kappa (float): Mean-reversion speed of the variance process.
            theta (float): Long-run average variance.
            sigma (float): Volatility of the variance process.
            rho (float): Correlation between the asset price and its variance.
            v0 (float): Initial variance.

        Returns:
            float: The value of the integral I.
        """
        integrand = lambda u: np.real(
            np.exp((1j * u + 0.5) * a) * HestonModel.phi(u - 0.5j, T, kappa, theta, sigma, rho, v0)
        ) / (u**2 + 0.25)
        I, _ = quad(integrand, 0, np.inf)
        return I

    @staticmethod
    def heston_call_price(S0, K, r, T, kappa, theta, sigma, rho, v0, opt_type='C'):
        """
        Compute the option price using the Heston model's single-integral formula.

        The pricing formula is given by:
            a = ln(S0/K) + (r - q)*T    (with q=0 assumed)
            Call = S0 - (K*exp(-r*T)/π)*I
        For a put option, the price is adjusted via put-call parity.

        Args:
            S0 (float): Current spot price of the underlying asset.
            K (float): Strike price of the option.
            r (float): Risk-free rate (as a decimal).
            T (float): Time to maturity in years.
            kappa (float): Mean-reversion speed of the variance process.
            theta (float): Long-run average variance.
            sigma (float): Volatility of the variance process.
            rho (float): Correlation between the asset price and its variance.
            v0 (float): Initial variance.
            opt_type (str, optional): Option type, 'C' for call and 'P' for put. Defaults to 'C'.

        Returns:
            float: The computed option price.
        """
        # Here, q (dividend yield) is assumed to be 0.
        a = np.log(S0 / K) + r * T  
        I = HestonModel.integral(a, T, kappa, theta, sigma, rho, v0)
        price = S0 - (K * np.exp(-r * T) / np.pi) * I
        if opt_type == 'P':
            # Adjust price for put option via put-call parity
            price = price - S0 + K * np.exp(-r * T)
        return price

    @staticmethod
    def loss_function(params, market_prices, strikes, spot, ttm, rf):
        """
        Compute the sum of squared errors between market prices and Heston model prices.

        Args:
            params (list): Heston model parameters [kappa, theta, sigma, rho, v0].
            market_prices (array-like): Observed market prices for the options.
            strikes (array-like): Strike prices corresponding to the market prices.
            spot (float): Current spot price of the underlying asset.
            ttm (float): Time to maturity in years.
            rf (float): Risk-free rate (as a decimal).

        Returns:
            float: The sum of squared errors.
        """
        kappa, theta, sigma, rho, v0 = params

        # Compute model call prices for each strike
        model_prices = np.array([
            HestonModel.heston_call_price(spot, K, rf, ttm, kappa, theta, sigma, rho, v0)
            for K in strikes
        ])

        error = np.sum((market_prices - model_prices) ** 2)
        return error

    @staticmethod
    def minimize_function(market_prices, strikes, long_iv, bs_iv, spot, ttm, rf):
        """
        Calibrate Heston model parameters for a single expiration by minimizing pricing error.

        Uses the L-BFGS-B algorithm to minimize the loss function over the parameter space.

        Args:
            market_prices (array-like): Observed market option prices.
            strikes (array-like): Option strike prices.
            long_iv (float): Long-term variance assumption (squared IV).
            bs_iv (float): Average squared Black-Scholes implied volatility.
            spot (float): Current spot price.
            ttm (float): Time to maturity in years.
            rf (float): Risk-free rate (as a decimal).

        Returns:
            array: The optimized parameters [kappa, theta, sigma, rho, v0].
        """
        guess = [2, long_iv, 0.2, -0.7, bs_iv]
        bounds = [
            (1.0, 3.0),    # kappa
            (0.01, 0.5),   # theta
            (0.1, 0.5),    # sigma
            (-0.99, -0.01),# rho
            (0.01, 0.5)    # v0
        ]

        result = minimize(HestonModel.loss_function, 
                          x0=guess, 
                          args=(market_prices, strikes, spot, ttm, rf), 
                          bounds=bounds, 
                          method='L-BFGS-B')

        return result.x
