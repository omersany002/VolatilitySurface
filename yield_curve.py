"""
Module: yield_curve.py
Assignment 4

Description:
    This module bootstraps a zero-coupon yield curve from coupon-paying bonds using 
    treasury bills and bonds data. It calculates discount factors, zero yields, and forward rates,
    and provides plotting functions to visualize the yield and forward curves.

Treasury Data Requirements:
    - This module requires two files containing bonds and bills data.
    - The data should be copied from the website:
        https://www.wsj.com/market-data/bonds/treasuries
    - After obtaining the data from the website, the files must be renamed to include the 
      date the data was obtained (e.g., "bills2025-03-16.txt" and "bonds2025-03-16.txt").
    - These files are expected to be in a tab-separated format with the required fields.

Attribution:
    The yield curve module was adapted from code provided by Dr. Louis R. Piccotti,
    Associate Professor of Finance, Spears School of Business, Oklahoma State University.
"""

import warnings
warnings.simplefilter("ignore")

from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

# Define paths for treasury bills and bonds data files
bills_path = r'treasuries\bills2025-03-16.txt'
bonds_path = r'treasuries\bonds2025-03-16.txt'


class YieldCurve:
    """
    Class to bootstrap a zero-coupon yield curve and generate yield and forward rate curves.

    Attributes:
        bill_path (str): File path for the treasury bills data.
        bond_path (str): File path for the bonds data.
        yrlen (int): Number of days in a year (default is 365).
        today (pd.Timestamp): The reference date extracted from the bills file path.
        bills (pd.DataFrame): Formatted treasury bills data.
        bonds (pd.DataFrame): Formatted bonds data with bootstrapped zero prices and yields.
        term (pd.DataFrame): Combined term structure of coupon yields.
        zeros (pd.DataFrame): Combined zero-coupon yield curve data.
    """

    def __init__(self, bill_path, bond_path, yrlen=365):
        """
        Initialize the YieldCurve instance.

        Args:
            bill_path (str): File path for the treasury bills data.
            bond_path (str): File path for the bonds data.
            yrlen (int, optional): Number of days in a year. Defaults to 365.
        """
        self.bill_path = bill_path
        self.bond_path = bond_path
        self.yrlen = yrlen

        # Extract the reference date from the file name
        self.today = pd.to_datetime(bill_path[len(bill_path)-14:len(bill_path)-4])

        # Format the treasury bills and bonds data
        self.bills = self.format_treasuries(self.bill_path, self.today)
        self.bonds = self.format_treasuries(self.bond_path, self.today)

        # Calculate treasury bill prices
        self.bills['Price'] = 1.0 / (1.0 + (self.bills['Asked yield'] / 100) * self.bills['Ttm'])

        # Append zeros from bills and bonds data
        self.term, self.zeros = self.appending_zeros()

    @staticmethod
    def format_treasuries(path, today, yrlen=365):
        """
        Read and format treasury data from a given file.

        The function reads the data, drops rows with missing 'Asked' values,
        adjusts prices for 32 notation, converts maturity dates, and calculates time to maturity.

        Args:
            path (str): File path for the treasury data.
            today (pd.Timestamp): The reference date for calculating time to maturity.
            yrlen (int, optional): Number of days in a year. Defaults to 365.

        Returns:
            pd.DataFrame: Formatted treasury data with a 'Ttm' column representing time to maturity.
        """
        df = pd.read_csv(path, sep='\t')

        # Replace 'n.a.' with NaN and drop rows with NaN in 'Asked'
        df['Asked'] = df['Asked'].replace('n.a.', np.nan)
        count = df['Asked'].isna().sum()
        print(f'{count} rows dropped for nan values')
        df = df.dropna()

        # Adjust for 32 notation
        numeric_asked = pd.to_numeric(df.Asked)
        integer_part, fractional_part = divmod(numeric_asked, 1)
        df.Asked = integer_part + fractional_part * 100 / 32

        # Convert maturity to datetime and ensure numeric types for yields
        df.Maturity = pd.to_datetime(df.Maturity)
        df['Asked yield'] = pd.to_numeric(df['Asked yield'])

        # Filter to drop consecutive securities with the same maturity date
        df = df[df.Maturity != df.Maturity.shift(1)]
        df.index = np.arange(1, len(df) + 1)
        df['Ttm'] = pd.to_numeric((df.Maturity - today) / dt.timedelta(yrlen))
        return df

    def bootstrap_zero_curve(self):
        """
        Bootstrap a zero-coupon yield curve from coupon-paying bonds.

        This method calculates a zero price for each bond by stripping coupons 
        using the closest matching treasury bill or bond price, and then computes
        the implied zero yield.

        Returns:
            pd.DataFrame: The bonds DataFrame with additional columns 'ZeroPrice' and 'ZeroYield'.
        """
        bonds = self.bonds[self.bonds.Ttm > 1]
        bonds.index = np.arange(1, len(bonds) + 1)

        bonds['ZeroPrice'] = pd.to_numeric(bonds.Asked) / 100  # Set the quoted price
        bonds.Coupon = pd.to_numeric(bonds.Coupon)  # Convert coupons to numeric
        i = 1  # Start with the first observation

        while i <= len(bonds):
            s = np.floor(pd.to_numeric((bonds.Maturity[i] - self.today) / dt.timedelta(self.yrlen)) * 2)
            while ((bonds.Maturity[i] - relativedelta(months=int(s * 6)) > self.today) &
                   (bonds.Maturity[i] - relativedelta(months=int(s * 6)) < bonds.Maturity[i])):
                # Calculate the coupon date
                cpndate = bonds.Maturity[i] - relativedelta(months=int(s * 6))
                # Determine the closest price for the coupon stripping
                if pd.to_numeric((cpndate - self.today) / dt.timedelta(self.yrlen)) < 1:
                    absdif = abs(self.bills.Maturity - cpndate)
                    df_val = self.bills.Price[absdif.idxmin()]
                else:
                    absdif = abs(bonds.Maturity - cpndate)
                    df_val = bonds.ZeroPrice[absdif.idxmin()]
                # Adjust ZeroPrice by stripping coupon and adding accrued interest
                if s == np.floor(pd.to_numeric((bonds.Maturity[i] - self.today) / dt.timedelta(self.yrlen)) * 2):
                    bonds.ZeroPrice[i] = bonds.ZeroPrice[i] + ((bonds.Coupon[i] / 100) / 2) * (1 - pd.to_numeric((cpndate - self.today) / dt.timedelta(30 * 6)))
                bonds.ZeroPrice[i] = bonds.ZeroPrice[i] - ((bonds.Coupon[i] / 100) / 2) * df_val
                s = s - 1
            bonds.ZeroPrice[i] = bonds.ZeroPrice[i] / (1 + ((bonds.Coupon[i] / 100) / 2))
            # Correct for numerical errors leading to jumps in the zero yield
            if i > 1 and (bonds.ZeroPrice[i] / bonds.ZeroPrice[i - 1] - 1) > 0.01:
                bonds.ZeroPrice[i] = 1 / ((1 + 1 / (bonds.ZeroPrice[i - 1] ** (1 / bonds.Ttm[i - 1])) - 1) ** bonds.Ttm[i])
            i = i + 1

        # Calculate the zero yield implied by the bootstrapped zero price
        bonds['ZeroYield'] = (1 / (bonds.ZeroPrice ** (1 / bonds.Ttm)) - 1) * 100
        return bonds

    def appending_zeros(self):
        """
        Append treasury bills and bootstrapped bonds data to form the complete zero-coupon yield curve.

        This method combines the treasury bills yields and bootstrapped bond zero yields,
        calculates moving averages, forward rates, and a polynomial fit for forward rates.

        Returns:
            tuple: A tuple containing:
                - term (pd.DataFrame): Combined coupon yield data with maturities.
                - zeros (pd.DataFrame): Combined zero-coupon yield data with additional computed columns.
        """
        self.bonds = self.bootstrap_zero_curve()

        term = pd.DataFrame((self.bills['Asked yield'])._append(self.bonds['Asked yield']))
        term['Maturity'] = (self.bills.Maturity)._append(self.bonds.Maturity)
        term.index = np.arange(1, len(term) + 1)

        zeros = pd.DataFrame((self.bills['Asked yield'])._append(self.bonds['ZeroYield']))
        zeros.columns = ['Yield']
        zeros['Price'] = (self.bills['Price'])._append(self.bonds['ZeroPrice'])
        zeros['Maturity'] = (self.bills['Maturity'])._append(self.bonds['Maturity'])
        zeros.index = np.arange(1, len(zeros) + 1)

        # Construct a 12-month centered moving average of yields
        zeros['MA'] = zeros.Yield.rolling(window=12, center=True, min_periods=0).mean()

        # Compute forward yields
        zeros["Fwrd"] = zeros.Yield
        i = 1
        while i <= len(zeros) - 1:
            ft = zeros.Maturity[i]
            fs = zeros.Maturity[i] + relativedelta(months=3)
            tau = pd.to_numeric((fs - ft) / dt.timedelta(self.yrlen))
            dif = pd.to_numeric(zeros.Maturity - fs)
            absdifs = abs(zeros.Maturity - fs)
            sgn = np.sign(dif[absdifs.idxmin()])
            if sgn == -1:
                ps = zeros.Price[absdifs.idxmin()] + (fs - zeros.Maturity[absdifs.idxmin()]) / (zeros.Maturity[absdifs.idxmin() + 1] - zeros.Maturity[absdifs.idxmin()]) * (zeros.Price[absdifs.idxmin() + 1] - zeros.Price[absdifs.idxmin()])
            if sgn == 1:
                ps = zeros.Price[absdifs.idxmin() - 1] + (fs - zeros.Maturity[absdifs.idxmin() - 1]) / (zeros.Maturity[absdifs.idxmin()] - zeros.Maturity[absdifs.idxmin() - 1]) * (zeros.Price[absdifs.idxmin()] - zeros.Price[absdifs.idxmin() - 1])
            if sgn == 0:
                ps = zeros.Price[absdifs.idxmin()]
            zeros.Fwrd[i] = (1 / tau) * (zeros.Price[i] / ps - 1) * 100
            if i == len(zeros) - 1:
                zeros.Fwrd[i + 1] = zeros.Fwrd[i]
            i = i + 1

        zeros['FwrdMA'] = zeros.Fwrd.rolling(window=6, center=True, min_periods=0).mean()  # Centered moving average for forward rates
        zeros['Ttm'] = pd.to_numeric((zeros.Maturity - self.today) / dt.timedelta(self.yrlen))  # Time to maturity in years

        # Polynomial fit of degree 9 for forward rates as a function of time to maturity
        PlyFit = np.polyfit(zeros.Ttm, zeros.Fwrd, 9)
        zeros['PlyFit'] = np.polyval(PlyFit, zeros.Ttm)

        return term, zeros

    def yield_curve(self):
        """
        Plot the yield curve and the term structure of discount factors.

        Generates two plots:
            1. Yield curve plot comparing coupon yields and bootstrapped zero yields.
            2. Term structure plot for discount factors from treasury bills and bonds.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.term.Maturity, self.term['Asked yield'], color='blue')
        plt.plot(self.zeros.Maturity, self.zeros.Yield, color="red")
        plt.ylim(0., 7.)
        plt.title("Yield curve" + " (" + str(self.today.date()) + ")")
        plt.xlabel("Maturity date")
        plt.ylabel("Interest rate")
        plt.gca().legend(('coupon', 'bootstrapped zero'), loc='lower right')
        plt.show()

        plt.plot(self.bills.Maturity[1:len(self.bills)], self.bills.Price[1:len(self.bills)],
                 self.bonds.Maturity[1:len(self.bonds)], self.bonds.ZeroPrice[1:len(self.bonds)],
                 color="black")
        plt.title("Term structure of discount factors" + " (" + str(self.today.date()) + ")")
        plt.xlabel("Maturity date")
        plt.ylabel("Discount factor")
        plt.show()

    def forward_curve(self):
        """
        Plot the forward rate curve along with a polynomial fit and moving average.

        Generates a plot that compares:
            - Coupon term structure yields.
            - Bootstrapped zero rates.
            - Polynomial fit of forward rates.
            - Centered moving average of forward rates.
        """
        plt.plot(self.term.Maturity, self.term['Asked yield'], color='blue')
        plt.plot(self.zeros.Maturity, self.zeros.Yield, color="red")
        plt.plot(self.zeros.Maturity, np.polyval(np.polyfit(self.zeros.Ttm, self.zeros.Fwrd, 9), self.zeros.Ttm), color="green")
        plt.plot(self.zeros.Maturity, self.zeros.FwrdMA, color="orange", linewidth=0.5)
        plt.ylim(-3., 10.)
        plt.title("Forward rate curve" + " (" + str(self.today.date()) + ")")
        plt.xlabel("Maturity date")
        plt.ylabel("Interest rate")
        plt.gca().legend(('coupon term structure', 'bootstrapped zero rates', 'P[F(t:T,T+3mo)], deg(P)=9',
                           'F(t,T,T+3mo), centered MA(6)'), loc='best')
        plt.show()


# Create an instance of the YieldCurve class and extract zeros data
yieldModel = YieldCurve(bills_path, bonds_path)
zeros = yieldModel.zeros
