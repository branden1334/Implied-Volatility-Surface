import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from numpy import log, sqrt, exp

# Page configuration
st.set_page_config(page_title="Implied Volatility Surface", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# Define the Black-Scholes model
class BlackScholes:
    def __init__(self, time_to_maturity, strike, current_price, volatility, interest_rate):
        self.time_to_maturity = time_to_maturity
        self.strike = strike
        self.current_price = current_price
        self.volatility = volatility
        self.interest_rate = interest_rate

    def calculate_prices(self):
        d1 = (log(self.current_price / self.strike) + (self.interest_rate + 0.5 * self.volatility ** 2) * self.time_to_maturity) / (self.volatility * sqrt(self.time_to_maturity))
        d2 = d1 - self.volatility * sqrt(self.time_to_maturity)
        call_price = self.current_price * norm.cdf(d1) - self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(d2)
        put_price = self.strike * exp(-self.interest_rate * self.time_to_maturity) * norm.cdf(-d2) - self.current_price * norm.cdf(-d1)
        return call_price, put_price

# Sidebar inputs
st.sidebar.title("📊 Implied Volatility Surface")
current_price = st.sidebar.number_input("Current Asset Price", value=100.0)
interest_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05)

# Adding multiple strikes and maturities
strike_prices = st.sidebar.text_input("Strike Prices (comma-separated)", "60, 80, 100, 120, 140")
maturities = st.sidebar.text_input("Maturities (comma-separated, in years)", "0.1, 0.2, 0.4, 0.6, 0.8, 1.0")
strike_prices = [float(x) for x in strike_prices.split(",")]
maturities = [float(x) for x in maturities.split(",")]

# Display inputs
input_data = {
    "Current Asset Price": [current_price],
    "Risk-Free Interest Rate": [interest_rate],
    "Strike Prices": [strike_prices],
    "Maturities": [maturities],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Hypothetical implied volatility function
def implied_volatility(strike, maturity):
    base_vol = 0.2
    vol = base_vol + 0.1 * (strike / current_price - 1) * (1 - maturity)
    return max(vol, 0.01)  # Ensure volatility is not negative

# Calculate implied volatility surface
vol_surface = np.zeros((len(maturities), len(strike_prices)))

for i, maturity in enumerate(maturities):
    for j, strike in enumerate(strike_prices):
        vol_surface[i, j] = implied_volatility(strike, maturity)

# Plot 3D surface
fig = go.Figure(data=[go.Surface(z=vol_surface, x=strike_prices, y=maturities)])
fig.update_layout(title='Implied Volatility Surface', autosize=True,
                  scene=dict(xaxis_title='Strike Price',
                             yaxis_title='Maturity',
                             zaxis_title='Implied Volatility'))

st.plotly_chart(fig, use_container_width=True)
