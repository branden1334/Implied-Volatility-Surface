import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from numpy import log, sqrt, exp
import seaborn as sns
import matplotlib.pyplot as plt

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
time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=1.0)
volatility = st.sidebar.number_input("Volatility (σ)", value=0.2)
interest_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05)

# Adding multiple strikes and maturities
strike_prices = st.sidebar.text_input("Strike Prices (comma-separated)", "90, 100, 110")
strike_prices = [float(x) for x in strike_prices.split(",")]

# Display inputs
input_data = {
    "Current Asset Price": [current_price],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [volatility],
    "Risk-Free Interest Rate": [interest_rate],
    "Strike Prices": [strike_prices],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate and display option prices
bs_models = [BlackScholes(time_to_maturity, strike, current_price, volatility, interest_rate) for strike in strike_prices]
prices = [bs_model.calculate_prices() for bs_model in bs_models]
call_prices, put_prices = zip(*prices)
call_prices = np.array(call_prices)
put_prices = np.array(put_prices)

# Display prices
for i, strike in enumerate(strike_prices):
    st.write(f"Strike Price: {strike}")
    st.write(f"Call Price: ${call_prices[i]:.2f}")
    st.write(f"Put Price: ${put_prices[i]:.2f}")

# Interactive heatmap for implied volatility
spot_min = st.sidebar.number_input('Min Spot Price', min_value=0.01, value=current_price*0.8, step=0.01)
spot_max = st.sidebar.number_input('Max Spot Price', min_value=0.01, value=current_price*1.2, step=0.01)
vol_min = st.sidebar.slider('Min Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*0.5, step=0.01)
vol_max = st.sidebar.slider('Max Volatility for Heatmap', min_value=0.01, max_value=1.0, value=volatility*1.5, step=0.01)

spot_range = np.linspace(spot_min, spot_max, 50)
vol_range = np.linspace(vol_min, vol_max, 50)

# Plot heatmap
def plot_heatmap(bs_models, spot_range, vol_range, strike_prices):
    call_prices = np.zeros((len(vol_range), len(spot_range), len(strike_prices)))
    put_prices = np.zeros((len(vol_range), len(spot_range), len(strike_prices)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            for k, bs_model in enumerate(bs_models):
                bs_temp = BlackScholes(
                    time_to_maturity=bs_model.time_to_maturity,
                    strike=strike_prices[k],
                    current_price=spot,
                    volatility=vol,
                    interest_rate=bs_model.interest_rate
                )
                call_price, put_price = bs_temp.calculate_prices()
                call_prices[i, j, k] = call_price
                put_prices[i, j, k] = put_price
    
    fig_call, axes_call = plt.subplots(1, len(strike_prices), figsize=(15, 8))
    fig_put, axes_put = plt.subplots(1, len(strike_prices), figsize=(15, 8))

    for k, strike in enumerate(strike_prices):
        sns.heatmap(call_prices[:, :, k], xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=False, fmt=".2f", cmap="viridis", ax=axes_call[k])
        axes_call[k].set_title(f'CALL (Strike={strike})')
        axes_call[k].set_xlabel('Spot Price')
        axes_call[k].set_ylabel('Volatility')
        
        sns.heatmap(put_prices[:, :, k], xticklabels=np.round(spot_range, 2), yticklabels=np.round(vol_range, 2), annot=False, fmt=".2f", cmap="viridis", ax=axes_put[k])
        axes_put[k].set_title(f'PUT (Strike={strike})')
        axes_put[k].set_xlabel('Spot Price')
        axes_put[k].set_ylabel('Volatility')
    
    return fig_call, fig_put

# Main Page
st.title("Implied Volatility Surface")

# Heatmap
col1, col2 = st.columns(2)

with col1:
    st.subheader("Call Price Heatmap")
    heatmap_fig_call, _ = plot_heatmap(bs_models, spot_range, vol_range, strike_prices)
    st.pyplot(heatmap_fig_call)

with col2:
    st.subheader("Put Price Heatmap")
    _, heatmap_fig_put = plot_heatmap(bs_models, spot_range, vol_range, strike_prices)
    st.pyplot(heatmap_fig_put)
