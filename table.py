import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from numpy import log, sqrt, exp
import matplotlib.pyplot as plt
import seaborn as sns

#######################
# Page configuration
st.set_page_config(
    page_title="Implied Volatility Surface",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px;
    width: auto;
    margin: 0 auto;
}
.metric-label {
    font-size: 1rem;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)

# Black-Scholes Model for implied volatility calculation
class BlackScholes:
    def __init__(self, S, K, T, r, market_price, option_type='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.market_price = market_price
        self.option_type = option_type

    def d1(self, sigma):
        return (log(self.S / self.K) + (self.r + 0.5 * sigma ** 2) * self.T) / (sigma * sqrt(self.T))

    def d2(self, sigma):
        return self.d1(sigma) - sigma * sqrt(self.T)

    def call_price(self, sigma):
        d1 = self.d1(sigma)
        d2 = self.d2(sigma)
        return self.S * norm.cdf(d1) - self.K * exp(-self.r * self.T) * norm.cdf(d2)

    def put_price(self, sigma):
        d1 = self.d1(sigma)
        d2 = self.d2(sigma)
        return self.K * exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    def implied_volatility(self):
        try:
            if self.option_type == 'call':
                price_func = self.call_price
            else:
                price_func = self.put_price
            implied_vol = brentq(lambda x: price_func(x) - self.market_price, 1e-6, 10)
            return implied_vol
        except Exception:
            return np.nan

# Sidebar for User Inputs
with st.sidebar:
    st.title("📈 Implied Volatility Surface")
    st.write("`Created by:`")
    linkedin_url = "https://www.linkedin.com/in/branden-bahk/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Bahk, Branden`</a>', unsafe_allow_html=True)

    S = st.number_input("Current Asset Price", value=100.0)
    r = st.number_input("Risk-Free Interest Rate", value=0.05)

    st.markdown("---")
    st.write("Option Market Data")
    strikes = st.text_area("Strike Prices (comma-separated)", value="90,100,110")
    maturities = st.text_area("Maturities (Years, comma-separated)", value="0.5,1,1.5")
    option_prices = st.text_area("Option Market Prices (comma-separated, same order as strikes)", value="12,10,8")
    option_type = st.selectbox("Option Type", ["call", "put"])

    strike_list = [float(x) for x in strikes.split(",")]
    maturity_list = [float(x) for x in maturities.split(",")]
    price_list = [float(x) for x in option_prices.split(",")]

# Calculate Implied Volatilities
implied_vols = []
for K in strike_list:
    row = []
    for T, market_price in zip(maturity_list, price_list):
        bs_model = BlackScholes(S, K, T, r, market_price, option_type)
        iv = bs_model.implied_volatility()
        row.append(iv)
    implied_vols.append(row)

# Convert to DataFrame for easier handling
iv_df = pd.DataFrame(implied_vols, index=strike_list, columns=maturity_list)

# Plot Implied Volatility Surface
fig = go.Figure(data=[go.Surface(z=iv_df.values, x=strike_list, y=maturity_list)])
fig.update_layout(title='Implied Volatility Surface', autosize=True,
                  scene=dict(xaxis_title='Strike Price', yaxis_title='Maturity (Years)', zaxis_title='Implied Volatility'),
                  margin=dict(l=65, r=50, b=65, t=90))

st.plotly_chart(fig)

# Display Table of Implied Volatilities
st.markdown("### Implied Volatility Table")
st.write(iv_df)
