import math
from scipy.stats import norm
import streamlit as st

def bsm_option_pricer(S, K, r, q, sigma, T):
    """
    Calculate the price of a European call or put option using the Black-Scholes-Merton formula.
    
        args:
            S: stock price
            K: strike price
            r: risk-free rate
            q: dividend yield
            sigma: volatility
            T: time to expiration
            
        Output:
            call_price: call option price
            put_price: put option price
    """

    d1 = calc_d1(S, K, r, q, sigma, T)
    d2 = calc_d2(d1, sigma, T)

    call_price = calc_call_price(S, K, r, q, T, d1, d2)
    call_delta = calc_delta_call(q, T, d1)
    call_theta = calc_theta_call(S, K, r, q, sigma, T, d1, d2)
    call_rho = calc_rho_call(K, r, T, d2)
    
    put_price = calc_put_price(S, K, r, q, T, d1, d2)
    put_delta = calc_delta_put(q, T, d1)
    put_theta = calc_theta_put(S, K, r, q, sigma, T, d1, d2)
    put_rho = calc_rho_put(K, r, T, d2)
    
    gamma = calc_gamma(S, q, sigma, T, d1)
    vega = calc_vega(S, q, T, d1)
    
    prices = {
        "call_price": call_price,
        "call_delta": call_delta,
        "call_theta": call_theta,
        "call_rho": call_rho,
        "put_price": put_price,
        "put_delta": put_delta,
        "put_theta": put_theta,
        "put_rho": put_rho,
        "gamma": gamma,
        "vega": vega
    }
    
    return prices


def calc_call_price(S, K, r, q, T, d1, d2):
    """
    Calculate the price of a European call option using the Black-Scholes-Merton formula.
    
    Formula:
        price = S * e^(-q*T) * N(d1) - K * e^(-r*T) * N(d2)
    
    Args:
        S (float): Stock price
        K (float): Strike price
        r (float): Risk-free rate
        q (float): Dividend yield
        T (float): Time to expiration
        d1 (float): d1 value from the Black-Scholes-Merton formula
        d2 (float): d2 value from the Black-Scholes-Merton formula
    
    Returns:
        float: Call option price
    """
    price = S * math.exp(-q*T) * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
    return price

def calc_put_price(S, K, r, q, T, d1, d2):
    """
    Calculate the price of a European put option using the Black-Scholes-Merton formula.
    
    Formula:
        price = K * e^(-r*T) * N(-d2) - S * e^(-q*T) * N(-d1)
    
    Args:
        S (float): Stock price
        K (float): Strike price
        r (float): Risk-free rate
        q (float): Dividend yield
        T (float): Time to expiration
        d1 (float): d1 value from the Black-Scholes-Merton formula
        d2 (float): d2 value from the Black-Scholes-Merton formula
    
    Returns:
        float: Put option price
    """
    price = K * math.exp(-r*T) * norm.cdf(-d2) - S * math.exp(-q*T) * norm.cdf(-d1)
    return price

def calc_d1(S, K, r, q, sigma, T):
    """
    Calculate d1 in the Black-Scholes-Merton formula.
    
    Formula:
        d1 = [ln(S/K) + (r - q + sigma^2 / 2) * T] / (sigma * sqrt(T))
    
    Args:
        S (float): Stock price
        K (float): Strike price
        r (float): Risk-free rate
        q (float): Dividend yield
        sigma (float): Volatility
        T (float): Time to expiration
    
    Returns:
        float: d1 value
    """
    nominator = math.log(S/K) + (r - q + sigma**2/2)*T
    denominator = sigma * math.sqrt(T)
    d1 = nominator / denominator
    return d1

def calc_d2(d1, sigma, T):
    """
    Calculate d2 in the Black-Scholes-Merton formula.
    
    Formula:
        d2 = d1 - sigma * sqrt(T)
    
    Args:
        d1 (float): d1 value from the Black-Scholes-Merton formula
        sigma (float): Volatility
        T (float): Time to expiration
    
    Returns:
        float: d2 value
    """
    d2 = d1 - sigma * math.sqrt(T)
    return d2

def calc_delta_call(q, T, d1):
    """
    Calculate the delta of a European call option.
    
    Formula:
        delta = e^(-q*T) * N(d1)
    
    Args:
        q (float): Dividend yield
        T (float): Time to expiration
        d1 (float): d1 value from the Black-Scholes-Merton formula
    
    Returns:
        float: Delta value for the call option
    """
    delta = math.exp(-q*T) * norm.cdf(d1)
    return delta


def calc_delta_put(q, T, d1):
    """
    Calculate the delta of a European put option.
    
    Formula:
        delta = e^(-q*T) * (N(d1) - 1)
    
    Args:
        q (float): Dividend yield
        T (float): Time to expiration
        d1 (float): d1 value from the Black-Scholes-Merton formula
    
    Returns:
        float: Delta value for the put option
    """
    delta = math.exp(-q*T) * (norm.cdf(d1) - 1)
    return delta


def calc_theta_call(S, K, r, q, sigma, T, d1, d2):
    """
    Calculate the theta of a European call option.
    A negative theta means the option price decreases as time passes.
    
    Formula:
        theta = -S * e^(-q*T) * n(d1) * sigma / (2 * sqrt(T)) - r * K * e^(-r*T) * N(d2) + q * S * e^(-q*T) * N(d1)
    
    Args:
        S (float): Stock price
        K (float): Strike price
        r (float): Risk-free rate
        q (float): Dividend yield
        sigma (float): Volatility
        T (float): Time to expiration
        d1 (float): d1 value from the Black-Scholes-Merton formula
        d2 (float): d2 value from the Black-Scholes-Merton formula
    
    Returns:
        float: Theta value for the call option
    """
    theta = -S * math.exp(-q*T) * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r*T) * norm.cdf(d2) + q * S * math.exp(-q*T) * norm.cdf(d1)
    return theta

def calc_theta_put(S, K, r, q, sigma, T, d1, d2):
    """
    Calculate the theta of a European put option.
    A negative theta means the option price decreases as time passes.
    
    Formula:
        theta = -S * e^(-q*T) * n(d1) * sigma / (2 * sqrt(T)) + r * K * e^(-r*T) * N(-d2) - q * S * e^(-q*T) * N(-d1)
    
    Args:
        S (float): Stock price
        K (float): Strike price
        r (float): Risk-free rate
        q (float): Dividend yield
        sigma (float): Volatility
        T (float): Time to expiration
        d1 (float): d1 value from the Black-Scholes-Merton formula
        d2 (float): d2 value from the Black-Scholes-Merton formula
    
    Returns:
        float: Theta value for the put option
    """
    theta = -S * math.exp(-q*T) * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) + r * K * math.exp(-r*T) * norm.cdf(-d2) - q * S * math.exp(-q*T) * norm.cdf(-d1)
    return theta

def calc_rho_call(K, r, T, d2):
    """
    Calculate the rho of a European call option.
    
    Formula:
        rho = K * T * e^(-r*T) * N(d2)
    
    Args:
        K (float): Strike price
        r (float): Risk-free rate
        T (float): Time to expiration
        d2 (float): d2 value from the Black-Scholes-Merton formula
    
    Returns:
        float: Rho value for the call option
    """
    rho = K * T * math.exp(-r*T) * norm.cdf(d2)
    return rho

def calc_rho_put(K, r, T, d2):
    """
    Calculate the rho of a European put option.
    
    Formula:
        rho = -K * T * e^(-r*T) * N(-d2)
    
    Args:
        K (float): Strike price
        r (float): Risk-free rate
        T (float): Time to expiration
        d2 (float): d2 value from the Black-Scholes-Merton formula
    
    Returns:
        float: Rho value for the put option
    """
    rho = -K * T * math.exp(-r*T) * norm.cdf(-d2)
    return rho

def calc_gamma(S, q, sigma, T, d1):
    """
    Calculate the gamma of a European call or put option.
    
    Formula:
        gamma = e^(-q*T) * n(d1) / (S * sigma * sqrt(T))
    
    Args:
        S (float): Stock price
        q (float): Dividend yield
        sigma (float): Volatility
        T (float): Time to expiration
        d1 (float): d1 value from the Black-Scholes-Merton formula
    
    Returns:
        float: Gamma value for the option
    """
    gamma = math.exp(-q*T) * norm.pdf(d1) / (S * sigma * math.sqrt(T))
    return gamma
    
def calc_vega(S, q, T, d1):
    """
    Calculate the vega of a European call or put option.
    
    Formula:
        vega = S * e^(-q*T) * n(d1) * sqrt(T)
    
    Args:
        S (float): Stock price
        q (float): Dividend yield
        T (float): Time to expiration
        d1 (float): d1 value from the Black-Scholes-Merton formula
    
    Returns:
        float: Vega value for the option
    """
    vega = S * math.exp(-q*T) * norm.pdf(d1) * math.sqrt(T)
    return vega 

# Utility function to format numbers
def format_number(value, decimal_places):
    if abs(value) < 1e-10:  # Adjust the threshold as needed
        return f"{0:.{decimal_places}f}"
    else:
        return f"{value:.{decimal_places}f}"

# Utility function to automatically update option prices
def update_values(bsm_option_pricer):
    st.session_state.decimal_places = st.session_state.get("decimal_places", 2)
    st.session_state.option_prices = bsm_option_pricer(
        st.session_state.underlying_price,
        st.session_state.strike_price,
        st.session_state.risk_free_rate,
        st.session_state.dividend_yield,
        st.session_state.volatility,
        st.session_state.time_to_expiration
    )


# Streamlit app
st.title("Black-Scholes-Merton Option Pricer")

# Inject custom CSS to hide links on hover
hide_links_css = """
<style>
h1:hover a, h2:hover a, h3:hover a, h4:hover a, h5:hover a, h6:hover a {
    display: none;
}
</style>
"""
st.markdown(hide_links_css, unsafe_allow_html=True)

# Initialize session state for reset flag
if "reset" not in st.session_state:
    st.session_state.reset = False
if "option_prices" not in st.session_state:
    st.session_state.option_prices = None
if "decimal_places" not in st.session_state:
    st.session_state.decimal_places = 2
if "calculate_pressed" not in st.session_state:
    st.session_state.calculate_pressed = False
    
# Default values
default_values = {
    "underlying_price": 100.0,
    "strike_price": 100.0,
    "risk_free_rate": 0.05,
    "dividend_yield": 0.00,
    "volatility": 0.2,
    "time_to_expiration": 1.0,
}

# Input fields with reset logic
underlying_price = default_values["underlying_price"] if st.session_state.reset else st.session_state.get("underlying_price", default_values["underlying_price"])
strike_price = default_values["strike_price"] if st.session_state.reset else st.session_state.get("strike_price", default_values["strike_price"])
risk_free_rate = default_values["risk_free_rate"] if st.session_state.reset else st.session_state.get("risk_free_rate", default_values["risk_free_rate"])
dividend_yield = default_values["dividend_yield"] if st.session_state.reset else st.session_state.get("dividend_yield", default_values["dividend_yield"])
volatility = default_values["volatility"] if st.session_state.reset else st.session_state.get("volatility", default_values["volatility"])
time_to_expiration = default_values["time_to_expiration"] if st.session_state.reset else st.session_state.get("time_to_expiration", default_values["time_to_expiration"])

st.session_state.underlying_price = st.number_input("Stock Price (S)", value=underlying_price, min_value=0.0, step=10.0)
st.session_state.strike_price = st.number_input("Strike Price (K)", value=strike_price, min_value=0.0, step=10.0)
st.session_state.risk_free_rate = st.number_input("Risk-Free Rate (r)", value=risk_free_rate)
st.session_state.dividend_yield = st.number_input("Dividend Yield (q)", value=dividend_yield, min_value=0.00, max_value=1.00, format="%.2f")
st.session_state.volatility = st.number_input("Volatility (σ)", value=volatility, min_value=0.01, max_value=1.00, format="%.2f")
st.session_state.time_to_expiration = st.number_input("Time to Expiration (T) in years", value=time_to_expiration, min_value=0.01, step=0.25)


button_col1, button_col2 = st.columns(2)

# Calculate button
with button_col1:
    if st.button("Calculate"):
        st.session_state.option_prices = bsm_option_pricer(
            st.session_state.underlying_price,
            st.session_state.strike_price,
            st.session_state.risk_free_rate,
            st.session_state.dividend_yield,
            st.session_state.volatility,
            st.session_state.time_to_expiration
        )
        st.session_state.calculate_pressed = True

# Reset button
with button_col2:
    if st.button("Reset"):
        st.session_state.reset = True  # Mark reset as True if button is pressed
        st.session_state.option_prices = None  # Reset the option prices
        st.session_state.calculate_pressed = False  # Reset the calculate flag
        st.rerun()  # Trigger a rerun
    else:
        st.session_state.reset = False  # Reset the reset flag


if st.session_state.calculate_pressed:
    update_values(bsm_option_pricer)
        
# Display results if available
if st.session_state.option_prices is not None:
    # Slider for selecting the number of decimal places
    decimal_places = st.slider("Decimal Places", min_value=0, max_value=6, value=st.session_state.decimal_places, key='decimal_places', on_change=update_values(bsm_option_pricer))

    option_prices = st.session_state.option_prices
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Call Option")
        st.write(f"Price: {format_number(option_prices['call_price'], decimal_places)}")
        st.write(f"Delta: {format_number(option_prices['call_delta'], decimal_places)}")
        st.write(f"Theta: {format_number(option_prices['call_theta'], decimal_places)}")
        st.write(f"Rho: {format_number(option_prices['call_rho'], decimal_places)}")

    with col2:
        st.subheader("Put Option")
        st.write(f"Price: {format_number(option_prices['put_price'], decimal_places)}")
        st.write(f"Delta: {format_number(option_prices['put_delta'], decimal_places)}")
        st.write(f"Theta: {format_number(option_prices['put_theta'], decimal_places)}")
        st.write(f"Rho: {format_number(option_prices['put_rho'], decimal_places)}")
    
    with col3:
        st.subheader("Common Greeks")
        st.write(f"Gamma: {format_number(option_prices['gamma'], decimal_places)}")
        st.write(f"Vega: {format_number(option_prices['vega'], decimal_places)}")

# To run the app, use the command: streamlit run app.py

# To Do:
# 1. Add greeks calculation to the option pricer - completed
# 2. Add user interface to the option pricer using Streamlit - halfway
# 3. Add a reset button to the option pricer - completed
# 4. Add a purchase price option that allows users to see the PnL against options - not started
# 5. Add heatmap on the effect of volaility and the underlying price on the option price - not started
# 6. Add show purchase price PnL in the heatmap - not started
