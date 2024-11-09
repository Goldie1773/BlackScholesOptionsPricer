import math
from scipy.stats import norm
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sqlalchemy import text, create_engine
from sqlalchemy.orm import sessionmaker

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
    st.session_state.purchase_price = st.session_state.get("purchase_price")
    st.session_state.option_prices = bsm_option_pricer(
        underlying_price,
        strike_price,
        risk_free_rate,
        dividend_yield,
        volatility,
        time_to_expiration,
    )


# Function to insert values into the Neon database
def insert_input(session, stock_price, strike_price, interest_rate, volatility, time_to_expiry):
    try:
        insert_input_query = text("""
            INSERT INTO Inputs (StockPrice, StrikePrice, InterestRate, Volatility, TimeToExpiry)
            VALUES (:stock_price, :strike_price, :interest_rate, :volatility, :time_to_expiry)
            RETURNING CalcID
        """)
        result = session.execute(insert_input_query, {
            'stock_price': round(float(stock_price), 6),
            'strike_price': round(float(strike_price), 6),
            'interest_rate': round(float(interest_rate), 6),
            'volatility': round(float(volatility), 6),
            'time_to_expiry': round(float(time_to_expiry), 6)
        })
        calc_id = result.fetchone()[0]
        session.commit()
        return calc_id
    except Exception as e:
        session.rollback()
        return None
    

# Function to insert option prices into the Neon database
def insert_output(session, volatility_shock, stock_price_shock, option_price, is_call_or_put, calc_id):
    try:
        insert_output_query = text("""
            INSERT INTO Outputs (VolatilityShock, StockPriceShock, OptionPrice, IsCallOrPut, CalcID)
            VALUES (:volatility_shock, :stock_price_shock, :option_price, :is_call_or_put, :calc_id)
        """)
        session.execute(insert_output_query, {
            'volatility_shock': round(float(volatility_shock), 6),
            'stock_price_shock': round(float(stock_price_shock), 6),
            'option_price': round(float(option_price), 6),
            'is_call_or_put': is_call_or_put,  # Should be 'Call' or 'Put'
            'calc_id': int(calc_id)
        })
    except Exception as e:
        session.rollback()
   
# Streamlit app
st.title("Black-Scholes-Merton Option Pricer")

# Create the SQLAlchemy engine using the connection URL from secrets.toml
DATABASE_URL = st.secrets["connections"]["neon"]["url"]
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session() # Create a new session

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
if "option_prices" not in st.session_state:
    st.session_state.option_prices = None
if "decimal_places" not in st.session_state:
    st.session_state.decimal_places = 2
    
# Default values
default_values = {
    "underlying_price": 100.00,
    "strike_price": 100.00,
    "risk_free_rate": 0.05,
    "dividend_yield": 0.00,
    "volatility": 0.20,
    "time_to_expiration": 1.0,
    "purchase_price": 0.00
}

input_col1, input_col2 = st.columns(2)

with input_col1:
    underlying_price = st.number_input("Stock Price (S)", value=default_values["underlying_price"], min_value=0.0, step=10.0)
    strike_price = st.number_input("Strike Price (K)", value=default_values["strike_price"], min_value=0.0, step=10.0)
    risk_free_rate = st.number_input("Risk-Free Rate (r)", value=default_values["risk_free_rate"])
    
with input_col2:
    dividend_yield = st.number_input("Dividend Yield (q)", value=default_values["dividend_yield"], min_value=0.00, max_value=1.00, format="%.2f")
    volatility = st.number_input("Volatility (σ)", value=default_values["volatility"], min_value=0.01, max_value=1.00, format="%.2f")
    time_to_expiration = st.number_input("Time to Expiration (T) in years", value=default_values["time_to_expiration"], min_value=0.01, step=0.25)


st.session_state.option_prices = bsm_option_pricer(
    underlying_price,
    strike_price,
    risk_free_rate,
    dividend_yield,
    volatility,
    time_to_expiration
)
        
# Display results if available
if st.session_state.option_prices is not None:
    option_prices = st.session_state.option_prices
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Call Option")
        st.write(f"Price: {format_number(option_prices['call_price'], st.session_state.decimal_places)}")
        st.write(f"Delta: {format_number(option_prices['call_delta'], st.session_state.decimal_places)}")
        st.write(f"Theta: {format_number(option_prices['call_theta'], st.session_state.decimal_places)}")
        st.write(f"Rho: {format_number(option_prices['call_rho'], st.session_state.decimal_places)}")

    with col2:
        st.subheader("Put Option")
        st.write(f"Price: {format_number(option_prices['put_price'], st.session_state.decimal_places)}")
        st.write(f"Delta: {format_number(option_prices['put_delta'], st.session_state.decimal_places)}")
        st.write(f"Theta: {format_number(option_prices['put_theta'], st.session_state.decimal_places)}")
        st.write(f"Rho: {format_number(option_prices['put_rho'], st.session_state.decimal_places)}")
    
    with col3:
        st.subheader("Common Greeks")
        st.write(f"Gamma: {format_number(option_prices['gamma'], st.session_state.decimal_places)}")
        st.write(f"Vega: {format_number(option_prices['vega'], st.session_state.decimal_places)}")
    
    # Slider for selecting the number of decimal places
    decimal_places_slider = st.slider("Decimal Places", min_value=0, max_value=6, value=st.session_state.decimal_places, key='decimal_places', on_change=update_values(bsm_option_pricer))
    
button_pp, button_pp_inp, button_op, button_pt, check_show_heatmap = st.columns((0.15, 0.3, 0.1, 0.2, 0.3))

with button_pp:
    st.write("Purchase Price")

with button_pp_inp:
    purchase_price = st.number_input('', value=None, min_value=0.00, step=1.00, format="%.2f", label_visibility="collapsed")

with button_op:
    st.write("Option Type")

with button_pt:
    purchase_type = st.radio("", ["Call", "Put"], label_visibility="collapsed")

with check_show_heatmap:
    if purchase_price in [None, 0.00]:
        show_heatmap = st.toggle("Show Heatmap", value=False, disabled=True)
    else:
        show_heatmap = st.toggle("Show Heatmap", value=False)

    
if purchase_price not in [None, 0.00]:
    st.subheader("Profit and Loss")
    if purchase_type == "Call":
        pnl = option_prices['call_price'] - purchase_price
    else:    
        pnl = option_prices['put_price'] - purchase_price
    color = "green" if pnl > 0 else "red"
    st.write(f"PnL: :{color}[**{format_number(pnl, st.session_state.decimal_places)}**]")
    
    if show_heatmap:
        # Generate ranges for strike price and volatility
        strike_prices = np.linspace(strike_price * 0.25, strike_price * 1.75, 15)
        volatilities = np.linspace(volatility * 0.25, volatility * 1.75, 15)

        # Round the strike_prices and volatilities for database insertion (6 decimal places)
        strike_prices_db = np.round(strike_prices, 6)
        volatilities_db = np.round(volatilities, 6)
        
        # Round the strike_prices and volatilities for display purposes (2 decimal places)
        strike_prices_display = np.round(strike_prices, 2)
        volatilities_display = np.round(volatilities, 2)

        # Initialize a DataFrame to store PnL values with rounded indices and columns for display
        pnl_data = pd.DataFrame(index=volatilities_display, columns=strike_prices_display)

        # Insert the input and get CalcID
        calc_id = insert_input(session, underlying_price, strike_price, risk_free_rate, volatility, time_to_expiration)
        
       # Calculate PnL for each combination using high precision values
        for vol_db, vol_disp in zip(volatilities_db, volatilities_display):
            for K_db, K_disp in zip(strike_prices_db, strike_prices_display):
                option_prices = bsm_option_pricer(
                    underlying_price,
                    K_db,
                    risk_free_rate,
                    dividend_yield,
                    vol_db,
                    time_to_expiration
                )
                if purchase_type == 'Call':
                    pnl = option_prices['call_price'] - purchase_price
                else:
                    pnl = option_prices['put_price'] - purchase_price
                
                # Store PnL in DataFrame with display precision
                pnl_data.at[vol_disp, K_disp] = pnl
                
                # Insert into Outputs with high precision values
                insert_output(session, vol_db, K_db, pnl, purchase_type, calc_id)

        session.commit()
        
        # Ensure pnl_data is of numeric type and round values to 2 decimal places
        pnl_data = pnl_data.astype(float).round(2)

        # Handle any missing values (if necessary)
        pnl_data.fillna(0, inplace=True)  # Replace NaN with zeros 
        
        # Calculate minimum PnL value and maximum absolute PnL value for symmetric scaling
        min_pnl = pnl_data.min().min()
        max_pnl = pnl_data.max().max()

        max_abs_pnl = max(abs(min_pnl), abs(max_pnl))
        norm = TwoSlopeNorm(vmin=min_pnl, vcenter=0, vmax=max_abs_pnl)

        # Create the heatmap with adjusted parameters
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            pnl_data,
            ax=ax,
            cmap='RdYlGn',
            norm=norm,
            annot=True,
            fmt=".2f",
            linecolor='black',
            cbar=False,
        )
        
        # Calculate the middle cell indices
        num_rows, num_cols = pnl_data.shape
        mid_row = num_rows // 2
        mid_col = num_cols // 2

        # Highlight the middle cell's annotation
        for text in ax.texts:
            x, y = text.get_position()
            col_idx = int(x - 0.5)
            row_idx = int(y - 0.5)
            if col_idx == mid_col and row_idx == mid_row:
                text.set_backgroundcolor('blue')
                text.set_color('white')
                break
            
        ax.set_xlabel('Strike Price (K)', fontweight='bold')
        ax.set_ylabel('Volatility (σ)', fontweight='bold')
        ax.set_title('PnL Heatmap', fontweight='bold', fontsize=16)

        st.pyplot(fig)

        session.close()                  
# To run the app, use the command: streamlit run bs_option_pricer.py