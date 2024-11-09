# BlackScholesOptionsPricer
Black Scholes Merton Options Pricing Model Developed in Python as a learning project
  Note: NOT intended for use in a professional environment

The application is capable of calculating profit and loss for both call and put options including common Greeks. 
The user can enter a purchase price for their option and the application will show their PnL including a heatmap showing the effect of volatility and stock price shock.

The results are then stored in a PostgreSQL Neon Database with the outputs mapped to the corresponding inputs.

To use the code:
  1. Setup virtual environment: python -m venv "Name of virtual environment"
  2. Install the requirements: pip install -r requirements.txt
  3. Run the code!

To run the Streamlit app:
  1. Use the following command in the terminal: streamlit run bs_option_pricer.py
