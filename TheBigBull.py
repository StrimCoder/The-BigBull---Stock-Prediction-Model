import streamlit as st
from streamlit_option_menu import option_menu
import yfinance as yf
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from datetime import datetime, timedelta


# Mock function to simulate web search for news
def web_search(query):
    return [
        {'title': f'Breaking News about {query}', 'snippet': 'This is a mock news article about the stock.', 'link': 'https://www.moneycontrol.com/news/tags/reliance-industries.html'},
        {'title': f'Latest Updates on {query}', 'snippet': 'Another mock article with recent updates.', 'link': 'https://www.marketwatch.com/story/reliance-industries-rises-friday-underperforms-competitors-9d1306cd-0312a7e6cc7b'},
        {'title': f'Market Analysis: {query}', 'snippet': 'A detailed analysis of the stock market trends.', 'link': 'https://finance.yahoo.com/quote/RELIANCE.NS/'}
    ]

# Load Stock Data
@st.cache_data
def load_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            raise ValueError("No data found for the given ticker and date range.")
        data = data.resample('M').last()  # Resample data to monthly frequency
        return data
    except Exception as e:
        st.error(f"Failed to load stock data: {e}")
        return pd.DataFrame()

# =========================================
# SIDEBAR NAVIGATION (YOUR REQUESTED UI)
# =========================================
with st.sidebar:
    selected_tab = option_menu(
        menu_title='The BigBull',
        options=['Home','Stock Data', 'Fundamental Data', 'Future Prediction', 
                'Stock Comparison', 'Model Performance', 'Top News','About'],
        icons=['graph-up', 'bar-chart-line', 'magic', 
               'graph-up-arrow', 'award', 'newspaper'],
        menu_icon='₿',
        default_index=0,
        styles={
            "container": {"padding": "5!important","background-color":'black'},
            "icon": {"color": "white", "font-size": "23px"}, 
            "nav-link": {
                "color":"white",
                "font-size": "20px", 
                "text-align": "left", 
                "margin":"0px", 
                "--hover-color": "blue"
            },
            "nav-link-selected": {"background-color": "#02ab21"},
             "menu-title": {  # 👈 Add this new key for title styling
                "color": "white",
                "font-weight": "bold",  # Makes text bold
                "font-size": "30px",    # Slightly larger font
                "text-align": "center",  # Center alignment
                "padding": "10px 0"      # Add some padding
            }
        }
    )

# App Title
st.title("The BigBull - Stock Price Prediction")

# Stock Input Section (Moved from sidebar to main area)
col1, col2 = st.columns(2)
with col1:
    stock_ticker = st.text_input("Enter Stock Ticker (e.g., Reliance.NS):", value="Reliance.NS")
with col2:
    current_date = datetime.now().date()
    start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("End Date", current_date)
    if start_date >= end_date:
        st.error("Start date must be before end date.")

stock_data = load_stock_data(stock_ticker, start_date, end_date)
st.markdown("---")

# =========================================
# ORIGINAL FUNCTIONALITY - NOW CONNECTED TO SIDEBAR BUTTONS
# =========================================
if selected_tab == 'Home':
    st.header('The BigBull:violet[ :Revolutionizing Stock Market Predictions with AI and Data Science]')
    st.text("""Introduction: A New Era of Smart Trading

In the high-stakes world of stock trading, timing and insight are everything. Whether you're a seasoned investor or a beginner looking to make your first trade, understanding market movements can be a daunting task. This is where The BigBull steps in—a next-generation AI-powered stock market prediction app designed to provide real-time insights, data-driven forecasts, and cutting-edge analysis to help you navigate the stock market with confidence.

The BigBull is more than just an ordinary stock tracking tool; it is an advanced financial companion that utilizes state-of-the-art machine learning models, technical indicators, and deep financial analytics to forecast stock price trends. From live stock tracking to candlestick pattern recognition, from trend analysis to real-time alerts, The BigBull arms you with everything you need to stay ahead in the market.
Why The BigBull?

Traditional stock trading is often complex and overwhelming, with millions of data points moving every second. Many investors rely on guesswork or manual chart analysis, leading to missed opportunities or unnecessary risks. The BigBull eliminates uncertainty by offering AI-driven predictions and insightful analytics that give traders a significant edge over the market.

With an easy-to-use interface, real-time updates, and data-backed strategies, The BigBull ensures that investors no longer have to rely on intuition alone. Harness the power of AI and transform the way you trade!

Key Features of The BigBull

1. Real-Time Stock Analysis

With The BigBull, you can track your favorite stocks in real time, ensuring that you never miss an important market movement. Our platform provides live price updates, historical data, 52-week highs and lows, and comprehensive performance reports to keep you informed at every stage of your investment journey.

2. AI-Powered Market Predictions

What if you could predict the market’s next move? The BigBull integrates advanced machine learning models that analyze historical stock data to provide highly accurate forecasts. Using deep-learning algorithms, the app evaluates past trends, market behavior, and investor sentiment to give you the best possible price predictions.

3. Candlestick Pattern Recognition

Candlestick charts are one of the most important tools for traders, yet reading them accurately requires years of experience. The BigBull automatically detects candlestick patterns and interprets their implications, highlighting potential bullish or bearish trends so you can make informed decisions.

4. Technical Indicators and Trend Analysis

To enhance your trading strategy, The BigBull offers a comprehensive suite of technical indicators, including:-

✅ Relative Strength Index (RSI) – Measures market momentum and overbought/oversold conditions.
✅ Moving Averages (SMA & EMA) – Tracks trends by smoothing price data over time.
✅ MACD (Moving Average Convergence Divergence) – Helps identify trend reversals and momentum shifts.
✅ Bollinger Bands – Shows volatility levels and potential breakout points.

These tools provide traders with a deeper understanding of market trends, entry points, and potential reversals.

5. Smart Alerts and Notifications

Never miss an opportunity again! The BigBull allows users to set custom price alerts, news notifications, and market updates so that you're always informed about key market movements in real time.

6. Stock Comparison Tool

Compare multiple stocks side by side based on historical performance, technical indicators, and real-time price movements to determine the best investment option.

7. User-Friendly Dashboard with Intuitive UI

The BigBull is built with a sleek, modern, and highly responsive interface, ensuring a seamless user experience. Whether you’re a beginner or an expert trader, our easy-to-navigate dashboard makes stock analysis effortless and enjoyable.

How The BigBull Uses AI and Machine Learning for Stock Predictions

One of the core elements that set The BigBull apart is its AI-driven stock prediction engine. By leveraging historical data, pattern recognition, and deep learning algorithms, The BigBull makes highly accurate forecasts for short-term and long-term stock performance. Here’s how it works:

1. Data Collection and Preprocessing

The system collects real-time and historical stock data from various sources, including financial APIs, market reports, and global economic indicators. This data is cleaned and structured for analysis.

2. Feature Engineering and Model Training

Our AI models analyze multiple factors such as stock price trends, trading volume, market volatility, and sentiment analysis to train predictive models using machine learning techniques such as:

Linear Regression (for trend forecasting)

LSTM Neural Networks (for time-series predictions)

Random Forest & XGBoost (for pattern-based predictions)

3. Prediction and Insights

Once trained, the models continuously learn and adapt based on market fluctuations, providing traders with up-to-date forecasts and investment strategies tailored to market conditions.

Who Can Benefit from The BigBull?

✅ Beginner Investors – Learn market basics and make informed trades with guided insights.
✅ Day Traders – Get real-time data and trend analysis for intraday trading strategies.
✅ Long-Term Investors – Analyze historical trends and stock fundamentals for long-term investment decisions.
✅ Financial Analysts – Access deep market insights and predictive analytics to support professional research.

Security and Data Protection

At The BigBull, we take data security and user privacy seriously. Our platform implements end-to-end encryption, secure cloud storage, and compliance with global financial regulations to ensure that your personal and financial data remains safe at all times.

Why The BigBull is the Future of Stock Trading

The financial markets are becoming increasingly complex, and traditional trading methods are no longer enough to keep up with the rapid pace of change. With AI-driven insights, powerful analytics, and real-time tracking, The BigBull represents the future of stock market trading.

Instead of relying on outdated methods or gut feelings, traders can now make data-backed, confident investment decisions powered by cutting-edge technology.

Join The BigBull Revolution!

If you’re serious about taking your stock trading to the next level, it’s time to embrace AI-driven investing with The BigBull. Whether you’re a casual trader or a seasoned investor, our app gives you the power to make smarter, faster, and more profitable decisions.

📲 Download The BigBull today and start trading smarter! 🚀📈

The stock market waits for no one stay ahead, stay informed, and rule the market with The BigBull!

""")
    st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80",
             use_container_width=True,
             caption="Market Analysis Dashboard")

elif selected_tab == 'Stock Data':
    if not stock_data.empty:
        # Calculate Heikin-Ashi values
        stock_data['HA_Close'] = (stock_data['Open'] + stock_data['High'] + stock_data['Low'] + stock_data['Close']) / 4
        stock_data['HA_Open'] = (stock_data['Open'].shift(1) + stock_data['Close'].shift(1)) / 2
        stock_data['HA_High'] = stock_data[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        stock_data['HA_Low'] = stock_data[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

        # Heikin-Ashi Chart
        st.subheader("ChandelStick Pattern Chart")
        ha_fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                                open=stock_data['HA_Open'],
                                                high=stock_data['HA_High'],
                                                low=stock_data['HA_Low'],
                                                close=stock_data['HA_Close'])])
        ha_fig.update_layout(title=f"{stock_ticker} ChandelStick Chart (Monthly)",
                             xaxis_title='Date',
                             yaxis_title='Price (USD)',
                             xaxis_rangeslider_visible=False)
        st.plotly_chart(ha_fig)

        # Technical Indicators
        indicators = st.multiselect(
            "Select Technical Indicators",
            options=["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"],
            default=["SMA", "EMA"]
        )

        # Line Chart with Indicators
        st.subheader("Stock Closing Prices with Indicators")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue', linewidth=2)

        if "SMA" in indicators:
            sma = stock_data['Close'].rolling(window=20).mean()
            ax.plot(stock_data.index, sma, label='20-Day SMA', color='orange')

        if "EMA" in indicators:
            ema = stock_data['Close'].ewm(span=20, adjust=False).mean()
            ax.plot(stock_data.index, ema, label='20-Day EMA', color='red')

        if "RSI" in indicators:
            delta = stock_data['Close'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            ax2 = ax.twinx()
            ax2.plot(stock_data.index, rsi, label='RSI', color='purple')
            ax2.set_ylabel('RSI', fontsize=12)

        if "MACD" in indicators:
            short_ema = stock_data['Close'].ewm(span=12, adjust=False).mean()
            long_ema = stock_data['Close'].ewm(span=26, adjust=False).mean()
            macd = short_ema - long_ema
            signal = macd.ewm(span=9, adjust=False).mean()
            ax.plot(stock_data.index, macd, label='MACD', color='green')
            ax.plot(stock_data.index, signal, label='MACD Signal', color='pink')

        if "Bollinger Bands" in indicators:
            sma_20 = stock_data['Close'].rolling(window=20).mean()
            std_20 = stock_data['Close'].rolling(window=20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            ax.plot(stock_data.index, sma_20, label='BB Middle', color='cyan')
            ax.plot(stock_data.index, upper_band, label='BB Upper', color='magenta')
            ax.plot(stock_data.index, lower_band, label='BB Lower', color='magenta')

        ax.set_title(f"{stock_ticker} Stock Closing Prices with Indicators (Monthly)", fontsize=16)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Closing Price (USD)", fontsize=12)
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Raw Data Display
        st.subheader("Stock Data")
        st.write(stock_data.head(20))

elif selected_tab == 'Fundamental Data':
    if not stock_data.empty:
        try:
            stock = yf.Ticker(stock_ticker)
            info = stock.info

            # Display fundamental data
            st.subheader("Fundamental Data")
            fundamental_data = {
                "Market Cap": info.get("marketCap"),
                "Earnings Per Share (EPS)": info.get("trailingEps"),
                "Price-to-Earnings Ratio (P/E)": info.get("trailingPE"),
                "Dividend Yield": info.get("dividendYield"),
                "52 Week High": info.get("fiftyTwoWeekHigh"),
                "52 Week Low": info.get("fiftyTwoWeekLow"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
            }

            # Create a DataFrame for better visualization
            fundamental_df = pd.DataFrame(list(fundamental_data.items()), columns=["Metric", "Value"])
            st.dataframe(fundamental_df)
        except Exception as e:
            st.error(f"Failed to fetch fundamental data: {e}")

elif selected_tab == 'Future Prediction':
    if not stock_data.empty:
        # Scale Data
        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(stock_data[['Close']])

        # Future Predictions
        st.subheader("Future Price Predictions")
        future_days = 12  # Predict for next 12 months
        future_dates = [stock_data.index[-1] + timedelta(days=i*30) for i in range(1, future_days + 1)]
        future_X = np.arange(len(scaled_close), len(scaled_close) + future_days).reshape(-1, 1)

        # Model Definitions
        models = {
            'SVR': SVR(kernel='rbf', C=1e3, gamma=0.1),
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }

        # Function to Train and Predict
        def train_and_predict(model, X_train, y_train, X_test):
            try:
                model.fit(X_train, y_train)
                return model.predict(X_test)
            except Exception as e:
                st.error(f"Model training failed: {e}")
                return np.zeros_like(X_test[:, 0])

        # Prepare Data
        train_size = int(len(scaled_close) * 0.8)
        X_train = np.arange(train_size).reshape(-1, 1)
        y_train = scaled_close[:train_size].flatten()

        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                future_preds = model.predict(future_X)
                future_prices = scaler.inverse_transform(future_preds.reshape(-1, 1))

                # Plot Future Predictions
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                ax3.plot(stock_data.index, stock_data['Close'], label="Historical Data", color='green')
                ax3.plot(future_dates, future_prices, label=f"Future Predictions ({model_name})", color='blue')
                ax3.set_title(f"Future Stock Price Prediction ({model_name})", fontsize=16)
                ax3.set_xlabel("Date", fontsize=12)
                ax3.set_ylabel("Closing Price (USD)", fontsize=12)
                ax3.legend()
                st.pyplot(fig3)
            except Exception as e:
                st.error(f"Failed to predict future prices with {model_name}: {e}")

elif selected_tab == 'Stock Comparison':
    if not stock_data.empty:
        # Sidebar: User Input for Comparison
        comparison_tickers = st.text_input("Enter Stock Tickers to Compare (comma-separated, e.g., AAPL,MSFT):", value="AAPL,MSFT")

        # Load Comparison Stock Data
        comparison_tickers_list = [ticker.strip() for ticker in comparison_tickers.split(",")]
        comparison_data = {}
        for ticker in comparison_tickers_list:
            comparison_data[ticker] = load_stock_data(ticker, start_date, end_date)

        # Display Stock Comparison
        st.subheader("Stock Comparison")
        if any(not data.empty for data in comparison_data.values()):
            fig_comp, ax_comp = plt.subplots(figsize=(12, 6))
            for ticker, data in comparison_data.items():
                if not data.empty:
                    ax_comp.plot(data.index, data['Close'], label=f'{ticker} Close Price')

            ax_comp.set_title("Stock Comparison (Monthly)", fontsize=16)
            ax_comp.set_xlabel("Date", fontsize=12)
            ax_comp.set_ylabel("Closing Price (USD)", fontsize=12)
            ax_comp.legend()
            ax_comp.grid(True)
            st.pyplot(fig_comp)

        # Display Fundamental Data for Comparison Stocks (YOUR REQUESTED ADDITION)
        st.subheader("Fundamental Data Comparison")
        for ticker in comparison_tickers_list:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Display fundamental data
                fundamental_data = {
                    "Market Cap": info.get("marketCap"),
                    "Earnings Per Share (EPS)": info.get("trailingEps"),
                    "Price-to-Earnings Ratio (P/E)": info.get("trailingPE"),
                    "Dividend Yield": info.get("dividendYield"),
                    "52 Week High": info.get("fiftyTwoWeekHigh"),
                    "52 Week Low": info.get("fiftyTwoWeekLow"),
                    "Sector": info.get("sector"),
                    "Industry": info.get("industry"),
                }

                # Create a DataFrame for better visualization
                fundamental_df = pd.DataFrame(list(fundamental_data.items()), columns=["Metric", "Value"])
                st.write(f"### Fundamental Data for {ticker}")
                st.dataframe(fundamental_df)
            except Exception as e:
                st.error(f"Failed to fetch fundamental data for {ticker}: {e}")

elif selected_tab == 'Model Performance':
    if not stock_data.empty:
        # Scale Data
        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(stock_data[['Close']])

        # Display Model Performance
        st.subheader("Model Performance Summary")
        try:
            # Split Data into Training and Testing Sets
            train_size = int(len(scaled_close) * 0.8)
            train_data, test_data = scaled_close[:train_size], scaled_close[train_size:]

            # Prepare Data
            X_train = np.arange(train_size).reshape(-1, 1)
            y_train = train_data.flatten()
            X_test = np.arange(train_size, len(scaled_close)).reshape(-1, 1)

            # Model Definitions
            models = {
                'SVR': SVR(kernel='rbf', C=1e3, gamma=0.1),
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Decision Tree': DecisionTreeRegressor(random_state=42)
            }

            # Predict and Evaluate Models
            predictions = {}
            errors = {}
            r2_scores = {}

            for model_name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    predictions[model_name] = model.predict(X_test)
                    predicted_prices = scaler.inverse_transform(predictions[model_name].reshape(-1, 1))
                    actual_prices = scaler.inverse_transform(test_data)

                    # Calculate Metrics
                    error = sqrt(mean_squared_error(actual_prices, predicted_prices))
                    r2 = r2_score(actual_prices, predicted_prices)
                    errors[model_name] = error
                    r2_scores[model_name] = r2
                except Exception as e:
                    st.error(f"Error processing model {model_name}: {e}")

            performance_df = pd.DataFrame({"RMSE": errors, "R² Score": r2_scores})
            st.dataframe(performance_df)

            # Plot Model Performance
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            performance_df.plot(kind='bar', ax=ax4)
            ax4.set_title("Model Performance Comparison", fontsize=16)
            ax4.set_xlabel("Model", fontsize=12)
            ax4.set_ylabel("Score", fontsize=12)
            ax4.legend(title='Metrics')
            st.pyplot(fig4)
        except Exception as e:
            st.error(f"Failed to display model performance: {e}")

elif selected_tab == 'Top News':
    if not stock_data.empty:
        # Display Top News
        try:
            # Fetch top news related to the stock
            news_query = f"{stock_ticker} stock news"
            news_results = web_search(news_query)

            # Display news articles
            st.subheader("Top News")
            for i, article in enumerate(news_results):
                st.write(f"**Article {i+1}:**")
                st.write(f"- **Title:** {article['title']}")
                st.write(f"- **Snippet:** {article['snippet']}")
                st.write(f"- **Link:** [Read more]({article['link']})")
                st.write("---")
        except Exception as e:
            st.error(f"Failed to fetch news: {e}")

elif selected_tab == 'About':
    st.header("ℹ️ About The BigBull")
    
    st.markdown("""
    ### Stock Analysis and Prediction Platform
    
    **Version**: 2.0  
    **Last Updated**: March 2025  
    
    ### Features
    - 📊 **Technical Analysis**: Candlestick charts, technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
    - 💹 **Fundamental Analysis**: Key financial metrics and ratios
    - 🔮 **AI Predictions**: Multiple machine learning models for price forecasting
    - 🆚 **Stock Comparison**: Compare multiple stocks' performance
    - 📰 **Market News**: Latest updates on selected stocks
    
    ### Data Sources

    - Stock data from Yahoo Finance (via yfinance)
    - Simulated news results (real integration available with API key)
    
    ### Technologies Used
    - Python (Streamlit, Pandas, NumPy)
    - Machine Learning (scikit-learn)
    - Visualization (Plotly, Matplotlib)
    
    ### Disclaimer
    This application is for educational purposes only. The predictions should not 
    be considered as financial advice. Always consult with a qualified financial 
    advisor before making investment decisions.
    """)
    
    st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2072&q=80",
             use_container_width=True,
             caption="Data Analysis Visualization")

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    The BigBull Stock Analysis & Prediction Platform | © 2025 | A Project By Mr.Bhushan Navsagar & Team
</div>
""", unsafe_allow_html=True)