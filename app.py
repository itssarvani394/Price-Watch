import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import time
from datetime import datetime, timedelta


# Function to load and preprocess data
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Upper_BB'] = data['MA20'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_BB'] = data['MA20'] - (data['Close'].rolling(window=20).std() * 2)
    data = data.dropna()
    return data


# Function for EDA plots
def create_eda_plots(data):
    # Closing Price with MA20 and Bollinger Bands
    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig_close.add_trace(
        go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='20-day MA', line=dict(color='orange')))
    fig_close.add_trace(go.Scatter(x=data.index, y=data['Upper_BB'], mode='lines', name='Upper BB',
                                   line=dict(color='gray', dash='dash')))
    fig_close.add_trace(go.Scatter(x=data.index, y=data['Lower_BB'], mode='lines', name='Lower BB',
                                   line=dict(color='gray', dash='dash')))
    fig_close.update_layout(title='Closing Price with 20-day MA and Bollinger Bands', xaxis_title='Date',
                            yaxis_title='Price')

    # Volume
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'))
    fig_volume.update_layout(title='Trading Volume Over Time', xaxis_title='Date', yaxis_title='Volume')

    # Returns
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Scatter(x=data.index, y=data['Returns'], mode='lines', name='Daily Returns'))
    fig_returns.update_layout(title='Daily Returns Over Time', xaxis_title='Date', yaxis_title='Returns')

    # Volatility
    fig_volatility = go.Figure()
    fig_volatility.add_trace(go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility'))
    fig_volatility.update_layout(title='20-Day Volatility Over Time', xaxis_title='Date', yaxis_title='Volatility')

    return fig_close, fig_volume, fig_returns, fig_volatility


# Anomaly detection functions (as previously defined)
def detect_zscore_anomalies(data, threshold=3):
    z_scores = np.abs((data['Close'] - data['Close'].mean()) / data['Close'].std())
    return z_scores > threshold


def detect_iforest_anomalies(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['Close', 'Volume', 'Returns', 'Volatility']])
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    return iso_forest.fit_predict(X) == -1


def detect_dbscan_anomalies(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['Close', 'Volume', 'Returns', 'Volatility']])
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    return dbscan.fit_predict(X) == -1


def detect_lstm_anomalies(data, sequence_length=20, threshold_percentile=95):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'Volume', 'Returns', 'Volatility']])

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 4)),
        Dense(4)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    predictions = model.predict(X)
    mse = np.mean(np.power(y - predictions, 2), axis=1)
    threshold = np.percentile(mse, threshold_percentile)

    anomalies = np.zeros(len(data))
    anomalies[sequence_length:] = mse > threshold
    return anomalies.astype(bool)


def detect_autoencoder_anomalies(data, threshold_percentile=95):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'Volume', 'Returns', 'Volatility']])

    input_dim = scaled_data.shape[1]
    encoding_dim = 2

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(8, activation="relu")(input_layer)
    encoder = Dense(4, activation="relu")(encoder)
    encoder = Dense(encoding_dim, activation="relu")(encoder)
    decoder = Dense(4, activation="relu")(encoder)
    decoder = Dense(8, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="linear")(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    autoencoder.fit(scaled_data, scaled_data, epochs=100, batch_size=32, shuffle=True, verbose=0)

    predictions = autoencoder.predict(scaled_data)
    mse = np.mean(np.power(scaled_data - predictions, 2), axis=1)
    threshold = np.percentile(mse, threshold_percentile)

    return mse > threshold


# Page configuration
st.set_page_config(
    page_title="Price Watch - Stock Anomaly Detection",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #5a6fd8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📈 Price Watch</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Advanced Stock Market Anomaly Detection System</p>', unsafe_allow_html=True)

# Sidebar with enhanced styling
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown("---")

# Stock selection with popular stocks
popular_stocks = {
    "GameStop (GME)": "GME",
    "Apple (AAPL)": "AAPL", 
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Google (GOOGL)": "GOOGL",
    "NVIDIA (NVDA)": "NVDA",
    "Meta (META)": "META"
}

selected_stock = st.sidebar.selectbox(
    "📊 Select Stock",
    options=list(popular_stocks.keys()),
    index=0
)
ticker = popular_stocks[selected_stock]

# Custom ticker input
custom_ticker = st.sidebar.text_input("🔍 Or enter custom ticker", value="", placeholder="e.g., BTC-USD, ETH-USD")

if custom_ticker:
    ticker = custom_ticker.upper()

# Date range with presets
st.sidebar.markdown("### 📅 Date Range")
date_preset = st.sidebar.selectbox(
    "Quick Select",
    ["Custom", "Last 6 months", "Last year", "Last 2 years", "Last 5 years"],
    index=3
)

if date_preset == "Custom":
    start_date = st.sidebar.date_input('Start date', pd.to_datetime('2020-01-01'))
    end_date = st.sidebar.date_input('End date', pd.to_datetime('2023-12-31'))
elif date_preset == "Last 6 months":
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=180)
elif date_preset == "Last year":
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
elif date_preset == "Last 2 years":
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=730)
elif date_preset == "Last 5 years":
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=1825)

# Algorithm settings
st.sidebar.markdown("### 🤖 Algorithm Settings")
st.sidebar.markdown("---")

zscore_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
iforest_contamination = st.sidebar.slider("Isolation Forest Contamination", 0.001, 0.1, 0.01, 0.001)
lstm_threshold = st.sidebar.slider("LSTM Threshold Percentile", 90, 99, 95, 1)
autoencoder_threshold = st.sidebar.slider("Autoencoder Threshold Percentile", 90, 99, 95, 1)

# Load data with progress indicator
with st.spinner(f'📊 Loading data for {ticker}...'):
    data = load_data(ticker, start_date, end_date)

# Display key metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📈 Current Price",
        value=f"${data['Close'].iloc[-1]:.2f}",
        delta=f"{data['Returns'].iloc[-1]:.2%}"
    )

with col2:
    st.metric(
        label="📊 Volume",
        value=f"{data['Volume'].iloc[-1]:,}",
        delta=f"{(data['Volume'].iloc[-1] - data['Volume'].mean()) / data['Volume'].mean():.1%}"
    )

with col3:
    st.metric(
        label="📉 Volatility",
        value=f"{data['Volatility'].iloc[-1]:.3f}",
        delta=f"{(data['Volatility'].iloc[-1] - data['Volatility'].mean()) / data['Volatility'].mean():.1%}"
    )

with col4:
    st.metric(
        label="📅 Data Points",
        value=f"{len(data):,}",
        delta=f"{(start_date - end_date).days} days"
    )

# EDA Section with tabs
st.markdown("---")
st.markdown("## 📊 Market Analysis")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Analysis", "📊 Volume Analysis", "📉 Returns Analysis", "📋 Statistics"])

with tab1:
    st.markdown("### Price Trends with Technical Indicators")
    fig_close, fig_volume, fig_returns, fig_volatility = create_eda_plots(data)
    
    # Enhanced price chart
    fig_close.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    st.plotly_chart(fig_close, use_container_width=True)

with tab2:
    st.markdown("### Trading Volume Analysis")
    fig_volume.update_layout(
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_volume, use_container_width=True)

with tab3:
    st.markdown("### Returns and Volatility Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_returns.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with col2:
        fig_volatility.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_volatility, use_container_width=True)

with tab4:
    st.markdown("### Statistical Summary")
    
    # Enhanced statistics display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Basic Statistics")
        st.dataframe(data.describe(), use_container_width=True)
    
    with col2:
        st.markdown("#### 🔗 Correlation Matrix")
        corr_matrix = data[['Close', 'Volume', 'Returns', 'Volatility']].corr()
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values, 
                x=corr_matrix.index, 
                y=corr_matrix.columns, 
                colorscale='RdYlBu_r',
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10}
            )
        )
        fig_corr.update_layout(
            height=400,
            title="Feature Correlation Matrix",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# Anomaly Detection Section
st.markdown("---")
st.markdown("## 🔍 Anomaly Detection Analysis")

# Progress bar for anomaly detection
progress_bar = st.progress(0)
status_text = st.empty()

# Detect anomalies with progress updates
status_text.text("🔍 Running Z-Score Analysis...")
progress_bar.progress(20)
zscore_anomalies = detect_zscore_anomalies(data, zscore_threshold)

status_text.text("🌲 Running Isolation Forest...")
progress_bar.progress(40)
iforest_anomalies = detect_iforest_anomalies(data)

status_text.text("🔍 Running DBSCAN Clustering...")
progress_bar.progress(60)
dbscan_anomalies = detect_dbscan_anomalies(data)

status_text.text("🧠 Training LSTM Network...")
progress_bar.progress(80)
lstm_anomalies = detect_lstm_anomalies(data, threshold_percentile=lstm_threshold)

status_text.text("🔄 Training Autoencoder...")
progress_bar.progress(100)
autoencoder_anomalies = detect_autoencoder_anomalies(data, threshold_percentile=autoencoder_threshold)

status_text.text("✅ Analysis Complete!")
time.sleep(1)
progress_bar.empty()
status_text.empty()


# Enhanced anomaly plots
def create_anomaly_plot(data, anomalies, title, color='red', symbol='circle'):
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'], 
        mode='lines', 
        name='Close Price',
        line=dict(color='#2E86AB', width=2)
    ))
    
    # Anomalies
    if anomalies.sum() > 0:
        fig.add_trace(go.Scatter(
            x=data.index[anomalies], 
            y=data['Close'][anomalies], 
            mode='markers', 
            name='Anomalies',
            marker=dict(color=color, size=10, symbol=symbol, line=dict(width=2, color='white'))
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# Anomaly detection results with tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Summary", "📈 Z-Score", "🌲 Isolation Forest", "🔍 DBSCAN", "🧠 LSTM", "🔄 Autoencoder"
])

with tab1:
    st.markdown("### 📊 Anomaly Detection Summary")
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Z-Score", f"{zscore_anomalies.sum()}", f"{zscore_anomalies.sum()/len(data)*100:.1f}%")
    with col2:
        st.metric("Isolation Forest", f"{iforest_anomalies.sum()}", f"{iforest_anomalies.sum()/len(data)*100:.1f}%")
    with col3:
        st.metric("DBSCAN", f"{dbscan_anomalies.sum()}", f"{dbscan_anomalies.sum()/len(data)*100:.1f}%")
    with col4:
        st.metric("LSTM", f"{lstm_anomalies.sum()}", f"{lstm_anomalies.sum()/len(data)*100:.1f}%")
    with col5:
        st.metric("Autoencoder", f"{autoencoder_anomalies.sum()}", f"{autoencoder_anomalies.sum()/len(data)*100:.1f}%")
    
    # Combined comparison plot
    st.markdown("### 🔍 All Models Comparison")
    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='#2E86AB', width=2)))
    
    if zscore_anomalies.sum() > 0:
        fig_combined.add_trace(go.Scatter(x=data.index[zscore_anomalies], y=data['Close'][zscore_anomalies], mode='markers', name='Z-Score', marker=dict(color='red', size=8, symbol='circle')))
    if iforest_anomalies.sum() > 0:
        fig_combined.add_trace(go.Scatter(x=data.index[iforest_anomalies], y=data['Close'][iforest_anomalies], mode='markers', name='Isolation Forest', marker=dict(color='green', size=8, symbol='square')))
    if dbscan_anomalies.sum() > 0:
        fig_combined.add_trace(go.Scatter(x=data.index[dbscan_anomalies], y=data['Close'][dbscan_anomalies], mode='markers', name='DBSCAN', marker=dict(color='blue', size=8, symbol='diamond')))
    if lstm_anomalies.sum() > 0:
        fig_combined.add_trace(go.Scatter(x=data.index[lstm_anomalies], y=data['Close'][lstm_anomalies], mode='markers', name='LSTM', marker=dict(color='purple', size=8, symbol='cross')))
    if autoencoder_anomalies.sum() > 0:
        fig_combined.add_trace(go.Scatter(x=data.index[autoencoder_anomalies], y=data['Close'][autoencoder_anomalies], mode='markers', name='Autoencoder', marker=dict(color='orange', size=8, symbol='star')))
    
    fig_combined.update_layout(
        title='All Models Comparison',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    st.plotly_chart(fig_combined, use_container_width=True)

with tab2:
    st.markdown("### 📈 Z-Score Anomaly Detection")
    st.plotly_chart(create_anomaly_plot(data, zscore_anomalies, 'Z-Score Anomalies', 'red', 'circle'), use_container_width=True)

with tab3:
    st.markdown("### 🌲 Isolation Forest Anomaly Detection")
    st.plotly_chart(create_anomaly_plot(data, iforest_anomalies, 'Isolation Forest Anomalies', 'green', 'square'), use_container_width=True)

with tab4:
    st.markdown("### 🔍 DBSCAN Anomaly Detection")
    st.plotly_chart(create_anomaly_plot(data, dbscan_anomalies, 'DBSCAN Anomalies', 'blue', 'diamond'), use_container_width=True)

with tab5:
    st.markdown("### 🧠 LSTM Anomaly Detection")
    st.plotly_chart(create_anomaly_plot(data, lstm_anomalies, 'LSTM Anomalies', 'purple', 'cross'), use_container_width=True)

with tab6:
    st.markdown("### 🔄 Autoencoder Anomaly Detection")
    st.plotly_chart(create_anomaly_plot(data, autoencoder_anomalies, 'Autoencoder Anomalies', 'orange', 'star'), use_container_width=True)

# Footer with additional information
st.markdown("---")
st.markdown("## 📋 Analysis Summary")

# Enhanced summary table
summary_data = {
    'Algorithm': ['Z-Score', 'Isolation Forest', 'DBSCAN', 'LSTM', 'Autoencoder'],
    'Anomalies Detected': [
        zscore_anomalies.sum(),
        iforest_anomalies.sum(),
        dbscan_anomalies.sum(),
        lstm_anomalies.sum(),
        autoencoder_anomalies.sum()
    ],
    'Percentage': [
        f"{zscore_anomalies.sum()/len(data)*100:.2f}%",
        f"{iforest_anomalies.sum()/len(data)*100:.2f}%",
        f"{dbscan_anomalies.sum()/len(data)*100:.2f}%",
        f"{lstm_anomalies.sum()/len(data)*100:.2f}%",
        f"{autoencoder_anomalies.sum()/len(data)*100:.2f}%"
    ],
    'Status': [
        '✅ Complete' if zscore_anomalies.sum() > 0 else '⚠️ No anomalies',
        '✅ Complete' if iforest_anomalies.sum() > 0 else '⚠️ No anomalies',
        '✅ Complete' if dbscan_anomalies.sum() > 0 else '⚠️ No anomalies',
        '✅ Complete' if lstm_anomalies.sum() > 0 else '⚠️ No anomalies',
        '✅ Complete' if autoencoder_anomalies.sum() > 0 else '⚠️ No anomalies'
    ]
}

summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <h4>📈 Price Watch - Advanced Stock Market Anomaly Detection</h4>
    <p>Powered by Machine Learning • Built with Streamlit • Data from Yahoo Finance</p>
    <p><em>This tool is for educational and research purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)