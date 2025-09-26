# Price Watch: Advanced Stock Market Anomaly Detection System

## Project Overview

**Price Watch** is a comprehensive stock market analysis platform that leverages cutting-edge machine learning techniques to detect anomalies and unusual patterns in stock price movements. This sophisticated system provides traders, analysts, and investors with powerful tools to identify market irregularities, potential trading opportunities, and risk assessment insights.

### 🎥 Demo Video

https://github.com/user-attachments/assets/d1b5c802-1635-44d7-b92c-d341fd908c91

**Local Demo Video:**
```html
<video width="100%" controls>
  <source src="assets/Demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

<video width="100%" controls>
  <source src="assets/Demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


### Key Highlights
- **Multi-Algorithm Approach**: Implements 5 different anomaly detection algorithms for comprehensive analysis
- **Real-Time Analysis**: Interactive Streamlit dashboard for live market analysis
- **Advanced Visualizations**: Rich, interactive charts and graphs for data exploration
- **Customizable Parameters**: Flexible configuration for different stocks and time periods
- **Professional-Grade**: Built with industry-standard libraries and best practices

### Why Price Watch?
In today's volatile financial markets, identifying unusual price movements and market anomalies is crucial for:
- **Risk Management**: Early detection of potential market disruptions
- **Trading Opportunities**: Spotting unusual patterns that may indicate profitable trades
- **Market Research**: Understanding market behavior and sentiment
- **Portfolio Optimization**: Making informed investment decisions

## 🚀 Core Features

### 📊 Advanced Data Analysis
- **Real-time Data Integration**: Seamless connection to Yahoo Finance API via yfinance
- **Comprehensive EDA**: Multi-dimensional exploratory data analysis with interactive visualizations
- **Technical Indicators**: Built-in calculation of moving averages, Bollinger Bands, volatility metrics
- **Statistical Analysis**: Detailed correlation matrices and descriptive statistics

### 🤖 Machine Learning Algorithms
Price Watch implements five state-of-the-art anomaly detection methods:

1. **Z-Score Analysis** 📈
   - Statistical outlier detection based on standard deviations
   - Ideal for identifying extreme price movements
   - Configurable threshold parameters

2. **Isolation Forest** 🌲
   - Unsupervised learning approach for anomaly detection
   - Excellent for detecting complex, non-linear patterns
   - Robust against high-dimensional data

3. **DBSCAN Clustering** 🔍
   - Density-based spatial clustering for noise detection
   - Identifies outliers in multi-dimensional feature space
   - Adaptive to different data distributions

4. **LSTM Neural Networks** 🧠
   - Deep learning approach using Long Short-Term Memory
   - Captures temporal dependencies in price sequences
   - Advanced pattern recognition capabilities

5. **Autoencoder Networks** 🔄
   - Neural network-based reconstruction error analysis
   - Learns normal patterns and flags deviations
   - Highly effective for complex anomaly detection

### 🎯 Interactive Dashboard
- **Real-time Visualization**: Live charts and graphs with Plotly integration
- **Customizable Parameters**: Adjustable timeframes, stock symbols, and algorithm settings
- **Comparative Analysis**: Side-by-side comparison of all detection methods
- **Export Capabilities**: Save results and visualizations for further analysis

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/price-watch.git
   cd price-watch
   ```

2. **Create Virtual Environment** (Highly Recommended)
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python -c "import streamlit, pandas, numpy, yfinance, plotly, sklearn, tensorflow; print('All dependencies installed successfully!')"
   ```

### 🚀 Quick Start
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Launch Price Watch
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 🎯 Getting Started with Price Watch

#### Option 1: Interactive Web Application (Recommended)
```bash
# Launch the main Price Watch dashboard
streamlit run app.py
```

**Features Available:**
- **Stock Selection**: Enter any stock ticker symbol (e.g., AAPL, TSLA, GME)
- **Date Range**: Select custom start and end dates for analysis
- **Real-time Analysis**: Get instant results with interactive visualizations
- **Method Comparison**: Compare all 5 anomaly detection algorithms side-by-side

#### Option 2: Jupyter Notebook Analysis
```bash
# For detailed analysis and experimentation
jupyter notebook Anomalies Detection.ipynb
```

**Advanced Features:**
- **Custom Algorithm Tuning**: Adjust parameters for each detection method
- **Statistical Analysis**: Deep dive into correlation matrices and distributions
- **Model Training**: Fine-tune neural network architectures
- **Export Results**: Save analysis results and visualizations

### 🎛️ Dashboard Navigation

1. **Sidebar Controls**
   - **Stock Ticker**: Enter any valid stock symbol
   - **Date Range**: Select analysis period
   - **Algorithm Settings**: Adjust detection sensitivity

2. **Main Dashboard Sections**
   - **📊 Exploratory Data Analysis**: Price charts, volume, returns, volatility
   - **🔍 Anomaly Detection**: Individual algorithm results
   - **📈 Comparative Analysis**: Side-by-side method comparison
   - **📋 Summary Statistics**: Performance metrics and anomaly counts

3. **Interactive Features**
   - **Zoom & Pan**: Navigate through time series data
   - **Hover Details**: Get precise values and timestamps
   - **Export Charts**: Save visualizations as images
   - **Real-time Updates**: Refresh data with new parameters

## 📁 Project Structure

```
price-watch/
├── 📊 app.py                          # Main Streamlit application
├── 📓 Anomalies Detection.ipynb       # Jupyter notebook for detailed analysis
├── 📋 requirements.txt                # Python dependencies
├── 📖 README.md                       # Project documentation
├── 🗂️ venv/                          # Virtual environment directory
│   ├── Scripts/                       # Windows activation scripts
│   ├── Lib/                          # Installed packages
│   └── pyvenv.cfg                    # Virtual environment config
└── 📁 data/                          # Stock data storage (auto-created)
    └── 📁 models/                    # Saved ML models (auto-created)
```

### 📄 File Descriptions

- **`app.py`**: Main Streamlit dashboard with interactive anomaly detection
- **`Anomalies Detection.ipynb`**: Comprehensive Jupyter notebook for analysis
- **`requirements.txt`**: All necessary Python packages and versions
- **`venv/`**: Isolated Python environment for dependency management
- **`data/`**: Automatically created directory for cached stock data
- **`models/`**: Directory for saving trained machine learning models

## 🔬 Technical Methodology

### 📊 Data Pipeline

1. **Data Acquisition**
   - **Source**: Yahoo Finance API via yfinance library
   - **Data Points**: OHLCV (Open, High, Low, Close, Volume)
   - **Time Range**: Configurable (default: 2020-2023)
   - **Frequency**: Daily stock data

2. **Data Preprocessing**
   - **Cleaning**: Handle missing values and outliers
   - **Feature Engineering**: Calculate technical indicators
     - Returns: Percentage change in closing prices
     - Volatility: 20-day rolling standard deviation
     - Moving Averages: 20-day simple moving average
     - Bollinger Bands: Upper and lower price bands
   - **Normalization**: StandardScaler for ML algorithms

3. **Exploratory Data Analysis (EDA)**
   - **Price Trends**: Interactive candlestick and line charts
   - **Volume Analysis**: Trading volume patterns over time
   - **Returns Distribution**: Statistical analysis of price changes
   - **Correlation Matrix**: Feature relationships and dependencies

### 🤖 Anomaly Detection Algorithms

#### 1. Z-Score Analysis
- **Method**: Statistical outlier detection
- **Threshold**: 3 standard deviations from mean
- **Use Case**: Extreme price movements
- **Formula**: `|x - μ| / σ > threshold`

#### 2. Isolation Forest
- **Method**: Unsupervised ensemble learning
- **Contamination**: 1% of data points
- **Features**: Close, Volume, Returns, Volatility
- **Advantage**: Handles high-dimensional data effectively

#### 3. DBSCAN Clustering
- **Method**: Density-based spatial clustering
- **Parameters**: eps=0.5, min_samples=5
- **Outlier Detection**: Points not assigned to any cluster
- **Advantage**: Adaptive to data distribution

#### 4. LSTM Neural Network
- **Architecture**: 50 LSTM units + Dense layer
- **Sequence Length**: 20 days
- **Training**: 50 epochs, batch size 32
- **Anomaly Detection**: Reconstruction error > 95th percentile
- **Advantage**: Captures temporal dependencies

#### 5. Autoencoder Network
- **Architecture**: Encoder-Decoder with 2D bottleneck
- **Layers**: 4→8→4→2→4→8→4
- **Training**: 100 epochs, batch size 32
- **Anomaly Detection**: Reconstruction error > 95th percentile
- **Advantage**: Learns normal patterns effectively

### 📈 Performance Evaluation

- **Metrics**: Anomaly count comparison across methods
- **Visualization**: Overlay anomalies on price charts
- **Comparison**: Side-by-side method evaluation
- **Validation**: Cross-validation with different time periods

## 📊 Results & Insights

### 🎯 Key Findings

Price Watch successfully identifies various types of market anomalies:

- **📈 Price Spikes**: Sudden, significant price movements
- **📉 Volume Anomalies**: Unusual trading volume patterns
- **🔄 Volatility Shifts**: Changes in market volatility regimes
- **⚡ Market Events**: Correlation with news, earnings, or external factors

### 📈 Performance Metrics

| Algorithm | Detection Rate | Precision | Use Case |
|-----------|---------------|-----------|----------|
| Z-Score | High | Medium | Extreme price movements |
| Isolation Forest | Medium | High | Complex patterns |
| DBSCAN | Medium | High | Density-based outliers |
| LSTM | High | High | Temporal patterns |
| Autoencoder | High | High | Reconstruction errors |

### 🔍 Real-World Applications

- **Risk Management**: Early warning system for portfolio protection
- **Trading Signals**: Identification of potential entry/exit points
- **Market Research**: Understanding of market behavior and sentiment
- **Compliance**: Detection of unusual trading activities

## 🎛️ Streamlit Dashboard Features

### 📊 Interactive Visualizations
- **Real-time Charts**: Live price data with technical indicators
- **Anomaly Overlays**: Visual markers for detected anomalies
- **Comparative Analysis**: Side-by-side algorithm comparison
- **Export Functionality**: Save charts and data for reports

### 🎯 User Interface
- **Stock Selection**: Any ticker symbol with instant data loading
- **Date Range Picker**: Flexible time period selection
- **Parameter Controls**: Adjustable algorithm sensitivity
- **Results Summary**: Comprehensive statistics and metrics

### 📱 Responsive Design
- **Mobile-Friendly**: Optimized for all screen sizes
- **Fast Loading**: Efficient data processing and caching
- **Intuitive Navigation**: User-friendly interface design
- **Professional Layout**: Clean, modern dashboard aesthetics

## 🚀 Future Enhancements

### 🔮 Planned Features
- **📰 Sentiment Analysis**: Integration with news and social media sentiment
- **📊 Advanced Indicators**: RSI, MACD, Stochastic Oscillator
- **🌐 Multi-Asset Support**: Cryptocurrency, Forex, Commodities
- **⚡ Real-time Processing**: Live data streaming and instant alerts
- **🤖 Ensemble Methods**: Combined algorithm voting systems
- **📱 Mobile App**: Native iOS and Android applications

### 🔧 Technical Improvements
- **⚡ Performance Optimization**: GPU acceleration for neural networks
- **🗄️ Database Integration**: Persistent storage for historical analysis
- **🔐 API Development**: RESTful API for third-party integrations
- **📈 Advanced Analytics**: Machine learning model performance tracking
- **🌍 Cloud Deployment**: Scalable cloud infrastructure

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### 🛠️ Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### 📝 Contribution Guidelines
- **Code Style**: Follow PEP 8 for Python code
- **Documentation**: Update README and docstrings for new features
- **Testing**: Add tests for new functionality
- **Issues**: Report bugs and suggest enhancements

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### 🏆 Core Libraries
- **yfinance**: Seamless Yahoo Finance data integration
- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced data visualization capabilities
- **TensorFlow/Keras**: Deep learning and neural network implementation
- **scikit-learn**: Machine learning algorithms and preprocessing

### 🌟 Community
- **Open Source Community**: For the incredible ecosystem of Python libraries
- **Financial Data Providers**: Yahoo Finance for reliable market data
- **Contributors**: All developers who have contributed to this project

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ for the financial analysis community

</div>

