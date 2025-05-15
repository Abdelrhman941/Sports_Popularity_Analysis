# Sports Popularity Analysis and Forecasting

## Overview
This project analyzes and forecasts sports popularity trends using data collected from Google Trends. It employs advanced time series analysis and machine learning techniques to predict future popularity trends for various sports, including Football, Baseball, Tennis, Table Tennis, and Volleyball.

## Dataset
The dataset (`major_events_full_updated.csv`) is collected from Google Trends and includes:
- Historical popularity data for multiple sports
- Major sporting events indicators
- YouTube views score (where available)
- Monthly data points from 2004 onwards

## Features

### Time Series Analysis
- Advanced ensemble forecasting combining Prophet and SARIMA models
- Anomaly detection and outlier handling
- Seasonal trend decomposition
- Statistical tests for stationarity

### Interactive Visualization
- Dynamic plotly-based visualizations
- Major events highlighting with star markers
- Confidence intervals for forecasts
- Training vs. Test data visualization

### Model Components
- **Prophet Models**: Specialized models for each sport (`/Model/prophet_model_*.pkl`)
- **SARIMA Models**: Sport-specific SARIMA models (`/Model/sarima_model_*.pkl`)
- **Ensemble Approach**: Combines predictions from multiple models for improved accuracy

## Project Structure
```
├── DataSets/
│   └── major_events_full_updated.csv
├── Model/
│   ├── prophet_model_*.pkl
│   └── sarima_model_*.pkl
├── plots/
│   ├── sports_plot.html
│   ├── Football_plot.html
│   └── sarima_forecast.html
├── gui.py
└── main.ipynb
```

## Technical Features
- **Modern UI**: Dark-themed Streamlit interface with custom styling
- **Data Preprocessing**: Handles missing values and outliers
- **Model Evaluation**: Includes MAE, MSE, and R² metrics
- **Interactive Components**: Dynamic sport selection and forecast period adjustment

## Dependencies
- streamlit
- prophet
- pandas
- numpy
- plotly
- statsmodels
- scikit-learn
- pmdarima

## Usage
To run the application:
```bash
streamlit run gui.py
```

## Analysis Capabilities
- Long-term trend analysis
- Seasonal pattern identification
- Major event impact assessment
- Cross-sport popularity comparison
- Future popularity forecasting

## Visualization Examples
- Time series plots with confidence intervals
- Seasonal decomposition visualizations
- Trend comparison across different sports
- Major events impact visualization

## Future Improvements
- Integration of additional data sources
- Enhanced feature engineering
- Real-time data updates
- Advanced anomaly detection
- Cross-platform compatibility