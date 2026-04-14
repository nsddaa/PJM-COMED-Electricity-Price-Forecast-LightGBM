# PJM Electricity Price Forecasting Model
> 📈 A LightGBM based hour-level electricity price prediction project for power trading strategy reference.

## Project Introduction
This project builds a machine learning model to predict the hourly spot price of electricity in the **PJM COMED** region (USA). 
The model results can provide data-driven decision support for power traders, including price forecasting and risk control.

## Tech Stack
- **Language**: Python 3.8+
- **Data Processing**: Pandas, NumPy
- **Modeling**: LightGBM (Regressor)
- **Visualization**: Matplotlib, Seaborn

## Project Structure
├── data/ # Dataset directory (include historical price/load/temperature)
├── results/ # Generated forecast charts and feature importance graphs
├── main.ipynb # Main training code (Jupyter Notebook)
├── requirements.txt # Python dependencies
└── README.md # Project description
