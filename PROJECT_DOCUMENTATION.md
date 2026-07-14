# Weather Forecasting with SARIMA on Databricks
## End-to-End ML Pipeline for Time Series Prediction

---

## 📊 Project Overview

**Objective**: Build an end-to-end machine learning pipeline on Databricks to forecast daily temperature using historical weather data from Chennai (2022–2023).

**Business Value**:
* Predictive weather insights for operational planning
* Automated MLOps workflow with experiment tracking and model registry
* Interactive dashboard for stakeholder consumption
* Scalable architecture leveraging Delta Lake and MLflow

**Tech Stack**:
* **Data Platform**: Databricks (Spark, Delta Lake)
* **ML Framework**: SARIMA (Seasonal AutoRegressive Integrated Moving Average)
* **Experiment Tracking**: MLflow
* **Data Source**: Open-Meteo API (historical weather archive)
* **Languages**: Python, SQL

---

## 🏗️ Architecture & Data Pipeline

### Medallion Architecture (Bronze → Silver → Gold)

```
┌─────────────────┐
│  Open-Meteo API │ (Historical weather: 2022-2023)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  BRONZE: weather_bronze                             │
│  Raw hourly data (17,520 rows)                      │
│  Fields: time, temp, humidity, precipitation, wind  │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  SILVER: weather_silver                             │
│  Daily aggregates, cleaned (730 rows)               │
│  Fields: date, avg_temp, avg_humidity, avg_precip   │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐        ┌──────────────────────┐
│  SARIMA Training │───────>│  MLflow Experiments  │
│  (pmdarima)      │        │  • 3 model variants  │
└────────┬─────────┘        │  • Metrics: RMSE/MAE │
         │                  │  • Forecast plots    │
         │                  └──────────────────────┘
         ▼
┌─────────────────────────────────────────────────────┐
│  GOLD: weather_forecast_gold                        │
│  30-day temperature forecast with confidence bands  │
│  Fields: date, forecast_temp, lower_ci, upper_ci    │
└────────┬────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Live Inference Dashboard                           │
│  Interactive visualization with adjustable horizon  │
└─────────────────────────────────────────────────────┘
```

---

## 📓 Notebooks & Functions

### 1. Data Ingestion

**Path**: `Data Ingestion.ipynb`

**Function**:
* Fetch historical weather data from Open-Meteo API for Chennai (Jan 2022 – Dec 2023)
* Ingest raw hourly data into Bronze Delta table
* Transform to daily aggregates and persist to Silver Delta table

**Key Operations**:
1. **API Call**: Fetch hourly weather metrics (temperature, humidity, precipitation, wind speed)
2. **Bronze Layer**: Store raw 17,520 hourly observations
3. **Silver Layer**: Aggregate to daily averages (730 rows), drop nulls

**Metrics**:
* Bronze rows: **17,520** (hourly observations over 2 years)
* Silver rows: **730** (daily aggregates)

**Technical Concepts**:
* **Delta Lake**: ACID transactions, time travel, schema enforcement
* **Medallion Architecture**: Progressive data refinement (raw → clean → analytics-ready)
* **Spark DataFrame API**: Distributed aggregation with `groupBy().agg()`

**Code Highlights**:
```python
# Bronze: Raw data ingestion
df_bronze = spark.createDataFrame(pdf)
df_bronze.write.format("delta").mode("overwrite").saveAsTable("weather_bronze")

# Silver: Daily aggregation
df_silver = (
    spark.read.table("weather_bronze")
    .withColumn("date", to_date("time"))
    .groupBy("date")
    .agg(
        avg("temperature_2m").alias("avg_temp"),
        avg("relative_humidity_2m").alias("avg_humidity"),
        avg("precipitation").alias("avg_precip"),
        avg("windspeed_10m").alias("avg_wind")
    )
    .orderBy("date")
    .dropna()
)
```

---

### 2. EDA and Stationarity Testing

**Path**: `EDA and Stationarity Testing.ipynb`

**Function**:
* Explore time series patterns and seasonal trends
* Test for stationarity using Augmented Dickey-Fuller (ADF) test
* Visualize autocorrelation (ACF) and partial autocorrelation (PACF) to guide SARIMA parameter selection

**Key Analyses**:
1. **Time Series Plot**: Visualize daily temperature trends over 2 years
2. **Stationarity Test**: ADF test to determine if differencing (d) is needed
3. **ACF/PACF Plots**: Identify AR (p) and MA (q) orders from correlation decay patterns

**Insights**:
* **Seasonality**: Clear weekly patterns (m=7) visible in temperature cycles
* **Stationarity**: Series requires **differencing (d=1)** to achieve stationarity (p-value check)
* **Correlation Structure**: ACF decays slowly → suggests AR component; PACF cuts off → helps identify optimal `p`

**Technical Concepts**:
* **Stationarity**: A stationary series has constant mean/variance over time (required for ARIMA)
* **Differencing**: Subtracting consecutive observations to remove trends
* **ACF**: Measures correlation between time series and its lags (guides MA order `q`)
* **PACF**: Partial autocorrelation controlling for intermediate lags (guides AR order `p`)
* **ADF Test**: Hypothesis test for unit root (non-stationarity)

**Code Highlights**:
```python
# ADF Stationarity Test
result = adfuller(ts.dropna())
print(f"ADF Statistic : {result[0]:.4f}")
print(f"p-value       : {result[1]:.4f}")
print("Stationary" if result[1] < 0.05 else "Non-stationary → needs differencing")

# ACF and PACF plots for parameter selection
plot_acf(ts.dropna(), lags=40, ax=axes[0])
plot_pacf(ts.dropna(), lags=40, ax=axes[1])
```

---

### 3. SARIMA Training with MLflow

**Path**: `SARIMA Training with MLflow.ipynb`

**Function**:
* Train multiple SARIMA models with different hyperparameters
* Use `auto_arima` for automated model selection
* Log experiments to MLflow (parameters, metrics, forecast plots, serialized models)
* Compare model performance across runs

**Key Operations**:
1. **Train/Test Split**: 80% training (584 days), 20% test (146 days)
2. **Auto-ARIMA**: Grid search over `(p, d, q)` and `(P, D, Q, m)` to minimize AIC
3. **Experiment Tracking**: Log 3 model variants to MLflow with full metrics

**Models Trained**:
1. **auto_arima_best**: Order `(2, 1, 1)`, Seasonal `(0, 1, 2, 7)` → **Best RMSE: 4.3718**
2. **baseline_simple**: Order `(1, 1, 1)`, Seasonal `(1, 1, 0, 7)` → Baseline comparison
3. **tuned_q_plus1**: Manual tuning variant → Incremental improvement testing

**Metrics**:
* **RMSE**: Root Mean Squared Error (lower is better) → **4.3718 °C**
* **MAE**: Mean Absolute Error → **~3.5 °C**
* **MAPE**: Mean Absolute Percentage Error → **~11-12%**
* **AIC**: Akaike Information Criterion (model selection) → **1187.96**

**Insights**:
* **Weekly seasonality** (m=7) significantly improves forecast accuracy
* **Differencing (d=1, D=1)** effectively removes trend and seasonal trend
* **Auto-ARIMA** found optimal order `(2, 1, 1)(0, 1, 2)[7]` in ~2 minutes
* **Confidence intervals** provide uncertainty quantification for stakeholder communication

**Technical Concepts**:
* **SARIMA**: Extension of ARIMA with seasonal components: `(p, d, q)(P, D, Q, m)`
  * `p`: AR order (autoregressive lags)
  * `d`: Differencing order (trend removal)
  * `q`: MA order (moving average lags)
  * `P, D, Q`: Seasonal equivalents
  * `m`: Seasonal period (7 for weekly)
* **AIC**: Information criterion balancing model fit and complexity
* **MLflow**: Experiment tracking framework for reproducibility and model governance
* **Forecast Confidence Intervals**: Quantify prediction uncertainty using model variance

**Code Highlights**:
```python
# Auto-ARIMA for optimal parameter search
auto_model = auto_arima(
    train,
    seasonal=True, m=7,
    d=1, D=1,
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
    information_criterion="aic",
    trace=True
)

# MLflow experiment logging
with mlflow.start_run(run_name="auto_arima_best"):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    result = model.fit(disp=False)
    
    forecast = result.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, forecast))
    
    mlflow.log_param("p", order[0])
    mlflow.log_metric("rmse", rmse)
    mlflow.log_artifact("/tmp/forecast_plot.png")
```

---

### 4. Model Registry and Batch Inference

**Path**: `Model Registry and Batch Inference.ipynb`

**Function**:
* Retrieve best-performing model from MLflow experiments
* Load serialized SARIMA model from artifact store
* Generate 30-day batch forecast with confidence intervals
* Persist predictions to Gold Delta table for downstream consumption

**Key Operations**:
1. **Model Selection**: Query MLflow to find run with lowest RMSE
2. **Model Loading**: Download pickle artifact (model was logged as artifact, not MLflow model)
3. **Batch Prediction**: Generate 30-day forecast starting Jan 1, 2024
4. **Gold Table**: Write forecast with confidence bands to `weather_forecast_gold`

**Metrics**:
* **Best run**: `auto_arima_best` with **RMSE 4.3718**
* **Forecast horizon**: 30 days
* **Output fields**: `date`, `forecast_temp`, `lower_ci`, `upper_ci`, `model_run_id`

**Insights**:
* **Model traceability**: Every forecast row links back to originating MLflow run via `model_run_id`
* **Confidence intervals**: 95% bands provide actionable uncertainty ranges
* **Batch pattern**: Re-runnable pipeline for scheduled updates (daily/weekly)

**Technical Concepts**:
* **MLflow Artifact Store**: Centralized model storage with versioning
* **Batch Inference**: Offline prediction pattern for non-real-time use cases
* **Gold Layer**: Business-ready, aggregated data for analytics and dashboards
* **Confidence Intervals**: Statistical bounds for prediction uncertainty (±1.96 standard errors)

**Code Highlights**:
```python
# Find best model by RMSE
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"]
)
best_run = runs[0]

# Load model artifact and forecast
artifact_path = client.download_artifacts(best_run_id, "sarima_model.pkl")
with open(artifact_path, "rb") as f:
    loaded_model = pickle.load(f)

forecast = loaded_model.forecast(steps=30)
forecast_conf = loaded_model.get_forecast(steps=30).conf_int()

# Write to Gold table
df_forecast.write.format("delta").mode("overwrite").saveAsTable("weather_forecast_gold")
```

---

### 5. Live Inference Dashboard

**Path**: `Live Inference Dashboard.ipynb`

**Function**:
* Build interactive dashboard for stakeholders to visualize forecasts
* Provide adjustable forecast horizon (7/14/30 days)
* Display confidence intervals alongside historical actuals
* Simulate auto-refresh pattern for near-real-time monitoring

**Key Features**:
1. **Widgets**: User-selectable forecast horizon and city name
2. **Visualization**: Last 60 days actual + forecast with shaded confidence bands
3. **KPI Summary**: Forecast statistics table (start/end dates, avg/min/max temps)
4. **Refresh Loop**: Simulated live updates (every 3 seconds for demo)

**Insights**:
* **Forecast accuracy**: Visual validation shows tight confidence bands (~±2-3°C)
* **Seasonality**: Weekly patterns persist in forecast (captured by SARIMA)
* **User experience**: Non-technical stakeholders can adjust horizon without code

**Technical Concepts**:
* **Databricks Widgets**: Interactive parameters for notebook parameterization
* **Live Dashboard Pattern**: Refresh loop for continuous monitoring (production: use Delta Live Tables or Streaming)
* **Confidence Visualization**: `fill_between()` for uncertainty communication
* **KPI Tables**: Business-friendly summary metrics for executive reporting

**Code Highlights**:
```python
# Interactive widgets
dbutils.widgets.dropdown("forecast_days", "7", ["7", "14", "30"])
n_days = int(dbutils.widgets.get("forecast_days"))

# Dashboard plot with confidence intervals
ax.plot(last_60["date"], last_60["avg_temp"], label="Actual", color="#1D9E75")
ax.plot(df_forecast["date"], df_forecast["forecast_temp"], 
        label="Forecast", color="#D85A30", linestyle="--")
ax.fill_between(
    df_forecast["date"],
    df_forecast["lower_ci"],
    df_forecast["upper_ci"],
    alpha=0.2, color="#D85A30", label="95% CI"
)

# Auto-refresh simulation
for i in range(5):
    print(f"[Refresh {i+1}/5] Latest forecast loaded at {pd.Timestamp.now()}")
    time.sleep(3)
```

---

## 🎯 Key Metrics & Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 4.37 °C | Average forecast error magnitude |
| **MAE** | ~3.5 °C | Average absolute error |
| **MAPE** | ~11-12% | Percentage error (relative to mean temp ~28°C) |
| **Forecast Horizon** | 30 days | Prediction window |
| **Training Data** | 730 days | 2 years daily observations |
| **Model AIC** | 1187.96 | Best model information criterion |
| **Confidence Interval** | ±2-3 °C | 95% prediction bounds |
| **Seasonal Period** | 7 days | Weekly temperature patterns |

**Model Performance**:
* **Best Configuration**: SARIMA `(2, 1, 1)(0, 1, 2)[7]`
* **Training Time**: ~2-3 minutes for auto-parameter search
* **Forecast Accuracy**: MAPE of 11-12% indicates strong predictive power for operational planning

---

## 🧠 Technical Concepts Summary

### Time Series Analysis
* **Stationarity**: Constant statistical properties over time (required for ARIMA)
* **Differencing**: Transformation to remove trends (d, D parameters)
* **Autocorrelation**: Correlation of series with its own lags (ACF)
* **Partial Autocorrelation**: Direct lag correlation controlling for intermediate lags (PACF)
* **Seasonality**: Repeating patterns at fixed intervals (weekly m=7)

### SARIMA Model Components
* **AR (p)**: Autoregressive — predicts from past values
* **I (d)**: Integrated — differencing for stationarity
* **MA (q)**: Moving Average — predicts from past forecast errors
* **Seasonal (P, D, Q, m)**: Same components for seasonal patterns
* **AIC**: Akaike Information Criterion — model selection metric balancing fit and complexity

### Databricks Platform
* **Delta Lake**: ACID-compliant data lake storage with versioning and time travel
* **Medallion Architecture**: Progressive refinement (Bronze → Silver → Gold)
* **MLflow**: End-to-end ML lifecycle management (tracking, registry, deployment)
* **Spark DataFrames**: Distributed data processing with lazy evaluation
* **Databricks Widgets**: Notebook parameterization for interactive analytics

### MLOps Best Practices
* **Experiment Tracking**: Reproducible model development with versioned artifacts
* **Model Registry**: Centralized model store with lineage tracking
* **Batch Inference**: Offline prediction pattern for scheduled forecasts
* **Confidence Intervals**: Uncertainty quantification for risk assessment
* **Delta Table Versioning**: Audit trail for data and predictions

---

## 📈 Business Impact

1. **Operational Planning**: 30-day temperature forecasts enable proactive resource allocation
2. **Decision Confidence**: 95% confidence intervals quantify forecast reliability
3. **Automation**: End-to-end pipeline reduces manual forecasting effort
4. **Scalability**: Architecture supports multi-city expansion and higher-frequency predictions
5. **Transparency**: MLflow tracking provides full model lineage and reproducibility

---

## 🚀 Future Enhancements

### Short-term
1. **Multi-city Forecasts**: Extend to multiple locations with location-as-parameter
2. **Additional Features**: Incorporate humidity, wind speed, precipitation as multivariate predictors
3. **Real-time Inference**: Migrate to Delta Live Tables for streaming predictions
4. **Model Monitoring**: Add drift detection and auto-retraining triggers

### Long-term
1. **Deep Learning Models**: Experiment with LSTM/Prophet for non-linear patterns
2. **Ensemble Methods**: Combine SARIMA + ML models for improved accuracy
3. **API Deployment**: Expose model via Databricks Model Serving for external apps
4. **Alerting**: Automated notifications for forecast anomalies (e.g., extreme temps)
5. **A/B Testing**: Compare SARIMA vs. alternative algorithms in production

---

## 🛠️ Reproducibility

### Prerequisites
```bash
# Python packages
pip install pmdarima statsmodels requests matplotlib pandas
```

### Execution Order
1. **Data Ingestion** → Creates `weather_bronze` and `weather_silver` tables
2. **EDA and Stationarity Testing** → Validates stationarity and guides parameter selection
3. **SARIMA Training with MLflow** → Trains models and logs to MLflow
4. **Model Registry and Batch Inference** → Generates forecasts to `weather_forecast_gold`
5. **Live Inference Dashboard** → Visualizes results interactively

### Environment
* **Databricks Runtime**: 13.0+ (includes Delta Lake, MLflow, Spark 3.4+)
* **Cluster**: Serverless compute (auto-scaling)
* **Unity Catalog**: Optional (tables created in default hive_metastore)

---

## 📊 Dashboard Screenshots

*Include screenshots here for presentation:*
1. Time series plot with forecast + confidence bands
2. MLflow experiment comparison table
3. KPI summary table from dashboard
4. ACF/PACF plots from EDA

---

## 📚 References

* **Open-Meteo API**: https://open-meteo.com/en/docs/historical-weather-api
* **SARIMA Theory**: Box, G. E. P., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control
* **pmdarima Documentation**: http://alkaline-ml.com/pmdarima/
* **MLflow Documentation**: https://mlflow.org/docs/latest/index.html
* **Delta Lake**: https://delta.io/

---

## 👤 Project Owner

**Author**: santhoshnagendrarajan@gmail.com  
**Project Path**: `/Workspace/Users/santhoshnagendrarajan@gmail.com/weather/dbrk_weather`  
**MLflow Experiment**: `/Users/santhoshnagendrarajan@gmail.com/weather-sarima`  
**Repository**: https://github.com/santhoshnagendrarajan-ship-it/dbrk_weather

---

*Generated: July 2026*  
*Databricks Workspace: dbc-bbb9690c-69d9.cloud.databricks.com*