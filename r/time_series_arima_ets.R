# gdp-time-series-forecasting
Time series forecasting of Germany GDP using ARIMA and ETS (R)
library(forecast)

# Load data
dane = read.table("data/pkb.txt", header = FALSE, sep = "\t")
head(dane)
str(dane)

# Create time series object (quarterly data)
pkb_ts <- ts(dane, start = c(1991, 1), frequency = 4)
plot(pkb_ts,
     main = "Germany GDP time series (1991–2023, quarterly data)",
     ylab = "GDP",
     xlab = "Year")

# Split data into training (80%) and test (20%) sets
n <- length(pkb_ts)
train_size <- round(0.8 * n)

pkb_train <- window(
  pkb_ts,
  start = c(1991, 1),
  end = c(1991 + (train_size %/% 4) - 1, train_size %% 4)
)
pkb_test <- window(
  pkb_ts,
  start = c(1991 + (train_size %/% 4), (train_size %% 4) + 1)
)

# Time series decomposition – trend and seasonality analysis
d <- decompose(pkb_train, type = "additive")
plot(d)

# Outlier detection
outliers <- tsoutliers(pkb_train)
print(outliers)

# Outlier removal
pkb_clean <- tsclean(pkb_train)

# Plot before and after cleaning
plot(pkb_train, col = "red")
lines(pkb_clean, col = "blue")
legend("topleft",
       legend = c("Original", "After tsclean()"),
       col = c("red", "blue"),
       lty = 1)

# Box-Cox transformation
lambda <- BoxCox.lambda(pkb_clean)
lambda  # ~0.95
pkb_bc <- BoxCox(pkb_clean, lambda = lambda)

# Differencing – remove trend and seasonality
pkb_bc_diff <- diff(pkb_bc, differences = 1)     # First-order differencing
pkb_bc_seas_diff <- diff(pkb_bc_diff, lag = 4)   # Remove quarterly seasonality
plot(pkb_bc_seas_diff, main = "GDP after differencing")

# Check outliers after differencing
tsoutliers(pkb_bc_seas_diff)

# Autocorrelation (ACF) and partial autocorrelation (PACF)
Acf(pkb_bc_seas_diff, lag.max = 40, main = "ACF")
Pacf(pkb_bc_seas_diff, lag.max = 40, main = "PACF")

# Lag plot
lag.plot(pkb_clean, lags = 4, main = "Lag plot of GDP values")

# Fit ARIMA(4,1,0) model
model_arima_4 <- Arima(pkb_bc_seas_diff, order = c(4, 1, 0))
summary(model_arima_4)
checkresiduals(model_arima_4)  # Residual diagnostics

# Fit ARIMA(4,1,1) model
model_arima_4_1 <- Arima(pkb_bc_seas_diff, order = c(4, 1, 1))
summary(model_arima_4_1)
checkresiduals(model_arima_4_1)

# Automatic ARIMA model selection
model_auto <- auto.arima(pkb_bc_seas_diff)
summary(model_auto)
checkresiduals(model_auto)

# Residual normality test
shapiro.test(residuals(model_auto))

# Histogram of residuals
hist(residuals(model_auto),
     main = "Histogram of ARIMA residuals",
     col = "lightblue")

# Ljung-Box test and residual ACF
Box.test(residuals(model_auto), type = "Ljung-Box", lag = 10)
Acf(residuals(model_auto), lag.max = 40, main = "ACF of ARIMA residuals")

# Forecasting next 8 quarters
forecast_auto <- forecast(model_auto, h = 8)
plot(forecast_auto, main = "GDP forecast for next 8 quarters")
lines(pkb_test, col = "red")  # Test data

# Forecast accuracy
accuracy(forecast_auto, pkb_test)

# Compare actual vs forecasted values
comparison <- data.frame(
  Year = time(pkb_test),
  Actual = as.numeric(pkb_test),
  Forecast = as.numeric(forecast_auto$mean),
  Error = as.numeric(pkb_test - forecast_auto$mean)
)
print(comparison)

###### 
# ETS model

model_ets <- ets(pkb_train)
summary(model_ets)

# ETS fitted values plot
plot(model_ets)

# Forecast using ETS model
forecast_ets <- forecast(model_ets, h = length(pkb_test))

# Accuracy of ETS model
accuracy(forecast_ets, pkb_test)

# ETS forecast vs actual values
plot(forecast_ets, main = "GDP forecast using ETS model")
lines(pkb_test, col = "red")
legend("topleft",
       legend = c("Actual", "Forecast"),
       col = c("red", "blue"),
       lty = 1,
       lwd = 2)
