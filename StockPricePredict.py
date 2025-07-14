import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.title("📈 Stock Price Prediction App")

# Load your CSV from your path
data = pd.read_csv("StockData.csv")
df = pd.DataFrame(data)

st.subheader("Raw Data")
st.write(df)

# Preprocess dates
df['Date'] = pd.to_datetime(df['Date'])
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

X = df[['Days']]
y = df['Stock_Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Show predictions
st.subheader("Predictions")
st.write(pd.DataFrame({'Actual': y_test.values, 'Predicted': predictions}))

# Metrics
r2 = r2_score(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

st.subheader("Model Performance")
st.write(f"**R² Score:** {r2:.4f}")
st.write(f"**Mean Squared Error:** {mse:.4f}")

# Plot
fig, ax = plt.subplots()
ax.scatter(X_test, y_test, color="blue", label="Actual")
ax.scatter(X_test, predictions, color="red", label="Predicted")
ax.set_xlabel("Days since start")
ax.set_ylabel("Stock Price")
ax.set_title("Actual vs Predicted Stock Price")
ax.legend()
ax.grid()

st.pyplot(fig)
