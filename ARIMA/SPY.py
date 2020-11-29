import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


df = pd.read_csv("SPY-2.csv") #load the SPY stock history data: 

#Load any Cross Correlation in our SPY Data

plt.figure()
lag_plot(df['Open'], lag=3)
plt.title('SPY Stock (Auto correlation plot with lag = 3)')
plt.show()

#Plot the SPY Stock Price in Last 5 Years

plt.plot(df["Date"], df["Close"])
plt.xticks(np.arange(0,1300, 150), df['Date'][0:1300:150])
plt.title("SPY Stock Price over 5 Years")
plt.xlabel("DATE")
plt.ylabel("PRICE")
plt.show()

#Build the Predictive ARIMA Model
#Divide the Data into a Training (%70), test (30%)

number_examples = len(df)
train_data_len = int(number_examples*0.7) 
X_train = df[0:train_data_len]['Close'].values
X_test = df[train_data_len:]['Close'].values

#ARIMA Parameters: p=4, d=1 and q=0.

history = X_train.tolist()
model_outputs = []
for index in range(len(X_test)):
    model = ARIMA(history, order=(4, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    model_outputs.append(output[0])
    history.append(X_test[index])
    

MSE_error = mean_squared_error(X_test, model_outputs)
print('Mean Squared Error is {}'.format(MSE_error))

#Plot the Predictions of the Model 

test_indexes = df[train_data_len:].index
plt.plot(test_indexes, model_outputs, color='blue', marker='o', linestyle='dashed',label='Predicted Price')
plt.plot(test_indexes, X_test, color='red', label='Actual Price')
plt.title('SPY Prices Prediction over 5 Years')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.xticks(np.arange(881, 380, 50), df.Date[881:380:50])
plt.legend()
plt.show()


