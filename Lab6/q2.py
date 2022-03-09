import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

#creating datframe
dframe = pd.read_csv('daily_covid_cases.csv')
train_sample_size = int(0.35* len(dframe))
train = dframe["new_cases"][:(len(dframe) - train_sample_size)].values
test = dframe["new_cases"][(len(dframe) - train_sample_size):].values

# 2-(a)
print('2(a):')

p_value = 5
# training the model
model = AutoReg(train, lags=p_value , old_names= False)
# fit/train the model
model_fit = model.fit()
# Getting  the coefficients
coef = model_fit.params
print('The coefficients obtained are', coef)

#using  these  coefficients  walk  forward  over  time  steps  in  test,  one step each time
history = train[len(train)-p_value:]
history = [history[i] for i in range(len(history))]
pred = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
  length = len(history)
  lag = [history[i] for i in range(length-p_value,length)]
  yhat = coef[0]  # Initialize to w0
  for d in range(p_value):
   yhat += coef[d+1] * lag[p_value-d-1]  # Add other values
  obs = test[t]
  pred.append(yhat)  #Append  predictions  to  compute  RMSE  later
  history.append(obs)  # Append actual test value to history, to be  used in next step.

# 2(b) (i)
# scatter plot between actual and predicted values
plt.scatter(pred, test)
plt.title('Actual vs Predicted values')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()

# 2(b) (ii)
# line plot between actual and predicted values
plt.plot(test,pred)
plt.title('Predicted vs Actual values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# 2(b) (iii)
#finding RMSE
rms = (math.sqrt(mean_squared_error(pred, test)) / np.mean(test))
print('RMSE(%):',round(rms*100 , 3))

# computing MAPE
mape = np.mean(np.abs((pred-test)/test))*100
print('MAPE:',round(mape,3))

