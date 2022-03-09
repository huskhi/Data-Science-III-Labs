import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

dframe = pd.read_csv('daily_covid_cases.csv')

train_sample_size = int(0.35* len(dframe))
train = dframe["new_cases"][:(len(dframe) - train_sample_size)].values
test = dframe["new_cases"][(len(dframe) - train_sample_size):].values

MAPE_values = []
RMSE_values = []
lag_value = [1,5,10,15,25]

for p_value in lag_value:

    # training the model
    model = AutoReg(train, lags=p_value, old_names=False)
    # fit/train the model
    model_fit = model.fit()
    # Getting  the coefficients
    coef = model_fit.params

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

    #computing and appendig RMSE values
    rms= (math.sqrt(mean_squared_error(pred, test)) / np.mean(test))
    RMSE_values.append(round(rms*100,3))

    # computing MAPE
    mape = np.mean(np.abs((pred-test)/test))*100
    MAPE_values.append(round(mape,3))

output = {'Lag value':lag_value,'MAPE' :mape , 'RMSE(%)':RMSE_values, }
print('Table 1\n',pd.DataFrame(output))

x_ticks = [1,2, 3,4,5 ]
# plotting time lag  vs. RMSE(%)
plt.bar(x_ticks,RMSE_values)
plt.title('RMSE(%) vs. time lag')
plt.ylabel('RMSE(%)')
plt.xlabel('Time Lag')
plt.xticks(x_ticks,lag_value)
plt.show()

# plotting time lag vs. MAPE
plt.xticks(x_ticks,lag_value)
plt.title('MAPE vs. time lag')
plt.ylabel('MAPE')
plt.xlabel('Time Lag')
plt.bar(x_ticks,RMSE_values)
plt.show()
