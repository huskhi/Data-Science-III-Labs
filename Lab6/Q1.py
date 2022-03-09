import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#creating datafrane
df = pd.read_csv('daily_covid_cases.csv')

#making list of month labels
l_names = ['Feb-20', 'Apr-20', 'Jun-20', 'Aug-20', 'Oct-20', 'Dec-20', 'Feb-21', 'Apr-21', 'Jun-21', 'Aug-21', 'Oct-21']

#making xticks to keep labels in correct position
x = [0]
for i in range(10):
    x.append(x[i] + 60)
print(df)

#plotting the cases graph
y_value = df ["new_cases"]
plt.plot(y_value)
plt.xticks (x, l_names)
plt.show()

#finding the lag1 time seires
lag1 = df['new_cases'].shift(-1)
print(lag1)

#b part
print('1(b) ')
correlation = round(lag1.corr(df['new_cases'] ) , 3)
print("Correlation value is " , correlation)

#making scatter plot
plt.scatter(lag1 ,df['new_cases'])
plt.title('one day lagged sequence vs. given time sequence')
plt.ylabel('one day lagged time sequence')
plt.xlabel('given time sequence')
plt.show()

#defining the corrleation fucntion
def corr_func(col_1 , col_2):
    correlation = col_1.corr(col_2)
    return round(correlation, 3)

#c part
x = []
y = []

#defining the lag values
lag_values = [1, 2, 3, 4 , 5 , 6]

#fidning correlation
for i in lag_values:
    col_1 = df['new_cases'].shift(-i)
    col_2 =  df ["new_cases"]
    y.append(corr_func(col_1 , col_2))
    x.append(i)
    print(i , " " , corr_func(col_1, col_2) )

plt.plot(x , y )
plt.show()

#d part
sm.graphics.tsa.plot_acf(df['new_cases'],lags=lag_values)
plt.xlabel('Lagged Values')
plt.ylabel('Correlation coefficients')
plt.show()


