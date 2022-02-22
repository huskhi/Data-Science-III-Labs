import pandas as pd
import matplotlib.pyplot as plt

#Name: Khushi Ladha
#roll no. B20013
#Mob No. 7665519043

#importing csv file tp dataframe format
df = pd.read_csv('pima-indians-diabetes.csv')

#plotting histogram for pregs values when class ==1
df_1 = df[df["class"]== 1]
(df_1["pregs"].hist(bins = 15))
plt.title('Histogram of Pregs column when class ==1')
plt.show()

#plotting histogram for pregs values when class ==0
df_0 = df[df["class"]== 0]
(df_0["pregs"].hist(bins = 15))
plt.title('Histogram of Pregs column when class ==0')
plt.show()

