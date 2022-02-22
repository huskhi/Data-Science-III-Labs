import pandas as pd
import matplotlib.pyplot as plt

#Name: Khushi Ladha
#roll no. B20013
#Mob No. 7665519043

#importing csv file tp dataframe format
df = pd.read_csv('pima-indians-diabetes.csv')

#histogram for hist column
df["pregs"].hist(bins = 20)
plt.title('Histogram of Pregs column')
plt.xlabel('Pregs')
plt.ylabel('Frequency')
plt.show()


#histogram for skin column
df["skin"].hist(bins = 20)
plt.title('Histogram of Skin column')
plt.xlabel('skin (in mm)')
plt.ylabel('Frequency')
plt.show()
