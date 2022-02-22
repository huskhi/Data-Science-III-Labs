import pandas as pd
import matplotlib.pyplot as plt

#Name: Khushi Ladha
#roll no. B20013
#Mob No. 7665519043

#importing csv file tp dataframe format
#class column dropped
df = pd.read_csv('pima-indians-diabetes.csv').drop("class", axis=1)

#renaming the columns to include units
df.columns = ["pregs" , "plas" , "pres (in mm Hg)" , "skin (in mm)" , "test (in mu U/mL) " , "BMI (in kg/m2)"  ,  "pedi" , "Age (in years)"]

#plotting individual boxplot for all columns except class
for col in list(df):
    df.boxplot(column= col)
    plt.ylabel("Values")
    plt.show()

