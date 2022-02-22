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

def plot_and_covariace(x_value):
    i = 0
    # to get name of all columns in a list
    column_names = list(df)

    for col in column_names :
        #2 a), b) scatter plot between ‘Age’/BMI and each of the other attributes, excluding ‘class’
        df.plot.scatter(x=x_value, y=col  )

        plt.xlabel(x_value)
        plt.ylabel(col)
        plt.show()

        #3 a) , b) covariance
        print("Correlation between" ,  x_value , " and ", col  , " is " , df[x_value].corr(df[col]))
    print("\n*****************************************\n")

plot_and_covariace("Age (in years)")
plot_and_covariace("BMI (in kg/m2)")

