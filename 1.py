import pandas as pd

#Name: Khushi Ladha
#roll no. B20013
#Mob No. 7665519043

#importing csv file tp dataframe format
#class column dropped
df = pd.read_csv('pima-indians-diabetes.csv').drop("class", axis=1)

#renaming the columns to include units
df.columns = ["pregs" , "plas" , "pres (in mm Hg)" , "skin (in mm)" , "test (in mu U/mL) " , "BMI (in kg/m2)"  ,  "pedi" , "Age (in years)"]

#for mean
print("Mean of required attributes are:- ")
print(df.mean())

#for median
print("**************************************", "\n", "\nMedian of required attributes are:- ")
print(df.median())

#for mode
print("**************************************", "\n", "\nMODE of required attributes are:- ")
print(df.mode(dropna = False , numeric_only= False ))

#for standard devaition
print("**************************************", "\n", "\nStandard Deviation of required attributes are:- ")
print(df.std())

#to find minimum
print("**************************************", "\n", "\nMinimum Values of required attributes are:- ")
print(df.min())

#to find maximum
print("**************************************", "\n", "\nMaximum Values of required attributes are:- ")
print(df.max())

