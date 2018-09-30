# Data processing template

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values #Independ variable
Y = dataset.iloc[:,3].values   #Dependent variable

#Splitting the dataset into the Training set and Test set
#her we are separete the dataset in 2 parts,  the part for a training
#and another part for test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#Now,  we need put the variable in same range
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_train)"""



