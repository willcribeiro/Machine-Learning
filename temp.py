# Data processing

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values #Independ variable
Y = dataset.iloc[:,3].values   #Dependent variable

# Taking care of missing data
from sklearn.preprocessing import Imputer
#Aqui estou subistituindo as info faltantas NaN,  pela média das colunas
imputer = Imputer(missing_values= 'NaN' , strategy= 'mean', axis= 0)
#Encaixando os locais dos dados faltantes
imputer = imputer.fit(X[:, 1:3])
#Aqui irei subistituir os dados faltantes pela média
X[:, 1:3] = imputer.transform(X[: , 1:3])

# Encoding categorical Data
# Change the string for a namber 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# For X
labelenconder_X = LabelEncoder() 
X[:, 0] = labelenconder_X.fit_transform(X[:, 0]) #Treoca as strings por numeros
onehotencoder = OneHotEncoder(categorical_features= [0]) #Dummy Encoding
X = onehotencoder.fit_transform(X).toarray()

#For Y 
labelenconder_Y = LabelEncoder()
Y = labelenconder_X.fit_transform(Y)

#Splitting the dataset into the Training set and Test set
#her we are separete the dataset in 2 parts,  the part for a training
#and another part for test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

#Now,  we need put the variable in same range
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_train)


