import pandas as pd
import sklearn.model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from pandas import read_csv
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import visuals as vs
filename='census.csv'
data=pd.read_csv(filename,sep=',')
features=data.drop('income',axis=1)
target=data['income']
print(features,target)
print(data.describe())
vs.distribution(data)
#plt.figure(figsize=(10,5))
#sns.heatmap(data.corr(),annot=True)
#plt.s
skewed_data=['capital-gain','capital-loss']
features=pd.DataFrame(data=features)
features[skewed_data]=features[skewed_data].apply(lambda x:np.log(x+1))

#plt.figure(figsize=(10,5))
#sns.heatmap(features.corr(),annot=True)
#plt.show()
scaler=MinMaxScaler()
name=['age','education-num','capital-gain','capital-loss','hours-per-week']
features=pd.DataFrame(data=features)
features[name]=scaler.fit_transform(features[name])
print(features.head(10))
#features.hist()
#plt.show()
#sns.heatmap(features.corr(),annot=True)
#plt.show()
features_name=['age','workclass','education_level','education-num','marital-status','occupation'
    ,'relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country']
features=pd.get_dummies(features)
print(features)
target=pd.get_dummies(target)
print(target)
encoded=list(features.columns)
print(format(len(encoded)))
print(encoded)
