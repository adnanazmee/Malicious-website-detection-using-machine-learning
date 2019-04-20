import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data=pd.read_csv('dataset.csv')

# clean up column names

data.columns = data.columns.\
    str.strip().\
    str.lower()

#____remove non-numeric columns
    
data = data.select_dtypes(['number']) 


#__extracting dependent and independent variable

x=data.drop(['type'],axis=1)
y=data['type']
x=np.nan_to_num(x) #____replace nan with zero and inf with finite numbers

#_____Input data visualization

#sns.heatmap(data.corr())
#sns.pairplot(data,hue='type',palette='Set2')
#sns.countplot(y='type',data=data)


#______Splitting the data into Training and test dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=44)


#_______Logistic Regression
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=42)
classifier.fit(x_train,y_train)

pred2 = classifier.predict(x_test)

#______calculate accuracy for logistic regression

print('For Logistic Regression accuracy score is ',accuracy_score(y_test,pred2))
print('For Logistic Regression confusion_matrix is: \n\n',confusion_matrix(y_test,pred2))
print ('For Logistic Regression Classification Report: \n\n',classification_report(y_test,pred2))


#___________KNN
from sklearn.neighbors import KNeighborsClassifier

classifier2=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier2.fit(x_train,y_train)

pred3=classifier2.predict(x_test)

#______calculate accuracy for KNN

print('For KNN accuracy score is ',accuracy_score(y_test,pred3))
print('For  KNN confusion_matrix is: \n\n',confusion_matrix(y_test,pred3))
print ('For  KNN Classification Report: \n\n',classification_report(y_test,pred3))




#_________Random forest
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)

pred1=rfc.predict(x_test)

#______calculate accuracy for random forest

#Classification Report
print('For Random Forest accuracy score is ',accuracy_score(y_test,pred1))
print('For Random Forest confusion_matrix is: \n\n',confusion_matrix(y_test,pred1))
print ('For Random Forest Classification Report: \n\n',classification_report(y_test,pred1))









