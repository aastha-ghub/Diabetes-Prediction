#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd

#import required modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


#load dataset
diab= pd.read_csv("/Users/Admin/Downloads/diabetes.csv")


# In[21]:


diab.head()


# In[22]:


#split dataset in features and target variable
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age', 'Glucose', 'BloodPressure', 'DiabetesPedigreeFunction']
x = diab[feature_cols] #features
y = diab.Outcome       #target variable


# In[23]:


#split x and y into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#random_state is the object that controls randomization during splitting


# In[24]:


#import the class
from sklearn.linear_model import LogisticRegression

#instantiate the model (using the default parameters)
logreg = LogisticRegression(solver='lbfgs', max_iter=400)

#solver='lbfgs' -algorithm to use in the optimization problem
#max_iter=400.  -max no of iterations taken for the solvers to converge

#fit the model with data
logreg.fit(x_train,y_train)

y_pred= logreg.predict(x_test)


# In[25]:


#import the metrics data

from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[26]:


class_names= [0,1] #name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
# the arrange() function is used to get evenly spaced values within a given interval
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
#create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap= "YlGnBu", fmt= 'g')

#heatmap() fmt parameter – add text on each cell. The annot only help to add numeric value on python
#heatmap cell but fmt parameter allows to add string (text) values on the cell.
#'g' General format

ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y= 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[27]:


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))  
print("Recall:", metrics.recall_score(y_test, y_pred))      


# In[28]:


y_pred_proba = logreg.predict_proba(x_test)[::,1]

#The function predict_proba() returns a numpy array of two columns. The first column is the probability
#that target=0 and the second column is the probability that target=1 . That is why we add [:,1] after
#predict_proba() in order to get the probabilities of target=1 .

fpr,tpr,_ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr, label='data 1, auc='+str(auc))
plt.legend(loc=4)
plt.show()

#The TPR defines how many correct positive results occur among all positive samples available during
#the test
#FPR, on the other hand, defines how many incorrect positive results occur among all negative samples
#available during the test
#Threshold corresponds to a point on the ROC curve that is colinear with adjacent points


# In[ ]:




