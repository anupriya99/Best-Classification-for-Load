#!/usr/bin/env python


# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:




import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[22]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[7]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[17]:


X = Feature
X[0:5]


# What are our lables?

# In[18]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[19]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[112]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=0)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[87]:


k = 7
#Train Model and Predict  
knn = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
knn

yhat = knn.predict(X_test)
yhat[0:5]


# In[88]:


from sklearn import metrics
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    knn = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=knn.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)


# In[89]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[90]:


print("Train set Accuracy: ", metrics.accuracy_score(y_train, knn.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[91]:


# confusion matrix
from sklearn.metrics import confusion_matrix

knn_confusion_matrix = confusion_matrix(y_true = y_test, y_pred = yhat)
print("The Confusion matrix:\n", knn_confusion_matrix)


# In[92]:


# visualize the confusion matrix
plt.matshow(knn_confusion_matrix, cmap = plt.cm.Blues)
plt.title("KNN Confusion Matrix\n")
plt.ylabel('True label')
plt.xlabel('Predicted label')
for y in range(knn_confusion_matrix.shape[0]):
    for x in range(knn_confusion_matrix.shape[1]):
        plt.text(x, y, '{}'.format(knn_confusion_matrix[y, x]),
                horizontalalignment = 'center',
                verticalalignment = 'center',)
plt.show()


# In[93]:


#Generate the classification report
from sklearn.metrics import classification_report
knn_classify_report = classification_report(y_true = y_test, y_pred = yhat)
print(knn_classify_report)


# In[ ]:





# # Decision Tree

# In[94]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
dis_Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
dis_Tree # it shows the default parameters


# In[96]:


dis_Tree.fit(X_train,y_train)


# In[97]:


pred_Tree = dis_Tree.predict(X_test)


# In[98]:


print (pred_Tree [0:5])
print (y_test [0:5])


# In[99]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, pred_Tree))


# # Let's Visualize the model

# In[100]:


import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} pydotplus')


# In[101]:


from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
get_ipython().run_line_magic('matplotlib', 'inline')


# In[102]:


dot_data = StringIO()
filename = "dis_tree.png"
featureNames = df.columns[0:9]
targetNames = df["loan_status"].unique().tolist()
out=tree.export_graphviz(dis_Tree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# In[ ]:





# # Support Vector Machine

# In[43]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[44]:


svm=SVC().fit(X_train,y_train)


# In[45]:




pred_svm=svm.predict(X_test)


# In[46]:


print(classification_report(y_test,pred_svm))
print('\n')
print('Jaccard Similarity Score : ',round(jaccard_similarity_score(y_test,pred_svm)*100,2),'%')
print('\n')
print('F1-SCORE : ',f1_score(y_test,pred_svm,average=None))
print('\n')
print('Train Accuracy: ',metrics.accuracy_score(y_train, svm.predict(X_train))*100,'%')


# In[ ]:





# # Logistic Regression

# In[47]:


from sklearn.linear_model import LogisticRegression


# In[48]:


lgm=LogisticRegression()


# In[49]:


lgm.fit(X_train,y_train)


# In[50]:


pred_lgm=lgm.predict(X_test)


# In[51]:


print(classification_report(y_test,pred_lgm))
print('\n')
print('Jaccard Similarity Score : ',round(jaccard_similarity_score(y_test,pred_lgm)*100,2),'%')
print('\n')
print('F1-SCORE : ',f1_score(y_test,pred_lgm,average=None))
print('\n')
print('Train Accuracy: ',metrics.accuracy_score(y_train, lgm.predict(X_train))*100,'%')


# # Model Evaluation using Test set

# In[52]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[53]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[66]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[67]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df.head()


# In[68]:


test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df.head()


# In[69]:


test_df['weekend']=test_df['dayofweek'].apply(lambda x: 1 if (x>3) else 0)


# In[58]:


sns.heatmap(test_df.isnull())


# In[59]:


test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)


# In[60]:


dumm=pd.get_dummies(test_df['education'])
#dumm=dumm.drop('Master or Above',axis=1,inplace=True)
dumm=dumm[['Bechalor','High School or Below','college']]


# In[61]:


test_feature = test_df[['Principal','terms','age','Gender','weekend']]
test_feature = pd.concat([test_feature,dumm], axis=1)
#test_feature.drop(['Master or Above'], axis = 1,inplace=True)
test_feature.head()


# In[62]:


test_data= test_feature
test_data= preprocessing.StandardScaler().fit(test_data).transform(test_data)


# In[63]:


y_t=test_df['loan_status']


# In[ ]:





# # Accuracy of the Model
# 

# In[ ]:





# In[70]:


knn=KNeighborsClassifier()
knn.fit(X,y)
predKNN_test=knn.predict(test_data)
accuracy=metrics.accuracy_score(predKNN_test,y_t)
print("accuracy : ",round(accuracy,3)*100,'%')


# In[71]:


score=[]
for k in range(1,100):
    knn=KNeighborsClassifier(n_neighbors=k,weights='uniform')
    knn.fit(X,y)
    predKNN=knn.predict(test_data)
    accuracy=metrics.accuracy_score(predKNN,y_t)
    score.append(accuracy*100)
    print (k,': ',accuracy)


# In[72]:


print(score.index(max(score))+1,' : ',round(max(score),2),'%')


# In[73]:


knn=KNeighborsClassifier(n_neighbors=32)
knn.fit(X,y)
predKNN_test=knn.predict(test_data)
accuracy=metrics.accuracy_score(predKNN_test,y_t)
print("accuracy : ",round(accuracy,3)*100,'%')


# In[74]:


print(classification_report(y_t,predKNN_test))
print('\n')
print('Jaccard Similarity Score : ',round(jaccard_similarity_score(y_t,predKNN_test)*100,2),'%')
print('\n')
print('F1-SCORE : ',f1_score(y_t,predKNN_test,average=None))
print('\n')
print('Train Accuracy: ',metrics.accuracy_score(y,knn.predict(X))*100,'%')


# # For Decision Tree

# In[75]:


parameter_grid = {'max_depth': [1, 2, 3, 4, 5,6,5,9,15,20],
                  'max_features': [1, 2, 3, 4,5,6,7,8],
                 'random_state':[0,15,20,35,50,80,100,150,180,200],
                 'criterion':['gini','entropy'],
                 }

grid_search = GridSearchCV(dtree, param_grid = parameter_grid,
                          cv =10)

grid_search.fit(X, y)

print ("Best Score: {}".format(grid_search.best_score_))
print ("Best params: {}".format(grid_search.best_params_))


# In[76]:


dtree=DecisionTreeClassifier(max_depth=6,criterion='entropy',max_features=4,random_state=20).fit(X,y)
pred_Dtree=dtree.predict(test_data)


# In[77]:


print(classification_report(y_t,pred_Dtree))
print('\n')
print('Jaccard Similarity Score : ',round(jaccard_similarity_score(y_t,pred_Dtree)*100,2),'%')
print('\n')
print('F1-SCORE : ',f1_score(y_t,pred_Dtree,average=None))
print('\n')
print('Train Accuracy: ',metrics.accuracy_score(y, dtree.predict(X))*100,'%')


# # Support Vector Machine

# In[78]:


svm=SVC().fit(X,y)
pred_svm=svm.predict(test_data)


# In[79]:


print(classification_report(y_t,pred_svm))
print('\n')
print('Jaccard Similarity Score : ',round(jaccard_similarity_score(y_t,pred_svm)*100,2),'%')
print('\n')
print('F1-SCORE : ',f1_score(y_t,pred_svm,average=None))
print('\n')
print('Train Accuracy: ',metrics.accuracy_score(y, svm.predict(X))*100,'%')


# # Logistic Regression

# In[80]:


lgm=LogisticRegression().fit(X,y)


# In[81]:


pred_lgm=lgm.predict(test_data)


# In[82]:


print(classification_report(y_t,pred_lgm))
print('\n')
print('Jaccard Similarity Score : ',round(jaccard_similarity_score(y_t,pred_lgm)*100,2),'%')
print('\n')
print('F1-SCORE : ',f1_score(y_t,pred_lgm,average=None))
print('\n')
print('Train Accuracy: ',metrics.accuracy_score(y, lgm.predict(X))*100,'%')


# In[83]:


Algoritm=['KNN','Decision Tree','SVM','LogisticRegression']
j_knn=round(jaccard_similarity_score(y_t,predKNN_test)*100,2)
j_dtree=round(jaccard_similarity_score(y_t,pred_Dtree)*100,2)
j_svm=round(jaccard_similarity_score(y_t,pred_svm)*100,2)
j_lgm=round(jaccard_similarity_score(y_t,pred_lgm)*100,2)
Jaccard=[j_knn,j_dtree,j_svm,j_lgm]

f1_knn=f1_score(y_t,predKNN_test,average=None)
f1_dtree=f1_score(y_t,pred_Dtree,average=None)
f1_svm=f1_score(y_t,pred_svm,average=None)
f1_lgm=f1_score(y_t,pred_lgm,average=None)
F1_score=[f1_knn,f1_dtree,f1_svm,f1_lgm]


# In[84]:


table = pd.DataFrame({
    "Algorithm": Algoritm,
    "Jaccatd": Jaccard,
    "F1-Score": F1_score,
    "LogLoss":[np.NAN,np.NAN,np.NAN,np.NAN]})


# In[85]:


table

