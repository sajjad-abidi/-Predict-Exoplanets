#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas


# In[2]:


pip install seaborn


# In[3]:


pip install seaborn


# In[4]:


pip install numpy


# In[5]:


pip install matplotlib


# In[6]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[9]:


train_df = pd.read_csv("exoplanet/exoTrain.csv")


# In[11]:


train_df.head(10)


# In[13]:


#checking the shape of the dataset
train_df.shape


# In[18]:


#checking the null values
train_df[train_df.isnull().any(axis=1)]


# In[19]:


sns.heatmap(train_df.isnull())


# In[21]:


#checking the labels
train_df['LABEL'].unique()


# In[23]:


#extracting index for stars labbeled as 2
list(train_df[train_df['LABEL']==2].index)


# In[27]:


plt.figure(figsize=(3, 5))
ax = sns.countplot(x='LABEL', data = train_df)
ax.bar_label(ax.containers[0])


# In[28]:


#replace 2 to 1 and 1 to 0..means coverting it into binary
train_df = train_df.replace({"LABEL" : {2:1, 1:0}})


# In[30]:


train_df.LABEL.unique()


# In[31]:


#checking the drop in flux
plot_df = train_df.drop(["LABEL"], axis = 1)


# In[32]:


plot_df


# In[33]:


x = range(1, 3198)
plot_df.iloc[3,:].values


# In[42]:


#plot a random star from plot df
time = range(1, 3198)
flux_val = plot_df.iloc[25,:].values
plt.figure(figsize = (18, 10))
plt.plot(time, flux_val, linewidth = 1)


# In[45]:


plt.figure(figsize = (25, 15))

for i in range(1, 4):
    plt.subplot(1, 4, i)
    sns.boxplot(data = train_df, x = "LABEL", y = 'FLUX.' + str(i))


# In[49]:


#drooping outliners

train_df.drop(train_df[train_df['FLUX.2'] > 0.25e6].index, axis = 0, inplace = True)


# In[50]:


sns.boxplot(data = train_df, x ='LABEL', y = 'FLUX.' + str(np.random.randint(1000)))


# In[63]:


#exctracting dependent and independent variables
x = train_df.drop(['LABEL'], axis = 1)
y = train_df.LABEL
print(f"Take a look over ~\n\nX train array:-\n{x.values}\n\nY train array:-\n{y.values}")


# In[64]:


#splitting data into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state =0)


# In[56]:


# feature scalling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.fit_transform(X_test)


# In[65]:


#checking the minimum, maximum and mean value after scaling
print("X_train after scaling ~\n")
print(f"Minimum:- {round(np.min(X_train_sc),2)}\nMean:- {round(np.mean(X_train_sc),2)}\nMax:- {round(np.max(X_train_sc), 2)}\n")
print("--------------------------------\n")
print("X_test after scaling ~\n")
print(f"Minimum:- {round(np.min(X_test_sc),2)}\nMean:- {round(np.mean(X_test_sc),2)}\nMax:- {round(np.max(X_test_sc), 2)}\n")


# In[57]:


np.min(X_train_sc), np.max(X_train_sc)


# In[58]:


#initiating the KNN model

from sklearn.neighbors import KNeighborsClassifier as KNC


# In[59]:


#choosing the K = 5
knn_classifier = KNC (n_neighbors = 5, metric = 'minkowski', p=2)


# In[66]:


# fitting the model
knn_classifier.fit(X_train_sc, y_train)


# In[67]:


#predicting

y_pred = knn_classifier.predict(X_test_sc)


# In[68]:


#results

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, auc, roc_curve

print("Validation accuracy: ", accuracy_score(y_test, y_pred))
print()
print("classification report: \n", classification_report(y_test, y_pred))


# In[69]:


#Confusion matrix
plt.figure(figsize=(15,11))
plt.subplots_adjust(wspace = 0.3)
plt.suptitle("KNN Performance before handling the imbalance in the data", color = 'r', weight = 'bold')
plt.subplot(221)
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="Set2",fmt = "d",linewidths=3, cbar = False,
           xticklabels=['nexo', 'exo'], yticklabels=['nexo','exo'], square = True)
plt.xlabel("True Labels", fontsize = 15, weight = 'bold', color = 'tab:pink')
plt.ylabel("Predicited Labels", fontsize = 15, weight = 'bold', color = 'tab:pink')
plt.title("CONFUSION MATRIX",fontsize=20, color = 'm')

#ROC curve and Area under the curve plotting
predicting_probabilites = knn_classifier.predict_proba(X_test_sc)[:,1]
fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("AUC :",auc(fpr,tpr)),color = "g")
plt.plot([1,0],[1,0],"k--")
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20, color = 'm')
plt.show()


# In[70]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter

ros = RandomOverSampler()
x_ros, y_ros = ros.fit_resample(x, y)  # Taking the original x, y as arguments

print(f"Before sampling:- {Counter(y)}")
print(f"After sampling:- {Counter(y_ros)}")


# In[71]:


y_ros.value_counts().plot(kind='bar', title='After aplying RandomOverSampler');


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(x_ros, y_ros, test_size = 0.3, random_state = 0)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[73]:


# Create function to fetch the optimal value of K
def optimal_Kval_KNN(start_k, end_k, x_train, x_test, y_train, y_test, progress = True):
    ''' 
    This function takes in the following arguments -
    start_k - start value of k
    end_k - end value of k
    x_train - independent training values for training the KNN
    x_test - independent testing values for prediction
    y_train - dependent training values for training KNN
    y_test - dependent testing values for computing error rate
    progress - if true shows the progress for each k (by default its set to True)
    '''
    # Header
    print(f"Fetching the optimal value of K in between {start_k} & {end_k} ~\n\nIn progress...")
    
    # Empty list to append error rate
    mean_err = []
    for K in range(start_k, end_k + 1):                         # Generates K from start to end-1 values
        knn = KNC(n_neighbors = K)                              # Build KNN for respective K value
        knn.fit(x_train, y_train)                               # Train the model
        err_rate = np.mean(knn.predict(x_test) != y_test)       # Get the error rate
        mean_err.append(err_rate)                               # Append it
        # If progress is true display the error rate for each K
        if progress == True:print(f'For K = {K}, mean error = {err_rate:.3}')
        
    # Get the optimal value of k and corresponding value of mean error
    k, val = mean_err.index(min(mean_err))+1, min(mean_err)
    
    # Footer
    print('\nDone! Here is how error rate varies wrt to K values:- \n')
    
    # Display how error rate changes wrt K values and mark the optimal K value
    plt.figure(figsize = (5,5))
    plt.plot(range(start_k,end_k + 1), mean_err, 'mo--', markersize = 8, markerfacecolor = 'c',
            linewidth = 1)          # plots all mean error wrt K values
    plt.plot(k, val, marker = 'o', markersize = 8, markerfacecolor = 'gold', 
             markeredgecolor = 'g') # highlits the optimal K
    plt.title(f"The optimal performance is obtained at K = {k}", color = 'r', weight = 'bold',
             fontsize = 15)
    plt.ylabel("Error Rate", color = 'olive', fontsize = 13)
    plt.xlabel("K values", color = 'olive', fontsize = 13)
    
    '''returns the optimal value of k'''
    return k


# In[74]:


k = optimal_Kval_KNN(1, 10, X_train_sc, X_test_sc, y_train, y_test)


# In[76]:


# Fiting the KNN Classifier Model on to the training data after

# Choosing K = 1
knn_classifier = KNC(n_neighbors=1,metric='minkowski',p=2)  
'''metric is to be by default minkowski for p = 2 to calculate the Eucledian distances'''

# Fit the model
knn_classifier.fit(X_train_sc, y_train)

# Predict
y_pred = knn_classifier.predict(X_test_sc)

# Results
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc

print('\nValidation accuracy of KNN is', accuracy_score(y_test,y_pred))
print("\n-------------------------------------------------------")
print ("\nClassification report :\n",(classification_report(y_test,y_pred)))

#Confusion matrix
plt.figure(figsize=(15,11))
plt.subplots_adjust(wspace = 0.3)
plt.suptitle("KNN Performance after handling the imbalance in the data", color = 'b', weight = 'bold')
plt.subplot(221)
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap="Set2",fmt = "d",linewidths=3, cbar = False,
           xticklabels=['nexo', 'exo'], yticklabels=['nexo','exo'], square = True)
plt.xlabel("True Labels", fontsize = 15, weight = 'bold', color = 'm')
plt.ylabel("Predicited Labels", fontsize = 15, weight = 'bold', color = 'm')
plt.title("CONFUSION MATRIX",fontsize=20, color = 'purple')

#ROC curve and Area under the curve plotting
predicting_probabilites = knn_classifier.predict_proba(X_test_sc)[:,1]
fpr,tpr,thresholds = roc_curve(y_test,predicting_probabilites)
plt.subplot(222)
plt.plot(fpr,tpr,label = ("AUC :",auc(fpr,tpr)),color = "g")
plt.plot([1,0],[1,0], 'k--')
plt.legend(loc = "best")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC - CURVE & AREA UNDER CURVE",fontsize=20, color = 'm')
plt.show()


# In[ ]:




