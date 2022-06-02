# -*- coding: utf-8 -*-

# Standard Data Science Imports
import numpy as np
import pandas as pd
from pandas import DataFrame

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Scikit-learn
import sklearn
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

# Loading the data set into Pandas dataframe
churn_df = pd.read_csv(r'C:\Users\Hydraconix\Desktop\DATA\churn_clean.csv')

# Examining fist five records of dataset
churn_df.head()

# Viewing DataFrame descriptive information
churn_df.info

# Getting an overview of descriptive stats
churn_df.describe()

# Getting data types of features
churn_df.dtypes

# Checking for null values
churn_df.isnull()

# Renaming the last 8 Survey Columns for better description of variables
churn_df.rename(columns = {'Item1' : 'TimelyResponse',
                           'Item2' : 'Fixes' ,
                           'Item3' : 'Replacements' ,
                           'Item4' : 'Reliability' ,
                           'Item5' : 'Options' ,
                           'Item6' : 'Respectfulness' ,
                           'Item7' : 'Courteous' ,
                           'Item8' : 'Listening'},
                          inplace=True)

# Converting ordinal categorical data into numeric variables
churn_df['DummyInternetService'] = churn_df.InternetService.map({'None' : 0, 'DSL' : 1, 'Fiber Optic' : 2})
churn_df['DummyContract'] = churn_df.Contract.map({'Month-to-month' : 0, 'One year' : 1, 'Two Year' : 2})
churn_df['DummyGender'] = churn_df.Gender.map({'Nonbinary' : 0, 'Male' : 1, 'Female' : 2})

# Histograms of continuous variables
churn_df[['Age', 'Bandwidth_GB_Year', 'Children',  'Contacts', 'Email', 'Income', 'MonthlyCharge',
          'Outage_sec_perweek', 'Tenure', 'Yearly_equip_failure','DummyContract','DummyGender','DummyInternetService']].hist()
plt.savefig('churn_pyplot.jpg')
plt.tight_layout()

# A scatterplot to get an idea of correlations between potentially related variables
sns.scatterplot(x=churn_df['MonthlyCharge'], y=churn_df['Churn'], color='green')
plt.show()

# A scatterplot to get an idea of correlations between potentially related variables
sns.scatterplot(x=churn_df['Outage_sec_perweek'], y=churn_df['Churn'], color='green')
plt.show()

# A scatterplot to get an idea of correlations between potentially related variables
sns.scatterplot(x=churn_df['Tenure'], y=churn_df['Churn'], color='green')
plt.show()

# Setting the plot style to ggplot
plt.style.use('ggplot')

# Countplots of categorical variables
plt.figure()
sns.countplot(x='DeviceProtection', hue='Churn', data=churn_df, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

plt.figure()
sns.countplot(x='Multiple', hue='Churn', data=churn_df, palette ='RdBu')
plt.xticks([0,1],['No','Yes'])
plt.show()

plt.figure()
sns.countplot(x='Techie', hue='Churn', data=churn_df, palette ='RdBu')
plt.xticks([0,1],['No','Yes'])
plt.show()

plt.figure()
sns.countplot(x='TechSupport', hue='Churn', data=churn_df, palette ='RdBu')
plt.xticks([0,1],['No','Yes'])
plt.show()

# A scatter matrix of the discrete variables for high level overview of potential relationships & distributions
churn_discrete = churn_df[['Churn','TimelyResponse', 'Fixes', 'Replacements', 'Reliability', 'Options', 
                           'Respectfulness', 'Courteous', 'Listening']]
pd.plotting.scatter_matrix(churn_discrete, figsize = [30, 30])

# An individual scatterplot for viewing relationship of key financial feature against target variable
sns.scatterplot(x = churn_df['TimelyResponse'], y = churn_df['Churn'], color='red')
plt.show()

sns.scatterplot(x = churn_df['Fixes'], y = churn_df['Churn'], color='red')
plt.show()

sns.scatterplot(x = churn_df['Replacements'], y = churn_df['Churn'], color='red')
plt.show()

# Converting binary categorical variables to numeric variables
churn_df['DummyChurn'] = [1 if v == 'Yes' else 0 for v in churn_df['Churn']]
churn_df['DummyTechie'] = [1 if v == 'Yes' else 0 for v in churn_df['Techie']]
churn_df['DummyPort_modem'] = [1 if v == 'Yes' else 0 for v in churn_df['Port_modem']]
churn_df['DummyTablet'] = [1 if v == 'Yes' else 0 for v in churn_df['Tablet']]
churn_df['DummyPhone'] = [1 if v == 'Yes' else 0 for v in churn_df['Phone']]
churn_df['DummyMultiple'] = [1 if v == 'Yes' else 0 for v in churn_df['Multiple']]
churn_df['DummyOnlineSecurity'] = [1 if v == 'Yes' else 0 for v in churn_df['OnlineSecurity']]
churn_df['DummyOnlineBackup'] = [1 if v == 'Yes' else 0 for v in churn_df['OnlineBackup']]
churn_df['DummyDeviceProtection'] = [1 if v == 'Yes' else 0 for v in churn_df['DeviceProtection']]
churn_df['DummyTechSupport'] = [1 if v == 'Yes' else 0 for v in churn_df['TechSupport']]
churn_df['DummyStreamingTV'] = [1 if v == 'Yes' else 0 for v in churn_df['StreamingTV']]
churn_df['DummyStreamingMovies'] = [1 if v == 'Yes' else 0 for v in churn_df['StreamingMovies']]
churn_df['DummyPaperlessBilling'] = [1 if v == 'Yes' else 0 for v in churn_df['PaperlessBilling']]

#Drop original categorical features from dataframe for further analysis
churn_df = churn_df.drop(columns=['Churn', 'Contract', 'DeviceProtection', 'Gender', 'InternetService', 
                                  'Multiple' , 'OnlineBackup', 'OnlineSecurity', 'PaperlessBilling', 
                                  'Phone', 'Port_modem', 'StreamingMovies', 'StreamingTV', 'Tablet', 
                                  'Techie', 'TechSupport'])

#Remove the other less meaningful categorical variables from dataset to provide fully numerical dataframe for further analysis
churn_df = churn_df.drop(columns=['CaseOrder','Customer_id', 'Interaction', 'UID', 'City', 'State', 'County', 'Zip', 'Lat', 'Lng', 'Population',
                                  'Area', 'TimeZone', 'Job', 'Marital', 'PaymentMethod'])

# Provide a copy of the prepared data set
churn_df.to_csv(r'C:\Users\Hydraconix\Desktop\'churn_prepared_log.csv')

# List features for analysis
features = (list(churn_df.columns[:-1]))
print('Features for analysis include: \n', features)

# Re-read fully numerical prepared dataset
churn_df = pd.read_csv(r'C:\Users\Hydraconix\Desktop\'churn_prepared_log.csv')
                       
# Set predictor features & target variable
X = churn_df.drop('DummyChurn', axis=1).values
y = churn_df['DummyChurn'].values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =1)

# Export X_train dataset
X_train_df = pd.DataFrame(X_train)
X_train_df.to_csv(r'C:\Users\Hydraconix\Desktop\X_train.csv')

# Export X_test dataset
X_test_df = pd.DataFrame(X_test)
X_test_df.to_csv(r'C:\Users\Hydraconix\Desktop\X_test.csv')

# Export y_train dataset
y_train_df = pd.DataFrame(y_train)
y_train_df.to_csv(r'C:\Users\Hydraconix\Desktop\y_train.csv')

# Export y_test dataset
y_test_df = pd.DataFrame(X_test)
y_test_df.to_csv(r'C:\Users\Hydraconix\Desktop\y_test.csv')

# Initialize KNN model 
knn = KNeighborsClassifier(n_neighbors = 7)

# Fit data to KNN model
knn.fit(X_train, y_train)

# Predict outcomes from test set
y_pred = knn.predict(X_test)

# Export y_pred dataset
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv(r'C:\Users\Hydraconix\Desktop\y_pred.csv')

# Print initial accuracy score of KNN model
print('Initial accuracy score KNN model: ', accuracy_score(y_test, y_pred))

# Compute classification metrics
print(classification_report(y_test, y_pred))

# Set steps for pipeline object
steps = [('scaler', StandardScaler()),
 ('knn', KNeighborsClassifier())]

# Initiate pipeline
pipeline = Pipeline(steps)

# Split dataframe
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Scale dateframe with pipeline object
knn_scaled = pipeline.fit(X_train_scaled, y_train_scaled)

# Predict from scaled dataframe
y_pred_scaled = pipeline.predict(X_test_scaled)

# Print new accuracy score of scaled KNN model
print('New accuracy score of scaled KNN model: {:0.3f}'.format(accuracy_score(y_test_scaled, y_pred_scaled)))

# Compute classification metrics after scaling
print(classification_report(y_test_scaled, y_pred_scaled))

#Confusion_matrix & generate results
cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)

# Visual confusion matrix
group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

# Set up parameters grid
param_grid = {'n_neighbors': np.arange(1, 50)}
# Re-initializing KNN for cross validation
knn = KNeighborsClassifier()
# Initializing GridSearch cross validation
knn_cv = GridSearchCV(knn , param_grid, cv=5)
# Fit model to 
knn_cv.fit(X_train, y_train)
# Print best parameters
print('Best parameters for this KNN model: {}'.format(knn_cv.best_params_))

# Generate model best score
print('Best score for this KNN model: {:.3f}'.format(knn_cv.best_score_))

# Fit it to the data
knn_cv.fit(X, y)

# Compute predicted probabilities: y_pred_prob
y_pred_prob = knn_cv.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("The Area under curve (AUC) on validation dataset is: {:.4f}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(knn_cv, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
