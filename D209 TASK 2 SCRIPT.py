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
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

 # Import model, splitting method & metrics from sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
  
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
          'Outage_sec_perweek', 'Tenure', 'Yearly_equip_failure','DummyInternetService', 'DummyContract','DummyGender']].hist()
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

# I will now set the plot style to ggplot
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

# Drop original categorical features from dataframe for further analysis
churn_df = churn_df.drop(columns=['Churn', 'Contract', 'DeviceProtection', 'Gender', 'InternetService', 
                                  'Multiple' , 'OnlineBackup', 'OnlineSecurity', 'PaperlessBilling', 
                                  'Phone', 'Port_modem', 'StreamingMovies', 'StreamingTV', 'Tablet', 
                                  'Techie', 'TechSupport'])

# Remove the other less meaningful categorical variables from dataset to provide fully numerical dataframe for further analysis
churn_df = churn_df.drop(columns=['CaseOrder','Customer_id', 'Interaction', 'UID', 'City', 'State', 'County', 'Zip', 'Lat', 'Lng', 'Population',
                                  'Area', 'TimeZone', 'Job', 'Marital', 'PaymentMethod'])

# Provide a copy of the prepared data set
churn_df.to_csv(r'C:\Users\Hydraconix\Desktop\'churn_prepared_dt.csv')

# List features for analysis
features = (list(churn_df.columns[:-1]))
print('Features for analysis include: \n', features)

# Re-read fully numerical prepared dataset
churn_df = pd.read_csv(r'C:\Users\Hydraconix\Desktop\'churn_prepared_dt.csv')

# Set predictor features & target variable
X = churn_df.drop('DummyChurn', axis=1).values
y = churn_df['DummyChurn'].values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state =1)

# Instantiate Decision Tree Regressor model
dt = DecisionTreeRegressor(max_depth = 8, min_samples_leaf = 0.1, random_state = 1)

# Fit dataframe to Decision Tree Regressor model
dt.fit(X_train, y_train)

# Predict Outcomes from test set
y_pred = dt.predict(X_test)

# Compute test set MSE
mse_dt = MSE(y_test, y_pred)

# Compute test set RMSE
rmse_dt = mse_dt**(1/2)

# Print initial RMSE
print('Initial RMSE score Decision Tree Regressor model: {:.3f}'.format(rmse_dt))

# Compute the coefficient of determination (R-squared)
scores = cross_val_score(dt, X, y, scoring='r2')

# Print R-squared value
print('Cross validation R-squared values: ', scores)

# Print Mean Squared Error
print('With a manual calculation, the Mean Squared Error: {:.3f} '.format(sum(abs(y_test - y_pred)**2)/len(y_pred)))
# Or

print('Using scikit-lean, the Mean Squared Error: {:.3f}'.format(MSE(y_test, y_pred)))

# Calculate & print the Root Mean Squared Error
RMSE = MSE(y_test, y_pred)**(1/2)

# Print the Root Mean Squared Error
print('Root Mean Squared Error: {:.3f} '.format(RMSE))

# Get parameters of Decision Tree Regression model for cross validation
dt.get_params()

# Define grid of hyperparameters 
params_dt = {'max_depth': [4, 6, 8],
 'min_samples_leaf': [0.1, 0.2],
 'max_features': ['log2', 'sqrt']}

# Re-intantiate Decision Tree Regressor for cross validation
dt = DecisionTreeRegressor()

# Instantiate GridSearch cross validation
dt_cv = GridSearchCV(estimator=dt,
 param_grid=params_dt,
 scoring='neg_mean_squared_error',
 cv=5,
 verbose=1,
 n_jobs=-1)

# Fit model to 
dt_cv.fit(X_train, y_train)

# Print best parameters
print('Best parameters for this Decision Tree Regressor model: {}'.format(dt_cv.best_params_))

# # Generate model best score
print('Best score for this Decision Tree Regressor model: {:.3f}'.format(dt_cv.best_score_))
