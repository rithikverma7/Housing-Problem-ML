#!/usr/bin/env python
# coding: utf-8

# ## Dragon Real Estate - Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")
# creates panda data frame 


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS']


# In[6]:


housing['CHAS'].value_counts()


# In[7]:


housing.describe()


# In[8]:


# for plotting histogram 
# %matplotlib inline
# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))


# ## Train-Test Splitting

# In[9]:


#for learning purposes
# import numpy as np
# def split_train_test(data, test_ratio):
#     np.random.seed(42) #only generates random indices once and don't change them
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data)*test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices] #iloc is a special way to retrive data
# this whole function is available in sklearn 
#train_set, test_set = split_train_test(housing,0.2)


# In[10]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set : {len(train_set)} \nRows in test set : {len(test_set)}")


# In[11]:


# However this type of splitting may cause some problem in the future 
# for eg. : CHAS feature has 471 0's and 35 1's 
# It may happen that CHAS has all values=1 in train set and proper training of model will not occur 
# we will use statified sampling on the basis of CHAS

from sklearn.model_selection import StratifiedShuffleSplit

shuffle = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_set, test_set in shuffle.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_set] #pandas.dataframe.loc[index] returns rows with index in series 
    strat_test_set = housing.loc[test_set]

# splits data acc. to CHAS feature value     
    


# In[12]:


strat_test_set['CHAS'].value_counts()


# In[13]:


strat_train_set['CHAS'].value_counts()


# In[14]:


95/7


# In[15]:


376/28


# In[16]:


#both splits give eual ratio of 0's and 1's

#always do this after splitting
housing = strat_train_set
housing.describe()


# ## LOOKING FOR CO-RELATIONS

# In[17]:


#relation between label i.e. Y and features OF X 
#using pearson corelation value lies b/w -1 to 1
# +1 : high positive corelation (feature with high +ve co relation if increases label also increases and vice versa )
# using pandas.df.corr


# In[18]:


corr_matrix = housing.corr()


# In[19]:


corr_matrix['MEDV'].sort_values(ascending = False)


# In[20]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
scatter_matrix(housing[attributes], figsize= (12,8))


# In[21]:


# insights from above curves : 
# diagnol elements are histogram instead of straight lines for better in sights 
# graph medv vs lstat shows high -ve corr
# grap rm vs medv shows high +ve corr


# In[22]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8) #alpha =0.8 will show dark color 


# In[23]:


# the reason we plotted this is ti see variations or problems in data 
# theres a horizontal cap at y=50 this hows in accuracy in data 
# for eq. too many outlet points (outside maine area of graph) should br removed inOrder to help model train better and so model doesnt get confused by these outlet points
# by graph insights of combined attributes we can even get help to select two attributes as one 


# In[24]:


#we can create an attribute of Tax/Room

#ousing["TAXRM"] = housing['TAX']/housing['RM']


# In[25]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[26]:


# I have got a good attribute with high -ve corelation
#housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[27]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# In[28]:


housing


# ## Missing Attributes
# 

# In[29]:


# To take care of missing attributes we have 3 options:
#     1. remove the data points with missing attributes
#     2. remove the whole attribute column which have a null vale
#     3. add a value for null values(0, mean or median)


# In[30]:


# option 1 : to be done if only very less rows(data) have null or missing values
housing.dropna(subset=["RM"]).shape
# a.shape #this function returns a new set of data but housing data remains same 
# if writtem like this housing.dropna(subset="RM",inplace = true) orignal data set will not be changed 


# In[31]:


# option2  : to remove the whole attribute column 
housing.drop("RM",axis=1).shape
#RM removed && orignal housind data is still the same 


# In[32]:


#option 3 : add a value for null values 
median = housing['RM'].median()
housing['RM'].fillna(median).shape #fills null with median value
#since inplace =true not written orignal dataset remains same i.e. no change to housing 
#we also need to add this median value to test set as well as new examples given
# sklearn has a class for this : Simple Imputer


# In[33]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
# now we need to fit this imputer on our datase
imputer.fit(housing)
imputer.statistics_


# In[34]:


# sklearn has a class for this : Simple Imputer
X = imputer.transform(housing) # X is a numpy array 
housing_tr = pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()


# ## Scikit-learn Design

# Primarly three types of objects :
# 1. Estimators : It estimates some parameter based on dataset.
#                 eg. Imputer : It has a fit() and tranform method()
#                 fit() method fits the dataset and calculate internal parameters (eg. median value)
# 2. Transformers : Tranform method takes input and return output based on the learning(calculated internal parameter) from fit()
#                   It also has has a convienence method called fit_transform() which fits and then tranform
#                     
# 3. predictors : eg. Linear Regression Model , fit() & predict() are two common methods
#                 fit() fits the data set into model & predict() predicts new value 
#                 also has a score() method to give score of predictions
#                 takes numpy array as input
# pipeline creation && serialisation too provided by scikit                

# ## Feature Scaling

# Primarly two types of scaling methods (produces value in 0 to 1):
# 
# 
#     1. Min-Max scaling (Normalisation)
#         value = (value - min)/(max - min) (In this method whole value change can change due to change in one value )
#     
#     
#     2. Standardization
#         value = (value - mean)/std (In this method variance = 1, If anyOne value changes it won't affect our value change.)

# ## CREATING A PIPELINE

# In[35]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# takes constructor parameter as a list 
my_pipline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    # .... add as many as you want in your pipeline
    ('std_scaler', StandardScaler())
])

housing_num_tr = my_pipline.fit_transform(housing)


# In[36]:


housing_num_tr.shape


# ## Selecting a desired model for Dragon Real Estates

# In[37]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor() # how easy it is to change model 
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[38]:


# model trained lets check for few values


# In[39]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipline.transform(some_data)


# In[40]:


model.predict(prepared_data)


# In[41]:


list(some_labels)


# ## evaluating the model

# In[42]:


import numpy as np
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)


# In[43]:


rmse #overfitting if mse = 0


# In[44]:


#high error change the model to decisiontree regressor 


# ## Using Better evaluation technique -  cross validation 

# In[45]:


# K-CV divide train set in K equal groups : to reduce high variance in training (to prevent overfitting)
# for i=0 to k-1 :
#     take ith group as test set and rest of groups as train set to train the model


# In[46]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
# cross_val_score(model, trainset, labels, scoring, K)
# cost function lower is better but CV requires utility(i.e. higher is better we negate the squared error to do so)
rmse_scores = np.sqrt(-scores)


# In[47]:


rmse_scores


# In[48]:


# very less errors and overfitting removed 
def print_scores(scores):
    print("Score : ", scores)
    print("Mean : ", scores.mean())
    print("Std : ", scores.std())


# In[49]:


print_scores(rmse_scores)


# ## Saving the Model 

# In[50]:


from joblib import dump, load
dump(model, 'Dragon.joblib') 


# ## Model Testing

# In[56]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipline.fit_transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
final_rmse = np.sqrt(final_mse)


# In[57]:


final_rmse


# In[58]:


prepared_data[0]


# In[ ]:




