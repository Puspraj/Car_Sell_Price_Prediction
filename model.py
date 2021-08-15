import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("car data.csv")

data.columns

final_dataset = data[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


final_dataset['Current_year']=2021

final_dataset['Number_Of_year']=final_dataset['Current_year'] - final_dataset['Year']

final_dataset.drop(['Year', 'Current_year'], axis=1, inplace=True)

final_dataset = pd.get_dummies(final_dataset, drop_first=True)

final_dataset.corr()

# Now split the dataset into independent feature and dependent feature
x = final_dataset.iloc[:, 1:]
y = final_dataset.iloc[:,0]

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x,y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()

# Hyper tunning
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

#Maximum number of leaves in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]

#Minimu number of samples required to split a node
min_samples_split = [2,5,10,15,100]

#Minimu number of samples required at each leaf node
min_samples_leaf = [1,2,5,10]


random_grid = {'n_estimators':n_estimators,
              'max_features':max_features,
              'max_depth':max_depth,
              'min_samples_split':min_samples_split,
              'min_samples_leaf':min_samples_leaf
              }

from sklearn.model_selection import RandomizedSearchCV

rf_random = RandomizedSearchCV(estimator=rfr, param_distributions=random_grid,n_iter=10,
                               scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=1)

rf_random.fit(x_train, y_train)


rfr_predict =rf_random.predict(x_test)
rfr_predict
sns.distplot(y_test-rfr_predict)

plt.scatter(y_test, rfr_predict)


from sklearn import metrics
print("MSE: ", metrics.mean_squared_error(y_test, rfr_predict))
print("MAE: ", metrics.mean_absolute_error(y_test, rfr_predict))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, rfr_predict)))

import pickle
file = open('randomForestRegressorModel.pkl', 'wb')
pickle.dump(rf_random, file)