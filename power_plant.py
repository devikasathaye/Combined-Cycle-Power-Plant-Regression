#!/usr/bin/env python
# coding: utf-8

# ## Name: Devika Jagadish Sathaye

# ### Importing Libraries

import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy import stats as st
from sklearn import neighbors, preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Download data from GIT local repository

get_ipython().system(' git clone https://github.com/devikasathaye/Combined-Cycle-Power-Plant-Regression')


# ### Read data from Excel file to dataframe

df = pd.read_excel('Combined-Cycle-Power-Plant-Regression/Folds5x2_pp.xlsx')
print(df.columns)
df.head(10) #Display the DataFrame


# ## (b) Exploring the data

# ## i.

# ### How many rows are in this data set?

num_of_rows=df.shape[0]
print("The number of rows in the given data set are",num_of_rows)


# ### How many columns are in this data set?

num_of_cols=df.shape[1]
print("The number of columns in the given data set are",num_of_cols)


# ### What do the rows and columns represent?

# The columns represent the features like- Temperature (T), Ambient Pressure (AP), Relative Humidity (RH), Exhaust Vacuum (V) and net hourly electrical energy output (EP) of the plant. Each row represents a sample/record/tuple. A row gives us the value for net hourly electrical energy output over 6 years for the given values of the other features like Temperature, Pressure, Humidity, Vacuum.

# ### ii. Pairwise Scatterplots

a=0

def diagfunc(x, **kws):
    global a
    ax = plt.gca()
    ax.annotate(df.columns[a], xy=(0.5, 0.5), xycoords=ax.transAxes)
    a=a+1

sns.set(context="paper")

g = sns.PairGrid(df).map_diag(diagfunc)
g = g.map_offdiag(plt.scatter)

for ax in g.axes.flatten():
    ax.set_ylabel('')
    ax.set_xlabel('')


# There exists a strong linear(negative) relationship between the following pairs- AT and PE- indicating high covariance.<br>The features AT and V have a linear(positive) relationship. And the features V and PE have a linear(negative) relationship.(medium covariance).<br>There is no relationship between other pairs of features(no or less covariance).

# ### iii. Mean, Median, Range, First and Third Quartiles, and Interquartile Ranges of each of the variables in the dataset.

mean=[]
for i in df.columns:
    mean.append(df[i].mean())

median=[]
for i in df.columns:
    median.append(df[i].median())

range_data=[]
for i in df.columns:
    rng=df[i].max()-df[i].min()
    range_data.append(rng)

fq=[]
for i in df.columns:
    fq.append(df[i].describe()[4])

tq=[]
for i in df.columns:
    tq.append(df[i].describe()[6])

iqr=[]
for i in range(len(df.columns)):
    iqr.append(tq[i]-fq[i])

stats=[]
stats.append(df.columns)
stats.append(mean)
stats.append(median)
stats.append(range_data)
stats.append(fq)
stats.append(tq)
stats.append(iqr)

stats=list(map(list,zip(*stats)))
statsDF=pd.DataFrame(stats)
statsDF.columns=['Feature','Mean','Median','Range','First Quartile','Third Quartile','Interquartile Range']
statsDF


# ## (c) Simple linear regression model, for each predictor

#Entire dataset is used for training/fitting. No train test split
df_X=df.drop(columns='PE')
df_y=pd.DataFrame(df['PE'])

columns=['AT', 'V', 'AP', 'RH', 'PE']
column_pred=['AT', 'V', 'AP', 'RH']


# Model of the form y = &beta;<sub>0</sub> + &beta;<sub>1</sub>X + &epsilon;

beta_uni=[] #Array of coefficient of each predictor taken individually
y = df_y['PE'] #Dataframe containing the dependent variable
j=0
for i in df_X.columns:
    print("Predictor-",columns[j])
    j=j+1
    X=df_X[i]
    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()
    predictions_uni = model.predict(X) # make the predictions by the model

    # Print out the statistics
    stat_uni=model.summary().tables[1]
    print(stat_uni)
    beta_uni.append(model.params[1])
    print("The accuracy is {}%".format((model.rsquared)*100))
    #For drawing the scatter plot and the regression line
    plt.scatter(X[i], y, color='black')
    plt.plot(X[i], predictions_uni, color='blue', linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()

print("Actual values\n", y.head(10))
print("Predicted values\n", predictions_uni.head(10))


# ### Describe your results. Is there a statistically significant association between the predictor and the response? Create some plots to back up your assertions.

# Considering &alpha;=0.05, and p=0.025.<br>
# For all features individually, t-values are high and p is low. Hence, all the features are significant features.<br>
# However, the magnitude of t values for AT and V is comparatively higher than that for AP and RH. So, AT and V are comparatively more significant than AP and RH. Same can be observed from the plots.

# ### Are there any outliers that you would like to remove from your data for each of these regression tasks?

#Removing outliers outside two standard deviations
data=df[(np.abs(st.zscore(df))<2).all(axis=1)]

X_out = data.drop('PE', axis=1)
y_out = data['PE']

print("After removing outliers")
# running simple regression again
for i in range(4):
    X_out_new = sm.add_constant(X_out[columns[i]])
    model = sm.OLS(y_out,X_out_new).fit()

    # print the Table of summary
    stats_out = model.summary().tables[1]
    print(stats_out)
    print("The accuracy is {}%".format((model.rsquared)*100))
    print("")


# From the tables, we can infer that removing outliers does not make much improvement in accuracy. So, removing outliers is not desirable.

# ## (d) Multiple Regression model to predict the response using all of the predictors.

# Model of the form y = &beta;<sub>0</sub> + &beta;<sub>1</sub>X<sub>1</sub> + &beta;<sub>2</sub>X<sub>2</sub> + &beta;<sub>3</sub>X<sub>3</sub> + &beta;<sub>4</sub>X<sub>4</sub> + &epsilon;

beta_mul=[] #Array of coefficient of each predictor

X = df_X #Dataframe of all the independent features and all the rows
y = df_y['PE'] #Dataframe containing the dependent variable

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions_mul = model.predict(X) # make the predictions by the model


print("Actual values\n", y.head(10))
print("Predicted values\n", predictions_mul.head(10))

#Print out the statistics
stats_mul=model.summary().tables[1]
print(stats_mul)

for i in range(1,5):
    beta_mul.append(model.params[i])
#print("The coefficients of the features are:", beta_mul)
print("The accuracy is {}%".format((model.rsquared)*100))


# Considering &alpha;=0.05, and p=0.025<br>
# The p values of all the features are 0. Hence, we can reject null hypothesis for all of them. All the features are statistically significant when taken together, since p is leass than 0.025 for all of them.

# ## (e) Comparison of results from (c) and (d). Plot of univariate vs. multivariate regression coefficients

# The accuracy of multiple regression model is more than that of univariate model.

n=['AT', 'V', 'AP', 'RH']
for i, txt in enumerate(n):
    x1=beta_uni[i]
    y1=beta_mul[i]
    plt.plot(x1, y1,'ko')
    plt.text(x1+0.05, y1+0.05, txt, fontsize=8)
plt.xlabel("Univariate Regression Coefficients")
plt.ylabel("Multiple Regression Coefficients")
plt.grid()
plt.show()


# ## (f) Non-linear association between any of the predictors and the response.

# Model of the form y = &beta;<sub>0</sub> + &beta;<sub>1</sub>X + &beta;<sub>2</sub>X<sup>2</sup> + &beta;<sub>3</sub>X<sup>3</sup> + &epsilon;

beta_nl=[]
poly = PolynomialFeatures(3)
for i in df_X.columns:
    print("Predictor-",i)
    col=df[i].values.reshape(-1,1)
    X_poly = poly.fit_transform(col)
    df_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names(i))

    print("Dataframe with non-linear terms-")
    print(df_poly.head())
    model = sm.OLS(y, df_poly).fit()
    predictions_nl = model.predict(df_poly) # make the predictions by the model

    # Print out the statistics
    stats_nl=model.summary().tables[1]
    print(stats_nl)
    print("The accuracy is {}%".format((model.rsquared)*100))
    beta_nl.append(model.params[1])
    print("")
    print("")

print("Actual values\n", y.head(10))
print("Predicted values\n", predictions_nl.head(10))


# Considering &alpha;=0.05, and p=0.025<br>
# From the p values in the table, we see that there is an evidence of non-linear association between the predictors and the response.<br>
# For predictor AT, AT, AT^2, AT^3 all are statistically significant.<br>
# For predictor V, V^2 is NOT statistically significant, as its p value is more than 0.025. V, V^3 are statistically significant.<br>
# For predictor AP, AP, AP^2, AP^3 all are statistically significant.<br>
# For predictor AT, RH, RH^2, RH^3 all are statistically significant.<br>

# ## (g) Evidence of association of interactions of predictors with the response?

poly = PolynomialFeatures(interaction_only=True)
X_poly = poly.fit_transform(df_X)
df_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names([i for i in df.columns]))
print("Dataframe with pairwise interaction terms-")
print(df_poly.head())
model = sm.OLS(y, df_poly).fit()

predictions_int = model.predict(df_poly) # make the predictions by the model

# Print out the statistics
stats_int=model.summary().tables[1]

print("")
print("")
print(stats_int)
print("The accuracy is {}%".format((model.rsquared)*100))


# From the p values in the table, we see that there is an evidence of association of interactions of predictors with the response. Considering &alpha;=0.05, and p=0.025, the terms 'V AP', 'AT RH', 'AT V' are statistically significant.

# ## (h) Improve the model using possible interaction terms or non-linear associations between the predictors and response.

# ### Train the regression model on a randomly selected 70% subset of the data with all predictors.

X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.30, random_state=42) #Train data=70%, test data=30%
print("Train data X_train\n", X_train.head(5))
print("Size of train data",X_train.shape)
print("")
print("Test data X_test\n", X_test.head(5))
print("Size of test data",X_test.shape)
print("")
print("Train labels y_train\n", y_train.head(5))
print("Size of train labels",y_train.shape)
print("")
print("Test labels y_test\n", y_test.head(5))
print("Size of test labels",y_test.shape)
print("")

beta_mul_h=[] #Array of coefficient of each predictor

X = X_train #Dataframe of all the independent features and all the rows
y = y_train #Dataframe containing the dependent variable

X = sm.add_constant(X)
X_test_new=sm.add_constant(X_test)

model = sm.OLS(y, X).fit()
predictions_mul_h = model.predict(X_test_new) # make the predictions by the model


print("Actual values\n", y_test.head(10))
print("Predicted values\n", predictions_mul_h.head(10))

#Print out the statistics
stats_mul_h=model.summary().tables[1]
print(stats_mul_h)

for i in range(1,5):
    beta_mul_h.append(model.params[i])
#print("The coefficients of the features are:", beta_mul)
print("The accuracy is {}%".format((model.rsquared)*100))

poly = PolynomialFeatures()
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)
df_poly = pd.DataFrame(X_poly_train, columns=poly.get_feature_names([i for i in df.columns]))
print("Dataframe with all possible interaction terms and quadratic nonlinearities")
print(df_poly.head())
y_train=list(y_train)
model = sm.OLS(y_train, df_poly).fit()
print("")
print("")
# Print out the statistics
stats_all=model.summary().tables[1]
print(stats_all)
print("The accuracy is {}%".format((model.rsquared)*100))
print("")
print("")

predictions_test_all = model.predict(X_poly_test) # make the predictions by the model
test_mse=mean_squared_error(y_test, predictions_test_all)
print("Test MSE",test_mse)

predictions_train_all = model.predict(X_poly_train) # make the predictions by the model
train_mse=mean_squared_error(y_train, predictions_train_all)
print("Train MSE",train_mse)


# Considering &alpha;=0.05, and p=0.025,<br>
# the terms 'RH^2', 'AP RH', 'AP^2', 'AT RH', 'AT V', 'AT^2', 'RH', 'AP' are statistically significant. Rest of the terms- 'V RH', 'V AP', 'V^2', 'AT AP', 'V', 'AT'- are statistically insignificant. But, we cannot remove the terms 'V', 'AT' since some interaction terms or quadratic nonlinearities involving these terms are statistically significant.

# ### Removing statistically insignificant terms one-by-one

# Removing the term 'V RH'

df_poly_new=df_poly.drop(columns=['V RH'])
X_poly_test=pd.DataFrame(X_poly_test)
X_poly_test.columns=df_poly.columns
X_poly_test_new=X_poly_test.drop(columns=['V RH'])
model = sm.OLS(y_train, df_poly_new).fit()

# Print out the statistics
stats_1=model.summary().tables[1]
print(stats_1)
print("The accuracy is {}%".format((model.rsquared)*100))
print("")
print("")

predictions_test_1 = model.predict(X_poly_test_new) # make the predictions by the model
test_mse_new=mean_squared_error(y_test, predictions_test_1)
print("Test MSE",test_mse_new)

predictions_train_1 = model.predict(df_poly_new) # make the predictions by the model
train_mse_new=mean_squared_error(y_train, predictions_train_1)
print("Train MSE",train_mse_new)


# Removing the term 'V^2'

df_poly_new=df_poly_new.drop(columns=['V^2'])
X_poly_test_new=X_poly_test_new.drop(columns=['V^2'])
model = sm.OLS(y_train, df_poly_new).fit()

# Print out the statistics
stats_2=model.summary().tables[1]
print(stats_2)
print("The accuracy is {}%".format((model.rsquared)*100))
print("")
print("")

predictions_test_2 = model.predict(X_poly_test_new) # make the predictions by the model
test_mse_new=mean_squared_error(y_test, predictions_test_2)
print("Test MSE",test_mse_new)

predictions_train_2 = model.predict(df_poly_new) # make the predictions by the model
train_mse_new=mean_squared_error(y_train, predictions_train_2)
print("Train MSE",train_mse_new)


# Removing the term 'V AP'

df_poly_new=df_poly_new.drop(columns=['V AP'])
X_poly_test_new=X_poly_test_new.drop(columns=['V AP'])
model = sm.OLS(y_train, df_poly_new).fit()

# Print out the statistics
stats_3=model.summary().tables[1]
print(stats_3)
print("The accuracy is {}%".format((model.rsquared)*100))
print("")
print("")

predictions_test_3 = model.predict(X_poly_test_new) # make the predictions by the model
test_mse_new=mean_squared_error(y_test, predictions_test_3)
print("Test MSE",test_mse_new)

predictions_train_3 = model.predict(df_poly_new) # make the predictions by the model
train_mse_new=mean_squared_error(y_train, predictions_train_3)
print("Train MSE",train_mse_new)


# Yes, the model can be improved using some significant interaction terms and quadratic nonlinearities, as we can see the Test MSE is decreased.

# ## (i) KNN Regression

# ### i. For Normalized and Raw features

#For normalized features
ErrorTrainNorm=[]
ErrorTestNorm=[]

minMaxSc=MinMaxScaler()
X_train_norm=minMaxSc.fit_transform(X_train)
X_test_norm=minMaxSc.transform(X_test)

for k in range(1, 101, 1):
    knn = neighbors.KNeighborsRegressor(n_neighbors=k)
    ErrorTrainNorm.append(1-(knn.fit(X_train_norm, y_train).score(X_train_norm, y_train)))
    ErrorTestNorm.append(1-(knn.score(X_test_norm, y_test)))

k=1+ErrorTestNorm.index(min(ErrorTestNorm))
print("k=",k)
print("Best Test Error with Normalized features=",min(ErrorTestNorm))

predictions_test_norm = knn.predict(X_test_norm) # make the predictions by the model
test_mse_new=mean_squared_error(y_test, predictions_test_norm)
print("Test MSE",test_mse_new)

#For raw features
ErrorTrainRaw=[]
ErrorTestRaw=[]

for k in range(1, 101, 1):
    knn = neighbors.KNeighborsRegressor(n_neighbors=k)
    ErrorTrainRaw.append(1-(knn.fit(X_train, y_train).score(X_train, y_train)))
    ErrorTestRaw.append(1-(knn.score(X_test, y_test)))

k=1+ErrorTestRaw.index(min(ErrorTestRaw))
print("k=",k)
print("Best Test Error with Raw features=",min(ErrorTestRaw))

predictions_test_raw = knn.predict(X_test_norm) # make the predictions by the model
test_mse_raw=mean_squared_error(y_test, predictions_test_raw)
print("Test MSE",test_mse_raw)


# ### Plot the train and test errors in terms of 1/k.

kv=[]
for i in range(1,101,1):
    kv.append(1/i)
plt.plot(kv, ErrorTrainRaw, label="ErrorTrainRaw")
plt.plot(kv, ErrorTestRaw, label="ErrorTestRaw")
plt.plot(kv, ErrorTrainNorm, label="ErrorTrainNorm")
plt.plot(kv, ErrorTestNorm, label="ErrorTestNorm")
plt.gca().legend(('ErrorTrainRaw','ErrorTestRaw','ErrorTrainNorm','ErrorTestNorm'))
plt.xlabel("1/k")
plt.ylabel("Error Rate")
plt.show()


# ## (j) Compare the results of KNN Regression with the linear regression model that has the smallest test error

# The best error rate for KNN regression is given by Normalized features for k=4.<br>
# The best error rate for linear regression is given by a model having significant interaction terms and quadratic non-linearities.

ErrorTrainNorm=[]
ErrorTestNorm=[]

minMaxSc=MinMaxScaler()
X_train_norm=minMaxSc.fit_transform(X_train)
X_test_norm=minMaxSc.transform(X_test)

knn = neighbors.KNeighborsRegressor(n_neighbors=k)
knn.fit(X_train_norm, y_train)
predictions_test_norm = knn.predict(X_test_norm) # make the predictions by the model
accu_knn=r2_score(y_test, predictions_test_norm)

print("Accuracy of KNN Regression with Normalized features is {}%".format(accu_knn*100))


# The accuracy for linear regression is given by a model having significant interaction terms and quadratic non-linearities is 93.83982931931386%

# Hence, the KNN Regression with normalized features performs better.
