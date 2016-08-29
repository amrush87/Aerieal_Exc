import numpy as np, sklearn
import pandas as pd

from sklearn.preprocessing import Imputer

#Reading data
data_2013 = pd.read_csv('https://aerialintel.blob.core.windows.net/recruiting/datasets/wheat-2013-supervised.csv', delimiter=',')
data_2014 = pd.read_csv('https://aerialintel.blob.core.windows.net/recruiting/datasets/wheat-2014-supervised.csv', delimiter=',')

"""print(pd.isnull(data_1).any(1).nonzero()[0])
print(pd.isnull(data_2).any(1).nonzero()[0])
"""

#Eliminating county and state columns
data_1 = data_2013.iloc[:,2:]
data_2 = data_2014.iloc[:,2:]

#Eliminating Date column
data_1 = data_1.drop('Date', axis=1)
data_2 = data_2.drop('Date', axis=1)

#Replacing Nan and empty cells with zeros
data_1 = data_1.replace([np.inf, -np.inf], np.nan)
data_2 = data_2.replace([np.inf, -np.inf], np.nan)

data_1 = data_1.fillna(value=0)
data_2 = data_2.fillna(value=0)


#sklearn.preprocessing.Imputer (impute through columns "axis = 0")
#Not sure I want to use this, taking median/mean would be affected by county.
#If I want to use this method, I have to consider each empty value to be median/mean of that particular county
#filling empty spaces with zeros would be more neutral in this case even though not the best solution. 
"""
imp = Imputer(missing_values='NaN', strategy='median', axis=0)
data_1 = imp.fit(data_1)
data_2 = imp.fit(data_2)
"""

print(data_1.shape, data_2.shape)

#separating Yield (Y1 and Y2) from Data (X1 and X2)
X1 = data_1.iloc[:,0:-1]
Y1 = data_1.iloc[:,-1]
X2 = data_2.iloc[:,0:-1]
Y2 = data_2.iloc[:,-1]

print(X1.shape, X2.shape)
print(Y1.shape, Y2.shape)

#Cross-Validation method and selecting best features
#I decided to go with another method (Lasso, below) because this would've been better with the regression method 
#I intended to use Stochastic Gradient Descent (SGDRegressor) first. However, that method gave horrible results. 
#The idea is that I thought that weather data being stochastic, a gradient descent would work. 
#I didn't explore it further. 

"""
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

X_1 = SelectKBest(f_regression, k=10).fit_transform(X1, np.reshape(Y1, (np.size(Y1),)))
X_2 = SelectKBest(f_regression, k=10).fit_transform(X2, np.reshape(Y2, (np.size(Y2),)))

x_train1, x_test1, y_train1, y_test1 = train_test_split(X1,Y1)
x_train2, x_test2, y_train2, y_test2 = train_test_split(X2,Y2)
"""

#Alternative method with better results. The idea is to create a n-fold cross-validated predictions. 
#I chose to build functions so it would be easier to test different sets of data. 

from sklearn.cross_validation import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

def build_regr (data, labels):
    #######
    #Function that Builds classifier. 
    #Lasso regression should work well with the number of dimensions and amount of data at hand
    #Choice is based on the flexibility of Lasso to select variables and regularize data
    #INPUTS:
    #data: training data
    #labels: training labels
    #####    
    regr = Lasso()
    regr.fit(data, labels)
    return regr

def classify (test_data, regr):
    ######
    #Function that gives out predictions of input test data given an input classifier
    #INPUTS:
    #test_data: data to be predicted given a classifier
    #regr: Classifier object built in build_regr
    #OUTPUTS:
    #predictions of test_data
    ######    
    return regr.predict(test_data)

def compare(predictions, test_labels,test_ind):
    #####
    #Calculating mean square error of predictions
    #INPUTS:
    #predictions:predicted values by the classifier
    #test_labels: test data labels
    #test_ind: the indices of the test data produced by KFold
    #OUTPUTS:
    #error: mean square error between test labels and predictions
    ####

    error = mean_squared_error((test_labels[test_ind]), (predictions))
    return error

def kf_regr (data,labels, n):
    ######
    #Function that divides the data to n-fold cross-validation, uses build_regr, classify, and compare
    #INPUTS: 
    #data
    #labels
    #n: number of folds
    #OUTPUTS:
    #kf:the indices of training and test data in each fold
    #pred_list: the list of predictions of each fold
    #error_list: the mean square error of each fold
    ########
    
    #KFold will create n sets of indices for n-fold training and test data
    kf = KFold(len(data), n_folds = n)
    itr = 0
    for train_ind, test_ind in kf:
        X_train, X_test = data.iloc[train_ind,:], data.iloc[test_ind,:] 
        y_train, y_test = labels.iloc[train_ind], labels.iloc[test_ind]
        regr = build_regr(X_train, y_train)
        pred = classify (X_test, regr)
        error = compare (pred, y_test, test_ind)
        if itr == 0:
            pred_arr = pred
            error_arr = error
        else:
            pred_arr = np.hstack((pred_arr, pred))
            error_arr = np.vstack((error_arr, error))
        print(itr)
        itr += 1
        
        
        
    return kf, pred_arr, error_arr

#Cross-validated predictions for the wheat-2013-supervised data set

kf1, pred1, error1 = kf_regr(X1, Y1, 10)
mean_err1 = np.mean(error1)
print(pred1)
print(error1)

print(pred1.shape, error1.shape)

#Cross-validated predictions for the wheat-2014-supervised data set

kf2, pred2, error2 = kf_regr(X2, Y2, 10)
mean_err2 = np.mean(error2)
print(pred2)
print(error2)

print(pred2.shape, error2.shape)

#Building a combined data set of wheat-2013-supervised and wheat-2014-supervised. 

XC = pd.concat((X1,X2), axis= 0, ignore_index = True)
YC = pd.concat((Y1,Y2), axis= 0, ignore_index = True)

print(XC.shape, YC.shape)

#Cross-validated predictions for the combined data set

kfC, predC, errorC = kf_regr(XC, YC, 10)
mean_errC = np.mean(errorC)
print(predC)
print(errorC)

print(predC.shape, errorC.shape)

#Appending results to to relevant data fields

final1 = pd.DataFrame(data_2013.iloc[:,[0,1,2,3,4,-1]])
final1['Predictions'] = pd.Series(pred1)
final1['Mean Error'] = pd.Series(mean_err1)
final2 = pd.DataFrame(data_2014.iloc[:,[0,1,2,3,4,-1]])
final2['predictions'] = pd.Series(pred2)
final2['Mean Error'] = pd.Series(mean_err2)
finalC = pd.concat((data_2013.iloc[:,[0,1,2,3,4,-1]], data_2014.iloc[:,[0,1,2,3,4,-1]]), axis= 0, ignore_index= True)
finalC['predictions'] = pd.Series(predC)
finalC['Mean Error'] = pd.Series(mean_errC)

print(final1.shape, final2.shape, finalC.shape)

#writing results into .csv files

final1.to_csv('predictions_2013.csv', sep=',')
final2.to_csv('predictions_2014.csv', sep=',')
finalC.to_csv('predictions_combined.csv', sep=',')
