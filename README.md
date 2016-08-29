# Aerieal_Exc

Description:

We have two data sets that contain location and time, raw features, and wheat yields of many counties around the country. The goal is to develop a machine learner that could predict the yield given the location and raw features. I must point out that I didn’t check the results after I produced them; mostly due time restriction. 

Data Preprocessing:

We have five location and time fields, and 20 different raw features. This is a regression problem. First thing I did, is to divide the data as such, two data arrays that contain the features (one for each year), and two arrays for targets (yield). 

First thing I noticed about the data is missing data cells. I simply filled them with zeros. Given more time, I would have filled every missing data cell with the median or mean of that feature within a few days’ range in that particular county (or maybe a range of latitudes and longitudes). 

Then I had to decide on how I want to divide my data into training and testing set. At first, I did a direct 0.75/0.25 on one set to try different regressors on that split and see how they perform. After I chose which regressor, I wanted to do some cross-validation. The goal I had in mind is that I wanted to predict each yield once, using data from other locations. I built an n-fold cross-validation program that takes each data instance into the test set once. The n-fold dividing method I chose is on the indices of the data, instead of on the dataframes themselves.  I needed to keep track with the predictions. Therefore, I kept the original sequential indices in interest to keep it simple and trackable; had I had more time, I would’ve randomized this process to avoid bias issues while still keeping track of the indices.

Regressor Selection:

Since I don’t know much about the relationships between the features, I wanted to select the most useful features (using feature selection methods) and do Stochastic Gradient Descent regression on them. The idea was that weather data are stochastic and chaotic in nature, a stochastic based regressor (even if it’s linear) should be able navigate through that. I tried a few different ways for feature selection; namely, RReliefF algorithm and SelectKBest from Scikit Learn. Both of which gave terrible result with SGDRegressor. 

Instead of tweaking, I decided to do Lasso regression, as it also has an internal feature selection and does automatic regularization. Like all regression tools, Lasso regression has many options. I opted to go with defaults just to check whether Lasso would give reasonable results or not. Had I had more time, I would have tweaked and tried a few of these options.

I must mention that I could’ve randomized the index selection in KFold, but then I would have needed to have developed a way to keep track of and fill predictions into the prediction array. This way, they just fall nicely in place. 


Main Program Logic: 

Create a matrix that contains the indices of training and testing instances of each iteration using sklearn.cross_validation.KFold
Go into a for loop that uses those indices as double-iterators. 
Create temporary X_train, Y_train, X_test, and Y_test arrays by using the current iteration of indices. 
Feed those arrays to the classifier and predictor.
Feed the predictor result, along with Y_test, to calculate error (mean square error). 
Record error and predictions into a new arrays.
Go back to 3; until all indices are tested once. 

Testing and results: 

I decided to run the program three times. Once on each data set separately (two), and once on a combined data set. The accompanying .csv files has the predicted and original yields. Also, I have included the mean error of each run. 

The predicted yields themselves seem to be hit and miss. I would try to confirm whether these “misses” are within reasonable range or not. However, all predictions cluster around within the counties. I would explore more if there is any kind of overfitting or underfitting. 

The error produced has a clear disparity between the 2013 set and the 2014 set. The 2013 set has a much higher mean square error. This is also can clearly be seen in the combined set results. I would want to investigate more into that had I had more time. 

Problems I faced: 

The error calculation for the combined data set broke down every time. It took me a few hours to figure out why. Turns out that, on concatenating both data sets, Pandas automatically keeps the original indexing as the combined data set moves from the first to the second. In other words, the Pandas dataframe index goes back to 1 in the middle of combined data set, even though that wouldn’t affect Python’s index to the data. That’s what caused the error calculation to break down, I’m still not sure why. However, it’s solved by asking Pandas to ignore indexing while concatenating. 

Had I had More time:

Instead of zeros, I would have filled the missing data with medians/means of the county within a few days’ range. The idea is that since weather is quite a localized and short-term variable, having medians/means is better than zeros. 
I would have figured out a way to incorporate ranges for the predicted yields, instead of single numbers. Or at least a general method for calculating the overall standard deviation of the predictions. The way I would approach this is to take each county’s predicted yields and report the median/mean and standard deviation. From then on, I would tweak the results as long as it makes sense to do so. 
I would have looked deeper into the relationship of longitude and latitude and yield. Having the yield county-specific must have a particular effect on those two feature. I would also ask for the altitude of that location, it could have an effect too. 
Try different error calculations and see which one makes more sense. As it stands, mean square error is the bigger the error is the worse with no clear indication of how worse. 
Tweak the Lasso regressor fitting step. Try different options. 
Of course, I would have produced some visualizations! 


What I learned: 

I learned many methods and techniques using Pandas. I have used it only a few times, but I typically transformed any .csv data into np.array. I am more comfortable with NumPy than I am with Pandas. I have been told that Pandas is very useful and powerful, so I challenged myself and used Pandas as much as I can. 
