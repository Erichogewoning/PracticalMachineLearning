# Machine Learning - Predicting type of barbell lifts
Eric Hogewoning  

## Synopsis
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 
In this project, we used data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://groupware.les.inf.puc-rio.br/har].

We used this data to train a (random forest) model. This model classifies the type of barbell lift based on the accelerometers. An (out of sample) accuracy of 99.1% was achieved.

##Loading Data

The data set for this project was taken from: [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv]




Loading and splitting the data set.


```r
pml_training_data <- read.csv("pml-training.csv", stringsAsFactors=TRUE)

## set seed for reproducability
set.seed(2015)

##data set is split randomly into 2 parts: 60% for training, 40% for testing.
InTrain = createDataPartition(pml_training_data$classe, p = 0.6)[[1]]
subtraining = pml_training_data[ InTrain,]
subtesting = pml_training_data[-InTrain,]
```


## Exploring the data

When deciding which columns to use for our model, we looked at the data set and noticed there are quite a few columns which contain mostly NA or blank values. Though possibly useful for prediction, we decided to first try to build a model without them. Fortunately, columns either contain mostly NA values or no NA's at all, making it easy to separate the two.

When looking for good predictor fields, we decided to exclude the first 7 columns, which contained data like: id, user_name, date, time and window. Since we want to predict the lift type based on the accelerometers, we leave the non-sensor data out.

The output variable 'classe' is categorical, which means that a classification algorithm is preferable to a regression algorithm.


```r
## count the number of NA's per column
na_count <-sapply(subtraining, function(y) sum(length(which(is.na(y)))))
head(na_count, 25)
```

```
##                    X            user_name raw_timestamp_part_1 
##                    0                    0                    0 
## raw_timestamp_part_2       cvtd_timestamp           new_window 
##                    0                    0                    0 
##           num_window            roll_belt           pitch_belt 
##                    0                    0                    0 
##             yaw_belt     total_accel_belt   kurtosis_roll_belt 
##                    0                    0                    0 
##  kurtosis_picth_belt    kurtosis_yaw_belt   skewness_roll_belt 
##                    0                    0                    0 
## skewness_roll_belt.1    skewness_yaw_belt        max_roll_belt 
##                    0                    0                11528 
##       max_picth_belt         max_yaw_belt        min_roll_belt 
##                11528                    0                11528 
##       min_pitch_belt         min_yaw_belt  amplitude_roll_belt 
##                11528                    0                11528 
## amplitude_pitch_belt 
##                11528
```

```r
str(subtraining$classe)
```

```
##  Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
```

## Preprocessing data

In the following code, the 'mostly NA' columns are dropped. For the remaining columns, the variance is calculated. Columns with near zero variance are also dropped, because they are unlikely to be useful predictors.
Finally, we also drop the first 7 columns, since they don't contain sensor data.


```r
traincleaned <- subset(subtraining, select=colnames(subtraining)[-which(na_count > 0)])
NZV <- nearZeroVar(traincleaned, saveMetrics=FALSE)

#removing the nearZeroValue columns as well as the now number, person, date and time.
traincleaned <- subset(traincleaned, select=colnames(traincleaned)[-c(1:7,NZV)])
```

## Training the Model

At this point we decide to prepare our data for a random forest model. Random forest does not require normalized data, so we do not apply centering and scaling. The random forest will use **cross-validation** by growing trees on various samples of the training data and combining their predictions.


```r
##train the model
modelFit <- train(traincleaned$classe ~ . , method="rf", data=traincleaned)

#from the test set, remove the same columns as from the training set
testcleaned <- subset(subtesting, select=colnames(subtesting)[-which(na_count > 0)])
testcleaned <- subset(testcleaned, select=colnames(testcleaned)[-c(1:7,NZV)])

##apply the model to the cleaned test set to create predictions.
predictions <- predict(modelFit, testcleaned)
```

## Results

The model is trained on all remaining (53) columns. The test set is given the same preparation as the training set.
Predictions are done on the remaining 40% of the data set and the predictions are matched with the actual values, as shown in the confusion matrix. 

The result is an accuracy of 99.1 %. (The out-of-sample error rate is 0.9% )


```r
##compare predictions with actual values in the test set.
confMatrix <- confusionMatrix(testcleaned$classe, predict(modelFit, testcleaned))
confMatrix$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2222    5    4    0    1
##          B   13 1497    8    0    0
##          C    0    5 1354    9    0
##          D    0    1   12 1273    0
##          E    0    1    4    7 1430
```

```r
confMatrix$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9910783      0.9887152      0.9887412      0.9930386      0.2848585 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```
