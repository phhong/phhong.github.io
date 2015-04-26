---
title: "Exercise manner prediction report"
author: "phhong"
output: html_document
---

## Introduction

This study investigates the data for barbell lifts activities of 6 participants collected by devices such as Jawbone Up, Nike FuelBand, and Fitbit. Based on the data, we build a model to predict the manner in which they did the exercise. We also estimate the out of sample error of the model with cross-validation.

## Data cleaning

After check the features in the dataset, we find that some of them are obviously irrelevant to the outcome such as user_name, timestamp and window. Some features are statistics of others and most of their values are NA so we don't involve them in the model either.


```r
# Load and clean training dataset
trainingRawData<-read.csv("pml-training.csv")
cleanTraining<-subset(trainingRawData, select=-1:-7)
cleanTraining<-subset(cleanTraining, select=grep("var_|stddev_|avg_|max_|min_|skewness_|amplitude_|kurtosis_",names(cleanTraining), invert=TRUE))
```

## Build model

First we divide the dataset into two part, training and testing. We will build the model based on training set and do prediction on testing set.


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## Loading required package: methods
```

```r
set.seed(15236)
inTrain<-createDataPartition(y=cleanTraining$classe,p=0.5,list=FALSE)
training<-cleanTraining[inTrain,]
testing<-cleanTraining[-inTrain,]
```

Since we want to estimate the accuracy with the training set, we use 10-fold cross-validation.


```r
trControl<-trainControl(method = "cv",number=10,allowParallel = TRUE)
```

Then we train the training data via random forest method with 100 trees generated.


```r
library(doParallel)
```

```
## Warning: package 'doParallel' was built under R version 3.1.3
```

```
## Loading required package: foreach
```

```
## Warning: package 'foreach' was built under R version 3.1.3
```

```
## Loading required package: iterators
```

```
## Warning: package 'iterators' was built under R version 3.1.3
```

```
## Loading required package: parallel
```

```r
cluster <- makeCluster(detectCores())
registerDoParallel(cluster)
set.seed(15646)
system.time(modelFit<-train(classe~.,data=training,method="rf",trControl = trControl,ntree=100, prox=TRUE))
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.3
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
##    user  system elapsed 
##   25.26    0.97  805.40
```

```r
stopCluster(cluster)
registerDoSEQ()
```

