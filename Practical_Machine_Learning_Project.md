Practical Machine Learning Project
==================================

The goal of this project is to predict the manner in which the people using movement devices did the exercise. This is the "classe" variable in the training set.

Data Manipulation
-----------------

This process removes the columns that have a majority of NA and blank values for both the test and training datasets. It also removes the first six columns so we are left with the raw movement data and the "classe" variable that is in the training set.

``` r
#pml-testing.csv & pml-training.csv files must be in working directory
test_read <- read.csv("pml-testing.csv", na.strings = c("NA","#DIV/0!",""))
testing <- test_read[,colSums(is.na(test_read))==0] #remove columns with NAs
testing[,7:59] <- sapply(testing[,7:59], as.numeric)
test_20 <- testing[,7:59]

train_read <- read.csv("pml-training.csv", na.strings = c("NA","#DIV/0!",""))
training <- train_read[,colSums(is.na(train_read))==0] #remove columns with NAs
training[,7:59] <- sapply(training[,7:59], as.numeric)
training <- training[,7:60]
```

Prediction Analysis
-------------------

``` r
library(caret)
library(rpart)
library(randomForest)
library(e1071)

set.seed(123)
inTrain <- createDataPartition(y=training$classe, p=.75, list=FALSE)
train <- training[inTrain,]
test <- training[-inTrain,]
```

Linear Discriminant Analysis

``` r
modlda <- train(classe~., data=training, method="lda")
predictLDA <- predict(modlda, test)
```

Recursive Partition

``` r
modrp <- rpart(classe~., data=training, method="class")
predictRP <- predict(modrp, test, type="class")
```

Random Forest

``` r
modrf <- randomForest(classe~., data=training)
predictRF <- predict(modrf, test)
```

Test to see which predictive model is most accurate.

``` r
confusionMatrix(predictLDA, test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1154  122   65   36   37
    ##          B   41  621   90   32  136
    ##          C   93  134  559  109   76
    ##          D  104   38  113  599   81
    ##          E    3   34   28   28  571
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.7145          
    ##                  95% CI : (0.7017, 0.7271)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6391          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8272   0.6544   0.6538   0.7450   0.6337
    ## Specificity            0.9259   0.9244   0.8982   0.9180   0.9768
    ## Pos Pred Value         0.8161   0.6750   0.5757   0.6406   0.8599
    ## Neg Pred Value         0.9309   0.9177   0.9247   0.9483   0.9222
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2353   0.1266   0.1140   0.1221   0.1164
    ## Detection Prevalence   0.2883   0.1876   0.1980   0.1907   0.1354
    ## Balanced Accuracy      0.8766   0.7894   0.7760   0.8315   0.8053

``` r
confusionMatrix(predictRP, test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1245  146   33   47   49
    ##          B   47  583   57   67   91
    ##          C   19   67  687  130   73
    ##          D   63  114   58  518  118
    ##          E   21   39   20   42  570
    ## 
    ## Overall Statistics
    ##                                          
    ##                Accuracy : 0.7347         
    ##                  95% CI : (0.7221, 0.747)
    ##     No Information Rate : 0.2845         
    ##     P-Value [Acc > NIR] : < 2.2e-16      
    ##                                          
    ##                   Kappa : 0.6637         
    ##  Mcnemar's Test P-Value : < 2.2e-16      
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8925   0.6143   0.8035   0.6443   0.6326
    ## Specificity            0.9216   0.9338   0.9286   0.9139   0.9695
    ## Pos Pred Value         0.8191   0.6899   0.7039   0.5947   0.8237
    ## Neg Pred Value         0.9557   0.9098   0.9572   0.9291   0.9214
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2539   0.1189   0.1401   0.1056   0.1162
    ## Detection Prevalence   0.3100   0.1723   0.1990   0.1776   0.1411
    ## Balanced Accuracy      0.9071   0.7740   0.8661   0.7791   0.8011

``` r
confusionMatrix(predictRF, test$classe)
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    0    0    0    0
    ##          B    0  949    0    0    0
    ##          C    0    0  855    0    0
    ##          D    0    0    0  804    0
    ##          E    0    0    0    0  901
    ## 
    ## Overall Statistics
    ##                                      
    ##                Accuracy : 1          
    ##                  95% CI : (0.9992, 1)
    ##     No Information Rate : 0.2845     
    ##     P-Value [Acc > NIR] : < 2.2e-16  
    ##                                      
    ##                   Kappa : 1          
    ##  Mcnemar's Test P-Value : NA         
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
    ## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
    ## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Rate         0.2845   0.1935   0.1743   0.1639   0.1837
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000

Since the Random Forest model has the highest accuracy, we'll us it to predict the classes for the initial 20 test cases.

``` r
predict(modrf, test_20)
```

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E
