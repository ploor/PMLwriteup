# PML Write Up

#Executive Summary
This is the Course Project Writeup for the Coursera Practical Machine Learning Course.  Here we will build a random forests model to predict "classe" from accelerometer data.

The model here predicted "classe" for the "Course Project: Submission" with 100% accuracy.  Cross validation was used on 2 levels:

- to validate the random forests model on a testing sample

- and within the training sample 4-fold cross validation is used when fitting the random forests model

The out of sample error was 0.9%, and the in sample error was 0.0%.

## Data Processing

First thing, set the seed to make this all repeatable.

```r
set.seed(1234)
```

Fields in the CSV stored as "#DIV/0!" are converted to NA.

```r
df <- read.csv("pml-training.csv"
              ,na.strings=c("#DIV/0!")
              ,stringsAsFactors=FALSE)
```

The non-accelerometer columns are removed.

```r
df <- df[,!(names(df) %in% c("X"
                            ,"user_name"
                            ,"raw_timestamp_part_1"
                            ,"raw_timestamp_part_2"
                            ,"cvtd_timestamp"
                            ,"new_window"
                            ,"num_window"))]
```

We then peel off the target, and then we convert all the remaining columns to numeric.

```r
classe <- as.data.frame(df$classe)

df <- df[,!(names(df) %in% c("classe"))]
options(warn=-1) # to hide warnings from coercion to numeric
df <- as.data.frame(sapply( df, as.numeric ))

df <- cbind(df,classe)
names(df)[names(df) == 'df$classe'] <- 'classe'
```

We also exclude all of the fields that are more than 90% missing.

```r
# drop columns where NA is more than 90%
df<-df[which(colMeans(is.na(df))<=.90)]
```


## Model Fitting

60% of the data is used to fit the model, and 40% is left for testing.

```r
library(caret)
inTrain=createDataPartition(y=df$classe,p=0.6,list=FALSE)
training<-df[inTrain,]
testing<-df[-inTrain,]
```

A random forests model is fit with 4-fold cross validation.

```r
modFit <- train(classe ~ ., data=training
               ,method="rf", prox=TRUE
               ,trControl=trainControl(method="cv"
                                      ,number=4
                                      ,allowParallel=TRUE))
```


## Model Evaluation

The out of sample error was 0.9% (99.1% accuracy), and the in sample error was 0.0% (100% accuracy).


```r
training.predict <- predict(modFit, newdata=training)
confusionMatrix(training.predict,training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
testing.predict <- predict(modFit, newdata=testing)
confusionMatrix(testing.predict,testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232   15    0    0    0
##          B    0 1500   11    0    0
##          C    0    3 1353   30    3
##          D    0    0    4 1254    2
##          E    0    0    0    2 1437
## 
## Overall Statistics
##                                         
##                Accuracy : 0.991         
##                  95% CI : (0.989, 0.993)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.989         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.988    0.989    0.975    0.997
## Specificity             0.997    0.998    0.994    0.999    1.000
## Pos Pred Value          0.993    0.993    0.974    0.995    0.999
## Neg Pred Value          1.000    0.997    0.998    0.995    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.191    0.172    0.160    0.183
## Detection Prevalence    0.286    0.193    0.177    0.161    0.183
## Balanced Accuracy       0.999    0.993    0.992    0.987    0.998
```


## Application to the Course Project Submission testing set

As the training set was processed, the testing set is likewise processed, and the predictions are listed below.

```r
dftest <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!"),stringsAsFactors=FALSE)

dftest <- dftest[,(names(dftest) %in% names(df))]
dftest <- as.data.frame(sapply( dftest, as.numeric ))

dftest.predict <- predict(modFit, newdata=dftest)
dftest.predict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

These predictions have 100% accuracy when submitted.

## Appendix
To aid in repeatability:

```r
sessionInfo()
```

```
## R version 3.1.1 (2014-07-10)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## 
## locale:
## [1] LC_COLLATE=Chinese (Traditional)_Taiwan.950 
## [2] LC_CTYPE=Chinese (Traditional)_Taiwan.950   
## [3] LC_MONETARY=Chinese (Traditional)_Taiwan.950
## [4] LC_NUMERIC=C                                
## [5] LC_TIME=Chinese (Traditional)_Taiwan.950    
## 
## attached base packages:
## [1] stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
## [1] knitr_1.6          e1071_1.6-3        randomForest_4.6-7
## [4] caret_6.0-30       ggplot2_1.0.0      lattice_0.20-29   
## 
## loaded via a namespace (and not attached):
##  [1] BradleyTerry2_1.0-5 brglm_0.5-9         car_2.0-20         
##  [4] class_7.3-10        codetools_0.2-8     colorspace_1.2-4   
##  [7] compiler_3.1.1      digest_0.6.4        evaluate_0.5.5     
## [10] foreach_1.4.2       formatR_0.10        grid_3.1.1         
## [13] gtable_0.1.2        gtools_3.4.1        htmltools_0.2.4    
## [16] iterators_1.0.7     lme4_1.1-7          MASS_7.3-33        
## [19] Matrix_1.1-4        minqa_1.2.3         munsell_0.4.2      
## [22] nlme_3.1-117        nloptr_1.0.0        nnet_7.3-8         
## [25] plyr_1.8.1          proto_0.3-10        Rcpp_0.11.2        
## [28] reshape2_1.4        rmarkdown_0.2.49    scales_0.2.4       
## [31] splines_3.1.1       stringr_0.6.2       tools_3.1.1
```
