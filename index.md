# Practical Machine Learning Course Project
Evan Raichek  
August 25, 2017  




### Introduction
This project predicts how people exercise using a data collected from personal activity devices. 6 participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways and data from accelerometers on the belt, forearm, arm, and dumbell was captured.

### Data Preparation / Feature Selection
The data consist of a training and test datasets with the dimensions 19622 and 20 observations respectively, each with 160 variables. Viewing the training dataset reveals many columns with significant number of NAs.  This is confirmed through a summary NAs in columns.


```
##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
##  0.0000  0.0000  0.9793  0.6132  0.9793  1.0000
```

Based on this all columns with 97% or more NAs are removed, as they will provide limited value to the model.  Next an evaluation of correlation identifies 7 additional columns that can be removed


```
## [1] "accel_belt_z"     "roll_belt"        "accel_belt_y"    
## [4] "accel_belt_x"     "gyros_dumbbell_x" "gyros_dumbbell_z"
## [7] "gyros_arm_x"
```
Finally, the first 7 columns, which contain metadata, are removed,  leaving 46 variables (45 predictors and 1 outcome variable) to use in the model.

### Model Creation
The model selected is Random Forest, with data split into 70% for training, and 30% for validation.  Tuning of the random forest is performed to choose the best performing model on the the test data.  The 3 version run are:  
  
- **rf**:  Random Forest with caret defaults, to see the what it recommends for "mtry" 
- **rf2**: Random Forest with mtry set at 7    
    * This is approximately the square root of the number of predictors  
    * As seen later, the default Random Forest does not run an mtry close to 7  
- **rf_gridsearch**: Random Forest with cross validation (10 folds), with mtry from 5 - 10  
    * Use **cross validation** to see if how results differ from the default bootstrap  
    * Bracket the mtry value of 7, to determine mtry with the best accuracy  

### Model Execution and Comparison
The 3 models are:


```r
rf <- train(classe ~., data=train, method="rf", prox=TRUE)

mtry <- 7
tunegrid <- expand.grid(.mtry=mtry)
rf2 <- train(classe ~., data=train, method="rf", tuneGrid=tunegrid, prox=TRUE)

control <- trainControl(method="cv", number=10,  search="grid")
tunegrid <- expand.grid(.mtry=c(5:10))
rf_gridsearch <- train(classe ~., data=train, method="rf", metric="Accuracy", tuneGrid=tunegrid, trControl=control)
```


Summary of the models are below, with the following hightlights:

- For **rf**  the accuracy never reaches 99%, and the mtry jumps from 2 to 23
- For **rf2** and **rf_gridsearch** accuracy exceeds 99% with the highest accurcy being **rf_gridseach** with mtry = 7
- The two bootstrap models each ran several hours, while *rf_gridsearch**, using cross validation, ran in 31 minutes
- OOB error rates are between .8% and .9 % for all models


```
## Random Forest 
## 
## 13737 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9874451  0.9841172
##   23    0.9886624  0.9856587
##   45    0.9798728  0.9745367
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 23.
```

```
## Random Forest 
## 
## 13737 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results:
## 
##   Accuracy  Kappa    
##   0.990213  0.9876153
## 
## Tuning parameter 'mtry' was held constant at a value of 7
```

```
## Random Forest 
## 
## 13737 samples
##    45 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12361, 12364, 12363, 12364, 12364, 12365, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    5    0.9934478  0.9917114
##    6    0.9940305  0.9924485
##    7    0.9942489  0.9927250
##    8    0.9941033  0.9925406
##    9    0.9941035  0.9925410
##   10    0.9941760  0.9926326
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 7.
```
OOB;s for each model  

- **rf**: 0.873%  
- **rf2**: 0.807%  
- **rf_gridsearch**: 0.819%  

### Model Evaluation against Validation data
The selected model **rf_gridsearch** with mtry =7 was run against the validation data, and yieled similar results, exceeding 99%.  
The predictors used in this final model are the first 7 variable shown in variable importance.



```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B    8 1127    4    0    0
##          C    0   11 1015    0    0
##          D    0    0   12  952    0
##          E    0    0    0    4 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9932          
##                  95% CI : (0.9908, 0.9951)
##     No Information Rate : 0.2856          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9914          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9952   0.9895   0.9845   0.9958   1.0000
## Specificity            0.9998   0.9975   0.9977   0.9976   0.9992
## Pos Pred Value         0.9994   0.9895   0.9893   0.9876   0.9963
## Neg Pred Value         0.9981   0.9975   0.9967   0.9992   1.0000
## Prevalence             0.2856   0.1935   0.1752   0.1624   0.1832
## Detection Rate         0.2843   0.1915   0.1725   0.1618   0.1832
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9975   0.9935   0.9911   0.9967   0.9996
```

![](index_files/figure-html/modelvalidation-1.png)<!-- -->
