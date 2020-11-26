##   Evaluation of machine learning methods for rock mass classification                          ##
##   System for rock mass classification developed by: Santos, A.E.M.;Lana, M.S.;Pereira, T.M.    ##
##   Graduate Program in Mineral Engineering - PPGEM                                              ##
##   Federal University of Ouro Preto - UFOP                                                      ##
##   Mining Engineering DepartmentDEMIN                                                           ##

#########################

## Script to ANN model ##

#########################

## choice of work directory ##
## Here the user chooses the folder where his database is and where the R history and the outputs will be saved ##
setwd("...")
## Packages used ##
library(neuralnet)
library(gmodels)
library(ggsn)
library(caret)
library(plyr)
library(multiROC)
library(pROC)
library(UBL)
## Choose your database in csv format ##
## If there is any difficulty in reading the database, please consult the documentation at "help(read.csv2)" ##
data_base <- read.csv2("dataset_dummies.csv",header=TRUE,dec = ".")
data_base = data_base[,2:26]
## conversion of variables for the balancing process ##
data_base$O1intact_rock_strength  <- as.factor(data_base$O1intact_rock_strength ) 
data_base$O2intact_rock_strength  <- as.factor(data_base$O2intact_rock_strength ) 
data_base$O3intact_rock_strength  <- as.factor(data_base$O3intact_rock_strength ) 
data_base$O4intact_rock_strength  <- as.factor(data_base$O4intact_rock_strength ) 
data_base$O5intact_rock_strength  <- as.factor(data_base$O5intact_rock_strength ) 
data_base$O6intact_rock_strength  <- as.factor(data_base$O6intact_rock_strength ) 
data_base$O1weathering <- as.factor(data_base$O1weathering)
data_base$O2weathering <- as.factor(data_base$O2weathering)
data_base$O3weathering <- as.factor(data_base$O3weathering)
data_base$O4weathering <- as.factor(data_base$O4weathering)
data_base$O5weathering <- as.factor(data_base$O5weathering)
data_base$O1discontinuity_spacing  <- as.factor(data_base$O1discontinuity_spacing )
data_base$O2discontinuity_spacing  <- as.factor(data_base$O2discontinuity_spacing )
data_base$O3discontinuity_spacing  <- as.factor(data_base$O3discontinuity_spacing )
data_base$O4discontinuity_spacing  <- as.factor(data_base$O4discontinuity_spacing )
data_base$O1discontinuity_persistence <- as.factor(data_base$O1discontinuity_persistence)
data_base$O2discontinuity_persistence <- as.factor(data_base$O2discontinuity_persistence)
data_base$O3discontinuity_persistence <- as.factor(data_base$O3discontinuity_persistence)
data_base$O4discontinuity_persistence <- as.factor(data_base$O4discontinuity_persistence)
data_base$O1discontinuity_aperture <- as.factor(data_base$O1discontinuity_aperture)
data_base$O2discontinuity_aperture <- as.factor(data_base$O2discontinuity_aperture)
data_base$O3discontinuity_aperture <- as.factor(data_base$O3discontinuity_aperture)
data_base$O4discontinuity_aperture <- as.factor(data_base$O4discontinuity_aperture)
data_base$N1presence_of_water <- as.factor(data_base$N1presence_of_water)
data_base$classification <- as.factor(data_base$classification)
## Procedure for balancing the database against classes ##
## If there is any difficulty in applying Smote, please consult the documentation at "help(SmoteClassif)" ##
balanced_data <- SmoteClassif(classification ~., 
                              data_base, 
                              C.perc = "balance", 
                              k = 5, 
                              repl = FALSE,
                              dist = "Overlap")
## Visualization of the balancing process ##
summary(balanced_data$classification)
plot(balanced_data$classification, ylim=c(0,1000))
## Assigning maximum interaction limit ##
maxit<-as.integer(1000000)
## separation of training and test samples ##
train_idx <- sample(nrow(balanced_data), 2/3 * nrow(balanced_data))
data_train <- balanced_data[train_idx, ]
data_test <- balanced_data[-train_idx, ]
## conversion of variables for the ANN training ##
data_train$O1intact_rock_strength  <- as.numeric(data_train$O1intact_rock_strength ) 
data_train$O2intact_rock_strength  <- as.numeric(data_train$O2intact_rock_strength ) 
data_train$O3intact_rock_strength  <- as.numeric(data_train$O3intact_rock_strength ) 
data_train$O4intact_rock_strength  <- as.numeric(data_train$O4intact_rock_strength ) 
data_train$O5intact_rock_strength  <- as.numeric(data_train$O5intact_rock_strength ) 
data_train$O6intact_rock_strength  <- as.numeric(data_train$O6intact_rock_strength ) 
data_train$O1weathering <- as.numeric(data_train$O1weathering)
data_train$O2weathering <- as.numeric(data_train$O2weathering)
data_train$O3weathering <- as.numeric(data_train$O3weathering)
data_train$O4weathering <- as.numeric(data_train$O4weathering)
data_train$O5weathering <- as.numeric(data_train$O5weathering)
data_train$O1discontinuity_spacing <- as.numeric(data_train$O1discontinuity_spacing)
data_train$O2discontinuity_spacing <- as.numeric(data_train$O2discontinuity_spacing)
data_train$O3discontinuity_spacing <- as.numeric(data_train$O3discontinuity_spacing)
data_train$O4discontinuity_spacing <- as.numeric(data_train$O4discontinuity_spacing)
data_train$O1discontinuity_persistence <- as.numeric(data_train$O1discontinuity_persistence)
data_train$O2discontinuity_persistence <- as.numeric(data_train$O2discontinuity_persistence)
data_train$O3discontinuity_persistence <- as.numeric(data_train$O3discontinuity_persistence)
data_train$O4discontinuity_persistence <- as.numeric(data_train$O4discontinuity_persistence)
data_train$O1discontinuity_aperture <- as.numeric(data_train$O1discontinuity_aperture)
data_train$O2discontinuity_aperture <- as.numeric(data_train$O2discontinuity_aperture)
data_train$O3discontinuity_aperture <- as.numeric(data_train$O3discontinuity_aperture)
data_train$O4discontinuity_aperture <- as.numeric(data_train$O4discontinuity_aperture)
data_train$N1presence_of_water <- as.numeric(data_train$N1presence_of_water)
data_test$O1intact_rock_strength  <- as.numeric(data_test$O1intact_rock_strength ) 
data_test$O2intact_rock_strength  <- as.numeric(data_test$O2intact_rock_strength ) 
data_test$O3intact_rock_strength  <- as.numeric(data_test$O3intact_rock_strength ) 
data_test$O4intact_rock_strength  <- as.numeric(data_test$O4intact_rock_strength ) 
data_test$O5intact_rock_strength  <- as.numeric(data_test$O5intact_rock_strength ) 
data_test$O6intact_rock_strength  <- as.numeric(data_test$O6intact_rock_strength ) 
data_test$O1weathering <- as.numeric(data_test$O1weathering)
data_test$O2weathering <- as.numeric(data_test$O2weathering)
data_test$O3weathering <- as.numeric(data_test$O3weathering)
data_test$O4weathering <- as.numeric(data_test$O4weathering)
data_test$O5weathering <- as.numeric(data_test$O5weathering)
data_test$O1discontinuity_spacing <- as.numeric(data_test$O1discontinuity_spacing)
data_test$O2discontinuity_spacing <- as.numeric(data_test$O2discontinuity_spacing)
data_test$O3discontinuity_spacing <- as.numeric(data_test$O3discontinuity_spacing)
data_test$O4discontinuity_spacing <- as.numeric(data_test$O4discontinuity_spacing)
data_test$O1discontinuity_persistence <- as.numeric(data_test$O1discontinuity_persistence)
data_test$O2discontinuity_persistence <- as.numeric(data_test$O2discontinuity_persistence)
data_test$O3discontinuity_persistence <- as.numeric(data_test$O3discontinuity_persistence)
data_test$O4discontinuity_persistence <- as.numeric(data_test$O4discontinuity_persistence)
data_test$O1discontinuity_aperture <- as.numeric(data_test$O1discontinuity_aperture)
data_test$O2discontinuity_aperture <- as.numeric(data_test$O2discontinuity_aperture)
data_test$O3discontinuity_aperture <- as.numeric(data_test$O3discontinuity_aperture)
data_test$O4discontinuity_aperture <- as.numeric(data_test$O4discontinuity_aperture)
data_test$N1presence_of_water <- as.numeric(data_test$N1presence_of_water)

## ANN training ##
nn <- neuralnet((classification == "class_I") 
                + (classification == "class_II") 
                + (classification == "class_III") 
                + (classification == "class_IV") 
                + (classification == "class_V")
                ~ O1intact_rock_strength + O2intact_rock_strength + O3intact_rock_strength + O4intact_rock_strength + O5intact_rock_strength + O6intact_rock_strength 
                + O1weathering + O2weathering + O3weathering + O4weathering + O5weathering 
                + O1discontinuity_spacing + O2discontinuity_spacing + O3discontinuity_spacing + O4discontinuity_spacing 
                + O1discontinuity_persistence + O2discontinuity_persistence + O3discontinuity_persistence + O4discontinuity_persistence 
                + O1discontinuity_aperture + O2discontinuity_aperture + O3discontinuity_aperture + O4discontinuity_aperture 
                + N1presence_of_water, 
                data = data_train, algorithm = "rprop+", err.fct = "sse", stepmax=maxit, 
                threshold =1, hidden = c(25), act.fct = "tanh", linear.output = FALSE)

## neural network plot ##
plot(nn, rep = "best",
     radius = 0.05, arrow.length = 0.2, intercept = TRUE,
     intercept.factor = 0.4, information = FALSE, information.pos = 0.1,
     col.entry.synapse = "blue4", col.entry = "black",
     col.hidden = "black", col.hidden.synapse = "black",
     col.out = "black", col.out.synapse = "brown2",
     col.intercept = "blue", fontsize = 14, dimension = 3,
     show.weights = FALSE)
## Overfitting and underfitting verification - Prediction of the model using the training sample ##
pred_train_nn <- predict(nn, data_train)
d_train_nn <- apply(pred_train_nn, 1, which.max)
d_train_nn <- mapvalues(d_train_nn,from = c(1,2,3,4,5), to = c("class_I", "class_II", "class_III", "class_IV", "class_V"))
d_train_nn <- as.factor(d_train_nn)
result_train_nn <- confusionMatrix(d_train_nn, data_train$classification)
roc_curve_train_nn <- multiclass.roc(response = data_train$classification, predictor = as.numeric(as.factor(d_train_nn)))
## Validation of the neural network trained with the test sample ##
pred_test_nn <- predict(nn, data_test)
d_test_nn <- apply(pred_test_nn, 1, which.max)
d_test_nn <- mapvalues(d_test_nn,from = c(1,2,3,4,5), to = c("class_I", "class_II", "class_III", "class_IV", "class_V"))
d_test_nn <- as.factor(d_test_nn)
result_test_nn <- confusionMatrix(d_test_nn, data_test$classification)
roc_curve_test_nn <- multiclass.roc(response = data_test$classification, predictor = as.numeric(as.factor(d_test_nn)))

## Confusion matrix - result in the training sample ##
result_train_nn
## Auc - result in the training sample ##
roc_curve_train_nn
## Confusion matrix - result in the test sample ##
result_test_nn
## Auc - result in the test sample ##
roc_curve_test_nn

## Geral metrics for model evaluation ##
accuracy_nn_train <- result_train_nn$overall['Accuracy']
lower_accuracy_nn_train <- result_train_nn$overall['AccuracyLower']
upper_accuracy_nn_train <- result_train_nn$overall['AccuracyUpper']
kappa_nn_train <- result_train_nn$overall['Kappa']
pvalue_accuracy_nn_train <- result_train_nn$overall['AccuracyPValue']
auc_nn_train <- roc_curve_train_nn$auc
vector_model_train_nn <- c("Artificial Neural Networks - Training sample performance",
                           round(accuracy_nn_train,3), 
                           round(lower_accuracy_nn_train,3),
                           round(upper_accuracy_nn_train,3),
                           round(kappa_nn_train,3),
                           round(auc_nn_train,3),
                           round(pvalue_accuracy_nn_train,10^20))

accuracy_nn_test <- result_test_nn$overall['Accuracy']
lower_accuracy_nn_test <- result_test_nn$overall['AccuracyLower']
upper_accuracy_nn_test <- result_test_nn$overall['AccuracyUpper']
kappa_nn_test <- result_test_nn$overall['Kappa']
pvalue_accuracy_nn_test <- result_test_nn$overall['AccuracyPValue']
auc_nn_test <- roc_curve_test_nn$auc
vector_model_test_nn <- c("Artificial Neural Networks - Test sample performance",
                          round(accuracy_nn_test,3), 
                          round(lower_accuracy_nn_test,3),
                          round(upper_accuracy_nn_test,3),
                          round(kappa_nn_test,3),
                          round(auc_nn_test,3),
                          round(pvalue_accuracy_nn_test,10^20))

compare_models <- rbind(vector_model_train_nn,
                        vector_model_test_nn)
rownames(compare_models) <- c("Artificial Neural Networks - Training sample performance", 
                              "Artificial Neural Networks - Test sample performance")
colnames(compare_models) <- c("Model",
                              "Accuracy", 
                              "Lower confidence interval - Accuracy",
                              "Upper confidence interval - Accuracy",
                              "kappa index",
                              "Auc value",
                              "p-Value")
compare_models <- as.data.frame(compare_models)
View(compare_models)

#########################

## Script to SVM model ##

#########################

## choice of work directory ##
## Here the user chooses the folder where his database is and where the R history and the outputs will be saved ##
setwd("...")
## Packages used ##
library(e1071)
library(multiROC)
library(pROC)
library(UBL)
library(caret)
## Choose your database in csv format ##
## If there is any difficulty in reading the database, please consult the documentation at "help(read.csv2)" ##
data_1 <- read.csv2("data_tot.csv",header=TRUE,dec = ".")
data = data_1[,2:8]

head(data)
## conversion of variables for the balancing process ##
data$intact_rock_strength <- as.factor(data$intact_rock_strength)
data$weathering <- as.factor(data$weathering)
data$discontinuity_spacing <- as.factor(data$discontinuity_spacing)
data$discontinuity_persistence <- as.factor(data$discontinuity_persistence)
data$discontinuity_aperture <- as.factor(data$discontinuity_aperture)
data$presence_of_water <- as.factor(data$presence_of_water)
data$rmr_class <- as.factor(data$rmr_class)
## Procedure for balancing the database against classes ##
## If there is any difficulty in applying Smote, please consult the documentation at "help(SmoteClassif)" ##
balanced_data <- SmoteClassif(rmr_class ~., 
                                  data, 
                                  C.perc = "balance", 
                                  k = 5, 
                                  repl = FALSE,
                                  dist = "Overlap")
## conversion of variables for the balancing process ##
balanced_data$intact_rock_strength <- as.numeric(balanced_data$intact_rock_strength)
balanced_data$weathering <- as.numeric(balanced_data$weathering)
balanced_data$discontinuity_spacing <- as.numeric(balanced_data$discontinuity_spacing)
balanced_data$discontinuity_persistence <- as.numeric(balanced_data$discontinuity_persistence)
balanced_data$discontinuity_aperture <- as.numeric(balanced_data$discontinuity_aperture)
balanced_data$presence_of_water <- as.numeric(balanced_data$presence_of_water)
## separation of training and test samples ##
train_idx <- sample(nrow(balanced_data), 2/3 * nrow(balanced_data))
data_train <- balanced_data[train_idx, ]
data_test <- balanced_data[-train_idx, ]
## SVM training ##
model_svm <- svm(
  rmr_class ~ .,
  data = data_train,
  kernel='radial',
  cost = 2, gamma = 1)
## Predict SVM model in data train ##
predict_model_svm <- predict(model_svm, newdata = data_train)
validation_predict_model_svm_train <- confusionMatrix(predict_model_svm, data_train$rmr_class)
curve_roc_svm_train <- multiclass.roc(response = data_train$rmr_class, predictor = as.numeric(as.factor(predict_model_svm)))
## Predict SVM model in data test ##
predict_model_svm <- predict(model_svm, newdata = data_test)
validation_predict_model_svm_test <- confusionMatrix(predict_model_svm, data_test$rmr_class)
curve_roc_svm_test <- multiclass.roc(response = data_test$rmr_class, predictor = as.numeric(as.factor(predict_model_svm)))
## Confusion matrix (data test) ##
validation_predict_model_svm_test
## Confusion matrix by Class (data test) ##
validation_predict_model_svm_test$byClass

##View SVM results ##
accuracy_svm_train <- validation_predict_model_svm_train$overall['Accuracy']
lower_accuracy_svm_train <- validation_predict_model_svm_train$overall['AccuracyLower']
upper_accuracy_svm_train <- validation_predict_model_svm_train$overall['AccuracyUpper']
kappa_accuracy_svm_train <- validation_predict_model_svm_train$overall['Kappa']
pvalue_accuracy_svm_train <- validation_predict_model_svm_train$overall['AccuracyPValue']
auc_svm_train <- curve_roc_svm_train$auc
vetor_model_train_svm <- c("SVM - Sample train",
                            round(accuracy_svm_train,3), 
                            round(lower_accuracy_svm_train,3),
                            round(upper_accuracy_svm_train,3),
                            round(kappa_accuracy_svm_train,3),
                            round(auc_svm_train,3),
                            round(pvalue_accuracy_svm_train,10^20))
accuracy_svm_test <- validation_predict_model_svm_test$overall['Accuracy']
lower_accuracy_svm_test <- validation_predict_model_svm_test$overall['AccuracyLower']
upper_accuracy_svm_test <- validation_predict_model_svm_test$overall['AccuracyUpper']
kappa_accuracy_svm_test <- validation_predict_model_svm_test$overall['Kappa']
pvalue_accuracy_svm_test <- validation_predict_model_svm_test$overall['AccuracyPValue']
auc_svm_test <- curve_roc_svm_test$auc
vetor_model_test_svm <- c("SVM - Sample test",
                           round(accuracy_svm_test,3), 
                           round(lower_accuracy_svm_test,3),
                           round(upper_accuracy_svm_test,3),
                           round(kappa_accuracy_svm_test,3),
                           round(auc_svm_test,3),
                           round(pvalue_accuracy_svm_test,10^20))
compare_models <- rbind(vetor_model_train_svm,
                         vetor_model_test_svm)
rownames(compare_models) <- c("SVM - Sample train", 
                               "SVM - Sample test")
colnames(compare_models) <- c("Model",
                               "Accuracy", 
                               "Lower confidence interval - Accuracy",
                               "Upper confidence interval - Accuracy",
                               "kappa",
                               "index-auc",
                               "p-Value")
## Final model data train x data test (vrification of overfitting) ##
compare_models <- as.data.frame(compare_models)
View(compare_models)

#################################

## Script to Naive Bayes model ##

#################################

## NB training ##
nb_classifier <- naiveBayes(data_train[,-7], data_train$rmr_class)
## Predict NB model in data train ##
sms_train_pred <- predict(nb_classifier, data_train[,-7])
sms_train_pred <- as.factor(sms_train_pred)
result_naive_bayes_train <- confusionMatrix(sms_train_pred, data_train$rmr_class)
curve_roc_naive_bayes_train <- multiclass.roc(response = data_train$rmr_class, predictor = as.numeric(as.factor(sms_train_pred)))
## Predict NB model in data test ##
sms_test_pred <- predict(nb_classifier, data_test[,-7])
sms_test_pred <- as.factor(sms_test_pred)
result_naive_bayes_test <- confusionMatrix(sms_test_pred, data_test$rmr_class)
curve_roc_naive_bayes_test <- multiclass.roc(response = data_test$rmr_class, predictor = as.numeric(as.factor(sms_test_pred)))
## Confusion matrix (data test) ##
result_naive_bayes_test
## Confusion matrix by class (data test) ##
result_naive_bayes_test$byClass
##View NB results ##
accuracy_naive_bayes_train <- result_naive_bayes_train$overall['Accuracy']
lower_accuracy_naive_bayes_train <- result_naive_bayes_train$overall['AccuracyLower']
upper_accuracy_naive_bayes_train <- result_naive_bayes_train$overall['AccuracyUpper']
kappa_accuracy_naive_bayes_train <- result_naive_bayes_train$overall['Kappa']
pvalue_accuracy_naive_bayes_train <- result_naive_bayes_train$overall['AccuracyPValue']
auc_naive_bayes_train <- curve_roc_naive_bayes_train$auc
vetor_model_train_naive_bayes <- c("Naive Bayes - Sample train",
                                   round(accuracy_naive_bayes_train,3), 
                                   round(lower_accuracy_naive_bayes_train,3),
                                   round(upper_accuracy_naive_bayes_train,3),
                                   round(kappa_accuracy_naive_bayes_train,3),
                                   round(auc_naive_bayes_train,3),
                                   round(pvalue_accuracy_naive_bayes_train,10^20))

accuracy_naive_bayes_test <- result_naive_bayes_test$overall['Accuracy']
lower_accuracy_naive_bayes_test <- result_naive_bayes_test$overall['AccuracyLower']
upper_accuracy_naive_bayes_test <- result_naive_bayes_test$overall['AccuracyUpper']
kappa_accuracy_naive_bayes_test <- result_naive_bayes_test$overall['Kappa']
pvalue_accuracy_naive_bayes_test <- result_naive_bayes_test$overall['AccuracyPValue']
auc_naive_bayes_test <- curve_roc_naive_bayes_test$auc
vetor_model_test_naive_bayes <- c("Naive Bayes - Sample test",
                                  round(accuracy_naive_bayes_test,3), 
                                  round(lower_accuracy_naive_bayes_test,3),
                                  round(upper_accuracy_naive_bayes_test,3),
                                  round(kappa_accuracy_naive_bayes_test,3),
                                  round(auc_naive_bayes_test,3),
                                  round(pvalue_accuracy_naive_bayes_test,10^20))


compare_models <- rbind(vetor_model_train_naive_bayes,
                        vetor_model_test_naive_bayes)
rownames(compare_models) <- c("Naive Bayes - Sample train", 
                              "Naive Bayes - Sample test")

colnames(compare_models) <- c("Model",
                              "Accuracy", 
                              "Lower confidence interval - accuracy",
                              "Upper confidence interval - accuracy",
                              "kappa",
                              "index-auc",
                              "p-Value")
compare_models <- as.data.frame(compare_models)
## Final model data train x data test (vrification of overfitting) ##
View(compare_models)

###################################

## Script to Random Forest model ##

###################################

## Packages used ##
library(randomForest)
## RF training ##
rf_model <- randomForest(data_train[-7], data_train$rmr_class, ntree = 100)
## Predict RF model in data train ##
model_pred_train <- predict(rf_model, data_train)
result_random_forest_train <- confusionMatrix(model_pred_train, data_train$rmr_class)
curve_roc_random_forest_train <- multiclass.roc(response = data_train$rmr_class, predictor = as.numeric(as.factor(model_pred_train)))
## Predict RF model in data test ##
model_pred_test <- predict(rf_model, data_test)
result_random_forest_test <- confusionMatrix(model_pred_test, data_test$rmr_class)
curve_roc_random_forest_test <- multiclass.roc(response = data_test$rmr_class, predictor = as.numeric(as.factor(model_pred_test)))
## Confusion matrix (data test) ##
result_random_forest_test
## Confusion matrix by class (data test) ##
result_random_forest_test$byClass
##View RF results ##
accuracy_random_forest_train <- result_random_forest_train$overall['Accuracy']
lower_accuracy_random_forest_train <- result_random_forest_train$overall['AccuracyLower']
upper_accuracy_random_forest_train <- result_random_forest_train$overall['AccuracyUpper']
kappa_accuracy_random_forest_train <- result_random_forest_train$overall['Kappa']
pvalue_accuracy_random_forest_train <- result_random_forest_train$overall['AccuracyPValue']
auc_random_forest_train <- curve_roc_random_forest_train$auc
vetor_model_train_random_forest <- c("Random Forest - Sample train",
                                      round(accuracy_random_forest_train,3), 
                                      round(lower_accuracy_random_forest_train,3),
                                      round(upper_accuracy_random_forest_train,3),
                                      round(kappa_accuracy_random_forest_train,3),
                                      round(auc_random_forest_train,3),
                                      round(pvalue_accuracy_random_forest_train,10^20))
accuracy_random_forest_test <- result_random_forest_test$overall['Accuracy']
lower_accuracy_random_forest_test <- result_random_forest_test$overall['AccuracyLower']
upper_accuracy_random_forest_test <- result_random_forest_test$overall['AccuracyUpper']
kappa_accuracy_random_forest_test <- result_random_forest_test$overall['Kappa']
pvalue_accuracy_random_forest_test <- result_random_forest_test$overall['AccuracyPValue']
auc_random_forest_test <- curve_roc_random_forest_test$auc
vetor_model_test_random_forest <- c("Random Forest - Sample test",
                                     round(accuracy_random_forest_test,3), 
                                     round(lower_accuracy_random_forest_test,3),
                                     round(upper_accuracy_random_forest_test,3),
                                     round(kappa_accuracy_random_forest_test,3),
                                     round(auc_random_forest_test,3),
                                     round(pvalue_accuracy_random_forest_test,10^20))
compare_models <- rbind(vetor_model_train_random_forest,
                        vetor_model_test_random_forest)
rownames(compare_models) <- c("Random Forest - Sample train", 
                              "Random Forest - Sample test")

colnames(compare_models) <- c("Model",
                               "Accuracy", 
                               "Lower confidence interval - Accuracy",
                               "Upper confidence interval - Accuracy",
                               "kappa",
                               "index-auc",
                               "p-Value")
compare_models <- as.data.frame(compare_models)
## Final model data train x data test (vrification of overfitting) ##
View(compare_models)
















