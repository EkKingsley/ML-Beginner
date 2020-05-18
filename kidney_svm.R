data <- read.csv("kidney_data.csv", stringsAsFactors = FALSE)
data = data[-1]

data$is_patient = factor(data$is_patient, 
                         levels = c(1, 2),
                         labels = c('yes', 'no'))

data$gender = as.factor(data$gender)
data$gender = factor(data$gender,
                     levels = c("Female", "Male"),
                     labels = c(0, 1))

dummy = dummyVars(is_patient ~., data = data)
new_data = as.data.frame(predict(dummy, newdata = data))

#new trial
#scaled age, sgot, sgpt
#log10'd tot_bilirubin, direct_bilirubin, tot_proteins, albumin, ag_ratio, alkphos

#scaling age feature which is normally distributed
new_data[c(1,9,10)] = scale(new_data[c(1,9,10)])

#log transform sgpt and sgot
new_data[c(4,5)] = log(new_data[c(4,5)])

#log10 transforming all other features
new_data[c(-1,-9,-10,-2,-3,-4,-5)] = 5 + new_data[c(-1,-9,-10,-2,-3,-4,-5)] - min(new_data[c(-1,-9,-10,-2,-3,-4,-5)])
new_data[c(-1,-9,-10,-2,-3,-4,-5)] = log10(new_data[c(-1,-9,-10,-2,-3,-4,-5)])

new_data$is_patient = data$is_patient

#fitting kernel svm to training and test sets
training_set = new_data_train
training_set$is_patient = new_data_train_labels 

test_set = new_data_test
test_set$is_patient = new_data_test_labels

classifier = svm(formula = is_patient ~.,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'radial')
#predicting the test set results
y_pred = predict(classifier, newdata = test_set[-12])

#making the confusion matrix
cm = table(test_set[, 12], y_pred)
confusionMatrix(test_set[, 12], y_pred)

CrossTable(test_set[, 12], y_pred)
Kappa(table(test_set[, 12], y_pred))

#applying k-fold cross validation
folds = createFolds(training_set$is_patient, k=10)
cv = lapply(folds, function(x){
  #in the next 2 lines we will separate the training set into its 10 pieces
  
  training_fold = training_set[-x, ] # training fold = training set - its sub test fold
  
  test_fold = training_set[x, ] # here we descrive the test fold individually
  
  #now apply (train) the classifier on the training fold
  classifier3 = svm(formula = is_patient ~.,
                    data = training_fold,
                    type = 'C-classification',
                    kernel = 'radial')
  
  #next step in the loop,calculate the predictions and cm and we equate the accuracy
  #note we are training on the trainig_fold and testing its accuracy on the test_fold
  
  y_pred3 = predict(classifier3, newdata = test_fold[-12])
  cm = table(test_fold[, 12], y_pred3)
  accuracy = (cm[1,1] + cm[2,2]) / (cm[1,1] + cm[2,2] + cm[1,2] + cm[2,1])
  return(accuracy)
})

knitr::include_graphics("CV.png")
cv # displays all accuracy of each fold
accuracy = mean(as.numeric(cv))
accuracy #0.7117276


#train and test data sets
x = new_data[1:11]
y = new_data[12]

random_sample = createDataPartition(new_data$is_patient, p = 0.70, list = FALSE)
new_data_train = x[random_sample, ]
new_data_train_labels = y[random_sample]
new_data_test = x[-random_sample, ]
new_data_test_labels = y[-random_sample]

#load svm library e1071
library(e1071)

# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = is_patient ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)







