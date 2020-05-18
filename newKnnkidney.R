data <- read.csv("kidney_data.csv", stringsAsFactors = FALSE)
data = data[-1]

data$gender = as.factor(data$gender)
data$gender = factor(data$gender,
                     levels = c("Female", "Male"),
                     labels = c(0, 1))

library(caret)

dummy = dummyVars(is_patient ~., data = data)
new_data = as.data.frame(predict(dummy, newdata = data))

#scaling age feature which is normally distributed
new_data[c(1,9,10)] = scale(new_data[c(1,9,10)])

#log transform sgpt and sgot
new_data[c(4,5)] = log(new_data[c(4,5)])

#log10 transforming all other features
new_data[c(-1,-9,-10,-2,-3)] = 5 + new_data[c(-1,-9,-10,-2,-3)] - min(new_data[c(-1,-9,-10,-2,-3)])
new_data[c(-1,-9,-10,-2,-3)] = log10(new_data[c(-1,-9,-10,-2,-3)])

new_data$is_patient <- data$is_patient

#train and test data sets, using age, tot and directs bilirubins
x = new_data[c(1,4,5)]
y = new_data[12]

random_sample = createDataPartition(new_data$is_patient, p = 0.70, list = FALSE)
new_data_train = x[random_sample, ]
new_data_train_labels = y[random_sample]
new_data_test = x[-random_sample, ]
new_data_test_labels = y[-random_sample]

library(class)
library(gmodels)
#build model
kidney_pred <- knn(new_data_train, new_data_test, new_data_train_labels, 26)

CrossTable(new_data_test_labels, kidney_pred)
Kappa(table(new_data_test_labels, kidney_pred))

#k-fold cross validation
trC = trainControl(method = "cv", number = 25)
fit = train(is_patient ~., 
            method = "knn", 
            tuneGrid = expand.grid(k = 10:30),
            trControl = trC,
            metric = "Accuracy",
            data = new_data)
fit
