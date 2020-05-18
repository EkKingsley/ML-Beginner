#logistic regression on Breast Cancer dataset
data = read.csv('bcw.csv')

str(data)
data = data[-1]
data$diagnosis = factor(data$diagnosis, levels = c('B', 'M'), labels = c(0, 1))
data[2:31] = scale(data[2:31])

library(caret)
data_partition = createDataPartition(data$diagnosis, p = 0.75, list = FALSE)

#splitting data into test and training sets
ndata_train = data[data_partition, 1:31]
data_train_label = data[data_partition, 1]
data_train_features = data[, -1]

ndata_test = data[-data_partition, ]
data_test_label = data[-data_partition, 1]

#finding correlations and removing features that are past the correlation cutoff of 0.75
data_correlations = cor(data_train_features)
findCorrelation(data_correlations, cutoff = 0.75)

#building the model
bc_model = glm(diagnosis ~., data = ndata_train[c(-8, -9, -7, -29, -28, -24, -22, -4, -27, -25, -3, -14, -12, -19, -17, -15, -18, -6, -11, -23)], family = binomial('logit'))
summary(bc_model)

#Assessing the model
train_predictions = predict(bc_model, newdata = ndata_train, type = "response")
train_class_predictions = as.numeric(train_predictions > 0.5)
mean(train_class_predictions == ndata_train$diagnosis)

test_predictions = predict(bc_model, newdata = ndata_test, type = "response")
test_class_predictions = as.numeric(test_predictions > 0.5)
mean(test_class_predictions == ndata_test$diagnosis)