data <- read.csv("kidney_data.csv", stringsAsFactors = FALSE)
data = data[-1]

data$is_patient = factor(data$is_patient, 
                            levels = c(1, 2),
                            labels = c('yes', 'no'))

data$gender = as.factor(data$gender)
data$gender = factor(data$gender,
                        levels = c("Female", "Male"),
                        labels = c(0, 1))

fit_glm <- glm(is_patient ~., data, family = binomial(link = 'logit'))
#generate summary
summary(fit_glm)

#train and test sets
train = data[1:409, ]
test = data[410:583, ]

#descriptive statistics with skimr
library(skimr)
skimmed = skim_to_wide(data)
skimmed

#scaling age feature which is normally distributed
x[1] = scale(x[1])

#log transforming the numerical features
x[c(-1,-7,-8)] = 5 + x[c(-1,-7,-8)] - min(x[c(-1,-7,-8)]) #transforming right skewed features
x[c(-1,-7,-8)] = log10(x[c(-1,-7,-8)])



x[c(7,8)] = 5 + x[c(7,8)] - min(x[c(7,8)])
x[c(7,8)] = log10(x[c(7,8)])

x[1] = scale(x[1])

#dummy variabling
dummy = dummyVars(is_patient ~., data = data)
new_data = as.data.frame(predict(dummy, newdata = data))

#scaling age feature which is normally distributed
new_data[1] = scale(new_data[1])

#log10 transforming all other features, both left and right skewed
new_data[-1] = 5 + new_data[-1] - min(new_data[-1])
new_data[-1] = log10(new_data[-1])

new_data$is_patient = data$is_patient

#train and test data sets
x = new_data[1:11]
y = new_data[12]

random_sample = createDataPartition(new_data$is_patient, p = 0.70, list = FALSE)
new_data_train = x[random_sample, ]
new_data_train_labels = y[random_sample]
new_data_test = x[-random_sample, ]
new_data_test_labels = y[-random_sample]

#build model
kidney5_pred <- knn(new_data_train, new_data_test, new_data_train_labels, 26)

CrossTable(new_data_test_labels, kidney5_pred)
Kappa(table(new_data_test_labels, kidney5_pred))

trC = trainControl(method = "cv", number = 25)
fit = train(is_patient ~., 
            method = "knn", 
            tuneGrid = expand.grid(k = 10:30),
            trControl = trC,
            metric = "Accuracy",
            data = new_data)
fit

#new trial
#scaled age, sgot, sgpt
#log10'd tot_bilirubin, direct_bilirubin, tot_proteins, albumin, ag_ratio, alkphos

newD$is_patient = data[11]

View(newD)
y = data[11]

random_sample = createDataPartition(data$is_patient, p = 0.75, list = FALSE)
newD_train = newD[random_sample, ]
newD_train_labels = y[random_sample]
newD_test = newD[-random_sample, ]
newD_test_labels = y[-random_sample]

#build model
newK_pred <- knn(newD_train, newD_test, newD_train_labels, 26)

CrossTable(newD_test_labels, newK_pred)
Kappa(table(newD_test_labels, newK_pred))

trC = trainControl(method = "cv", number = 25)
fit = train(is_patient ~., 
            method = "knn", 
            tuneGrid = expand.grid(k = 10:30),
            trControl = trC,
            metric = "Accuracy",
            data = newD)
fit








