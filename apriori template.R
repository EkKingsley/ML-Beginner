#Market Basket Analysis using Apriori Algorithm

#Step 1 - Collecting data
#Dataset is groceries.csv
#Containing 9,835 transactions

#Step 2 - Exploring and preparing the data
#using the read.transactions() function in the association rules
#(arules) package, we read in the groceries data file.
#cannot use the normal data frame structure for processing transactional data
#need to create a sparse matrix data structure from the transactional data
# which is suitable for the algorithm
library(arules)
groceries = read.transactions("groceries.csv", sep = ",")

#check out basic info about the groceries dataset
summary(groceries)

#can check out some of the contents of the sparse matrix
#using the inspect() fxn in combination with vector opertors
#the first five transactons cn be viewed as follows:
inspect(groceries[1:5])

#can visualize item support - item freq. plots
#use itemFrequencyPlot, support param to require items to appear
#in a specified minimum proportion of transactions
itemFrequencyPlot(groceries, support = 0.1)
#can limit plot a specific number of items using the topN param
itemFrequencyPlot(groceries, topN = 10)

#can plot the sparse matrix using the image functin and can also specify the items using vector operators
image(groceries[1:100])

#Step 3 - training a model on the data
#using the apriori() in the arules pkg
groceryrules = apriori(groceries, parameter = list(
  support = 0.006,
  confidence = 0.25,
  minlen = 2
))
groceryrules

#step 4 - evaluating the model
summary(groceryrules)

#inspect the grocery rules
inspect(groceryrules[1:5])

#step 5 - improving the model performance
#sort the set of rules with sort() to display 
#those with highest or lowest values of quality measure comes first
inspect(sort(groceryrules, by="lift")[1:5])

#Taking subsets of the association rules
berryrules = subset(groceryrules, items %in% "berries")
inspect(berryrules)

#can save the association rules to a file or dataframe
#to file using write():
write(groceryrules, file = "groceryRules.csv", sep = ",", quote = TRUE, row.names = FALSE)

#to data frame using as():
groceryrules.df = as(groceryrules, "data.frame")