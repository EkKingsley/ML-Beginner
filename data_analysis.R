#loading text files of a reasonable size
library('hflights')

#writing an 18.5MBtext file to disk using hglights package
#which includes some data on all flights departing from Houston in 2011
write.csv(hflights, 'hflights.csv', row.names = F)
str(hflights)

#loading data files larer than memoery with sqldf package
library(sqldf)
read.csv.sql('hflights.csv')

#loading a subset of files
df = read.csv.sql('hflights.csv', sql = "select * from file where Dest = '\"BNA\"'")

#loading data from databases : MySQL
library(RMySQL)

#connection varibale and refoer to the mysql connection as con
con <- dbConnect(dbDriver('MySQL'), user = 'user', password = 'password', dbname = 'hflights_db')

#using dbWriteTable() to write the hflights dataframe to the con
dbWriteTable(con, name = 'hflights', value = hflights) # imports df to mysql

#using dbReadTable() to read the data from the table in mysql
dbReadTable(con, 'hflights')

#or can do so with a direct sql command passed to dbGetQuery from same DBI package
dbGetQuery(con, 'select * from hflights')

#Using a graphical user interface to connect to the databases
#using the dbConnect package:
library(dbConnect)
DatabaseConnect()


#filtering and summarizing data
    #filter rows and columns in data framse
    #summarize and aggregate data
    #improve performance of such tasks with the dplyr and data.table packages 
#dropping needless dat ausing sql-like approach of the sqldf pkg
sqldf("SELECT * FROM mtcars WHERE am=1 AND vs=1")#row.names = False by default

#dropping needless data using subset() tool in base R
subset(mtcars, am == 1 & vs == 1)

#removing some columns
subset(mtcars, am ==1 & vs == 1, select = hp:wt)#select cols hp to wt

#using a faster method from dplyr pkg
library(dplyr)
filter(hflights, Dest == 'BNA')

#removing specific cols
str(select(filter(hflights, Dest == 'BNA'), DepTime:ArrTime))

#rownames are not preserved in dplyr, so if 
#required, copy names to explicit var b4 passing 
#them to dplyr or directly to data.table as follows:
mtcars$rownames <- rownames(mtcars) # save rownames of the data
select(filter(mtcars, hp>300), c(rownames, hp))

#Aggregation, summarizing data, splitting data into subsets
#by a grouping variable, then computing summary stats fr them separately
#most basic way is to call the aggregate() is to pass the numeric vector to be aggregated,
#and a factor variable to define the splits fr de function passed in the FUN argument applied.
#Now,lets see the averae ratio of diverted flights on each weekday:
aggregate(hflights$Diverted, by = list(hflights$DayOfWeek), FUN = mean)

#using the formula notation in aggregate to see column names
aggregate(Diverted ~ DayOfWeek, data = hflights, FUN = mean)


#Restructuring Data
    #Transposing matrices
    #Splitting, applying and joining data
    #Computing margins of tables
    #Merging data frames
    #Casting and melting data

#Transposing matrices, using t()
m = matrix(1:9, 3)
t(m)

#filtering data by matching string, using dplyr pkg
#dplyr pkg provides funcs to select some columns of the data
#based on column name patterns.
#for example, keeping only the variables ending with the string, 'delay':
library(dplyr)
library(hflights)
str(select(hflights, ends_with("delay")))

#can select cols of data using starts_with('...')
str(select(hflights, starts_with("Div")))

#can use contains(), looking for substrings
str(select(hflights, contains("T", ignore.case = F)))

#Rearranging data
#sorting the hflights data, based on the actual elapsed
#time for each of the quarter million flights:
str(arrange(hflights, ActualElapsedTime))
#or
hflights %>% arrange(ActualElapsedTime) %>% str

hflights %>%
    arrange(ActualElapsedTime) %>%
    select(ActualElapsedTime, Dest) %>%
    subset(Dest != 'AUS') %>%
    head %>%
    str
