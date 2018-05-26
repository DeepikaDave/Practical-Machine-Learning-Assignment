#The goal of your project is to predict the manner in which they did the exercise. 
#This is the "classe" variable in the training set. 

install.packages("caret")
install.packages("dplyr")
install.packages("data.table")
library(caret)
library(dplyr)
library(stringr)

#First we need to tell read.csv to treat empty columns as NA by using na.strings = c("", "NA")

modeldata=read.csv("C:/Users/Vaio/Desktop/Coursera-John Hopkins/8-Practical Machine Learning/Final course project-8/pml-training.csv",na.strings = c(""," ",NA,"NA"))
validationdata=read.csv("C:/Users/Vaio/Desktop/Coursera-John Hopkins/8-Practical Machine Learning/Final course project-8/pml-training-2.csv",na.strings = c(""," ",NA,"NA"))


View(modeldata)
dim(modeldata)
head(modeldata)
View(validationdata)
dim(validationdata)
head(validationdata)


#in this trainingdata dataset there are some columns whose all values are  "NA" 
#removing columns completely filled with na
d=colSums(is.na(modeldata))
sum(print(d==0))
sum(print(d!=0))

#d=0 variables can be considered and rest others to be discarded
#select only those whose sum of colum values having NA is zero

newmodeldata=modeldata[,d==0]
dim(newmodeldata)
View(newmodeldata)
newmodeldata$classe=as.factor(newmodeldata$classe)

#after removing columns which are compltely filled with na values we are left with 60 variables


#similarly in this testdata set there are some columns whose all values are  "NA" 
#removing columns completely filled with na
e=colSums(is.na(validationdata))
sum(print(e==0))
finalvalidationdata=validationdata[,e==0]
dim(finalvalidationdata)
#after removing columns which are compltely filled with na values we are left with 60 variables

dim(newmodeldata)
dim(finalvalidationdata)

#creating data partition in the training dataset
intrain=createDataPartition(y=newmodeldata$classe,p=0.70,list=FALSE)
trainingdata=newmodeldata[intrain,]
testingdata=newmodeldata[-intrain,]
x=trainingdata[,-c(60,1,2,3,4,5,6,7)]#all independent variables
y=trainingdata[,60]#depemdent variable "classe"



#now making the type of all the variables of training and validation  data same
sum(sapply(trainingdata,class)==sapply(testingdata,class)) 
sum(sapply(trainingdata,class)==sapply(finalvalidationdata,class)) #3 values not matching

which((sapply(trainingdata,class)!=sapply(finalvalidationdata,class)))
which((sapply(testingdata,class)!=sapply(finalvalidationdata,class)))

trainingdata$magnet_dumbbell_z=as.numeric(trainingdata$magnet_dumbbell_z)
trainingdata$magnet_forearm_y=as.integer(trainingdata$magnet_forearm_y)
trainingdata$magnet_forearm_z=as.integer(trainingdata$magnet_forearm_z)

testingdata$magnet_dumbbell_z=as.numeric(testingdata$magnet_dumbbell_z)
testingdata$magnet_forearm_y=as.integer(testingdata$magnet_forearm_y)
testingdata$magnet_forearm_z=as.integer(testingdata$magnet_forearm_z)

finalvalidationdata$magnet_dumbbell_z=as.numeric(finalvalidationdata$magnet_dumbbell_z)
finalvalidationdata$magnet_forearm_y=as.integer(finalvalidationdata$magnet_forearm_y)
finalvalidationdata$magnet_forearm_z=as.integer(finalvalidationdata$magnet_forearm_z)

finalvalidationdata$classe=as.factor(finalvalidationdata$classe)

which((sapply(trainingdata,class)!=sapply(finalvalidationdata,class)))
which((sapply(testingdata,class)!=sapply(finalvalidationdata,class)))

class(trainingdata$magnet_dumbbell_z)
class(testingdata$magnet_dumbbell_z)
class(finalvalidationdata$magnet_dumbbell_z)

class(trainingdata$magnet_forearm_y)
class(testingdata$magnet_forearm_y)
class(finalvalidationdata$magnet_forearm_y)

class(trainingdata$magnet_forearm_z)
class(testingdata$magnet_forearm_z)
class(finalvalidationdata$magnet_forearm_z)

sum(sapply(trainingdata,class)==sapply(testingdata,class)) 
sum(sapply(trainingdata,class)==sapply(finalvalidationdata,class))



#applying model random forest as dependent variable is char type and independent variables are numeric and char types

#configure parallel processing
library(parallel)
library(doParallel)


#configure train control object
tcontrol <- trainControl(method = "cv",number = 5,allowParallel = TRUE)#the number that specifies the quantity of folds for k-fold cross-validation, and allowParallel which tells caret to use the cluster that we've registered in the previous step.

#develop training model
set.seed(1001)
modelrf=train(x,y,data=trainingdata,trControl = tcontrol,method = "rf")#train the model, using the trainControl() object that we just created.

#De-register parallel processing cluster
#After processing the data, we explicitly shut down the cluster by calling the stopCluster() and registerDoSEQ() functions. The registerDoSEQ() function is required to force R to return to single threaded processing.

stopCluster(cluster)
registerDoSEQ() 

#prediction
modelrf
p1=predict(modelrf,testingdata)
confusionMatrix(testingdata$classe,p1)


#finalprediction on 20 test cases
finalvalidationdata1=finalvalidationdata[,-c(60,1,2,3,4,5,6,7)]
p2=predict(modelrf,finalvalidationdata1)
p2


output:
[1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
