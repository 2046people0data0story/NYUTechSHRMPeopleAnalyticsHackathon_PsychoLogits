#load the data - median/mean, median, and KNN imputed training sets
#change physician and verysat from integer (interval) to factor (nominal) class

data1 <- read.csv("C:/Users/Vin/Desktop/R/hackathon/MEDIAN&MEAN_training.csv")
str(data1)
data1$physician <- as.factor(data1$physician)
data1$verysat <- as.factor(data1$verysat)

data2 <- read.csv("C:/Users/Vin/Desktop/R/hackathon/MEDIAN_imputed_training.csv")
str(data2)
data2$Specialty_num <- as.factor(data2$Specialty_num)
data2$Group_num <- as.factor(data2$Group_num)
data2$physician <- as.factor(data2$physician)
data2$verysat <- as.factor(data2$verysat)

data3 <- read.csv("C:/Users/Vin/Desktop/R/hackathon/KNN_imputed_training.csv")
str(data3)
data3$physician <- as.factor(data3$physician)
data3$verysat <- as.factor(data3$verysat)


# corr matrix // get rid of the "#" if you want to run the code

#library(corrplot)
#source("http://www.sthda.com/upload/rquery_cormat.r")
#rquery.cormat(apply(data1[,3:38], 2, as.numeric))

#somecor <- data1[,c("q2","q6","q57","q20","q14","q17","q19","q16","q18","q54","q53","q58","q34","q35")]
#rquery.cormat(apply(somecor, 2, as.numeric))

#cor(data1[,"q17"], data1[,"q14"])


#split datasets into training/test using an 75/25 split

library(caret)

set.seed(100)

train1ind <- createDataPartition(data1$q58, p = .75, list = FALSE, times = 1)

train1 <- data1[train1ind,]
test1 <- data1[-train1ind,]

train2ind <- createDataPartition(data2$q58, p = .75, list = FALSE, times = 1)

train2 <- data2[train2ind,]
test2 <- data2[-train2ind,]

train3ind <- createDataPartition(data3$q58, p = .75, list = FALSE, times = 1)

train3 <- data3[train3ind,]
test3 <- data3[-train3ind,]


#linear regression

reg1 <- lm(q58 ~., data = train1[,c(-1,-2)])
summary(reg1)

reg2 <- lm(q58 ~., data = train2[,c(-1,-2)])
summary(reg2)

reg3 <- lm(q58 ~., data = train3[,c(-1,-2)])
summary(reg3)

regp1 <- predict(reg1, newdata = test1)
compare1 <- data.frame(cbind(actual=test1$q58, predicted=regp1))
accuracy1 <- cor(compare1$actual,compare1$predicted)
accuracy1 #.5959

regp2 <- predict(reg2, newdata = test2)
compare2 <- data.frame(cbind(actual=test2$q58, predicted=regp2))
accuracy2 <- cor(compare2$actual,compare2$predicted)
accuracy2 #.6483

regp3 <- predict(reg3, newdata = test3)
compare3 <- data.frame(cbind(actual=test3$q58, predicted=regp3))
accuracy3 <- cor(compare3$actual,compare3$predicted)
accuracy3 #.6124

#randomforest

library(randomForest)

rf1 <- randomForest(as.factor(q58)~., data = train1, ntree= 1200, mtry = 12, proximity = TRUE
                   , importance = TRUE)

rf2 <- randomForest(as.factor(q58)~., data = train2, ntree= 1200, mtry = 12, proximity = TRUE
                    , importance = TRUE)

rf3 <- randomForest(as.factor(q58)~., data = train3, ntree= 1200, mtry = 12, proximity = TRUE
                    , importance = TRUE)
#rf1
#plot(rf1)
#varImpPlot(rf)

#tuneRF(test1[,-38], test1$q58, plot = TRUE, ntreetry = 1200, trace = TRUE, stepfactor = .5, improve = .01)

library(pROC)

rfp1 <- predict(rf1, newdata = test1)
table(rfp1,test1$q58)
multiclass.roc(test1$q58, predict(rf1, test1, type = 'prob')) #AUC .7295
confusionMatrix(rfp1,as.factor(test1$q58)) #Accuracy : 0.5948, Kappa : 0.3769

rfp2 <- predict(rf2, newdata = test2)
table(rfp2,test2$q58)
multiclass.roc(test2$q58, predict(rf2, test2, type = 'prob')) #AUC .7796
confusionMatrix(rfp2,as.factor(test2$q58)) #Accuracy : 0.5821, Kappa : 0.3637

rfp3 <- predict(rf3, newdata = test3)
table(rfp3,test3$q58)
multiclass.roc(test3$q58, predict(rf3, test3, type = 'prob')) #AUC .7904
confusionMatrix(rfp3,as.factor(test3$q58)) #Accuracy : 0.6604, Kappa : 0.4798 

#balance data using SMOTE

library(tidyverse)

train1$q58 <- as.integer(train1$q58)
train1m <- mutate(train1,
       q58T = ifelse(train1$q58 < 4,0,1))
table(train1$q58)
table(train1$q58T)

library(DMwR)

train1m$q58T <- as.factor(train1m$q58T)
train1m$q58 <- as.factor(train1m$q58)
train1b <- SMOTE(q58T ~ ., train1m, perc.over = 600)
table(train1b$q58T)
table(train1b$q58)
train1b$q58 <- as.integer(train1b$q58)

rf1b <- randomForest(as.factor(q58) ~ ., data = train1b[,-39], ntree= 1200, mtry = 12, proximity = TRUE
                   , importance = TRUE)

#tuneRF(test1[,c(-38,-39)], test1$q58, plot = TRUE, ntreetry = 550, trace = TRUE, stepfactor = .5, improve = .01)

rfp1b <- predict(rf1b, newdata = test1)
table(rfp1b,test1$q58)
multiclass.roc(test1$q58, predict(rf1b, test1, type = 'prob')) # AUC 0.7349
confusionMatrix(rfp1b,as.factor(test1$q58)) # Accuracy : 0.5725, Kappa : 0.3444

#try out SMOTE & Random Forest with KNN dataset

train3$q58 <- as.integer(train3$q58)
train3m <- mutate(train3,
                  q58T = ifelse(train3$q58 < 4,0,1))
table(train3$q58)
table(train3$q58T)

train3m$q58T <- as.factor(train3m$q58T)
train3m$q58 <- as.factor(train3m$q58)
train3b <- SMOTE(q58T ~ ., train3m, perc.over = 600)
table(train3b$q58T)
table(train3b$q58)
train3b$q58 <- as.integer(train3b$q58)

rf2b <- randomForest(as.factor(q58) ~ ., data = train3b[,-39], ntree= 1200, mtry = 12, proximity = TRUE
                     , importance = TRUE)

#tuneRF(test1[,c(-38,-39)], test1$q58, plot = TRUE, ntreetry = 550, trace = TRUE, stepfactor = .5, improve = .01)

rfp2b <- predict(rf2b, newdata = test3)
table(rfp2b,test3$q58)
multiclass.roc(test3$q58, predict(rf2b, test3, type = 'prob')) #AUC 0.7837
confusionMatrix(rfp2b,as.factor(test3$q58)) # Accuracy : 0.6157, Kappa : 0.4223 

#svm

library(e1071)

svm1 <- svm(q58~., data = train1, kernel = "linear", gamma =.5, cost =10, type = "C-classification")

svm2 <- svm(q58~., data = train2, kernel = "linear", gamma =.5, cost =10, type = "C-classification")

svm3 <- svm(q58~., data = train3, kernel = "linear", gamma =.5, cost =10, type = "C-classification")

#tune parameters of gamma and cost for best fit

#tune.out <- tune(svm, q58~., data = train1, kernel ="linear",
                 #ranges = list(cost = c(0.1,1,10,100,1000), gamma = c(0.5,1,2,3,4)))

#summary(tune.out)

svmp1 <- predict(svm1, newdata = test1)
table(svmp1,test1$q58)
confusionMatrix(svmp1,as.factor(test1$q58)) # Accuracy : 0.5465, Kappa : 0.3163

svmp2 <- predict(svm2, newdata = test2)
table(svmp2,test2$q58)
confusionMatrix(svmp2,as.factor(test2$q58)) # Accuracy : 0.5112, Kappa : 0.2696

svmp3 <- predict(svm3, newdata = test3)
table(svmp3,test3$q58)
confusionMatrix(svmp3,as.factor(test3$q58)) # Accuracy : 0.6231, Kappa : 0.4225

#j48

library(RWeka)
library(caTools)

train1$q58 <- as.factor(train1$q58)
train2$q58 <- as.factor(train2$q58)
train3$q58 <- as.factor(train3$q58)

j1 <- J48(q58~., data = train1)

j2 <- J48(q58~., data = train2)

j3 <- J48(q58~., data = train3)


jp1 <- predict(j1, test1)
table(jp1,test1$q58)
confusionMatrix(jp1,as.factor(test1$q58)) # Accuracy : 0.5651, Kappa : 0.3402

jp2 <- predict(j2, test2)
table(jp2,test2$q58)
confusionMatrix(jp2,as.factor(test2$q58)) # Accuracy : 0.4851, Kappa : 0.2288

jp3 <- predict(j3, test3)
table(jp3,test3$q58)
confusionMatrix(jp3,as.factor(test3$q58)) # Accuracy : 0.5746, Kappa : 0.3453