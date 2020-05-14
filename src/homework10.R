#==========================================
##利用caret建模
##包括数据加载，数据预处理，挑选特征值， 拆分数据集，建模以及模型评估
##==========================================

rm(list = ls())
library(caret)


#加载数据集
data <- read.csv("data/npcl11.csv")
str(data)
head(data)

#数据预处理
library(Hmisc, quietly=TRUE)
contents(data)
summary(data)

library(fBasics, quietly=TRUE)
skewness(data, na.rm=TRUE)

#向数据集内插入缺失值
library(skimr)
skimmed <- skim_to_wide(data)
skimmed[, 2:12]

preProcess_missingdata_model <- preProcess(data[,2:12], 
                                           method='knnImpute')
preProcess_missingdata_model

library(RANN)
data_NA <- predict(preProcess_missingdata_model, 
                   newdata = data)
anyNA(data_NA)

#独热编码
dummies_model <- dummyVars(loss_rate ~ ., 
                           data= data_NA)
data_NA_dum_mat <- predict(dummies_model, 
                           newdata = data_NA)
data_NA_dum <- data.frame(data_NA_dum_mat)
loss_rate <- data_NA$loss_rate
data_clean <- cbind(loss_rate,data_NA_dum)
head(data_clean)


#数据转换
library(tidyverse)
data_class <- data [,-1] %>% 
  mutate(loss_rate = 
           case_when(loss_rate >= 0.4 ~ 'serious',
                     loss_rate < 0.4 ~ 'normal')) %>% 
  rename(loss_degree=loss_rate)
head(data_class)

#保存预处理后的数据为csv格式到本地
write.csv(data_class, file = "data/data_class.csv")



##挑选特征值

#可视化特征值的重要性
x = as.matrix(data_class[, 1:11])
y = as.factor(data_class$loss_degree)

featurePlot(x, y, plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation = "free"), 
                          y = list(relation="free")))

featurePlot(x, y, plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

##估算特征值重要性的不同方法 (caret)
#自动选择最具预测性的特征值子集
options(warn=-1)
set.seed(1234)

subsets <- c(1:5, 8, 11)
ctrl <- rfeControl(functions = rfFuncs, #随机森林算法
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)
ImProfile <- rfe(x, y, 
                 sizes=subsets, 
                 rfeControl=ctrl)#运行随森林算法算法
print(ImProfile)
predictors(ImProfile)#列出选中的特征值
plot(ImProfile, type=c("g", "o"))# 利用结果作图

# 检索删除冗余的特征值
corr_Matrix <- cor(data_class[,1:11])
print(corr_Matrix)
highlyCorr <- findCorrelation(corr_Matrix, cutoff=0.5)
print(highlyCorr)

#按重要性对特征值排序
control <- trainControl(method="repeatedcv", 
                        number=10, repeats=3)# cross-validation
model <- train(loss_degree~., 
               data=data_class, 
               method="rf", 
               preProcess="scale", 
               trControl=control)# train the model
importance <- varImp(model, scale=FALSE)
print(importance)# 总结重要的特征值
plot(importance)# 重要性作图


##训练和调整模型

#拆分数据集
set.seed(1234)
train_idx <- createDataPartition(data_class$loss_degree, p=0.75, list=FALSE)
training <- data_class[train_idx,]
test <- data_class[-train_idx,]

##构建随机森林模型并评估其性能
#构建模型
set.seed(1234)
rf_fit <- train(as.factor(loss_degree) ~ IA + PA + CA + Q + G, 
                data = training, 
                method = "rf")
rf_fit
plot(rf_fit)

#性能评估
rf_pred <- predict(rf_fit, test)
rf_pred
confusionMatrix(reference = as.factor(test$loss_degree), 
                data = rf_pred,
                mode = "everything")

#设置 tuneLength 或者 tuneGrid 来调整模型预测能力

ctrl <- trainControl(
  method = 'cv',                  
  number = 5,                     
  savePredictions = 'final',
  classProbs = T,                  
  summaryFunction=twoClassSummary) 

rf_fit <- train(as.factor(loss_degree) ~., 
                data = training, 
                method = "rf", 
                tuneLength = 5,
                trControl = ctrl,
                verbose = FALSE
)

#评估随机森林模型预测能力
rf_pred <- predict(rf_fit, test)
rf_pred
confusionMatrix(reference = as.factor(test$loss_degree), 
                data = rf_pred,
                mode = "everything")

library(MLeval)
x <- evalm(rf_fit)
x$roc
x$stdres




#建模并比较不同模型
#设置训练的控制条件
ctrl <- trainControl(method = "repeatedcv",   
                     number = 5,	
                     summaryFunction=twoClassSummary,	
                     classProbs=TRUE,
                     allowParallel = TRUE)



##同时训练多种模型
set.seed(1234)  
rpart_fit = train(as.factor(loss_degree) ~.,
                  data=training, 
                  method='rpart', 
                  tuneLength=15, 
                  trControl = ctrl)

svm_fit = train(as.factor(loss_degree) ~ .,
                data=training, 
                method='svmRadial', 
                tuneLength=15, 
                trControl = ctrl)

#比较不同模型的性能
models_compare <- resamples(list(rpart = rpart_fit, SVM = svm_fit))
str(models_compare)
summary(models_compare)
scales <- list(x=list(relation="free"), y=list(relation="free"))
library(lattice)
bwplot(models_compare, scales=scales)


# 堆叠算法
library(caretEnsemble)

ctrl <- trainControl(method="repeatedcv", 
                     number=10, 
                     repeats=3,
                     savePredictions=TRUE, 
                     classProbs=TRUE)

algorithmList <- c('rf', 'rpart', 'svmRadial')

set.seed(1234)
models <- caretList(as.factor(loss_degree) ~ .,
                    data=training, 
                    trControl=ctrl, 
                    methodList=algorithmList) 
results <- resamples(models)
str(results)
summary(results)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

