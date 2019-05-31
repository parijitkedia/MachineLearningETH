# strategy 1: bins of 5 years -> classification
# remove background (black) voxels


library(oro.nifti)
library(quantmod)
library(lubridate)
library(e1071)
library(graphics)
library(EBImage)
library(pixmap)
library(png)

setwd("Documents/Master/ETH/Machine Learning/Project")

###########################################################################

# creating 2d images by slicing 3d train images
nbTrain <- length(list.files("data/set_train"))
for(i in 1:nbTrain){       
        img <- readNIfTI(paste("data/set_train/train_",i,".nii",sep=""))
        png(paste("data/img_2d/img",i,sep=""))
        orthographic(img)
        dev.off()
}

###########################################################################

# reading in 2d images, convert to grayscale, resize, creare feature vector
# and build matrix with feature vectors of all training data

feature_train <- c()
for(i in 1:nbTrain){
        img <- readImage(paste("data/img_2d/img",i,sep=""), type="png")
        img <- channel(img, "gray")
        # resize to 1/8 of original dimension
        img <- resize(img, dim(img)[1]/8)
        mat <- as.array(img)
        # matrix to feature vector
        vec <- c(mat)
        # feature vectors of all training data
        feature_train <<- rbind(feature_train, vec)
}

###########################################################################
###########################################################################
# Building the models
###########################################################################
###########################################################################

# dependant variable age
y <- read.csv("data/targets.csv", header=F)[,]
# feature 1 engineering: sum total pixel intensity
feature1 <- apply(feature_train,1,sum)
# feature 2 engineering: sum number of pixels with intensity < 0.1
feature2 <- apply(feature_train,1,function(x) sum(x<0.1))
#feature 3 engineering: 
feature3 <- apply(feature_train,1,mean)
# model 1: linear regression model
fit1 <- lm(y ~ feature1+feature2)
summary(fit1)
# model 2: polynomial regression model
fit2 <- lm(y ~ feature1+feature2+I(feature2^2))
summary(fit2)
# model 3: polynomial regression model inc f3
fit3 <- lm(y ~ feature1+feature2+I(feature2^2)+feature3)
summary(fit3)
compare <- rbind(y, round(predict(fit)), y==round(predict(fit)))
sum(compare[3,])

###########################################################################
###########################################################################
# Predicting Test Data
###########################################################################
###########################################################################

# creating 2d images by slicing 3d test images
nbTest <- length(list.files("data/set_test"))
for(i in 1:nbTest){       
        img <- readNIfTI(paste("data/set_test/test_",i,".nii",sep=""))
        png(paste("data/img_2d_test/img",i,sep=""))
        orthographic(img)
        dev.off()
}

###########################################################################

# reading in 2d images, convert to grayscale, resize, creare feature vector
# and build matrix with feature vectors of all test data

feature_test <- c()
for(i in 1:nbTest){
        img <- readImage(paste("data/img_2d_test/img",i,sep=""), type="png")
        img <- channel(img, "gray")
        # resize to 1/8 of original dimension
        img <- resize(img, dim(img)[1]/8)
        mat <- as.array(img)
        # matrix to feature vector
        vec <- c(mat)
        # feature vectors of all training data
        feature_test <<- rbind(feature_test, vec)
}

###########################################################################
# prediction of test data with model 2
feature1 <- apply(feature_test,1,sum)
# feature 2 engineering: sum number of pixels with intensity < 0.1
feature2 <- apply(feature_test,1,function(x) sum(x<0.1))
test <- data.frame(cbind(feature1, feature2),row.names=NULL)
pred <- round(predict(fit2, newdata=test))
submit <- cbind(1:nbTest, pred)
colnames(submit) <- c("ID", "Prediction")
write.csv(submit, "submit.csv", row.names=F)
# Your submission scored 213.31429