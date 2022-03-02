

featuresFile <- "../dataset/features.txt"

## Save features names
features <- read.table(featuresFile)[["V2"]]

## Factor labels
activities <- c("Walking","Walking upstairs","Walking downstairs","sitting","standing","lying")


########## TRAINING SET ################

trainFile <- "../dataset/train/X_train.txt"
labelsFile <- "../dataset/train/y_train.txt"

ex <- t(read.table(trainFile))
row.names(ex) <- c(1:nrow(ex))
labels <- read.table(labelsFile)
labels <- labels[["V1"]]


## SAMPLES SEPARATION 
walking <- which(labels == 1)
walking_up <- which(labels == 2)
walking_down <- which(labels == 3)
sitting <- which(labels == 4)
standing <- which(labels == 5)
laying <- which(labels == 6)


########## TESTING SET ################

testFile <- "../dataset/test/X_test.txt"
testLabelsFile <- "../dataset/test/y_test.txt"

exTest <- t(read.table(testFile))
row.names(exTest) <- c(1:nrow(exTest))
testLabels <- read.table(testLabelsFile)
testLabels <- testLabels[["V1"]]

testWalking <- which(testLabels == 1)
testWalking_up <- which(testLabels == 2)
testWalking_down <- which(testLabels == 3)
testSitting <- which(testLabels == 4)
testStanding <- which(testLabels == 5)
testLaying <- which(testLabels == 6)

