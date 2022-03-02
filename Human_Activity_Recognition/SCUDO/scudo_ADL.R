install.packages("igraph")
BiocManager::install("rScudo")
BiocManager::install("RCy3")


########## DATA PREPARATION #################

# Set your working directory
setwd(getSrcDirectory()[1])

# Run script dedicated to data preparation
source("data_preparation.R")

## TEST 1: All activities
inds <- c(sample(walking,100),
          sample(walking_up,100),
          sample(walking_down,100),
          sample(sitting,100),
          sample(standing,100),
          sample(laying,100))

dat <- ex[,inds]
y <- labels[inds]
f <- factor(y, labels = activities)


## TEST 2: walking, walking upstairs and walking downstairs
inds <- c(sample(walking,100),
          sample(walking_up,100),
          sample(walking_down,100))
dat <- ex[,inds]
y <- labels[inds]
f <- factor(y, labels = activities[c(1,2,3)])


## TEST 3: sitting, standing and lying
inds <- c(sample(sitting,100),
          sample(standing,100),
          sample(laying,100))
dat <- ex[,inds]
y <- labels[inds]
f <- factor(y, labels = activities[c(4,5,6)])


## TEST 4: moving vs resting => (walking,walking upstairs,wolking downstairs) vs (sitting,standing,lying)
inds <- c(sample(walking,50),
          sample(walking_up,50),
          sample(walking_down,50),
          sample(sitting,50),
          sample(standing,50),
          sample(laying,50))
dat <- ex[,inds]
y <- c(rep(1,150),rep(2,150))
f <- factor(y, labels = c("moving","resting"))



########## SCUDO ANALYSIS #################

## Set parameters
nTop <- 60
nBottom <- 180
N <- 0.5
p <- 0.05

library("caret")

set.seed(123)
trainData <- dat
testData <- exTest #[sample(nrow(exTest)),]
fTest <- factor(testLabels, labels = activities)

# inTrain <- createDataPartition(f, list = FALSE)
# trainData <- dat[, inTrain]
# testData <- dat[, -inTrain]

## Analyze training set
library("rScudo")
trainRes <- scudoTrain(trainData, 
                       groups = f,
                       nTop = nTop, 
                       nBottom = nBottom, 
                       alpha = p)

## Generate and plot map of training samples
trainNet <- scudoNetwork(trainRes, N = N)
scudoPlot(trainNet, vertex.label = NA)
# scudoCytoscape(trainNet)  #Open in Cytoscape

## Perform validation using testing samples
testRes <- scudoTest(trainRes, 
                     testData, 
                     f[-inTrain],
                     nTop = nTop, 
                     nBottom = nBottom)

testNet <- scudoNetwork(testRes, N = N)
scudoPlot(testNet, vertex.label = NA)

## Identify clusters on map
# library("igraph")
# testClust <- igraph::cluster_spinglass(trainNet, spins = 3)
# plot(testClust, trainNet, vertex.label = NA)

## Perform classification
classRes <- scudoClassify(trainData, 
                          testData, 
                          N = N,
                          nTop = nTop, 
                          nBottom = nBottom,
                          trainGroups = f[inTrain], 
                          alpha = p)

caret::confusionMatrix(classRes$predicted, f[-inTrain])






########## CROSS VALIDATION #################

## OPTION 1: Feature selection using Kruskal Test

virtControl <- rowMeans(trainData)
trainDataNorm <- trainData / virtControl
pVals <- apply(trainDataNorm, 1, function(x) {
  stats::kruskal.test(x, f[inTrain])$p.value})
trainDataNorm <- t(trainDataNorm[pVals <= p, ])

## OPTION 2: Use feature selected by scudoTrain

sel_feat <- selectedFeatures(trainRes)
length(sel_feat)
trainDataReduced <- t(trainData[sel_feat,])

## Start cross validation

cl <- parallel::makePSOCKcluster(2)
doParallel::registerDoParallel(cl)

model <- scudoModel(nTop = (1:10)*20, 
                    nBottom = (1:10)*20,
                    N = (1:6)*0.1)

control <- caret::trainControl(method = "cv", 
                               number = 5,
                               summaryFunction = caret::multiClassSummary)

cvRes <- caret::train(x = trainDataReduced, 
                      y = f, 
                      method = model,
                      trControl = control)

parallel::stopCluster(cl)

## Classify using the best parameter tuning found by CV

nTop <- cvRes$bestTune$nTop
nBottom <- cvRes$bestTune$nBottom
N <- cvRes$bestTune$N

classStage <- scudoClassify(trainData, 
                            testData[,1:1000], 
                            N = N, 
                            nTop = nTop, 
                            nBottom = nBottom, 
                            f, 
                            alpha = p)

caret::confusionMatrix(classStage$predicted, fTest[1:1000])





