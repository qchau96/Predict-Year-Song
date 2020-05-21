rm(list=ls())
library(dplyr)
library(tidyr)
library(randomForest)
library(ggplot2)
library(gridExtra)
library(glmnet)
library(data.table)
df = fread("/Users/chauchu/Downloads/YearPredictionMSD.txt", header = F)
df = sample_n(df,5000)
write.table(df, "/Users/chauchu/Downloads/data1.txt", row.names = F, col.names=F )
data = fread("/Users/chauchu/Downloads/data1.txt", header = F)
data = data %>% rename(year=V1)
data = data %>% select(-year) %>%
  mutate_all(.funs = function(x) {x / sqrt(mean((x - mean(x))^2))}) %>%
  mutate(year=data$year)
n = dim(data)[1]
p = dim(data)[2]-1

y = log(data[,p+1])
X = data.matrix(data[,-(p+1)])

n.train        =     floor(0.8*n)
n.test         =     n-n.train

M              =     10
Rsq.test.rf    =     rep(0,M)  # rf= randomForest
Rsq.train.rf   =     rep(0,M)
Rsq.test.en    =     rep(0,M)  #en = elastic net
Rsq.train.en   =     rep(0,M)
Rsq.test.rd    =     rep(0,M)  #rd = ridge
Rsq.train.rd   =     rep(0,M)
Rsq.test.ls    =     rep(0,M)  #ls = lasso
Rsq.train.ls   =     rep(0,M)

en.time = 0
ls.time = 0
rid.time = 0
rf.time = 0
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  # fit elastic-net and calculate and record the train and test R squares 
  a=0.5 # elastic-net
  start.time = Sys.time ()
  cv.fit           =     cv.glmnet(X.train, y.train, alpha = a, nfolds = 10)
  fit              =     glmnet(X.train, y.train, alpha = a, lambda = cv.fit$lambda.min) #lambda give the smallest cross-validation
  end.time = Sys.time()
  y.train.hat      =     predict(fit, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat       =     predict(fit, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)  
  res.test.en      =     y.test - y.test.hat
  res.train.en     =     y.train - y.train.hat
  en.time = en.time + (end.time - start.time)
  # fit lasso and calculate and record the train and test R squares (alpha =1)
  start.time = Sys.time ()
  cv.lasso         =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10)
  lasso.fit        =     glmnet(X.train, y.train, alpha = 1,lambda = cv.lasso$lambda.min)
  end.time = Sys.time()
  y.train.hat      =     predict(lasso.fit, newx = X.train, type = "response", cv.lasso$lambda.min)
  y.test.hat       =     predict(lasso.fit, newx = X.test, type = "response", cv.lasso$lambda.min)
  Rsq.test.ls[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.ls[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  res.test.ls      =     y.test - y.test.hat
  res.train.ls     =     y.train - y.train.hat
  ls.time = ls.time + (end.time - start.time)
  # fit ridge and calculate and record the train and test R squares (alpha =0)
  start.time = Sys.time ()
  cv.ridge         =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10)
  ridge.fit        =     glmnet(X.train, y.train, alpha = 0,lambda = cv.ridge$lambda.min)
  end.time = Sys.time()
  y.train.hat      =     predict(ridge.fit, newx = X.train, type = "response", cv.ridge$lambda.min)
  y.test.hat       =     predict(ridge.fit, newx = X.test, type = "response", cv.ridge$lambda.min)
  Rsq.test.rd[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rd[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  res.test.rd      =     y.test - y.test.hat
  res.train.rd     =     y.train - y.train.hat
  rid.time = rid.time + (end.time - start.time)
  # fit RF and calculate and record the train and test R squares 
  start.time = Sys.time ()
  rf               =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  end.time = Sys.time()
  y.test.hat       =     predict(rf, X.test)
  y.train.hat      =     predict(rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat)^2)/mean((y - mean(y))^2)
  res.test.rf      =     y.test - y.test.hat
  res.train.rf     =     y.train - y.train.hat
  rf.time = rf.time + (end.time - start.time)
  #  cat(sprintf("m=%3.f|  Rsq.test.rf=%.2f, Rsq.test.en = %.2f, Rsq.test.ls = %.2f, Rsq.test.rd=%.2f | 
  #               Rsq.train.rf=%.2f, Rsq.train.en=%.2f, Rsq.train.ls=%.2f, Rsq.train.rd=%.2f| \n", m,  
  #              Rsq.test.rf[m], Rsq.test.en[m], Rsq.test.ls[m],  Rsq.test.rd[m],  
  #              Rsq.train.rf[m], Rsq.train.en[m], Rsq.train.ls[m], Rsq.train.rd[m]))
  #  cat(sprintf("m=%3.f|  res.test.rf=%.2f, res.test.en = %.2f, res.test.ls = %.2f, res.test.rd=%.2f | 
  #               res.train.rf=%.2f, res.train.en=%.2f, res.train.ls=%.2f, res.train.rd=%.2f| \n", m,  
  #              res.test.rf[m], res.test.en[m], res.test.ls[m],  res.test.rd[m],  
  #              res.train.rf[m], res.train.en[m], res.train.ls[m], res.train.rd[m]))
}
model_time = data.frame(lasso = ls.time, ridge = rid.time, elnet = en.time, rf = rf.time)


plot(cv.fit, main = 'Elastic Net')
plot(cv.lasso, main ='Lasso')
plot(cv.ridge, main ='Ridge')

#Plot R-squared
rsq_train = data.frame(lasso = Rsq.train.ls, ridge = Rsq.train.rd,
                       elnet = Rsq.train.en, rf = Rsq.train.rf, dataset="train")
rsq_test = data.frame(lasso = Rsq.test.ls, ridge = Rsq.test.rd,
                      elnet = Rsq.test.en, rf = Rsq.test.rf, dataset="test")
rsq_models = rbind(rsq_train, rsq_test)
rsq_plot = rsq_models %>%
  gather(model, rsquared, lasso:rf) %>%
  ggplot(aes(x=model, y=rsquared, fill=model)) +
  geom_boxplot() +
  facet_wrap(~dataset)

print(rsq_plot)

#Plot Residuals
resid_train = data.frame(ls = res.train.ls, rd = res.train.rd,
                         en = res.train.en, rf = res.train.rf, dataset="train")
resid_train = resid_train %>% rename(lasso = X1, ridge = X1.1, elnet = s0)
resid_test = data.frame(lasso = res.test.ls, ridge = res.test.rd,
                        elnet = res.test.en, rf = res.test.rf, dataset="test")
resid_test = resid_test %>% rename(lasso = X1, ridge = X1.1, elnet = s0)
resid_models = rbind(resid_train, resid_test)
resid_plot = resid_models %>%
  gather(model, residuals, lasso:rf) %>%
  ggplot(aes(x=model, y=residuals, fill=model)) +
  geom_boxplot() +
  facet_wrap(~dataset)
print(resid_plot)

bootstrapSamples =     100
beta.rf.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.en.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.ls.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)
beta.rd.bs       =     matrix(0, nrow = p, ncol = bootstrapSamples)

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  # fit bs rf
  rf               =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.rf.bs[,m]   =     as.vector(rf$importance[,1])
  # fit bs en
  a                =     0.5 # elastic-net
  cv.fit           =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, nfolds = 10)
  fit              =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = a, lambda = cv.fit$lambda.min)  
  beta.en.bs[,m]   =     as.vector(fit$beta)
  # fit bs ls - alpha = 1
  cv.lasso         =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = 1, nfolds = 10)
  ls               =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = 1, lambda = cv.lasso$lambda.min)  
  beta.ls.bs[,m]   =     as.vector(ls$beta)
  # fit bs rd - alpha = 0
  cv.ridge         =     cv.glmnet(X.bs, y.bs, intercept = FALSE, alpha = 0, nfolds = 10)
  rd               =     glmnet(X.bs, y.bs, intercept = FALSE, alpha = 0, lambda = cv.ridge$lambda.min)  
  beta.rd.bs[,m]   =     as.vector(rd$beta)
  
  #cat(sprintf("Bootstrap Sample %3.f \n", m))
  
}
# calculate bootstrapped standard errors / r bounds

rf.bs.sd    = apply(beta.rf.bs, 1, "sd")
en.bs.sd    = apply(beta.en.bs, 1, "sd")
ls.bs.sd    = apply(beta.ls.bs, 1, "sd") 
rd.bs.sd    = apply(beta.rd.bs, 1, "sd")

# fit rf to the whole data
rf               =     randomForest(X, y, mtry = sqrt(p), importance = TRUE)

# fit en to the whole data
a=0.5 # elastic-net
cv.fit           =     cv.glmnet(X, y, alpha = a, nfolds = 10)
fit              =     glmnet(X, y, alpha = a, lambda = cv.fit$lambda.min)

#fit ls to the whole data - alpha = 1
cv.lasso         =     cv.glmnet(X, y, alpha = 1, nfolds = 10)
ls               =     glmnet(X, y, alpha = 1, lambda = cv.lasso$lambda.min)

#fit rd to the whole data - alpha = 0
cv.ridge         =     cv.glmnet(X, y, alpha = 0, nfolds = 10)
rd               =     glmnet(X, y, alpha = 0, lambda = cv.ridge$lambda.min)

betaS.rf               =     data.frame(c(1:p), as.vector(rf$importance[,1]), 2*rf.bs.sd)
colnames(betaS.rf)     =     c( "feature", "value", "err")

betaS.en               =     data.frame(c(1:p), as.vector(fit$beta), 2*en.bs.sd)
colnames(betaS.en)     =     c( "feature", "value", "err")

betaS.ls               =     data.frame(c(1:p), as.vector(ls$beta), 2*ls.bs.sd)
colnames(betaS.ls)     =     c( "feature", "value", "err")

betaS.rd               =     data.frame(c(1:p), as.vector(rd$beta), 2*rd.bs.sd)
colnames(betaS.rd)     =     c( "feature", "value", "err")

rfPlot =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.1) +ggtitle('RandomForest')


enPlot =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.1)+ggtitle('ElasticNet')

lsPlot =  ggplot(betaS.ls, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.1)+ggtitle('Lasso')

rdPlot =  ggplot(betaS.rd, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.1)+ggtitle('Ridge')

#Plot coefficients
grid.arrange(rfPlot, enPlot,lsPlot, rdPlot, nrow = 4)

