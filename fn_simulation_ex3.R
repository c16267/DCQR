rm(list=ls())
#setwd("/Users/shin/Dropbox/PhD/research/(241012)[JAS] DCQR/code")

library(mvtnorm)
library(VGAM)
library(nor1mix)

################################################################################
# Fixed parameter setting
################################################################################
# Seed
seed.train=123
seed.val=456
seed.test=789

# Kernel Parameters
# Lambda set
Lambda.set=exp(seq(-1, 0, .5))

# Sigma2 set
Sigma2.set=exp(seq(-3,3,1))

# check!!
esp.weight=1e-10;
n.round.sic=50;
n.round.sel=10;



################################################################################
# Error distribution 
################################################################################

# 1. normal
sigma.esp=1


# 3. gaussian mixture
obj=norMix(mu=c(0,0), sigma = sqrt(c(25,1)), w=c(0.1,0.9))

# 4. T
DF=3

# 5. lognormal 
# 6. Exponential
Rate=1

################################################################################
# Changed parameter setting
################################################################################
# Changed parameter

n.model=10;

K=9
tau.set=seq(1:K)/(K+1)
num.tau <- length(tau.set)
n.train=100; n.val=1000; n.test=10000;
Replication=100;
n.start <- rep.ini <- 5

# Error distribution
#err.dist="N";   # 1. normal N(0,1)
#err.dist="Mix"; # 3. gaussian mixture 
err.dist="T";   # 4. T with df=3
#err.dist="LN";  # 5. lognormal
#err.dist="Exp"; # 6. exponential


################################################################################
# Storage
################################################################################
# KRR, CKQR
Best.krr=matrix(,Replication,2);
Best.ckqr=matrix(,Replication,2)

MAE.ckqr=matrix(,Replication,K);
MSE=matrix(,Replication,n.model)
colnames(MSE)=c("KRR", "CKQR", "  ", "D-LSE", "D-MQR", "D-CQR", "", "","","")

# Deep-MQR, D-CQR
MAE.dmqr=matrix(,Replication,K);
MAE.dcqr=matrix(,Replication,K);
MAE.dcqr2=matrix(,Replication,K);

MAE=matrix(,Replication,4);
colnames(MAE)=c("Oracle", "D-CQR", "GD-CQR", "AGD-CQR")

time_mat=matrix(,Replication, 4)
colnames(time_mat)=c("Oracle", "D-CQR", "GD-CQR", "AGD-CQR")

TMPE <- matrix(, nrow=3, ncol=num.tau)

TMPE.list <- list()


W0.hat.PDCQR <- list()
W1.hat.PDCQR <- list()

W0.hat.APDCQR <- list()
W1.hat.APDCQR <- list()

nc <- c(1,1,1,0,0,0,0,0,0,0,0,0);nic <- 1-nc
var.val.PDCQR.mat <- matrix(,nrow=Replication, ncol=length(nc))
var.val.APDCQR.mat <-  matrix(,nrow=Replication, ncol=length(nc))

NC.val <- matrix(, Replication, 4)
NIC.val <- matrix(, Replication, 4)
NT.val <- matrix(, Replication, 4)


###################################################################
for (rep in 1:Replication){
################################################################################
# Reading Data
################################################################################
# 1. Training Data
 set.seed(seed.train+rep)
 
 #uniform random covariates
 X.train1 <- runif(n.train,-1, 1)
 X.train2 <- runif(n.train,-1, 1)
 X.train3 <- runif(n.train,-1, 1)
 X.train4 <- runif(n.train,-1, 1)
 X.train5 <- runif(n.train,-1, 1)
 X.train6 <- runif(n.train,-1, 1)
 X.train7 <- runif(n.train,-1, 1)
 X.train8 <- runif(n.train,-1, 1)
 X.train9 <- runif(n.train,-1, 1)
 X.train10 <- runif(n.train,-1, 1)
 X.train11<- runif(n.train,-1, 1)
 X.train12 <- runif(n.train,-1, 1)
 
 
 X.train <- cbind(X.train1,X.train2, X.train3, X.train4, X.train5, X.train6, X.train7, X.train8,
                  X.train9,X.train10,X.train11,X.train12)

 
 # Errors
 if (err.dist=="N") {Esp.train=as.matrix(rnorm(n.train,0,sigma.esp))}
 if (err.dist=="T") {Esp.train=as.matrix(rt(n.train,df=DF)); Esp.train=scale(Esp.train)};
 if (err.dist=="Mix") {Esp.train=as.matrix(rnorMix(n.train, obj)); Esp.train=scale(Esp.train)};
 if (err.dist=="Exp") {
   x = rexp(n.train, 1); u = 1 - exp(-1*x); z = qnorm(u)
   Esp.train=as.matrix(z)
 }
 if (err.dist=="LN") {Esp.train <- rlnorm(n.train, meanlog = 0, sdlog = 1); Esp.train=scale(Esp.train)};
 if (err.dist=="chi") {Esp.train=as.matrix(rchisq(n.train, 3)); Esp.train=scale(Esp.train)};
 
 
 Y.train <- 0.1 + 0.1*X.train[,1]^3 + 0.5*X.train[,2]^2 + X.train[,1] + X.train[,2] + X.train[,3] + Esp.train
 


# 2. Validation Data
 set.seed(seed.val+rep)
 #X.val=runif(n.val, -5, 5)
 #X.val=runif(n.val, -1, 1)
 X.val1 <- runif(n.val, -1, 1)
 X.val2 <- runif(n.val, -1, 1)
 X.val3 <- runif(n.val, -1, 1)
 X.val4 <- runif(n.val, -1, 1)
 X.val5 <- runif(n.val, -1, 1)
 X.val6 <- runif(n.val, -1, 1)
 X.val7 <- runif(n.val, -1, 1)
 X.val8 <- runif(n.val, -1, 1)
 X.val9 <- runif(n.val, -1, 1)
 X.val10 <- runif(n.val, -1, 1)
 X.val11 <- runif(n.val, -1, 1)
 X.val12 <- runif(n.val, -1, 1)
 X.val <- cbind(X.val1,X.val2,X.val3,X.val4,X.val5,X.val6,X.val7,X.val8,
                X.val9,X.val10,X.val11,X.val12)
 # Errors
 
 if (err.dist=="N") {Esp.val=as.matrix(rnorm(n.val,0,sigma.esp))}
 if (err.dist=="T") {Esp.val=as.matrix(rt(n.val,df=DF))}
 if (err.dist=="Mix") {Esp.val=as.matrix(rnorMix(n.val, obj)); Esp.val=scale(Esp.val)};
 if (err.dist=="Exp") {
   x = rexp(n.val, 1); u = 1 - exp(-1*x); z = qnorm(u)
   Esp.val=as.matrix(z)
 }
 if (err.dist=="LN") {Esp.val <- rlnorm(n.val, meanlog = 0, sdlog = 1);Esp.val=scale(Esp.val)};
 if (err.dist=="chi") {Esp.val=as.matrix(rchisq(n.val, 3)); Esp.val=scale(Esp.val)};
 
 Y.val <- 0.1 + 0.1*X.val[,1]^3 + 0.5*X.val[,2]^2 + X.val[,1] + X.val[,2] + X.val[,3] + Esp.val
 
# 3. Test Data
  set.seed(seed.test+rep)
  X.test1 <- runif(n.test, -1, 1)
  X.test2 <- runif(n.test, -1, 1)
  X.test3 <- runif(n.test, -1, 1)
  X.test4 <- runif(n.test, -1, 1)
  X.test5 <- runif(n.test, -1, 1)
  X.test6 <- runif(n.test, -1, 1)
  X.test7 <- runif(n.test, -1, 1)
  X.test8 <- runif(n.test, -1, 1)
  X.test9 <- runif(n.test, -1, 1)
  X.test10 <- runif(n.test, -1, 1)
  X.test11 <- runif(n.test, -1, 1)
  X.test12 <- runif(n.test, -1, 1)
  X.test <- cbind(X.test1,X.test2,X.test3,X.test4,X.test5,X.test6,X.test7,X.test8,
                  X.test9,X.test10,X.test11,X.test12)
  
  #ord=order(X.test,decreasing = F)
  #X.test=X.test[ord]
  
  Esp.test=matrix(,n.test,K); 
  if(err.dist=="N"){for(k in 1:K){Esp.test[,k]=qnorm(tau.set[k],0,sigma.esp)}}
  if(err.dist=="T"){for(k in 1:K){Esp.test[,k]=qt(tau.set[k],df=3)}}
  if (err.dist=="Mix"){
    quantiles <- qnorMix(tau.set, obj)
    quantiles <- quantiles - mean(quantiles)
    for(ii in 1:n.test){Esp.test[ii,] <- quantiles}
  }
  if (err.dist=="Exp"){for(k in 1:K){Esp.test[,k]=qexp(tau.set[k],1)}; Esp.test = Esp.test-mean(Esp.test) } #MEAN 0
  if (err.dist=="LN"){
    quantiles <- qlnorm(tau.set, meanlog = 0, sdlog = 1)
    quantiles <- quantiles - mean(quantiles)
    for(ii in 1:n.test){Esp.test[ii,] <- quantiles}
  }
  
  
  Y.test=matrix(,n.test,K)
  
  M.test <- 0.1 + 0.1*X.test[,1]^3 + 0.5*X.test[,2]^2 + X.test[,1] + X.test[,2] + X.test[,3]
  
  for(k in 1:K){Y.test[,k]=M.test+Esp.test[,k]}

  ###################
  ##1. ORACLE - D-CQR ##
  ###################
  model <- 1
  mod.oracle <- 0; Criteria_mod.oracle <- Inf; Best_i_mod.oracle <- 0
  true_p <- 3
  Bsize <- 16
  n.epoch <- 6000
  lr_rate <- 0.008
  

#parameter tuning
    for(i in 1:n.start){
    set.seed(1365*rep*i)
    a1=-1; b1=1
    W1_ini=matrix(runif(Bsize*(true_p+1), a1, b1), nrow=Bsize, ncol=(true_p+1)) #size adjusting
    
    a2=-1; b2=1
    W2_ini=matrix(runif(Bsize*(Bsize+1), a2, b2), nrow=Bsize, ncol=Bsize+1)
    
    a3=-1; b3=1
    W0_ini=matrix(runif(1*Bsize, a3, b3), nrow=1, ncol=Bsize)
    B0_ini=matrix(runif(K, a3, b3), nrow=1, ncol=K);
    B0_ini=B0_ini[order(B0_ini)]
    
      mod.oracle <- check2_cqr_val(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train[,1:3]), y=Y.train, Epoch=n.epoch, Bsize=Bsize, rate=lr_rate,
                                           X_val=cbind(1, X.val[,1:3]), y_val=Y.val, Taus=tau.set)
      criteria=min(mod.oracle$MPE_val)
      if(criteria < Criteria_mod.oracle){
        result.mod.oracle <- mod.oracle; Criteria_mod.oracle=criteria; Best_i_mo.oracle=i
      }
    plot(1:length(result.mod.oracle$MPE_val), result.mod.oracle$MPE_train, cex=.5, type='l', ylim=c(min(result.mod.oracle$MPE_train)-.1, max(result.mod.oracle$MPE_train)+.1 ), col="blue", main="Oracle")
    lines(result.mod.oracle$MPE_val, cex=.5,  col="red");
    }
  
  
  W1hat_oracle <- mod.oracle$Weight1
  W0hat_oracle <- mod.oracle$Weight0
  y.test.dcqr=cCheck2_predict(W1=result.mod.oracle$Weight1, W2=result.mod.oracle$Weight2, W0=result.mod.oracle$Weight0, B0=result.mod.oracle$WeightB0, X=cbind(1, X.test[,1:3]), y=1, Taus=tau.set)$Pred
  mae.dcqr=abs(Y.test-t(y.test.dcqr))
  
  MAE.dcqr[rep,]=colMeans(mae.dcqr)
  MAE[rep, model]=mean(mae.dcqr)
  

  ###################
  ##2. D-CQR ##
  ###################
  model <- 2
  mod.2 <- 0;
  Criteria_mod.2 <- Inf;
  Best_i_mod.2 <- 0
  p <- ncol(X.train)
  Bsize <- 16
  n.epoch <- 7000
  lr_rate <- 0.005
  
  
  #parameter tuning
  for(i in 1:n.start){
    set.seed(1365*rep*i)
    a1=-1; b1=1
    W1_ini=matrix(runif(Bsize*(p+1), a1, b1), nrow=Bsize, ncol=(p+1)) #size adjusting
    
    a2=-1; b2=1
    W2_ini=matrix(runif(Bsize*(Bsize+1), a2, b2), nrow=Bsize, ncol=Bsize+1)
    
    a3=-1; b3=1
    W0_ini=matrix(runif(1*Bsize, a3, b3), nrow=1, ncol=Bsize)
    B0_ini=matrix(runif(K, a3, b3), nrow=1, ncol=K);
    B0_ini=B0_ini[order(B0_ini)]
    
    mod.2 <- check2_cqr_val(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train), y=Y.train, Epoch=n.epoch, Bsize=Bsize, rate=lr_rate,
                                 X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set)
    criteria=min(mod.2$MPE_val)
    if(criteria < Criteria_mod.2){
      result.mod.2 <- mod.2; Criteria_mod.2=criteria; Best_i_mod.2=i
    }
    plot(1:length(result.mod.2$MPE_val), result.mod.2$MPE_train, cex=.5, type='l', ylim=c(min(result.mod.2$MPE_train)-.1, max(result.mod.2$MPE_train)+.1 ), col="blue", main="D-CQR")
    lines(result.mod.2$MPE_val, cex=.5,  col="red"); 
  }
  
  
  W1hat_dcqr <- result.mod.2$Weight1
  W0hat_dcqr <- result.mod.2$Weight0
  y.test.dcqr2=cCheck2_predict(W1=result.mod.2$Weight1, W2=result.mod.2$Weight2, W0=result.mod.2$Weight0, B0=result.mod.2$WeightB0, X=cbind(1, X.test), y=1, Taus=tau.set)$Pred
  mae.dcqr2=abs(Y.test-t(y.test.dcqr2))
  
  MAE.dcqr2[rep,]=colMeans(mae.dcqr2)
  MAE[rep, model]=mean(mae.dcqr2)

  
  
  
###################################################################
# 3. GroupLasso Penalized Deep Composite Quantile Regression (GD-CQR)
###################################################################
model=3
result3=0
Bsize <- 16
lr_rate <- 0.01
n.epoch <- 5000
#gamma <- 0.2  # L2 norm penalty
Criteria.mod3 <- Criteria <- Inf;
Best_i<-Best_mod3<- 0; 
Weis <- rep(1, (p+1))

#############
#tuning gamma
#############
  gamma_grid <- exp(seq(-3,0.3, by=0.2))
  gamma_mat_pnmqr <- matrix(NA, ncol=length(gamma_grid), nrow = n.start)
  rownames(gamma_mat_pnmqr) <- 1:n.start
  colnames(gamma_mat_pnmqr) <- gamma_grid
  
  for(uu in 1:length(gamma_grid)){
    for(i in 1:rep.ini){
      set.seed(1365*uu)
      a1=-1; b1=1
      W1_ini=matrix(runif(Bsize*(p+1), a1, b1), nrow=Bsize, ncol=(p+1)) #size adjusting
  
      a2=-1; b2=1
      W2_ini=matrix(runif(Bsize*(Bsize+1), a2, b2), nrow=Bsize, ncol=Bsize+1)
  
      a3=-1; b3=1
      W0_ini=matrix(runif(1*Bsize, a3, b3), nrow=1, ncol=Bsize)
      B0_ini=matrix(runif(K, a3, b3), nrow=1, ncol=K);
      B0_ini=B0_ini[order(B0_ini)]
  
      result <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train), y=Y.train, Epoch=n.epoch, Bsize=Bsize, rate=lr_rate,
                            X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set, gamma=gamma_grid[uu], Weights = Weis)
  
      criteria=min(result$MPE_val)
      if(criteria < Criteria){
        result3=result; Criteria=criteria; Best_mod3=i
      }
      cv_check <- cCheck2_predict_test_grlasso(W1=result3$Weight1, W2=result3$Weight2, W0=result3$Weight0, B0=result3$B0, X=cbind(1, X.test), y=Y.test, Taus=tau.set, gamma=gamma_grid[uu], ww=result3$Weight1)$Pred
      mae.pdmqr <- abs(Y.test-t(cv_check))
      gamma_mat_pnmqr[i,uu] <- mean(colMeans(mae.pdmqr))
    }
  }
  
  best_gamma_pnmqr <- gamma_grid[which.min(colMeans(gamma_mat_pnmqr))]

  #after tuning gamma
  Criteria.mod3 <- Criteria <- Inf;
  Best_mod3<- 0; 
  n.start <- 5
  for(i in 1:n.start){
    set.seed(1365*rep*i)
    a1=-1; b1=1
    W1_ini=matrix(runif(Bsize*(p+1), a1, b1), nrow=Bsize, ncol=(p+1)) #size adjusting
    
    a2=-1; b2=1
    W2_ini=matrix(runif(Bsize*(Bsize+1), a2, b2), nrow=Bsize, ncol=Bsize+1)
    
    a3=-1; b3=1
    W0_ini=matrix(runif(1*Bsize, a3, b3), nrow=1, ncol=Bsize)
    B0_ini=matrix(runif(K, a3, b3), nrow=1, ncol=K);
    B0_ini=B0_ini[order(B0_ini)]
  
    mod3 <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train), y=Y.train, Epoch=n.epoch, Bsize=Bsize, rate=lr_rate,
                             X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set, gamma=best_gamma_pnmqr, Weights = Weis)
    criteria=min(mod3$MPE_val)
    if(criteria < Criteria.mod3){
      result.mod3 <- mod3; Criteria.mod3=criteria; Best_mod3=i
    }
  } # END i  
  
  plot(1:length(result.mod3$MPE_val), result.mod3$MPE_train, cex=.5, type='l', ylim=c(min(result.mod3$MPE_train)-.1, max(result.mod3$MPE_train)+.1 ), col="blue", main="GD-CQR")
  lines(result.mod3$MPE_val, cex=.5,  col="red");
  
  W1hat_PDCQR <- result.mod3$Weight1
  W0hat_PDCQR <- result.mod3$Weight0
  B0hat_PDCQR <- result.mod3$B0
  
  pred_model3 <- cCheck2_predict_test_grlasso(W1=W1hat_PDCQR, W2=result.mod3$Weight2, W0=result.mod3$Weight0, B0=B0hat_PDCQR, X=cbind(1, X.test), y=Y.test, Taus=tau.set, gamma=best_gamma_pnmqr, ww=W1hat_PDCQR)$Pred
  mae.pdmqr <- abs(Y.test-t(pred_model3))
  MAE[rep, model] <- mean(colMeans(mae.pdmqr))
  
  
  # Model selection
  W1.hat.A <- round(W1hat_PDCQR[,-1],3)
  var.val <- matrix(as.numeric(W1.hat.A!=0),ncol=p)
  var.val.PDCQR <- matrix(ifelse(colMeans(abs(W1.hat.A)) <= 0.001*5, 0, 1),ncol=p)
  m.val <- colSums(var.val.PDCQR)
  NC.val[rep,model] <- sum((m.val*nc)!=0);
  NIC.val[rep,model] <- sum((m.val*nic)!=0);
  NT.val[rep,model] <- ifelse(NC.val[rep,model]==3 & NIC.val[rep,model]==0, 1, 0)
  
  var.val.PDCQR.mat[rep, ]  <- var.val.PDCQR
  W0.hat.PDCQR[[rep]] <- W0hat_PDCQR
  W1.hat.PDCQR[[rep]] <- W1hat_PDCQR


  ###################################################################
  # 4. Adaptive GroupLasso Penalized Deep Composite Quantile Regression (AGD-CQR)
  ###################################################################
  model=4
  result4=0
  Bsize <- 16
  lr_rate <- 0.01
  n.epoch <- 8000
  #gamma <- 0.2  # L2 norm penalty
  Criteria.mod4 <- Criteria <- Inf;
  Best_mod4 <- 0; 
  Weis <- glasso_weight(W_ini=W1hat_dcqr)
  
  ############  
  #tuning gamma
  ############
  
  gamma_grid <- exp(seq(-2,0.5,0.2))
  gamma_mat_apnmqr <- matrix(, ncol=length(gamma_grid), nrow = n.start)
  colnames(gamma_mat_apnmqr) <- gamma_grid

  for(uu in 1:length(gamma_grid)){
    for(i in 1:n.start){
    set.seed(1365*uu*i)
    a1=-1; b1=1
    W1_ini=matrix(runif(Bsize*(p+1), a1, b1), nrow=Bsize, ncol=(p+1)) #size adjusting

    a2=-1; b2=1
    W2_ini=matrix(runif(Bsize*(Bsize+1), a2, b2), nrow=Bsize, ncol=Bsize+1)

    a3=-1; b3=1
    W0_ini=matrix(runif(1*Bsize, a3, b3), nrow=1, ncol=Bsize)
    B0_ini=matrix(runif(K, a3, b3), nrow=1, ncol=K);
    B0_ini=B0_ini[order(B0_ini)]

    mod4 <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train), y=Y.train, Epoch=n.epoch, Bsize=Bsize, rate=lr_rate,
                              X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set, gamma=gamma_grid[uu], Weights = Weis)
    criteria=min(mod4$MPE_val)
    if(criteria < Criteria.mod4){
      result.mod4 <- mod4; Criteria.mod4=criteria; Best_mod4=i
    }
    cv_check <- cCheck2_predict_test_grlasso(W1=mod4$Weight1, W2=mod4$Weight2, W0=mod4$Weight0, B0=mod4$B0, X=cbind(1, X.test), y=Y.test, Taus=tau.set, gamma=gamma_grid[uu], ww=mod4$Weight1)$Pred
    mae.apdmqr <- abs(Y.test-t(cv_check))
    gamma_mat_apnmqr[i,uu] <- mean(colMeans(mae.apdmqr))
    }
  }

  best_gamma_apnmqr <- gamma_grid[which.min(colMeans(gamma_mat_apnmqr))]

  ############  
  #after tuning gamma
  ############
  
  Criteria.mod4 <- Criteria <- Inf;
  Best_mod3<- 0; 
  n.start <- 5
  
  for(i in 1:n.start){
    set.seed(1365*rep*i)
    a1=-1; b1=1
    W1_ini=matrix(runif(Bsize*(p+1), a1, b1), nrow=Bsize, ncol=(p+1)) #size adjusting
    
    a2=-1; b2=1
    W2_ini=matrix(runif(Bsize*(Bsize+1), a2, b2), nrow=Bsize, ncol=Bsize+1)
    
    a3=-1; b3=1
    W0_ini=matrix(runif(1*Bsize, a3, b3), nrow=1, ncol=Bsize)
    B0_ini=matrix(runif(K, a3, b3), nrow=1, ncol=K);
    B0_ini=B0_ini[order(B0_ini)]
    
    mod4 <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train), y=Y.train, Epoch=n.epoch, Bsize=Bsize, rate=lr_rate,
                              X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set, gamma=best_gamma_apnmqr, Weights = Weis)
    criteria=min(mod4$MPE_val)
    if(criteria < Criteria.mod4){
      result.mod4 <- mod4; Criteria.mod4=criteria; Best_mod4=i
    }
  
  }
  
  W1hat_APDCQR <- result.mod4$Weight1
  W0hat_APDCQR <- result.mod4$Weight0
  B0hat_APDCQR <- result.mod4$B0
  
  pred_model4 <- cCheck2_predict_test_grlasso(W1=W1hat_APDCQR, W2=result.mod4$Weight2, W0=result.mod4$Weight0, B0=B0hat_APDCQR, X=cbind(1, X.test), y=Y.test, Taus=tau.set, gamma=best_gamma_apnmqr, ww=W1hat_APDCQR)$Pred
  amae.pdmqr=abs(Y.test-t(pred_model4))
  MAE[rep, model] <- mean(colMeans(amae.pdmqr))
  
  
  # Model selection
  W1.hat.A <- round(W1hat_APDCQR[,-1],3)
  var.val.APDCQR <- matrix(ifelse(colMeans(abs(W1.hat.A)) <= 0.001*5, 0, 1),ncol=p)
  m.val <- colSums(var.val.APDCQR)
  NC.val[rep,model] <- sum((m.val*nc)!=0);
  NIC.val[rep,model] <- sum((m.val*nic)!=0);
  NT.val[rep,model] <- ifelse(NC.val[rep,model]==3 & NIC.val[rep,model]==0, 1, 0)
  
  var.val.APDCQR.mat[rep, ]  <- var.val.APDCQR
  W0.hat.APDCQR[[rep]] <- W0hat_APDCQR
  W1.hat.APDCQR[[rep]] <- W1hat_APDCQR
  
  print(paste("Replication :", rep))

}


#########################################################################################################################


