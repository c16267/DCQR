rm(list=ls())

library(mvtnorm)
library(VGAM)
library(nor1mix)
library(qrnn)
library(dplyr)
library(tidyr)
library(scatterplot3d)

################################################################################
# Fixed parameter setting
################################################################################
Replication <- 100;
nepoch <- 7000

tau.set <- seq(0.1, 0.9, by=0.1)
K <- length(tau.set)
q <- num.tau <- length(tau.set)
nc <- c(0,0,0,0,0,1,1);nic <- 1-nc

TMPE <- matrix(, nrow=4, ncol=num.tau)
colnames(TMPE) <- c("NM_tau1","NM_tau2","NM_tau3","NM_tau4","NM_tau5","NM_tau6","NM_tau7","NM_tau8","NM_tau9")
n.start <- 5

NC.val <- matrix(, Replication, 4)
NIC.val <- matrix(, Replication, 4)
NT.val <- matrix(, Replication, 4)
TMPE.list <- list()
W0.hat.GDCQR <- list()
W1.hat.GDCQR <- list()

W0.hat.AGDCQR <- list()
W1.hat.AGDCQR <- list()

var.val.GDCQR.list <- matrix(,nrow=Replication, ncol=length(nc))
var.val.AGDCQR.list <-  matrix(,nrow=Replication, ncol=length(nc))

time_mat <- matrix(, nrow=Replication, ncol=4)
rownames(time_mat) <- 1:Replication
colnames(time_mat) <- c("D-MQR", "D-CQR", "GD-CQR", "AGD-CQR")

var.val.GDCQR.mat <- matrix(,nrow=Replication, ncol=length(nc))
var.val.AGDCQR.mat <-  matrix(,nrow=Replication, ncol=length(nc))


# Deep-MQR, D-CQR
MAE.dmqr=matrix(,Replication,K);
MAE.dcqr=matrix(,Replication,K);

MAE=matrix(,Replication,4);
colnames(MAE)=c("D-MQR", "D-CQR", "GD-CQR", "AGD-CQR")

time_mat=matrix(,Replication,4)
colnames(time_mat)=c("D-MQR", "D-CQR", "GD-CQR", "AGD-CQR")


###################################################################
for (rep in 1:Replication){
  ################################################################################
  # Reading Data
  ################################################################################
  # 1. Data load and split
  set.seed(rep)
  #load data
  data(YVRprecip)
  y <- YVRprecip$precip
  x <- scale(YVRprecip$ncep)
  x.center <- attr(x, "scaled:center")
  x.scale <- attr(x, "scaled:scale")
  x <- as.data.frame(x)[,2:3]
  
  noise.mat <- matrix(runif(length(y)*5,-5,5), ncol=5)
  x <- cbind(noise.mat, x)
  colnames(x) <- c("X1","X2","X3","X4","X5","sh700","z500")
  
  # factor(format(YVRprecip$date,"%Y"))
  #data split
  tic <-as.numeric(format(YVRprecip$date,"%Y")) >= 1998
  x <- x[tic,]
  y <- y[tic,]
  
  ind <- sample(1:nrow(x), nrow(x)*0.7)
  train <- ind[1:(length(ind)*0.7)]
  X.train <- x[train,]; 
  Y.train <- y[train]
  
  X.val <- X.train[-train,]
  Y.val <- Y.train[-train]
  
  n.train <- nrow(X.train); n.val <- nrow(X.val); 
  p <- ncol(X.train)
  
  X.test <- x[-ind,]
  Y.test <- y[-ind]
  n.test <-nrow(X.test);
  
  Esp.test <- matrix(NA,n.test,q)
  
  for(k in 1:q){Esp.test[,k]=qt(tau.set[k],df=3)}
  
  Y.test <- Y.test + Esp.test #mean zero error
  abs.Y.test <- abs(Y.test)
  abs.Y.test <- abs.Y.test+ 10^-1
  sign_Y <- sign(Y.test)
  Y.test <- sign_Y*log(abs(abs.Y.test))
  
  
  Lambda.set=exp(seq(-1, 0.1, 0.3))
  Sigma2.set=exp(seq(-1, 0.1 ,0.3))
  
  
  ################################################################################
  # 0. Kernel Ridge Regression (KRR)
  ################################################################################
  #tuning
  ###################################################################
  # 1. Deep Multiple Quantile Regression (D-MQR)
  ###################################################################
  model=1
  result1=0; Criteria=Inf; Best_i=0
  lr_rate <- 0.008
  nepoch <- 5000
  Bsize <- node_1 <- 10; node_2 <- 10
  n.start <- 5
  
  for(i in 1:n.start){
    
    set.seed(1365*rep*i)
    a1=-1; b1=1
    W1_ini=matrix(runif(node_1*(p+1), a1, b1), nrow=node_1, ncol=p+1)
    
    a2=-1; b2=1
    W2_ini=matrix(runif(node_2*(node_1+1), a2, b2), nrow=node_2, ncol=node_1+1)
    
    a3=-1; b3=1
    W0_ini=matrix(runif(K*(node_2+1), a3, b3), nrow=K, ncol=(node_2+1))
    
    start_time_dmqr <- Sys.time()
    result=check2_mqr_val(W1=W1_ini, W2=W2_ini, W0=W0_ini, X=cbind(1,X.train), y=Y.train, Epoch=nepoch, Bsize, rate=lr_rate,
                          X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set)
    
    end_time_dmqr <- Sys.time()
    time_mat[rep, model] <-   as.numeric(end_time_dmqr - start_time_dmqr, units = "secs")
    criteria=min(result$MPE_val)
    if(criteria < Criteria){
      result1=result; Criteria=criteria; Best_i=i
    }
    
  } # END i
  
  plot(1:length(result1$MPE_val), result1$MPE_train, cex=.5, type='l', ylim=c(min(result1$MPE_train), max(result1$MPE_train) ), col="blue", main="D-MQR")
  lines(result1$MPE_val, cex=.5,  col="red");
  
  
  result=result1
  y.test.dmqr=mCheck2_predict(W1=result$Weight1, W2=result$Weight2, W0=result$Weight0,  X=cbind(1, X.test), y=1, Taus=tau.set)
  mae.dmqr=abs(Y.test-t(  y.test.dmqr$Pred))
  MAE[rep, model]=mean(colMeans(mae.dmqr))
  
  
  data_mqr <- data.frame(sh700 = X.test[,1], z500= X.test[,2], tau0.3=y.test.dmqr$Pred[3,], tau0.5=y.test.dmqr$Pred[5,],tau0.7=y.test.dmqr$Pred[7,])
  data_mqr_long <- gather(data_mqr, quantile, precipitation, tau0.3:tau0.7, factor_key = T)

  colors <- c("black", "blue", "red")
  colors <- colors[as.numeric(data_mqr_long$quantile)]
  shapes = c(1, 2, 4)
  shapes <- shapes[as.numeric(data_mqr_long$quantile)]
  s3d <- scatterplot3d(  data_mqr_long[,c(1,2,4)], color=colors, pch=shapes, angle=160, main="D-MQR", scale.y = .5)
  legend(s3d$xyz.convert(3.5, 2, 2.0), legend = expression(tau[3]==0.3, tau[5]==0.5, tau[7]==0.7),
         col =  c("black", "blue", "red"), pch = c(1,2,4), cex=1.0)

  

  ###################################################################
  # 2. Deep Composite Quantile Regression (D-CQR)
  ###################################################################
  model=2
  result2=0; Criteria=Inf; Best_i=0
  lr_rate <- 0.001
  nepoch <- 2000
  n.start <- 5
  node_1 <- Bsize <- 10; node_2 <- 10
  
  for(i in 1:n.start){
    set.seed(1365*rep*i)
    a1=-1; b1=1
    W1_ini=matrix(runif(node_1*(p+1), a1, b1), nrow=node_1, ncol=p+1)
    
    a2=-1; b2=1
    W2_ini=matrix(runif(node_2*(node_1+1), a2, b2), nrow=node_2, ncol=node_1+1)
    
    a3=-1; b3=1
    W0_ini=matrix(runif(1*node_2, a3, b3), nrow=1, ncol=node_2)
    B0_ini=matrix(runif(K, a3, b3), nrow=1, ncol=K); B0_ini=B0_ini[order(B0_ini)]
    start_time_dcqr <- Sys.time()
    
    result=check2_cqr_val(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train), y=Y.train, Epoch=nepoch, Bsize, rate=lr_rate,
                          X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set)
    end_time_dcqr <- Sys.time()
    time_mat[rep, model] <-   as.numeric(end_time_dcqr - start_time_dcqr, units = "secs")
    criteria=min(result$MPE_val)
    if(criteria < Criteria){
      result2=result; Criteria=criteria; Best_i=i
    }
    
  } # END i
  
  plot(1:length(result2$MPE_val), result2$MPE_train, cex=.5, type='l', ylim=c(min(result2$MPE_train)-.1, max(result2$MPE_train) ), col="blue", main="D-CQR")
  lines(result2$MPE_val, cex=.5,  col="red");
  
  
  # quantile ft.
  y.test.dcqr=cCheck2_predict(W1=result$Weight1, W2=result$Weight2, W0=result$Weight0, B0=result$WeightB0, X=cbind(1, X.test), y=1, Taus=tau.set)$Pred
  mae.dcqr=abs(Y.test-t(y.test.dcqr))
  MAE.dcqr[rep,]=colMeans(mae.dcqr)
  MAE[rep, model]=mean(mae.dcqr)
  
  W1hat_dcqr <- result2$Weight1
  W0hat_dcqr <- result2$Weight0
  
  
  
  data_cqr <- data.frame(sh700 = X.test[,1], z500= X.test[,2], tau0.3=y.test.dcqr[3,], tau0.5=y.test.dcqr[5,],tau0.7=y.test.dcqr[7,])
  data_cqr_long <- gather(data_mqr, quantile, precipitation, tau0.3:tau0.7, factor_key = T)

  colors <- c("black", "blue", "red")
  colors <- colors[as.numeric(data_mqr_long$quantile)]
  shapes = c(19,17,18) 
  shapes <- shapes[as.numeric(data_mqr_long$quantile)]
  
  s3d <- scatterplot3d(  data_cqr_long[,c(1,2,4)], color=colors, pch=shapes, angle=45, main="D-CQR", scale.y = .3,
                         grid=TRUE, box=FALSE, col.grid = "gray", zlab="log_precipitation", zlim=c(-0.3, 7))
  legend(s3d$xyz.convert(4.5, 2.2, 7.5), legend = expression(tau[3]==0.3, tau[5]==0.5, tau[7]==0.7),
         col =  c("black", "blue", "red"), pch = c(19,17,18), cex=1.0)
  

  ###################################################################
  # 3. Group Lasso Penalized Deep Composite Quantile Regression (GD-CQR)
  ###################################################################
  model=3
  result3=0
  lr_rate <- 0.007
  nepoch <- 3000
  node_1 <- Bsize <- 10; node_2 <- 10
  n.start <- 5
  Criteria.mod3 <- Criteria <- Inf;
  Best_i<-Best_mod3<- 0; 
  Weis <- rep(1, (p+1))
  
  #############
  #tuning gamma
  #############
  gamma_grid <- exp(seq(-3.5, -1, by=0.5))
  gamma_mat_pnmqr <- matrix(NA, ncol=length(gamma_grid), nrow = n.start)
  rownames(gamma_mat_pnmqr) <- 1:n.start
  colnames(gamma_mat_pnmqr) <- gamma_grid
  
  for(uu in 1:length(gamma_grid)){
    for(i in 1:n.start){
      set.seed(1365*uu*i)
      a1=-1; b1=1
      W1_ini=matrix(runif(Bsize*(p+1), a1, b1), nrow=Bsize, ncol=(p+1)) #size adjusting
      
      a2=-1; b2=1
      W2_ini=matrix(runif(node_2*(node_1+1), a2, b2), nrow=node_2, ncol=node_1+1)
      
      a3=-1; b3=1
      W0_ini=matrix(runif(1*node_2, a3, b3), nrow=1, ncol=node_2)
      B0_ini=matrix(runif(K, a3, b3), nrow=1, ncol=K);
      B0_ini=B0_ini[order(B0_ini)]
      
      result <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train), y=Y.train, Epoch=nepoch, Bsize=Bsize, rate=lr_rate,
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
    start_time_gdcqr <- Sys.time()
    
    mod3 <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train), y=Y.train, Epoch=nepoch, Bsize=Bsize, rate=lr_rate,
                              X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set, gamma=best_gamma_pnmqr, Weights = Weis)
    end_time_gdcqr <- Sys.time()
    time_mat[rep, model] <-   as.numeric(end_time_gdcqr - start_time_gdcqr, units = "secs")
    
    criteria=min(mod3$MPE_val)
    if(criteria < Criteria.mod3){
      result.mod3 <- mod3; Criteria.mod3=criteria; Best_mod3=i
    }
  } # END i  
  
  plot(1:length(result.mod3$MPE_val), result.mod3$MPE_train, cex=.5, type='l', ylim=c(min(result.mod3$MPE_train)-.1, max(result.mod3$MPE_train)+.1 ), col="blue", main="GD-CQR")
  lines(result.mod3$MPE_val, cex=.5,  col="red");
  
  W1hat_GDCQR <- result.mod3$Weight1
  W0hat_GDCQR <- result.mod3$Weight0
  B0hat_GDCQR <- result.mod3$B0
  
  pred_model3 <- cCheck2_predict_test_grlasso(W1=W1hat_GDCQR, W2=result.mod3$Weight2, W0=result.mod3$Weight0, B0=B0hat_GDCQR, X=cbind(1, X.test), y=Y.test, Taus=tau.set, gamma=best_gamma_pnmqr, ww=W1hat_GDCQR)$Pred
  mae.pdmqr <- abs(Y.test-t(pred_model3))
  MAE[rep, model] <- mean(colMeans(mae.pdmqr))
  
  
  # Model selection
  W1.hat.A <- round(W1hat_GDCQR[,-1],3)
  #var.val <- matrix(as.numeric(W1.hat.A!=0),ncol=p)
  var.val.GDCQR <- matrix(ifelse(colMeans(abs(W1.hat.A)) <= 0.001*5, 0, 1),ncol=p)
  m.val <- colSums(var.val.GDCQR)
  NC.val[rep,model] <- sum((m.val*nc)!=0);
  NIC.val[rep,model] <- sum((m.val*nic)!=0);
  NT.val[rep,model] <- ifelse(NC.val[rep,model]==3 & NIC.val[rep,model]==0, 1, 0)
  
  var.val.GDCQR.mat[rep, ]  <- var.val.GDCQR
  W0.hat.GDCQR[[rep]] <- W0hat_GDCQR
  W1.hat.GDCQR[[rep]] <- W1hat_GDCQR
  
  
  ###################################################################
  # 4. Adaptive GroupLasso Penalized Deep Composite Quantile Regression (AGD-CQR)
  ###################################################################
  model=4
  result4=0
  node_1 <- Bsize <- 10; node_2 <- 10
  lr_rate <- 0.005
  n.epoch <- 3000
  n.start <- 5
  
  
  #gamma <- 0.2  # L2 norm penalty
  Criteria.mod4 <- Criteria <- Inf;
  Best_mod4<- 0; 
  Weis <- glasso_weight(W_ini=W1hat_dcqr)
  
  
  #after tuning gamma
  gamma_grid <- exp(seq(-4, .5, by=0.5))
  gamma_mat_apnmqr <- matrix(, ncol=length(gamma_grid), nrow = n.start)
  colnames(gamma_mat_apnmqr) <- gamma_grid
  
  #############
  #tuning gamma
  #############
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
     plot(1:length(result.mod4$MPE_val), result.mod4$MPE_train, cex=.5, type='l', ylim=c(min(result.mod4$MPE_train)-.1, max(result.mod4$MPE_train)+.1 ), col="blue", main="AGD-CQR")
     lines(result.mod4$MPE_val, cex=.5,  col="red"); 
  }
  
  best_gamma_apnmqr <- gamma_grid[which.min(colMeans(gamma_mat_apnmqr))]
  
  #after tuning gamma
  Criteria.mod4 <- Criteria <- Inf;
  
  
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
    start_time_agdcqr <- Sys.time()
    mod4 <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train), y=Y.train, Epoch=nepoch, Bsize=Bsize, rate=lr_rate,
                              X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set, gamma=best_gamma_apnmqr, Weights = Weis)
    end_time_agdcqr <- Sys.time()
    time_mat[rep, model] <-   as.numeric(end_time_agdcqr - start_time_agdcqr, units = "secs")
    
    criteria=min(mod4$MPE_val)
    if(criteria < Criteria.mod4){
      result.mod4 <- mod4; Criteria.mod4=criteria; Best_mod4=i
    }
  }
  
  plot(1:length(result.mod4$MPE_val), result.mod4$MPE_train, cex=.5, type='l', ylim=c(min(result.mod4$MPE_train)-.1, max(result.mod4$MPE_train)+.1 ), col="blue", main="AGD-CQR")
  lines(result.mod4$MPE_val, cex=.5,  col="red");
  
  W1hat_AGDCQR <- result.mod4$Weight1
  W0hat_AGDCQR <- result.mod4$Weight0
  B0hat_AGDCQR <- result.mod4$B0
  
  pred_model4 <- cCheck2_predict_test_grlasso(W1=W1hat_AGDCQR, W2=result.mod4$Weight2, W0=result.mod4$Weight0, B0=B0hat_AGDCQR, X=cbind(1, X.test), y=Y.test, Taus=tau.set, gamma=best_gamma_apnmqr, ww=W1hat_AGDCQR)
  amae.pdmqr=abs(Y.test-t(pred_model4$Pred))
  MAE[rep, model] <- mean(colMeans(amae.pdmqr))
  
  
  # Model selection
  W1.hat.A <- round(W1hat_AGDCQR[,-1],3)
  var.val.AGDCQR <- matrix(ifelse(colMeans(abs(W1.hat.A)) <= 0.001*5, 0, 1),ncol=p)
  m.val <- colSums(var.val.AGDCQR)
  NC.val[rep,model] <- sum((m.val*nc)!=0);
  NIC.val[rep,model] <- sum((m.val*nic)!=0);
  NT.val[rep,model] <- ifelse(NC.val[rep,model]==3 & NIC.val[rep,model]==0, 1, 0)
  
  var.val.AGDCQR.mat[rep, ]  <- var.val.AGDCQR
  W0.hat.AGDCQR[[rep]] <- W0hat_AGDCQR
  W1.hat.AGDCQR[[rep]] <- W1hat_AGDCQR
  
  ######
  #plot
  #######
  data_nmqr <- data.frame(sh700 = X.test[,1], z500= X.test[,2], tau0.3=pred_model4$Pred[3,], tau0.5=pred_model4$Pred[5,],tau0.7=pred_model4$Pred[7,])
  data_nmqr_long <- gather(data_nmqr, quantile, precipitation, tau0.3:tau0.7, factor_key = T)
  
  colors <- c("black", "blue", "red")
  colors <- colors[as.numeric(data_nmqr_long$quantile)]
  shapes = c(19,17,18) 
  shapes <- shapes[as.numeric(data_nmqr_long$quantile)]
  s3d <- scatterplot3d(  data_nmqr_long[,c(1,2,4)], color=colors, pch=shapes, angle=45, main="AGD-CQR", scale.y=.3, grid=TRUE, box=FALSE,
                         col.grid = "gray", zlab="log_precipitation")
  legend(s3d$xyz.convert(4.5, 2.2, 2.7), legend = expression(tau[3]==0.3, tau[5]==0.5, tau[7]==0.7),
         col =  c("black", "blue", "red"), pch =  c(19,17,18) , cex=1.0)
 
  
  print(paste("Replication :", rep))
  
  
} # End of rep



