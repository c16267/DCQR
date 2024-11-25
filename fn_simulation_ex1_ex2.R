rm(list=ls()) 
#setwd("/Users/shin/Dropbox/PhD/research/(241012)[JAS] DCQR/code")
#source("main_functions.r")
library(mvtnorm)
library(VGAM)
library(nor1mix)
library(ggplot2)
library(nor1mix)

################################################################################
# Fixed parameter setting
################################################################################
# Seed
seed.train=1234
seed.val=4567
seed.test=7891

# Kernel Parameters
# Lambda set
Lambda.set=exp(seq(-2, 0, .2))

# Sigma2 set
Sigma2.set=exp(seq(-1, 1 ,0.3))


################################################################################
# Error distribution 
################################################################################

# 1. normal
sigma.esp=1

# 4. T
DF=3

# 6. Exponential
Rate=1


# 3. gaussian mixture
obj <- norMix(mu=c(0,0), sigma = sqrt(c(25,1)), w=c(0.1,0.9))

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
rep.ini=5

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

MAE=matrix(,Replication,n.model);
colnames(MAE)=c("KRR", "CKQR", "  ", "D-LSE", "D-MQR", "D-CQR", "", "","","")

time_mat=matrix(,Replication,n.model)
colnames(time_mat)=c("KRR", "CKQR", "  ", "D-LSE", "D-MQR", "D-CQR", "", "","","")


###################################################################
for (rep in 1:Replication){
  ################################################################################
  # Reading Data
  ################################################################################
  # 1. Training Data
  set.seed(seed.train+rep)
  X.train=runif(n.train, -5, 5)
  #X.train=runif(n.train, -1, 1)
  
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
  
  Y.train=1+2*(1 - X.train + 2*X.train^2)*exp(-0.5*X.train^2)+Esp.train
  
  
  # 2. Validation Data
  set.seed(seed.val+rep)
  X.val=runif(n.val, -5, 5)
  
  
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
  
  
  Y.val=1+2*(1 - X.val + 2*X.val^2)*exp(-0.5*X.val^2)+Esp.val
  
  
  # 3. Test Data
  set.seed(seed.test+rep)
  X.test=runif(n.test, -5, 5)
  ord=order(X.test,decreasing = F)
  X.test=X.test[ord]
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

  if (err.dist=="chi"){for(k in 1:K){Esp.test[,k]=qchisq(tau.set[k],3)}; Esp.test=scale(Esp.test)}
  
  
  Y.test=matrix(,n.test,K)
  M.test=1+2*(1 - X.test + 2*X.test^2)*exp(-0.5*X.test^2)
  for(k in 1:K){Y.test[,k]=M.test+Esp.test[,k]}
  
  
  ################################################################################
  # 1. Kernel Ridge Regression (KRR)
  ################################################################################
  #tuning
  CV_krr_MSE <- matrix(, nrow=length(Lambda.set), ncol=length(Sigma2.set))
  model=1
  for(k in 1:length(Lambda.set)){
    for(s in 1:length(Sigma2.set)){
      model.sel.krr=Val.KRR2(X.T=X.train,Y.T=Y.train,X.V=X.val,Y.V=Y.val,L.S=Lambda.set[k],S2.S=Sigma2.set[s])
      y.test.krr=krr.predict(X.train, X.test, Model=model.sel.krr$Best.model)
      resid.krr=M.test-y.test.krr
      CV_krr_MSE[k,s]=mean(resid.krr^2)
    }
  }
  ind <- which(CV_krr_MSE==min(CV_krr_MSE), arr.ind=T)
  best_lambda <- Lambda.set[ind[1]]
  best_sigma <- Sigma2.set[ind[2]]
  
  #after tuning: fitting
  start_time_krr <- Sys.time()
  model.sel.krr=Val.KRR2(X.T=X.train,Y.T=Y.train,X.V=X.val,Y.V=Y.val,L.S=best_lambda, S2.S=best_sigma)
  end_time_krr <- Sys.time()
  time_mat[rep, model] <-   as.numeric(end_time_krr - start_time_krr, units = "secs")
  
  # mean ft.
  y.test.krr=krr.predict(X.train, X.test, Model=model.sel.krr$Best.model)
  resid.krr=M.test-y.test.krr
  MSE[rep,model]=mean(resid.krr^2)
  
  
  ################################################################################
  # 2. Kernel Composite Quantile Regression (KCQR)
  ################################################################################
  model=2
  #lambda, sigma tuning.
  CV_kcqr_MSE <- matrix(, nrow=length(Lambda.set), ncol=length(Sigma2.set))
  for(k in 1:length(Lambda.set)){
    for(s in 1:length(Sigma2.set)){
      model.sel.ckqr=Val.CKQR2(X.T=X.train,Y.T=Y.train,X.V=X.val,Y.V=Y.val,TAUS=tau.set,L.S=Lambda.set[k],S2.S=Sigma2.set[s])
      y.test.ckqr=ckqr.predict.M(X.train, X.test, Model=model.sel.ckqr$Best.model)
      resid.ckqr=M.test-y.test.ckqr[,2]
      CV_kcqr_MSE[k,s]=mean(resid.ckqr^2)
    }
  }
  ind <- which(CV_kcqr_MSE==min(CV_kcqr_MSE), arr.ind=T)
  best_lambda <- Lambda.set[ind[1]]
  best_sigma <- Sigma2.set[ind[2]]
  
  
  #Time checking
  start_time_ckqr <- Sys.time()
  model.sel.ckqr=Val.CKQR2(X.T=X.train,Y.T=Y.train,X.V=X.val,Y.V=Y.val,TAUS=tau.set,L.S=best_lambda,S2.S=best_sigma)
  end_time_ckqr <- Sys.time()
  time_mat[rep, model] <-   as.numeric(end_time_ckqr - start_time_ckqr, units = "secs")
  
  # mean ft.
  y.test.ckqr=ckqr.predict.M(X.train, X.test, Model=model.sel.ckqr$Best.model)
  resid.ckqr=M.test-rowMeans(y.test.ckqr)
  MSE[rep,model]=mean(resid.ckqr^2)
  
  # quantile ft.
  y.test.ckqr=ckqr.predict.Q(X.train, X.test, Model=model.sel.ckqr$Best.model)
  mae.ckqr=abs(Y.test-y.test.ckqr)
  MAE.ckqr[rep,]=colMeans(mae.ckqr)
  MAE[rep,model]=mean(mae.ckqr)
  
  
  #plotting
  colfunc<-colorRampPalette(c("red","yellow","springgreen","royalblue"))
  
  par(mar=c(5,5,5,5),oma=c(.5,.5,.5,.5))
  plot(X.test, y.test.ckqr[,1], main="KCQR",
       type='l', ylim=c(min(Y.test[,1])-0.5, max(Y.test[,9])+0.5), ylab=expression(Y[tau[k]]),
       xlab="X", lwd=2, lty=2, col=colfunc(10)[1])
  lines(X.test, y.test.ckqr[,2], col=colfunc(10)[2], lwd=2, lty=3)
  lines(X.test, y.test.ckqr[,3], col=colfunc(10)[3], lwd=2, lty=4)
  lines(X.test, y.test.ckqr[,4], col=colfunc(10)[4], lwd=2, lty=5)
  lines(X.test, y.test.ckqr[,5], col=colfunc(10)[5], lwd=2, lty=6)
  lines(X.test, y.test.ckqr[,6], col=colfunc(10)[7], lwd=2, lty=7)
  lines(X.test, y.test.ckqr[,7], col=colfunc(10)[8], lwd=2, lty=8)
  lines(X.test, y.test.ckqr[,8], col=colfunc(10)[9], lwd=2, lty=9)
  lines(X.test, y.test.ckqr[,9], col=colfunc(10)[10], lwd=2, lty=10)

  # lines(X.test, Y.test[,1], col="gray", lwd=1.5)
  # lines(X.test, Y.test[,2], col="gray", lwd=1.5)
  # lines(X.test, Y.test[,3], col="gray", lwd=1.5)
  # lines(X.test, Y.test[,4], col="gray", lwd=1.5)
  # lines(X.test, Y.test[,5], col="gray", lwd=1.5)
  # lines(X.test, Y.test[,6], col="gray", lwd=1.5)
  # lines(X.test, Y.test[,7], col="gray", lwd=1.5)
  # lines(X.test, Y.test[,8], col="gray", lwd=1.5)
  # lines(X.test, Y.test[,9], col="gray", lwd=1.5)

  legend("topright", legend=c(expression(tau[1 ]),expression(tau[2 ]),expression(tau[3 ]), expression(tau[4 ]),expression(tau[5 ]),expression(tau[6 ]),
                              expression(tau[7 ]),expression(tau[8 ]),expression(tau[9 ])),
         lwd=3, col=colfunc(10)[-6], lty=seq(2,10,1), cex=1.5 )
  grid(nx = NULL, ny = NULL,
       lty = 2,      # Grid line type
       col = "gray", # Grid line color
       lwd = 1)      # Grid line width


  ###################################################################
  # 4. Deep LSE (D-LSE)
  ###################################################################
  model=4
  lr_rate <- 0.05
  result4=0; Criteria=Inf; Best_i=0
  for(i in 1:rep.ini){
    Bsize <- node_1 <- 10; node_2 <- 10
    set.seed(1365*rep*i)
    a1=-1; b1=1
    W1_ini=matrix(runif(node_1*2, a1, b1), nrow=node_1, ncol=2)
    
    a2=-1; b2=1
    W2_ini=matrix(runif(node_2*(node_1+1), a2, b2), nrow=node_2, ncol=node_1+1)
    
    a3=-1; b3=1
    W0_ini=matrix(runif(1*(node_2+1), a3, b3), nrow=1, ncol=node_2+1)
    
    #Time consuming
    start_time_dlse <- Sys.time()
    result=mLSE2_val(W1=W1_ini, W2=W2_ini, W0=W0_ini, X=cbind(1,X.train), y=Y.train, Epoch=10000, Bsize, rate=lr_rate,
                     X_val=cbind(1, X.val), y_val=Y.val)
    end_time_dlse <- Sys.time()
    time_mat[rep, model] <-   as.numeric(end_time_dlse - start_time_dlse, units = "secs")
    criteria=min(result$MSE_val)
    if(criteria < Criteria){
      result4=result; Criteria=criteria; Best_i=i
    }
    
  } # END i
  
  # plot(1:length(result4$MSE_val), result4$MSE_train, cex=.5, type='l', ylim=c(min(result4$MSE_train)-1, max(result4$MSE_train)+1 ), col="blue", main="D-LSE")
  # lines(result4$MSE_val, cex=.5,  col="red"); #abline(v=c(100,200,300,400,500,600,700,800,900,1000),lty=2)

  result=result4
  y.test.dlse=mLSE2_predict(W1=result$Weight1, W2=result$Weight2, W0=result$Weight0,  X=cbind(1, X.test), y=M.test)
  MSE[rep,model]=(y.test.dlse$MSE)
  
  
  
  ###################################################################
  # 5. Deep Multiple Quantile Regression (D-MQR)
  ###################################################################
  model=5
  result5=0; Criteria=Inf; Best_i=0
  lr_rate <- 0.05
  for(i in 1:rep.ini){
    Bsize <- node_1 <- 10; node_2 <-10
    set.seed(1365*rep*i)
    a1=-1; b1=1
    W1_ini=matrix(runif(node_1*2, a1, b1), nrow=node_1, ncol=2)
    
    a2=-1; b2=1
    W2_ini=matrix(runif(node_2*(node_1+1), a2, b2), nrow=node_2, ncol=node_1+1)
    
    a3=-1; b3=1
    W0_ini=matrix(runif(K*(node_2+1), a3, b3), nrow=K, ncol=(node_2+1))
    
    start_time_dmqr <- Sys.time()
    result=check2_mqr_val(W1=W1_ini, W2=W2_ini, W0=W0_ini, X=cbind(1,X.train), y=Y.train, Epoch=5000, Bsize, rate=lr_rate,
                          X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set)
    
    end_time_dmqr <- Sys.time()
    time_mat[rep, model] <-   as.numeric(end_time_dmqr - start_time_dmqr, units = "secs")
    criteria=min(result$MPE_val)
    if(criteria < Criteria){
      result5=result; Criteria=criteria; Best_i=i
    }
    
  } # END i
  
  # plot(1:length(result5$MPE_val), result5$MPE_train, cex=.5, type='l', ylim=c(min(result5$MPE_train)-1, max(result5$MPE_train)+1 ), col="blue", main="D-MQR")
  # lines(result5$MPE_val, cex=.5,  col="red"); #abline(v=c(100,200,300,400,500,600,700,800,900,1000),lty=2)

  #quantile fit
  result=result5
  y.test.dmqr=mCheck2_predict_dmqr(W1=result$Weight1, W2=result$Weight2, W0=result$Weight0,  X=cbind(1, X.test), y=Y.test, Taus=tau.set)
  MAE.dmqr[rep,]=rowMeans(abs(y.test.dmqr$Resid))
  MAE[rep, model]=mean(rowMeans(abs(y.test.dmqr$Resid)))
  
  #plotting
  par(mar=c(5,5,5,5),oma=c(.5,.5,.5,.5))
  plot(X.test, y.test.dmqr$Pred[1,], main="D-MQR",
       type='l', ylim=c(min(Y.test[,1])-0.5, max(Y.test[,9])+0.5), ylab=expression(Y[tau[k]]),
       xlab="X", lwd=2, lty=2, col=colfunc(10)[1])
  lines(X.test, y.test.dmqr$Pred[2,], col=colfunc(10)[2], lwd=2, lty=3)
  lines(X.test, y.test.dmqr$Pred[3,], col=colfunc(10)[3], lwd=2, lty=4)
  lines(X.test, y.test.dmqr$Pred[4,], col=colfunc(10)[4], lwd=2, lty=5)
  lines(X.test, y.test.dmqr$Pred[5,], col=colfunc(10)[5], lwd=2, lty=6)
  lines(X.test, y.test.dmqr$Pred[6,], col=colfunc(10)[7], lwd=2, lty=7)
  lines(X.test, y.test.dmqr$Pred[7,], col=colfunc(10)[8], lwd=2, lty=8)
  lines(X.test, y.test.dmqr$Pred[8,], col=colfunc(10)[9], lwd=2, lty=9)
  lines(X.test, y.test.dmqr$Pred[9,], col=colfunc(10)[10], lwd=2, lty=10)


  legend("topright", legend=c(expression(tau[1 ]),expression(tau[2 ]),expression(tau[3 ]), expression(tau[4 ]),expression(tau[5 ]),expression(tau[6 ]),
                              expression(tau[7 ]),expression(tau[8 ]),expression(tau[9 ])),
         lwd=3, col=colfunc(10)[-6], lty=seq(2,10,1), cex=1.5 )
  grid(nx = NULL, ny = NULL,
       lty = 2,      # Grid line type
       col = "gray", # Grid line color
       lwd = 1)      # Grid line width
  # 
  # 
  ###################################################################
  # 6. Deep Composite Quantile Regression (D-CQR)
  ###################################################################
  model=6
  lr_rate <- 0.05
  result6=0; Criteria=Inf; Best_i=0
  
  for(i in 1:rep.ini){
    Bsize <- node_1 <- 10; node_2 <- 10
    set.seed(1365*rep*i)
    a1=-1; b1=1
    W1_ini=matrix(runif(node_1*2, a1, b1), nrow=node_1, ncol=2)
    
    a2=-1; b2=1
    W2_ini=matrix(runif(node_2*(node_1+1), a2, b2), nrow=node_2, ncol=node_1+1)
    
    a3=-1; b3=1
    W0_ini=matrix(runif(1*node_2, a3, b3), nrow=1, ncol=node_2)
    B0_ini=matrix(runif(K, a3, b3), nrow=1, ncol=K); B0_ini=B0_ini[order(B0_ini)]
    start_time_dcqr <- Sys.time()
    
    result=check2_cqr_val(W1=W1_ini, W2=W2_ini, W0=W0_ini, B0=B0_ini, X=cbind(1,X.train), y=Y.train, Epoch=7000, Bsize, rate=lr_rate,
                          X_val=cbind(1, X.val), y_val=Y.val, Taus=tau.set)
    end_time_dcqr <- Sys.time()
    time_mat[rep, model] <-   as.numeric(end_time_dcqr - start_time_dcqr, units = "secs")
    criteria=min(result$MPE_val)
    if(criteria < Criteria){
      result6=result; Criteria=criteria; Best_i=i
    }
    
  } # END i

  # plot(1:length(result6$MPE_val), result6$MPE_train, cex=.5, type='l', ylim=c(min(result6$MPE_train)-1, max(result6$MPE_train)+1 ), col="blue", main="DCQR")
  # lines(result6$MPE_val, cex=.5,  col="red");
   
    
  # mean ft.
  result=result6
  y.test.dcqr=mean_dcqr_predict(W1=result$Weight1, W2=result$Weight2, W0=result$Weight0, B0=result$WeightB0, X=cbind(1, X.test), y=M.test)
  MSE[rep,model]=y.test.dcqr$MSE
  
  
  # quantile ft.
  y.test.dcqr_qt=cCheck2_predict_qt(W1=result$Weight1, W2=result$Weight2, W0=result$Weight0, B0=result$WeightB0, X=cbind(1, X.test), y=Y.test, Taus=tau.set)
  mae.dcqr=abs(y.test.dcqr_qt$Resid)
  MAE.dcqr[rep,]=colMeans(mae.dcqr)
  MAE[rep, model]=mean(mae.dcqr)
  

  par(mar=c(5,5,5,5),oma=c(.5,.5,.5,.5))
  plot(X.test, y.test.dcqr_qt$Pred[1,], main="DCQR",
       type='l', ylim=c(min(Y.test[,1])-0.5, max(y.test.dcqr_qt$Pred[9,])+0.5), ylab=expression(Y[tau[k]]),
       xlab="X", lwd=2, lty=2, col=colfunc(5)[1])
  lines(X.test, y.test.dcqr_qt$Pred[2,], col=colfunc(10)[2], lwd=2, lty=3)
  lines(X.test, y.test.dcqr_qt$Pred[3,], col=colfunc(10)[3], lwd=2, lty=4)
  lines(X.test, y.test.dcqr_qt$Pred[4,], col=colfunc(10)[4], lwd=2, lty=5)
  lines(X.test, y.test.dcqr_qt$Pred[5,], col=colfunc(10)[5], lwd=2, lty=6)
  lines(X.test, y.test.dcqr_qt$Pred[6,], col=colfunc(10)[7], lwd=2, lty=7)
  lines(X.test, y.test.dcqr_qt$Pred[7,], col=colfunc(10)[8], lwd=2, lty=8)
  lines(X.test, y.test.dcqr_qt$Pred[8,], col=colfunc(10)[9], lwd=2, lty=9)
  lines(X.test, y.test.dcqr_qt$Pred[9,], col=colfunc(10)[10], lwd=2, lty=10)

  legend("topright", legend=c(expression(tau[1 ]),expression(tau[2 ]),expression(tau[3 ]), expression(tau[4 ]),expression(tau[5 ]),expression(tau[6 ]),
                              expression(tau[7 ]),expression(tau[8 ]),expression(tau[9 ])),
         lwd=3, col=colfunc(10)[-6], lty=seq(2,10,1), cex=1.5 )
  grid(nx = NULL, ny = NULL,
       lty = 2,      # Grid line type
       col = "gray", # Grid line color
       lwd = 1)      # Grid line width

  
  print(paste("Replication :", rep))
  
} 


