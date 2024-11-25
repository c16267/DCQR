

####################################################################
# Sigmoid
####################################################################

Sigmoid=function(x){1/(1+exp(-x))}

obj_penalty=function(W_iter, Wei){
  
  W_iter2=W_iter^2
  W_iter2_sum=colSums(W_iter2)
  Wei[1]=0
  obj_p=sum(Wei*sqrt(W_iter2_sum))
  
  return(obj_p)
}

glasso_weight=function(W_ini){
  
  W_ini2=W_ini^2
  W_ini2_sum=colSums(W_ini2)
  return(1/(sqrt(W_ini2_sum)))
  
}

glasso_penalty=function(W_iter, Wei){
  
  W_iter2=W_iter^2
  W_iter2_sum=colSums(W_iter2)
  norms=(sqrt(W_iter2_sum))
  
  for(i in 1:ncol(W_iter)){W_iter[,i]=(Wei[i]*W_iter[,i])/norms[i]}
  return(W_iter)
  
}
####################################################################
# Deep-mean regression functions (D-LSE)
####################################################################
mLSE2_predict=function(W1, W2, W0, X, y){
  
  # ??????
  z1=W1%*%t(X);
  y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
  z2=W2%*%y1_add
  y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
  z0=W0%*%y2_add
  y0_hat=(z0)    # linear activate ft.
  
  pred=t(y0_hat);
  resid=y-pred
  cmse=colMeans(resid^2)
  mse=mean(cmse)
  
  result=list(Pred=pred, Resid=resid, CMSE=cmse, MSE=mse)
  return(result)
}

mLSE2_val <- function(W1, W2, W0, X, y, Epoch, Bsize, rate, X_val, y_val){
  W1 <- as.matrix(W1);
  old_W1 <- W1
  W2=as.matrix(W2);W0=as.matrix(W0); 
  X=as.matrix(X); y=as.matrix(y);                     # mini-batch?? ?????ϰ?
  X_ini=X; y_ini=y                                    # pred???? ????
  X_val=as.matrix(X_val); y_val=as.matrix(y_val)      # validation 
  n_Epoch=Epoch; n_Bsize=Bsize; eta=rate
  n=nrow(X); n_iter=ceiling(n/n_Bsize); 
  
  mse_train=c(); mse_val=c()
  mse_cri=Inf

  for(e in 1:n_Epoch){
    n_start=1;n_end=n_Bsize
    
    # epoch ???? ???? ????ġ
    set.seed(1365+e)
    idx=sample(1:n)
    X=X[idx,];y=y[idx]  
    
    for(t in 1:n_iter){
      
      # ??ġ ??????
      X_batch=X[n_start:n_end,];y_batch=y[n_start:n_end]
      n_batch=nrow(X_batch)
      
      
      # ??????
      z1=W1%*%t(X_batch);
      y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
      z2=W2%*%y1_add
      y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
      z0=W0%*%y2_add
      y0_hat=(z0)    # linear activate ft.
      
      
      # ?????? : ????, ??Ÿ ????, ??????
      e0=t(y_batch) - y0_hat
      delta0=(e0)    # linear activate ft.
      
      dW0=eta*delta0%*%t(y2_add)/n_batch        # ??Ÿ*??Ÿ*?Է?
      W0 = W0+dW0
      
      # ��????(2) : ????, ??Ÿ ????, ??????
      e2=t(W0)%*%delta0;e2_del=e2[-1,]
      delta2=(y2_hat*(1-y2_hat)*e2_del)
      
      dW2=eta*delta2%*%t(y1_add)/n_batch
      W2 = W2+dW2
      
      
      # ��????(1) : ????, ??Ÿ ????, ??????
      e1=t(W2)%*%delta2;e1_del=e1[-1,]
      delta1=(y1_hat*(1-y1_hat)*e1_del)
      
      dW1=eta*delta1%*%(X_batch)/n_batch
      W1 =W1+dW1
      
      # batch step + 
      n_start=n_start+n_Bsize
      n_end=ifelse(t==n_iter-1, n, n_Bsize*(t+1))
      
    } # END t  
    
    mse_train[e]=mLSE2_predict(W1, W2, W0, X=X_ini, y=y_ini)$MSE
    mse_val[e]  =mLSE2_predict(W1, W2, W0, X=X_val, y=y_val)$MSE
    
    if(mse_val[e]<=mse_cri){
      mse_cri=mse_val[e]
      best_epoch=e; 
      W1_final=W1; W2_final=W2; W0_final=W0
    } # End if
    
   
    del <- max(abs(W1[,2]-old_W1[,2])/abs(old_W1[,2]))
    if(del < 10^-5) {
      break
    }
    old_W1 <- W1
    #print(paste("epochs:",e, "del", del))
  }
  result=list(Weight1=W1_final, Weight2=W2_final, Weight0=W0_final, 
              MSE_train=mse_train, MSE_val=mse_val, Best_epoch=best_epoch)
  return(result)
}# END mLSE2_val


################################################################################
# Deep-multiple quantile regression functions (D-MQR)
################################################################################
mCheck2_predict=function(W1, W2, W0, X, y, Taus){

  # ??????
  taus=Taus
  z1=W1%*%t(X);
  y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
  z2=W2%*%y1_add
  y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
  z0=W0%*%y2_add
  y0_hat=(z0)    # linear activate ft.
  pred=(y0_hat);
  resid=pred;
  for(i in 1:nrow(resid)){resid[i,]=(y-pred[i,])*(taus[i]-as.numeric(y-pred[i,]<0))}
  cmpe=rowMeans(resid)  # col mean prediction error for check loss
  mpe=mean(cmpe)        #  mean prediction error for check loss

  result=list(Pred=pred, Resid=resid, CMPE=cmpe, MPE=mpe)
  return(result)
}


mCheck2_predict_dmqr=function(W1, W2, W0, X, y, Taus){
  
  # ??????
  taus=Taus
  z1=W1%*%t(X);
  y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
  z2=W2%*%y1_add
  y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
  z0=W0%*%y2_add
  y0_hat=(z0)    # linear activate ft.
  pred=(y0_hat);
  resid=pred;
  #for(i in 1:nrow(resid)){resid[i,]=(y[,i]-pred[i,])*(taus[i]-as.numeric(y[,i]-pred[i,]<0))}
  for(i in 1:nrow(resid)){resid[i,]=(y[,i]-pred[i,])}
  cmpe=rowMeans(resid)  # col mean prediction error for check loss
  mpe=mean(cmpe)        #  mean prediction error for check loss
  
  result=list(Pred=pred, Resid=resid, CMPE=cmpe, MPE=mpe)
  return(result)
}

############################
################
check2_mqr_val=function(W1, W2, W0, X, y, Epoch, Bsize, rate, X_val, y_val, Taus){
  
  W1=as.matrix(W1);old_W1 <- W1
  W2=as.matrix(W2);W0=as.matrix(W0); 
  X=as.matrix(X); y=as.vector(y)                       # mini-batch?? ?????ϰ?  
  X_ini=X; y_ini=y                                     # prediction???? ????
  X_val=as.matrix(X_val); y_val=as.matrix(y_val)       # validation
  n_Epoch=Epoch; n_Bsize=Bsize; eta=rate
  n=nrow(X); n_iter=ceiling(n/n_Bsize); 
  min_epoch=2000                                      # stop ��?? ?߰?
  taus=Taus
  
  mpe_train=c(); mpe_val=c()
  mpe_cri=Inf
  diff_max=c(); WS1=W1                                 # stop ��?? ?߰?
  
  for(e in 1:n_Epoch){
    n_start=1;n_end=n_Bsize
    
    # epoch ???? ???? ????ġ
    set.seed(1365+e)
    idx=sample(1:n)
    X=X[idx,];y=y[idx]  
    
    for(t in 1:n_iter){
      
      # ??ġ ??????
      X_batch=X[n_start:n_end,];y_batch=y[n_start:n_end]
      n_batch=nrow(X_batch)
      
      
      # ??????
      z1=W1%*%t(X_batch);                             # 1??° hidden layer?????? ?????? ????, ??:??????, ??:?????? ??   
      y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)      # y1_add : hidden layer???? bias node ?߰?, ??:??????+1
      z2=W2%*%y1_add                                  # 2??° hidden layer?????? ?????? ????, ??:??????, ??:?????? ??
      y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)      # activate ft : sigmoid
      z0=W0%*%y2_add                                  # ???? layer?????? ?????? ????, ??:??????, ??:?????? ??
      y0_hat=(z0)                                     # ȸ?͸????????? linear activate ft. ????
      
      # ?????? : ????, ??Ÿ ????, ??????
      e0=check_error(YBAT=y_batch, YHAT=y0_hat, 
                     TAUS=taus)                       # [tau-I(y-yhat<0)] : ??Ʈ ????~~
      delta0=(e0)                                     # linear activate ft.
      
      dW0=eta*delta0%*%t(y2_add)/n_batch              # ??Ÿ*??Ÿ*?Է? : (?????? x ?????ͼ?) X (?????ͼ? x ??????)
      W0 = W0+dW0
      
      # ��????(2) : ????, ??Ÿ ????, ??????     
      e2=t(W0)%*%delta0; e2_del=e2[-1,]               # error ???? : bias ��??, ?????? x ?????ͼ?
      delta2=(y2_hat*(1-y2_hat)*e2_del)               # sigmoid activate ft.,   ?????? x ?????ͼ?
      
      dW2=eta*delta2%*%t(y1_add)/n_batch              # ??Ÿ*??Ÿ*?Է? : (?????? x ?????ͼ?) X (?????ͼ? x ??????)
      W2 = W2+dW2
      
      
      # ��????(1) : ????, ??Ÿ ????, ??????
      e1=t(W2)%*%delta2;e1_del=e1[-1,]                # error ???? : bias ��??, ?????? x ?????ͼ?
      delta1=(y1_hat*(1-y1_hat)*e1_del)
      
      dW1=eta*delta1%*%(X_batch)/n_batch
      W1 =W1+dW1
      
      # batch step + 
      n_start=n_start+n_Bsize
      n_end=ifelse(t==n_iter-1, n, n_Bsize*(t+1))
      
    } # END t
    
    mpe_train[e]=mCheck2_predict(W1, W2, W0, X=X_ini, y=y_ini, Taus=taus)$MPE   # mean prediction error for check loss
    mpe_val[e]  =mCheck2_predict(W1, W2, W0, X=X_val, y=y_val, Taus=taus)$MPE
    
    if(mpe_val[e]<=mpe_cri){
      mpe_cri=mpe_val[e]
      best_epoch=e; 
      W1_final=W1; W2_final=W2; W0_final=W0
    } # End if
    
    del <- max(abs(W1[,2]-old_W1[,2])/abs(old_W1[,2]))
    if(del < 10^-5) {
      break
    }
    old_W1 <- W1
    #print(paste("epochs:",e, "del", del))
    
    # stop ��?? ?߰?
    #diff_max[e]=max(abs(WS1-W1))                      # stop ��?? ?߰?
    #WS1=W1                                            # stop ��?? ?߰?   
    #if(e>min_epoch & diff_max[e]<1e-4){break}         # stop ��?? ?߰?
    #print(e)
    
  } # END e
  
  result=list(Weight1=W1_final, Weight2=W2_final, Weight0=W0_final, 
              MPE_train=mpe_train, MPE_val=mpe_val, Best_epoch=best_epoch)
  return(result)
  
  
} # END check2_mqr_val



################################################################################
# Deep-composite quantile regression functions (DCQR)
################################################################################
check_error=function(YBAT, YHAT, TAUS){

  E=YHAT
  for(i in 1:nrow(YHAT)){E[i,]=TAUS[i]-as.numeric(YBAT-YHAT[i,]<0)}

  return(E)

}

mean_dcqr_predict=function(W1, W2, W0, B0, X, y){
  
  # ??????
  z1=W1%*%t(X);
  y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
  z2=W2%*%y1_add
  y2_hat=Sigmoid(z2); y2_add=rbind(y2_hat)
  z0=W0%*%y2_add
  y0_hat=(z0)    # linear activate ft.
  mean_qt <- mean(B0)
  pred=t(y0_hat+mean_qt);
  #pred=t(y0_hat);
  resid=y-pred
  
  mse=mean(resid^2)
  
  result=list(Pred=pred, Resid=resid, MSE=mse)
  return(result)
}

cCheck2_predict=function(W1, W2, W0, B0, X, y, Taus){
  
  # ??????
  taus=Taus; Q=length(taus);z0=matrix(, Q, nrow(X))    ## cqr ??��
  z1=W1%*%t(X);
  y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
  z2=W2%*%y1_add
  y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
  
  zz0=W0%*%y2_hat                                 # ???? layer?????? ?????? ????, ??:??????, ??:?????? ??
  for(q in 1:Q){z0[q,]=B0[q]+zz0}
  y0_hat=(z0)                                     # ȸ?͸????????? linear activate ft. ????
  
  pred=(y0_hat);
  resid=pred
  for(i in 1:nrow(resid)){resid[i,]=(y-pred[i,])*(taus[i]-as.numeric(y-pred[i,]<0))}
  cmpe=colMeans(resid)  # col mean prediction error for check loss
  mpe=mean(cmpe)        #  mean prediction error for check loss
  
  result=list(Pred=pred, Resid=resid, CMPE=cmpe, MPE=mpe, Mpred=zz0)  # Mpred : mean ft. 
  return(result)
}

cCheck2_predict_qt=function(W1, W2, W0, B0, X, y, Taus){
  
  # ??????
  taus=Taus; Q=length(taus);z0=matrix(, Q, nrow(X))    ## cqr ??��
  z1=W1%*%t(X);
  y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
  z2=W2%*%y1_add
  y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
  
  zz0=W0%*%y2_hat                                 # ???? layer?????? ?????? ????, ??:??????, ??:?????? ??
  for(q in 1:Q){z0[q,]=B0[q]+zz0}
  y0_hat=(z0)                                     # ȸ?͸????????? linear activate ft. ????
  
  pred=(y0_hat);
  resid <- matrix(,ncol=length(taus),nrow=ncol(pred))
  #for(i in 1:ncol(resid)){resid[,i]=(y[,i]-pred[i,])*(taus[i]-as.numeric(y[,i]-pred[i,]<0))}
  for(i in 1:ncol(resid)){resid[,i]=(y[,i]-pred[i,])}
  cmpe=colMeans(resid)  # col mean prediction error for check loss
  mpe=mean(cmpe)        #  mean prediction error for check loss
  
  result=list(Pred=pred, Resid=resid, CMPE=cmpe, MPE=mpe, Mpred=zz0)  # Mpred : mean ft. 
  return(result)
}


check2_cqr_val=function(W1, W2, W0, B0, X, y, Epoch, Bsize, rate, X_val, y_val, Taus){
  
  W1=as.matrix(W1);
  W2=as.matrix(W2);W0=as.matrix(W0); 
  B0=as.vector(B0)             ## cqr ??��
  X=as.matrix(X); y=as.vector(y)                       # mini-batch?? ?????ϰ?  
  X_ini=X; y_ini=y                                     # prediction???? ????
  X_val=as.matrix(X_val); y_val=as.matrix(y_val)       # validation
  n_Epoch=Epoch; n_Bsize=Bsize; eta=rate
  n=nrow(X); n_iter=ceiling(n/n_Bsize); 
  min_epoch=5000                                      # stop ��?? ?߰?
  taus=Taus; Q=length(taus)     ## cqr ??�� 
  
  mpe_train=c(); mpe_val=c()
  mpe_cri=Inf
  diff_max=c(); WS1=W1; BS0 =B0                                 # stop ��?? ?߰?
  for(e in 1:n_Epoch){
    n_start=1;n_end=n_Bsize
    
    # epoch ???? ???? ????ġ
    set.seed(1365+e)
    idx=sample(1:n)
    X=X[idx,];y=y[idx]  
    
    for(t in 1:n_iter){
      
      # ??ġ ??????
      X_batch=X[n_start:n_end,];y_batch=y[n_start:n_end]
      n_batch=nrow(X_batch)
      z0=matrix(, Q, n_batch)    ## cqr ??��
      
      # ??????
      z1=W1%*%t(X_batch);                             # 1??° hidden layer?????? ?????? ????, ??:??????, ??:?????? ??   
      y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)      # y1_add : hidden layer???? bias node ?߰?, ??:??????+1
      z2=W2%*%y1_add                                  # 2??° hidden layer?????? ?????? ????, ??:??????, ??:?????? ??
      y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)      # activate ft : sigmoid
      
      zz0=W0%*%y2_hat                                 # ???? layer?????? ?????? ????, ??:??????, ??:?????? ??
      for(q in 1:Q){z0[q,]=B0[q]+zz0}
      y0_hat=(z0)                                     # ȸ?͸????????? linear activate ft. ????
      
      # ?????? : ????, ??Ÿ ????, ??????                                                                         ## cqr ??��
      e0=check_error(YBAT=y_batch, YHAT=y0_hat, 
                     TAUS=taus)                       # [tau-I(y-yhat<0)] : ??Ʈ ????~~
      ddelta0=(e0)                                     # linear activate ft.
      delta0=colSums(e0)
      
      dW0=eta*delta0%*%t(y2_hat)/n_batch              # ??Ÿ*??Ÿ*?Է? : (?????? x ?????ͼ?) X (?????ͼ? x ??????)
      W0 = W0+dW0
      
      dB0=eta*rowMeans(ddelta0) 
      B0 = B0+dB0

      # ��????(2) : ????, ??Ÿ ????, ??????     
      e2=t(W0)%*%delta0; e2_del=e2                    # error ???? : bias ��??, ?????? x ?????ͼ?                  ## cqr ??��
      delta2=(y2_hat*(1-y2_hat)*e2_del)               # sigmoid activate ft.,   ?????? x ?????ͼ?
      
      dW2=eta*delta2%*%t(y1_add)/n_batch              # ??Ÿ*??Ÿ*?Է? : (?????? x ?????ͼ?) X (?????ͼ? x ??????)
      W2 = W2+dW2
      
      
      # ��????(1) : ????, ??Ÿ ????, ??????
      e1=t(W2)%*%delta2;e1_del=e1[-1,]                # error ???? : bias ��??, ?????? x ?????ͼ?
      delta1=(y1_hat*(1-y1_hat)*e1_del)
      
      dW1=eta*delta1%*%(X_batch)/n_batch
      W1 =W1+dW1
      
      # batch step + 
      n_start=n_start+n_Bsize
      n_end=ifelse(t==n_iter-1, n, n_Bsize*(t+1))
      
    } # END t
    
    mpe_train[e]=cCheck2_predict(W1, W2, W0, B0, X=X_ini, y=y_ini, Taus=taus)$MPE   # mean prediction error for check loss
    mpe_val[e]  =cCheck2_predict(W1, W2, W0, B0, X=X_val, y=y_val, Taus=taus)$MPE
    
    if(mpe_val[e]<=mpe_cri){
      mpe_cri=mpe_val[e]
      best_epoch=e; 
      W1_final=W1; W2_final=W2; W0_final=W0; B0_final=B0                                               # cqr ??�� 
    } # End if

    del <- max(abs(W1[,2]-WS1[,2])/abs(WS1[,2]))
    del2 <- max(abs(B0-BS0)/abs(BS0))
    if(del < 10^-5 & del2 < 10^-5) {
      break
    }
    WS1 <- W1
    BS0 <- B0
    count <- 0
    if(e %% 100 == 0){
      count <- count + 1
    #print(paste("epochs:",e, "del1", del, "del2", del2))
      }
  } # END e
  
  result=list(Weight1=W1_final, Weight2=W2_final, Weight0=W0_final, WeightB0=B0_final, 
              MPE_train=mpe_train, MPE_val=mpe_val, Best_epoch=best_epoch)
  return(result)
  
  
} # END check2_cqr_val

#################################################################################
# penalized DCQR
#################################################################################

check2_cqr_glasso=function(W1, W2, W0, B0, X, y, Epoch, Bsize, rate, X_val, y_val, Taus, gamma, Weights){
  
  W1=as.matrix(W1);W2=as.matrix(W2); W0=as.matrix(W0); 
  B0=as.vector(B0)
  X=as.matrix(X); y=as.vector(y)                       
  X_ini=X; y_ini=y                                     
  X_val=as.matrix(X_val); y_val=as.matrix(y_val)       
  n_Epoch=Epoch; n_Bsize=Bsize; eta=rate
  n=nrow(X); n_iter=ceiling(n/n_Bsize); 
  min_epoch=3000                                
  weights=Weights
  mpe_train=c(); mpe_val=c()
  mpe_cri=10^4
  diff_max=c(); WS1=W1                                
  taus=Taus; Q=length(taus)
  
  for(e in 1:n_Epoch){
    n_start=1;n_end=n_Bsize
    
    set.seed(1365+e)
    idx=sample(1:n)
    X=X[idx,];y=y[idx]  
    
    for(t in 1:n_iter){
      
      X_batch=X[n_start:n_end,];y_batch=y[n_start:n_end]
      n_batch=nrow(X_batch)
      z0=matrix(, Q, n_batch)
      
      
      z1=W1%*%t(X_batch)
      y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
      z2=W2%*%y1_add
      y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
      zz0=W0%*%y2_hat                                 
      for(q in 1:Q){z0[q,]=B0[q]+zz0}
      y0_hat=(z0) 
      
      e0=check_error(YBAT=y_batch, YHAT=y0_hat,TAUS=Taus)
      ddelta0=(e0)                                    
      delta0=colSums(e0)
      
      ########################## diff of Penalty(L2 norm)############
      penalty = -gamma*glasso_penalty(W_iter=W1, Wei=weights); penalty[,1]=0
      dP3 <- penalty
      #################################################################
      
      dW0=eta*delta0%*%t(y2_hat)/n_batch      
      W0 = W0+dW0
      
      dB0=eta*rowMeans(ddelta0) 
      B0 = B0+dB0
      
      e2=t(W0)%*%delta0; e2_del=e2                    
      delta2=(y2_hat*(1-y2_hat)*e2_del)               
      
      dW2=eta*delta2%*%t(y1_add)/n_batch              
      W2 = W2+dW2
      
      e1=t(W2)%*%delta2;e1_del=e1[-1,]       
      delta1=(y1_hat*(1-y1_hat)*e1_del)
      
      loss <- delta1%*%(X_batch)/n_batch
      dW1 <- eta*(loss+dP3)
      W1 =W1+dW1
      
      n_start=n_start+n_Bsize
      n_end=ifelse(t==n_iter-1, n, n_Bsize*(t+1))
      
    } # END t
    
    mpe_train[e]=cCheck2_predict_grlasso(W1, W2, W0, B0, X=X_ini, y=y_ini, Taus, gamma, Weights)$MPE   
    mpe_val[e]  =cCheck2_predict_grlasso(W1, W2, W0, B0, X=X_val, y=y_val, Taus, gamma, Weights)$MPE
    
    if(mpe_val[e]<=mpe_cri){
      mpe_cri=mpe_val[e]
      best_epoch=e; 
      W1_final=W1; W2_final=W2; W0_final=W0; B0_final= B0
    } # End if
    
    delta <- mean(abs(W1 - WS1)/abs(WS1)) 
    if(e > min_epoch){
      if(delta < 1e-4 | mean(mpe_val[(e-200):e]) > mean(mpe_val[(e-600):(e-400)])
         |abs(mean(mpe_val[(e-200):e]) - mean(mpe_val[(e-600):(e-400)]) ) < 1e-4 )  {break}
      #if(delta < 1e-4)  {break}         
    }
    #print(paste("epochs:",e, "del", delta))
  } # END e
  
  result=list(Weight1=W1_final, Weight2=W2_final, Weight0=W0_final, B0=B0_final,
              MPE_train=mpe_train, MPE_val=mpe_val, Best_epoch=best_epoch)
  return(result)
} 

#############################################3

cCheck2_predict_grlasso <- function(W1, W2, W0, B0, X, y, Taus, gamma, Weights){
  weights <- Weights
  taus=Taus; Q=length(taus);z0=matrix(, Q, nrow(X))
  z1=W1%*%t(X);
  y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
  z2=W2%*%y1_add
  y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
  zz0=W0%*%y2_hat                                
  for(q in 1:Q){z0[q,]=B0[q]+zz0}
  y0_hat=(z0)  
  
  pred=(y0_hat);
  resid=pred;
  for(i in 1:nrow(resid)){resid[i,]=(y-pred[i,])*(taus[i]-as.numeric(y-pred[i,]<0))}
  


  ############################# Penalty part(L2 norm) ########################
  P=W1
  P2=matrix(, nrow=1, ncol=ncol(P))
  for(j in 1:ncol(P)){
    #P2[j]=(1/norm(ww[,j], type="2"))*norm(as.matrix(abs(P)[,j]), type="2")  #revison!
    P2[j]=obj_penalty(W_iter=W1, Wei=weights)
  }
  totP2=gamma*sum(P2)
  ###########################################################################
  
  cmpe=rowMeans(resid+totP2)  
  mpe=mean(cmpe)        
  
  result=list(Pred=pred, Resid=resid, CMPE=cmpe, MPE=mpe, Mpred=zz0)
  return(result)
}


###########################################

cCheck2_predict_test_grlasso=function(W1, W2, W0, B0, X, y, Taus, gamma, ww){
  taus=Taus; Q=length(taus);z0=matrix(, Q, nrow(X)) 
  z1=W1%*%t(X);
  y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
  z2=W2%*%y1_add
  y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
  zz0=W0%*%y2_hat                                 # ???? layer?????? ?????? ????, ??:??????, ??:?????? ??
  for(q in 1:Q){z0[q,]=B0[q]+zz0}
  y0_hat=(z0)                                     # ȸ?͸????????? linear activate ft. ????
  
  pred=(y0_hat);
  resid=pred;
  
  #for(rr in 1:nrow(resid)){resid[rr,] <- (y[,rr]-pred[rr,])*(Taus[rr]-as.numeric(y[,rr]-pred[rr,]<0))}
  for(rr in 1:nrow(resid)){resid[rr,] <- abs(y[,rr]-pred[rr,])}
  
  ############################# Penalty part(L2 norm) ########################
  P=W1
  P2=matrix(, nrow=1, ncol=ncol(P))
  for(j in 1:ncol(P)){
    P2[j]=(1/norm(ww[,j], type="2"))*norm(as.matrix(abs(P)[,j]), type="2")  #revison!
  }
  totP2=gamma*sum(P2)
  ##################################################################
  
  cmpe=rowMeans(resid+totP2)  # col mean prediction error for check loss
  mpe=mean(cmpe)     #  mean prediction error for check loss
  result=list(Pred=pred, Resid=resid, CMPE=cmpe, MPE=mpe)
  return(result)
}


#################################################################################
# penalized DMQR
#################################################################################
mCheck2_predict_grlasso <- function(W1, W2, W0, X, y, Taus, gamma, Weights){
  weights <- Weights
  
  taus=Taus
  z1=W1%*%t(X);
  y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
  z2=W2%*%y1_add
  y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
  z0=W0%*%y2_add
  y0_hat=(z0)    # linear activate ft.
  pred=(y0_hat);
  resid=pred;
  for(i in 1:nrow(resid)){resid[i,]=(y-pred[i,])*(taus[i]-as.numeric(y-pred[i,]<0))}
  
  ############################# Penalty part(L2 norm) ########################
  P=W1
  P2=matrix(, nrow=1, ncol=ncol(P))
  for(j in 1:ncol(P)){
    #P2[j]=(1/norm(ww[,j], type="2"))*norm(as.matrix(abs(P)[,j]), type="2")  #revison!
    P2[j]=obj_penalty(W_iter=W1, Wei=weights)
  }
  totP2=gamma*sum(P2)
  ###########################################################################
  
  cmpe=rowMeans(resid+totP2)  
  mpe=mean(cmpe) 
  
  result=list(Pred=pred, Resid=resid, CMPE=cmpe, MPE=mpe)
  return(result)
}
############################
mCheck2_predict_test_grlasso=function(W1, W2, W0, X, y, Taus, gamma, ww){
  taus=Taus; 
  z1=W1%*%t(X);
  y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)
  z2=W2%*%y1_add
  y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)
  z0=W0%*%y2_add
  y0_hat=(z0)    # linear activate ft.
  pred=(y0_hat);                                    # ȸ?͸????????? linear activate ft. ????
  
  resid=pred;
  
  #for(rr in 1:nrow(resid)){resid[rr,] <- (y[,rr]-pred[rr,])*(Taus[rr]-as.numeric(y[,rr]-pred[rr,]<0))}
  for(rr in 1:nrow(resid)){resid[rr,] <- abs(y[,rr]-pred[rr,])}
  
  ############################# Penalty part(L2 norm) ########################
  P=W1
  P2=matrix(, nrow=1, ncol=ncol(P))
  for(j in 1:ncol(P)){
    P2[j]=(1/norm(ww[,j], type="2"))*norm(as.matrix(abs(P)[,j]), type="2")  #revison!
  }
  totP2=gamma*sum(P2)
  ##################################################################
  
  cmpe=rowMeans(resid+totP2)  # col mean prediction error for check loss
  mpe=mean(cmpe)     #  mean prediction error for check loss
  result=list(Pred=pred, Resid=resid, CMPE=cmpe, MPE=mpe)
  return(result)
}

################
check2_mqr_glasso=function(W1, W2, W0, X, y, Epoch, Bsize, rate, X_val, y_val, Taus, gamma,  Weights){
  weights <- Weights
  W1=as.matrix(W1);old_W1 <- W1
  W2=as.matrix(W2);W0=as.matrix(W0); 
  X=as.matrix(X); y=as.vector(y)                       # mini-batch?? ?????ϰ?  
  X_ini=X; y_ini=y                                     # prediction???? ????
  X_val=as.matrix(X_val); y_val=as.matrix(y_val)       # validation
  n_Epoch=Epoch; n_Bsize=Bsize; eta=rate
  n=nrow(X); n_iter=ceiling(n/n_Bsize); 
  min_epoch=1000                                    # stop ��?? ?߰?
  taus=Taus
  
  mpe_train=c(); mpe_val=c()
  mpe_cri=Inf
  diff_max=c(); WS1=W1                                 # stop ��?? ?߰?
  
  for(e in 1:n_Epoch){
    n_start=1;n_end=n_Bsize
    
    # epoch ???? ???? ????ġ
    set.seed(1365+e)
    idx=sample(1:n)
    X=X[idx,];y=y[idx]  
    
    for(t in 1:n_iter){
      
      # ??ġ ??????
      X_batch=X[n_start:n_end,];y_batch=y[n_start:n_end]
      n_batch=nrow(X_batch)
      
      
      # ??????
      z1=W1%*%t(X_batch);                             # 1??° hidden layer?????? ?????? ????, ??:??????, ??:?????? ??   
      y1_hat=Sigmoid(z1); y1_add=rbind(1,y1_hat)      # y1_add : hidden layer???? bias node ?߰?, ??:??????+1
      z2=W2%*%y1_add                                  # 2??° hidden layer?????? ?????? ????, ??:??????, ??:?????? ??
      y2_hat=Sigmoid(z2); y2_add=rbind(1,y2_hat)      # activate ft : sigmoid
      z0=W0%*%y2_add                                  # ???? layer?????? ?????? ????, ??:??????, ??:?????? ??
      y0_hat=(z0)                                     # ȸ?͸????????? linear activate ft. ????
      
      # ?????? : ????, ??Ÿ ????, ??????
      e0=check_error(YBAT=y_batch, YHAT=y0_hat, 
                     TAUS=taus)                       # [tau-I(y-yhat<0)] : ??Ʈ ????~~
      delta0=(e0)                                     # linear activate ft.
      
      ########################## diff of Penalty(L2 norm)############
      
      penalty = -gamma*glasso_penalty(W_iter=W1, Wei=weights); penalty[,1]=0
      dP3 <- penalty
      #################################################################
      
      
      dW0=eta*delta0%*%t(y2_add)/n_batch              # ??Ÿ*??Ÿ*?Է? : (?????? x ?????ͼ?) X (?????ͼ? x ??????)
      W0 = W0+dW0
      
      # ��????(2) : ????, ??Ÿ ????, ??????     
      e2=t(W0)%*%delta0; e2_del=e2[-1,]               # error ???? : bias ��??, ?????? x ?????ͼ?
      delta2=(y2_hat*(1-y2_hat)*e2_del)               # sigmoid activate ft.,   ?????? x ?????ͼ?
      
      dW2=eta*delta2%*%t(y1_add)/n_batch              # ??Ÿ*??Ÿ*?Է? : (?????? x ?????ͼ?) X (?????ͼ? x ??????)
      W2 = W2+dW2
      
      
      # ��????(1) : ????, ??Ÿ ????, ??????
      e1=t(W2)%*%delta2;e1_del=e1[-1,]                # error ???? : bias ��??, ?????? x ?????ͼ?
      delta1=(y1_hat*(1-y1_hat)*e1_del)
      
      loss <- delta1%*%(X_batch)/n_batch
      dW1 <- eta*(loss+dP3)
      W1 =W1+dW1

      
      # batch step + 
      n_start=n_start+n_Bsize
      n_end=ifelse(t==n_iter-1, n, n_Bsize*(t+1))
      
    } # END t
    
    mpe_train[e]=mCheck2_predict_grlasso(W1, W2, W0, X=X_ini, y=y_ini, Taus,  gamma, Weights)$MPE   
    mpe_val[e]  =mCheck2_predict_grlasso(W1, W2, W0, X=X_val, y=y_val, Taus,  gamma,  Weights)$MPE
    
    if(mpe_val[e]<=mpe_cri){
      mpe_cri=mpe_val[e]
      best_epoch=e; 
      W1_final=W1; W2_final=W2; W0_final=W0
    } # End if
    
    delta <- mean(abs(WS1 - W1)/abs(W1)) 
    if(e > min_epoch){
      if(delta < 1e-4 | mean(mpe_val[(e-200):e]) > mean(mpe_val[(e-600):(e-400)])
         |abs(mean(mpe_val[(e-200):e]) - mean(mpe_val[(e-600):(e-400)]) ) < 1e-4 )  {break}
      #if(delta < 1e-4)  {break}         
    }
    #print(paste("epochs:",e, "del", delta))
    
  } # END e
  
  result=list(Weight1=W1_final, Weight2=W2_final, Weight0=W0_final, 
              MPE_train=mpe_train, MPE_val=mpe_val, Best_epoch=best_epoch)
  return(result)
  
  
} # END check2_mqr_val




#################################################################################
# Kernel Method
#################################################################################
library(lpSolve)
library(quadprog)
################################################################################
# Kernel Ridge Regression (KRR)
################################################################################
krr.eq = function(Covariate, Response, Kernel = "radial", Lambda = 1, Sigma2=1, Degree=2, Epsilon = 1e-6){
  # Input Data
  X=as.matrix(Covariate); Y=as.matrix(Response);
  kernel=Kernel; c.value=1/Lambda; sigma2=Sigma2; degree=Degree; epsilon=Epsilon
  n=nrow(X); 
  
  # compute kernel matrix
  KM = krr.kernel(X1=X, X2=X, Kernel=kernel,Sigma2=sigma2,Degree=degree)
  KM2 = (1/c.value)*diag(n)
  EQM = KM+KM2; EQM = cbind(1,EQM); EQM = rbind(1,EQM); EQM[1,1]=0
  bvec = rbind(0,Y) 
  
  sol = solve(EQM)%*%bvec
  sol0=sol
  b.hat=sol0[1];sol0=sol0[-1]
  beta.hat=sol0[1:n];sol0=sol0[-(1:n)]
  
  # prepare output
  result = list(b = b.hat, beta=beta.hat,Kernel=kernel,Sigma2=sigma2,Degree=degree)
  return(result)
}

################################################################################
krr.kernel = function(X1, X2, Kernel = "radial", Sigma2=1, Degree=2) {
  x=as.matrix(X1);u=as.matrix(X2);ker=Kernel;sigma2=Sigma2;degree=Degree
  if (ker == "linear"){ K = (x %*% t(u))}
  if (ker == "poly")  {	K = (1 + x %*% t(u))^degree }
  if (ker == "radial")	{
    a = as.matrix(apply(x^2, 1, 'sum'))
    b = as.matrix(apply(u^2, 1, 'sum'))
    one.a = matrix(1, ncol = nrow(b))      
    one.b = matrix(1, ncol = nrow(a))
    K1 = one.a %x% a
    K2 = x %*% t(u)
    K3 = t(one.b %x% b)
    #K = exp(-gamma*(K1 - 2 * K2 + K3))
    K = exp(-(K1 - 2 * K2 + K3)/(2*sigma2))
  }
  return(K)
}    

################################################################################
krr.predict = function(X, new.X, Model){
  x = as.matrix(X); new.x = as.matrix(new.X); model=Model
  K = krr.kernel(X1=x, X2=new.x, Kernel=model$Kernel,Sigma2=model$Sigma2,Degree=model$Degree)
  y.hat = model$b + t(K)%*%model$beta
  return(y.hat)   
}

################################################################################
Val.KRR2=function(X.T, Y.T, X.V, Y.V, L.S, S2.S=0.5, KER="radial"){
  # Input data
  x.train=X.T; y.train=Y.T; x.val=X.V; y.val=Y.V; l.set=L.S; s2.set=S2.S; ker=KER
  ain=nrow(x.train); n.val=nrow(x.val); n.s2=length(s2.set); n.l=length(l.set)
  
  # Storiage
  s.loss=c();upper=Inf
  for (t in 1:n.l){
    # Validation
    model=krr.eq(Covariate=x.train,Response=y.train,Kernel=ker,Lambda=l.set[t],Sigma2=s2.set)      
    y.val.hat=krr.predict(x.train, x.val, Model=model)
    resid=y.val-y.val.hat;
    s.loss[t]=t(resid)%*%resid
    if(s.loss[t]<=upper){upper=s.loss[t]; best.lambda=l.set[t]; best.sigma2=s2.set; best.model=model}
  } # End of lambda
  
  # Reporting   
  result = list(Best.lambda=best.lambda, Best.sigma2=best.sigma2, Best.model=best.model)
  return(result)
}

################################################################################
# Kernel Quantile Regression (KQR)
################################################################################
kqr.qp = function(Covariate, Response, Tau, Kernel = "radial", Lambda = 1, Sigma2=1, Degree=2, Epsilon = 1e-6){
  # Input Data
  X=as.matrix(Covariate); Y=as.matrix(Response); tau=Tau; 
  kernel=Kernel; lambda=Lambda; sigma2=Sigma2; degree=Degree; epsilon=Epsilon
  n=nrow(X); n.var=(1+n)+n+n
  
  # compute kernel matrix
  KM = kqr.kernel(X1=X, X2=X, Kernel=kernel,Sigma2=sigma2,Degree=degree)
  Z.KM=rbind(0,cbind(0,KM)); O.KM=cbind(1,KM)
  
  # prepare QP
  dvec = -c(rep(0,n+1),rep(tau,n),rep((1-tau),n))
  Dmat = matrix(0,n.var,n.var);	Dmat[1:(n+1),1:(n+1)]=Z.KM
  Dmat=2*lambda*Dmat; diag(Dmat)=diag(Dmat)+epsilon
  Amat.equal = cbind(O.KM,diag(1,n),-diag(1,n))
  Amat.greater=cbind(matrix(0,2*n,n+1),diag(1,2*n))
  Amat = rbind(Amat.equal,Amat.greater);Amat=t(Amat)
  bvec = c(Y, rep(0, 2*n))
  # find alpha by QP
  sol = solve.QP(Dmat, dvec, Amat, bvec, meq = n, factorized = F)$solution
  
  # compute the index and the number of support vectors
  sol0=sol
  b.hat=sol0[1];sol0=sol0[-1]
  beta.hat=sol0[1:n];sol0=sol0[-(1:n)]
  u.hat=sol0[1:n];sol0=sol0[-(1:n)]
  v.hat=sol0[1:n]
  
  # prepare output
  result = list(b = b.hat, beta=beta.hat, Kernel=kernel,Sigma2=sigma2,Degree=degree)
  return(result)
}

################################################################################
kqr.kernel = function(X1, X2, Kernel = "radial", Sigma2=1, Degree=2) {
  x=as.matrix(X1);u=as.matrix(X2);ker=Kernel;sigma2=Sigma2;degree=Degree
  if (ker == "linear"){ K = (x %*% t(u))}
  if (ker == "poly")  {	K = (1 + x %*% t(u))^degree }
  if (ker == "radial")	{
    a = as.matrix(apply(x^2, 1, 'sum'))
    b = as.matrix(apply(u^2, 1, 'sum'))
    one.a = matrix(1, ncol = nrow(b))      
    one.b = matrix(1, ncol = nrow(a))
    K1 = one.a %x% a
    K2 = x %*% t(u)
    K3 = t(one.b %x% b)
    #K = exp(-gamma*(K1 - 2 * K2 + K3))
    K = exp(-(K1 - 2 * K2 + K3)/(2*sigma2))
  }
  return(K)
}    

################################################################################
kqr.predict = function(X, new.X, Model){
  x = as.matrix(X); new.x = as.matrix(new.X); model=Model
  K = kqr.kernel(X1=x, X2=new.x, Kernel=model$Kernel,Sigma2=model$Sigma2,Degree=model$Degree)
  y.hat = model$b + t(K)%*%model$beta
  return(y.hat)   
}

################################################################################
Val.KQR2=function(X.T, Y.T, X.V, Y.V, TAU, L.S, S2.S=0.5, KER="radial"){
  # Input data
  x.train=X.T; y.train=Y.T; x.val=X.V; y.val=Y.V; tau=TAU; l.set=L.S; s2.set=S2.S; ker=KER
  n.train=nrow(x.train); n.val=nrow(x.val); n.s2=length(s2.set); n.l=length(l.set)
  
  # Storiage
  s.loss=c();upper=Inf
  for (t in 1:n.l){
    # Validation
    model=kqr.qp(Covariate=x.train,Response=y.train,Tau=tau,Kernel=ker,Lambda=l.set[t],Sigma2=s2.set)      
    y.val.hat=kqr.predict(x.train, x.val, Model=model) ; 
    resid=y.val-y.val.hat;pos.index=which(resid>0)
    if(length(pos.index)>0){s.loss[t]=sum(resid[pos.index])*tau+sum(resid[-pos.index])*(tau-1)
    } else {s.loss[t]=sum(resid*(tau-1))
    }        
    if(s.loss[t]<=upper){upper=s.loss[t]; best.lambda=l.set[t]; best.sigma2=s2.set; best.model=model}
  } # End of lambda
  
  # Reporting   
  result = list(Best.lambda=best.lambda, Best.sigma2=best.sigma2, Best.model=best.model)
  return(result)
}

################################################################################
# Kernel Composite Quantiles Regression funcitons (KCQR)
################################################################################
ckqr.qp = function(Covariate, Response, Taus, Kernel = "radial", Lambda = 1, Sigma2=1, Degree=2, Epsilon = 1e-6){
  # Input Data
  X=as.matrix(Covariate); Y=as.matrix(Response); taus=Taus; 
  kernel=Kernel; lambda=Lambda; sigma2=Sigma2; degree=Degree; epsilon=Epsilon
  n=nrow(X); q=length(taus); n.var=(q+n)+(2*n*q)
  
  # compute kernel matrix
  KM = kqr.kernel(X1=X, X2=X, Kernel=kernel,Sigma2=sigma2,Degree=degree)
  OZ=kronecker(diag(q),rep(1,n));
  # prepare QP
  dvec = c(rep(0,(q+n)))
  for(k in 1:q){dvec=c(dvec,rep(taus[k],n),rep((1-taus[k]),n))}
  dvec=-dvec  
  Dmat = matrix(0,n.var,n.var);	Dmat[(q+1):(q+n),(q+1):(q+n)]=KM
  Dmat=2*lambda*Dmat; diag(Dmat)=diag(Dmat)+epsilon
  Amat.equal1 = cbind(OZ,kronecker(rep(1,q),KM)) 
  Amat.equal2 = kronecker(diag(q),cbind(diag(1,n),-diag(1,n)))
  Amat.equal = cbind(Amat.equal1,Amat.equal2) 
  Amat.greater=cbind(matrix(0,2*n*q,(n+q)),kronecker(diag(2*q),diag(n)))
  Amat = rbind(Amat.equal, Amat.greater); Amat=t(Amat)
  bvec = c(rep(Y,q),rep(0, 2*n*q))
  # find alpha by QP
  #time consuming
  sol = solve.QP(Dmat, dvec, Amat, bvec, meq = n*q, factorized = F)$solution
  
  # compute the index and the number of support vectors
  sol0=sol;
  b.hat=sol[1:q];sol0=sol0[-(1:q)]
  beta.hat=sol0[1:n];sol0=sol0[-(1:n)]
  u.hat=matrix(,n,q);v.hat=matrix(,n,q)
  for(k in 1:q){  
    u.hat[,k]=sol0[1:n];sol0=sol0[-(1:n)]
    v.hat[,k]=sol0[1:n];sol0=sol0[-(1:n)]
  }
  b.hat1=mean(b.hat)
  resid=Y-t(KM)%*%beta.hat
  b.hat2=mean(resid)
  
  # prepare output
  result = list(b=b.hat,b1=b.hat1,b2=b.hat2, beta=beta.hat,Kernel=kernel,Sigma2=sigma2,Degree=degree)
  return(result)
}
################################################################################
ckqr.predict.M = function(X, new.X, Model){
  x = as.matrix(X); new.x = as.matrix(new.X); model=Model; q=length(model$b)
  K = krr.kernel(X1=x, X2=new.x, Kernel=model$Kernel,Sigma2=model$Sigma2,Degree=model$Degree)
  y.hat1 = model$b1 + t(K)%*%model$beta
  y.hat2 = model$b2 + t(K)%*%model$beta
  y.hat=cbind(y.hat1,y.hat2)
  return(y.hat)   
}
################################################################################
ckqr.predict.Q = function(X, new.X, Model){
  x = as.matrix(X); new.x = as.matrix(new.X); model=Model; q=length(model$b)
  K = krr.kernel(X1=x, X2=new.x, Kernel=model$Kernel,Sigma2=model$Sigma2,Degree=model$Degree)
  y.hat=matrix(,nrow(new.x),q)
  for(k in 1:q){y.hat[,k] = model$b[k] + t(K)%*%model$beta }
  return(y.hat)   
}

################################################################################
Val.CKQR2=function(X.T, Y.T, X.V, Y.V, TAUS, L.S, S2.S=0.5, KER="radial"){
  # Input data
  x.train=X.T; y.train=Y.T; x.val=X.V; y.val=Y.V; taus=TAUS; l.set=L.S; s2.set=S2.S; ker=KER
  n.train=nrow(x.train); n.val=nrow(x.val); n.s2=length(s2.set); n.l=length(l.set)
  num.taus=length(taus)
  
  # Storiage
  ts.loss=c();upper=Inf
  for (t in 1:n.l){
    # Validation
    model=ckqr.qp(Covariate=x.train,Response=y.train,Taus=taus,Kernel=ker,Lambda=l.set[t],Sigma2=s2.set)      
    y.val.hat=ckqr.predict.Q(x.train, x.val, Model=model)
    s.loss=c();
    for(k in 1:num.taus){
      resid=y.val-y.val.hat[,k];pos.index=which(resid>0)
      if(length(pos.index)>0){s.loss[k]=sum(resid[pos.index])*taus[k]+sum(resid[-pos.index])*(taus[k]-1)
      } else {s.loss[k]=sum(resid*(taus[k]-1))
      }        
    } # End of K
    ts.loss[t]=sum(s.loss)
    if(ts.loss[t]<=upper){upper=ts.loss[t]; best.lambda=l.set[t]; best.sigma2=s2.set; best.model=model}
  } # End of lambda
  
  # Reporting   
  result = list(Best.lambda=best.lambda, Best.sigma2=best.sigma2, Best.model=best.model)
  return(result)
}