################################################################################
# DCQR fitting functions
# Two hidden layers, one regression function, one intercept per quantile.
# X has n rows. Residuals are K by n and predictions are n by K.
################################################################################

################################################################################
# 1. Check loss
################################################################################
.act = function(z, act) {
  switch(act,
         sigmoid = 1/(1 + exp(-z)),
         tanh = tanh(z),
         relu = pmax(z, 0),
         stop("unknown activation"))
}

.act_deriv = function(a, z, act) {
  switch(act, sigmoid=a*(1 - a), tanh=1 - a^2, relu=(z > 0)*1)
}

rho_tau = function(u, tau) u * (tau - (u < 0))

rho_h_mat = function(U, taus, h=0) {
  loss = U * (taus - (U < 0))
  if (h > 0) {
    ind = abs(U) <= h
    tau.mat = matrix(taus, nrow(U), ncol(U))
    loss[ind] = U[ind]^2/(4*h) + (tau.mat[ind] - 0.5)*U[ind] + h/4
  }
  loss
}

psi_h_mat = function(U, taus, h=0) {
  psi = taus - (U < 0)
  if (h > 0) {
    ind = abs(U) <= h
    tau.mat = matrix(taus, nrow(U), ncol(U))
    psi[ind] = U[ind]/(2*h) + tau.mat[ind] - 0.5
  }
  psi
}

.resid_mat = function(y, f, b) {
  K = length(b)
  n = length(y)
  matrix(y - f, K, n, byrow=TRUE) - b
}

composite_loss = function(U, taus, h=0, w=NULL) {
  loss.i = colMeans(rho_h_mat(U, taus, h))
  if (is.null(w)) return(mean(loss.i))
  sum(w * loss.i)/sum(w)
}

# Weighted empirical quantiles of the training residuals
wquantile = function(x, tau, w=NULL) {
  if (is.null(w)) return(as.numeric(quantile(x, tau, type=1, names=FALSE)))
  ord = order(x)
  cum.w = cumsum(w[ord])/sum(w)
  sapply(tau, function(t) x[ord][which(cum.w >= t)[1]])
}

col_norms = function(M) sqrt(colSums(M^2))

################################################################################
# 2. Initial weights and network derivatives
################################################################################
init_params = function(p, K, hidden=c(32, 32), seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  node_1 = hidden[1]
  node_2 = hidden[2]
  init.weight = function(nr, nc, fan.in, fan.out) {
    bound = sqrt(6/(fan.in + fan.out))
    matrix(runif(nr * nc, -bound, bound), nr, nc)
  }
  W1 = init.weight(node_1, p + 1, p, node_1)
  W2 = init.weight(node_2, node_1 + 1, node_1, node_2)
  W0 = init.weight(1, node_2, node_2, 1)
  list(W1=W1, W2=W2, W0=W0, b=rep(0, K))
}

forward = function(par, X, act) {
  # Column 1 in W1 and W2 holds the bias.
  Z1 = par$W1[, -1, drop=FALSE] %*% t(X) + par$W1[, 1]
  A1 = .act(Z1, act)
  Z2 = par$W2[, -1, drop=FALSE] %*% A1 + par$W2[, 1]
  A2 = .act(Z2, act)
  f = as.vector(par$W0 %*% A2)
  list(Z1=Z1, A1=A1, Z2=Z2, A2=A2, f=f)
}

loss_gradient = function(par, X, y, w, taus, h, act, wd) {
  K = length(taus)
  net = forward(par, X, act)
  U = .resid_mat(y, net$f, par$b)
  psi = psi_h_mat(U, taus, h) * rep(w, each=K)
  df = matrix(-colSums(psi)/(K * sum(w)), 1)
  gb = -rowSums(psi)/(K * sum(w))

  gW0 = df %*% t(net$A2)
  d2 = (t(par$W0) %*% df) * .act_deriv(net$A2, net$Z2, act)
  gW2 = d2 %*% t(rbind(1, net$A1))
  d1 = (t(par$W2[, -1, drop=FALSE]) %*% d2) * .act_deriv(net$A1, net$Z1, act)
  gW1 = d1 %*% cbind(1, X)

  if (wd > 0) {
    gW2[, -1] = gW2[, -1] + wd * par$W2[, -1]
    gW0 = gW0 + wd * par$W0
  }
  list(W1=gW1, W2=gW2, W0=gW0, b=gb)
}

################################################################################
# 3. DCQR, group Lasso DCQR and adaptive group Lasso DCQR
################################################################################
# lambda = 0 for DCQR. G-DCQR uses unit weights and AG-DCQR supplies omega.
# The validation monitor uses ordinary check loss. The objective monitor uses
# the penalized training loss. The final intercepts use training residuals.
dcqr_fit = function(X, y, taus, X_val=NULL, y_val=NULL, w=NULL, w_val=NULL,
                    lambda=0, omega=NULL, wd=1e-3, h=0, act="sigmoid",
                    hidden=c(32, 32), lr=0.05, momentum=0.9, lr_decay=0.999, batch=32,
                    epochs=1500, min_epochs=100, patience=100, tol=1e-3,
                    monitor=NULL, full_batch=200, freeze_from=Inf,
                    par_init=NULL, seed=1, verbose=FALSE) {
  X = as.matrix(X)
  y = as.numeric(y)
  n = nrow(X)
  p = ncol(X)
  K = length(taus)
  if (is.null(w)) w = rep(1, n)
  if (is.null(omega)) omega = rep(1, p)
  if (!is.null(X_val)) {
    X_val = as.matrix(X_val)
    if (is.null(w_val)) w_val = rep(1, nrow(X_val))
  }
  if (is.null(monitor)) {
    monitor = if (lambda > 0 || is.null(X_val)) "objective" else "val"
  }

  # Initial values
  par = par_init
  if (is.null(par_init)) {
    par = init_params(p, K, hidden, seed)
    par$b = wquantile(y, taus, w)
  }
  vel = list(W1=0*par$W1, W2=0*par$W2, W0=0*par$W0, b=0*par$b)
  n.batch = ceiling(n/batch)
  lr.rate = lr
  zero.group = NULL

  # First-layer block soft threshold. The first column is the bias.
  group.step = function(par, step) {
    if (lambda > 0) {
      norm.j = col_norms(par$W1[, -1, drop=FALSE])
      shrink = ifelse(norm.j > 0, pmax(0, 1 - step*lambda*omega/norm.j), 0)
      par$W1[, -1] = sweep(par$W1[, -1, drop=FALSE], 2, shrink, "*")
    }
    par
  }

  loss.train = loss.val = obj.train = rep(NA_real_, epochs)
  best.loss = Inf
  best.par = par
  best.epoch = 0
  n.wait = 0

  # Training
  for (e in seq_len(epochs)) {
    set.seed(seed * 100003 + e)
    ind = sample.int(n)
    if (e >= freeze_from && is.null(zero.group)) {
      zero.group = col_norms(par$W1[, -1, drop=FALSE]) == 0
    }

    for (i in seq_len(n.batch)) {
      block = ind[((i - 1)*batch + 1):min(i*batch, n)]
      g = loss_gradient(par, X[block, , drop=FALSE], y[block], w[block], taus, h, act, wd)
      for (name in names(vel)) {
        vel[[name]] = momentum * vel[[name]] + g[[name]]
        par[[name]] = par[[name]] - lr.rate * vel[[name]]
      }
      par = group.step(par, lr.rate)
      if (!is.null(zero.group) && any(zero.group)) par$W1[, c(FALSE, zero.group)] = 0
    }
    lr.rate = lr.rate * lr_decay

    f.train = forward(par, X, act)$f
    U.train = .resid_mat(y, f.train, par$b)
    loss.train[e] = composite_loss(U.train, taus, 0, w)
    group.pen = lambda * sum(omega * col_norms(par$W1[, -1, drop=FALSE]))
    ridge.pen = 0.5 * wd * (sum(par$W2[, -1]^2) + sum(par$W0^2))
    obj.train[e] = composite_loss(U.train, taus, h, w) + group.pen + ridge.pen
    if (!is.null(X_val)) {
      f.val = forward(par, X_val, act)$f
      loss.val[e] = composite_loss(.resid_mat(y_val, f.val, par$b), taus, 0, w_val)
    }

    # Stopping rule
    stop.now = FALSE
    if (monitor == "val") {
      if (loss.val[e] < best.loss - 1e-8) {
        best.loss = loss.val[e]
        best.par = par
        best.epoch = e
        n.wait = 0
      } else {
        n.wait = n.wait + 1
      }
      stop.now = e >= min_epochs && n.wait >= patience
    } else {
      best.epoch = e
      if (e >= max(min_epochs, 2*patience)) {
        recent = mean(obj.train[(e - patience + 1):e])
        previous = mean(obj.train[(e - 2*patience + 1):(e - patience)])
        stop.now = (previous - recent) < tol * abs(recent)
      }
    }
    if (verbose && e %% 100 == 0) {
      n.active = sum(col_norms(par$W1[, -1, drop=FALSE]) > 0)
      cat(sprintf("epoch %4d  obj %.4f  train %.4f  val %.4f  active %d\n",
                  e, obj.train[e], loss.train[e], ifelse(is.null(X_val), NA, loss.val[e]), n.active))
    }
    if (stop.now) break
  }
  if (monitor == "val") par = best.par

  # Final full-batch steps for penalized fits
  if (lambda > 0 && full_batch > 0) {
    for (i in seq_len(full_batch)) {
      g = loss_gradient(par, X, y, w, taus, h, act, wd)
      for (name in names(vel)) par[[name]] = par[[name]] - lr * g[[name]]
      par = group.step(par, lr)
    }
  }

  # Refit the quantile intercepts on training residuals
  f.train = forward(par, X, act)$f
  par$b = wquantile(y - f.train, taus, w)
  norm.j = col_norms(par$W1[, -1, drop=FALSE])
  result = list(par=par, act=act, taus=taus, h=h, lambda=lambda, omega=omega,
                wd=wd, hidden=hidden, monitor=monitor, best_epoch=best.epoch, n_epochs_run=e,
                train_curve=loss.train[seq_len(e)], val_curve=loss.val[seq_len(e)],
                obj_curve=obj.train[seq_len(e)], group_norm=norm.j, active=norm.j > 0,
                q_null=wquantile(y, taus, w))
  class(result) = "dcqr"
  result
}

################################################################################
# 4. Prediction and test error
################################################################################
predict_f = function(fit, X) forward(fit$par, as.matrix(X), fit$act)$f

predict_q = function(fit, X) {
  f = predict_f(fit, X)
  pred = outer(f, fit$par$b, "+")
  colnames(pred) = paste0("tau", fit$taus)
  pred
}

predict_mean = function(fit, X) rowMeans(predict_q(fit, X))

eval_fit = function(fit, X, y, w=NULL) {
  pred = predict_q(fit, X)
  K = ncol(pred)
  if (is.null(w)) w = rep(1, length(y))
  U = y - pred
  mpe = mpe0 = coverage = numeric(K)
  for (k in seq_len(K)) {
    tau = fit$taus[k]
    mpe[k] = sum(w * rho_tau(U[, k], tau))/sum(w)
    mpe0[k] = sum(w * rho_tau(y - fit$q_null[k], tau))/sum(w)
    coverage[k] = sum(w * (y <= pred[, k]))/sum(w)
  }
  crossings = 0
  if (K > 1) crossings = sum(apply(pred, 1, function(q) any(diff(q) < 0)))
  list(mpe=setNames(mpe, fit$taus), mpe_composite=mean(mpe),
       R1=setNames(1 - mpe/mpe0, fit$taus), coverage=setNames(coverage, fit$taus),
       crossings=crossings, n_active=sum(fit$active))
}

################################################################################
# 5. Adaptive weights and lambda grid
################################################################################
adaptive_weights = function(fit0, gamma=1, eps=1e-4) {
  w = (1/(fit0$group_norm + eps))^gamma
  w/mean(w)
}

# Increase in training loss when an input group is set to zero
lambda_max = function(fit_ref, X, y, taus, w=NULL, omega=NULL, h=0) {
  X = as.matrix(X)
  if (is.null(w)) w = rep(1, length(y))
  if (is.null(omega)) omega = rep(1, ncol(X))
  par = fit_ref$par
  norm.j = col_norms(par$W1[, -1, drop=FALSE])
  f = forward(par, X, fit_ref$act)$f
  loss0 = composite_loss(.resid_mat(y, f, par$b), taus, h, w)
  lambda.j = numeric(ncol(X))

  for (j in seq_len(ncol(X))) {
    if (norm.j[j] != 0) {
      par.j = par
      par.j$W1[, j + 1] = 0
      f.j = forward(par.j, X, fit_ref$act)$f
      loss.j = composite_loss(.resid_mat(y, f.j, par$b), taus, h, w)
      lambda.j[j] = max(loss.j - loss0, 0)/(omega[j] * norm.j[j])
    }
  }
  max(lambda.j)
}

dcqr_path = function(X, y, taus, X_val, y_val, lambda0, omega, par_start,
                     w=NULL, w_val=NULL, nlambda=12, ratio=10^(-2.5),
                     auto_top=TRUE, rule=c("1se", "min"), keep_all=FALSE, ...) {
  rule = match.arg(rule)
  if (is.null(w_val)) w_val = rep(1, nrow(X_val))
  lambda.top = lambda0
  par = par_start

  # Find the upper end of the lambda grid
  if (auto_top) {
    for (i in seq_len(8L)) {
      fit = dcqr_fit(X, y, taus, X_val, y_val, w=w, w_val=w_val, lambda=lambda.top,
                     omega=omega, par_init=par, ...)
      if (sum(fit$active) == 0 || i == 8L) break
      lambda.top = 2 * lambda.top
      par = fit$par
    }
  }

  # Warm starts from the largest lambda to the smallest
  lambda.set = lambda.top * 10^seq(0, log10(ratio), length.out=nlambda)
  fit.list = vector("list", nlambda)
  loss.val = se.val = rep(NA_real_, nlambda)
  for (i in seq_along(lambda.set)) {
    fit = dcqr_fit(X, y, taus, X_val, y_val, w=w, w_val=w_val,
                   lambda=lambda.set[i], omega=omega, par_init=par, ...)
    pred.val = predict_f(fit, X_val)
    U.val = .resid_mat(y_val, pred.val, fit$par$b)
    # This library path uses the ordinary check loss at validation.
    loss.i = colMeans(rho_h_mat(U.val, fit$taus, 0))
    loss.val[i] = sum(w_val * loss.i)/sum(w_val)
    se.val[i] = sqrt(sum(w_val * (loss.i - loss.val[i])^2)/sum(w_val)/length(loss.i))
    par = fit$par
    if (keep_all) {
      fit.list[[i]] = fit
    } else {
      fit.list[[i]] = list(active=fit$active, group_norm=fit$group_norm,
                           par=fit$par, q_null=fit$q_null)
    }
  }

  best = which.min(loss.val)
  if (rule == "1se") best = min(which(loss.val <= loss.val[best] + se.val[best]))
  best.fit = fit.list[[best]]
  if (!keep_all) {
    best.fit = c(best.fit, list(act=fit$act, taus=taus, h=fit$h, lambda=lambda.set[best],
                                omega=omega, wd=fit$wd, hidden=fit$hidden))
    class(best.fit) = "dcqr"
  }
  n.active = sapply(fit.list, function(f) sum(f$active))
  list(fit=best.fit, lambda=lambda.set[best], lambdas=lambda.set, val_loss=loss.val,
       val_se=se.val, n_active=n.active, rule=rule, path=fit.list)
}

################################################################################
# 6. Standardize predictors using the training data
################################################################################
standardize_split = function(X_train, ...) {
  x.center = colMeans(X_train)
  x.scale = apply(X_train, 2, sd)
  x.scale[x.scale == 0] = 1
  standardize = function(X) sweep(sweep(as.matrix(X), 2, x.center, "-"), 2, x.scale, "/")
  extra = list(...)
  for (i in seq_along(extra)) extra[[i]] = standardize(extra[[i]])
  c(list(train=standardize(X_train), mu=x.center, sd=x.scale), extra)
}

################################################################################
# 7. Single training, validation and test split
################################################################################
dcqr_one_split = function(Xtr, ytr, Xva, yva, Xte, yte, taus,
                          wtr=NULL, wva=NULL, wte=NULL, n_init=3, nlambda=12, single_taus=NULL,
                          hidden=c(32, 32), act="sigmoid", wd=1e-3, h=0, lr=0.05,
                          batch=32, epochs=1500, patience=100, seed=1) {
  # DCQR pilot
  p = ncol(Xtr)
  fit.list = vector("list", n_init)
  loss.val = numeric(n_init)
  for (i in seq_len(n_init)) {
    fit.list[[i]] = dcqr_fit(Xtr, ytr, taus, Xva, yva, w=wtr, w_val=wva, lambda=0,
                             wd=wd, h=h, act=act, hidden=hidden, lr=lr, batch=batch,
                             epochs=epochs, patience=patience, seed=seed * 1000 + i)
    loss.val[i] = eval_fit(fit.list[[i]], Xva, yva, wva)$mpe_composite
  }
  fit.dcqr = fit.list[[which.min(loss.val)]]

  # G-DCQR
  w.g = rep(1, p)
  lambda.g = lambda_max(fit.dcqr, Xtr, ytr, taus, wtr, w.g, h)
  fit.g = dcqr_path(Xtr, ytr, taus, Xva, yva, lambda.g, w.g, fit.dcqr$par,
                    nlambda=nlambda, w=wtr, w_val=wva, wd=wd, h=h, act=act,
                    hidden=hidden, lr=lr, batch=batch, epochs=epochs, patience=patience, seed=seed)

  # AG-DCQR
  w.ag = adaptive_weights(fit.dcqr)
  lambda.ag = lambda_max(fit.dcqr, Xtr, ytr, taus, wtr, w.ag, h)
  fit.ag = dcqr_path(Xtr, ytr, taus, Xva, yva, lambda.ag, w.ag, fit.dcqr$par,
                     nlambda=nlambda, w=wtr, w_val=wva, wd=wd, h=h, act=act,
                     hidden=hidden, lr=lr, batch=batch, epochs=epochs, patience=patience, seed=seed)

  result = list(DCQR=fit.dcqr, G=fit.g, AG=fit.ag,
                test=list(DCQR=eval_fit(fit.dcqr, Xte, yte, wte),
                          G=eval_fit(fit.g$fit, Xte, yte, wte),
                          AG=eval_fit(fit.ag$fit, Xte, yte, wte)))

  # Optional single-quantile AG fits
  if (!is.null(single_taus)) {
    single = vector("list", length(single_taus))
    for (k in seq_along(single_taus)) {
      tau = single_taus[[k]]
      pilot = dcqr_fit(Xtr, ytr, tau, Xva, yva, w=wtr, w_val=wva, lambda=0,
                       wd=wd, h=h, act=act, hidden=hidden, lr=lr, batch=batch,
                       epochs=epochs, patience=patience, seed=seed * 1000 + 1)
      w.k = adaptive_weights(pilot)
      lambda.k = lambda_max(pilot, Xtr, ytr, tau, wtr, w.k, h)
      fit.k = dcqr_path(Xtr, ytr, tau, Xva, yva, lambda.k, w.k, pilot$par,
                        nlambda=nlambda, w=wtr, w_val=wva, wd=wd, h=h, act=act,
                        hidden=hidden, lr=lr, batch=batch, epochs=epochs, patience=patience, seed=seed)
      single[[k]] = list(tau=tau, fit=fit.k$fit, lambda=fit.k$lambda,
                         test=eval_fit(fit.k$fit, Xte, yte, wte))
    }
    names(single) = paste0("tau", single_taus)
    result$single = single
  }
  result
}

################################################################################
# 8. Repeated splits and summary measures
################################################################################
dcqr_resample = function(X, y, taus, B=100, split=c(0.6, 0.2, 0.2), w=NULL,
                         single_taus=NULL, cores=1, seed0=2026, ...) {
  X = as.matrix(X)
  n = nrow(X)
  if (is.null(w)) w = rep(1, n)
  one.split = function(rep) {
    set.seed(seed0 + rep)
    ind = sample.int(n)
    n.train = floor(split[1] * n)
    n.val = floor(split[2] * n)
    train = ind[1:n.train]
    val = ind[(n.train + 1):(n.train + n.val)]
    test = ind[(n.train + n.val + 1):n]
    X.scaled = standardize_split(X[train, , drop=FALSE], va=X[val, , drop=FALSE],
                                 te=X[test, , drop=FALSE])
    start.time = Sys.time()
    result = dcqr_one_split(X.scaled$train, y[train], X.scaled$va, y[val],
                            X.scaled$te, y[test], taus, wtr=w[train], wva=w[val], wte=w[test],
                            single_taus=single_taus, seed=rep, ...)
    result$time = as.numeric(difftime(Sys.time(), start.time, units="secs"))
    result$idx = list(train=train, val=val, test=test)
    result
  }
  if (cores > 1 && requireNamespace("parallel", quietly=TRUE)) {
    return(parallel::mclapply(seq_len(B), one.split, mc.cores=cores))
  }
  result.list = vector("list", B)
  for (rep in seq_len(B)) result.list[[rep]] = one.split(rep)
  result.list
}

selection_frequency = function(res, method=c("AG", "G"), var_names=NULL) {
  method = match.arg(method)
  var.mat = t(sapply(res, function(r) r[[method]]$fit$active))
  if (!is.null(var_names)) colnames(var.mat) = var_names
  list(freq=colMeans(var.mat), n_active=rowSums(var.mat), matrix=var.mat)
}

selection_frequency_single = function(res, var_names=NULL) {
  tau.names = names(res[[1]]$single)
  frequency = sapply(tau.names, function(t) {
    var.mat = t(sapply(res, function(r) r$single[[t]]$fit$active))
    colMeans(var.mat)
  })
  if (!is.null(var_names)) rownames(frequency) = var_names
  frequency
}

jaccard_stability = function(A) {
  B = nrow(A)
  total = 0
  pairs = 0
  for (i in 1:(B - 1)) {
    for (j in (i + 1):B) {
      size = sum(A[i, ] | A[j, ])
      total = total + ifelse(size == 0, 1, sum(A[i, ] & A[j, ])/size)
      pairs = pairs + 1
    }
  }
  total/pairs
}

test_metric = function(res, method=c("DCQR", "G", "AG"), what=c("mpe", "coverage", "R1")) {
  method = match.arg(method)
  what = match.arg(what)
  value = t(sapply(res, function(r) r$test[[method]][[what]]))
  list(mean=colMeans(value), se=apply(value, 2, sd)/sqrt(nrow(value)), matrix=value)
}

intercept_summary = function(res, method="AG") {
  B = t(sapply(res, function(r) r[[method]]$fit$par$b))
  list(mean=colMeans(B), q025=apply(B, 2, quantile, 0.025), q975=apply(B, 2, quantile, 0.975))
}

################################################################################
# 9. Variable importance and partial dependence
################################################################################
grad_x = function(fit, X) {
  X = as.matrix(X)
  par = fit$par
  net = forward(par, X, fit$act)
  d2 = .act_deriv(net$A2, net$Z2, fit$act) * as.vector(par$W0)
  d1 = .act_deriv(net$A1, net$Z1, fit$act) * (t(par$W2[, -1, drop=FALSE]) %*% d2)
  t(par$W1[, -1, drop=FALSE]) %*% d1
}

grad_importance = function(fit, X) sqrt(rowMeans(grad_x(fit, X)^2))
avg_derivative = function(fit, X) rowMeans(grad_x(fit, X))

perm_importance = function(fit, X, y, w=NULL, R=10, seed=1) {
  X = as.matrix(X)
  loss0 = eval_fit(fit, X, y, w)$mpe_composite
  set.seed(seed)
  importance = numeric(ncol(X))
  for (j in seq_len(ncol(X))) {
    loss.perm = numeric(R)
    for (rep in seq_len(R)) {
      X.perm = X
      X.perm[, j] = sample(X.perm[, j])
      loss.perm[rep] = eval_fit(fit, X.perm, y, w)$mpe_composite
    }
    importance[j] = mean(loss.perm) - loss0
  }
  importance
}

partial_dependence = function(fit, X, j, grid=NULL, n_grid=40) {
  X = as.matrix(X)
  if (is.null(grid)) {
    lim = quantile(X[, j], c(0.02, 0.98))
    grid = seq(lim[1], lim[2], length.out=n_grid)
  }
  value = sapply(grid, function(xj) {
    X.new = X
    X.new[, j] = xj
    mean(predict_f(fit, X.new))
  })
  data.frame(x=grid, f=value)
}

################################################################################
# 10. Location-shift check on test residuals
################################################################################
location_shift_diag = function(fit, X_test, y_test, taus_diag=c(0.1, 0.25, 0.5, 0.75, 0.9),
                               var_names=NULL) {
  resid.test = y_test - predict_f(fit, X_test)
  if (!requireNamespace("quantreg", quietly=TRUE)) {
    warning("quantreg not installed: returning residuals only")
    return(list(resid=resid.test))
  }
  df = data.frame(r=resid.test, as.data.frame(X_test))
  if (!is.null(var_names)) names(df)[-1] = var_names
  rq_fit = quantreg::rq(r ~ ., data=df, tau=taus_diag)
  slopes = coef(rq_fit)[-1, , drop=FALSE]
  test = tryCatch(quantreg::anova.rq(rq_fit, joint=TRUE), error=function(e) NULL)
  list(resid=resid.test, slopes=slopes, joint_test=test)
}

################################################################################
# 11. Stability selection on complementary half samples
################################################################################
stability_selection = function(X, y, taus, lambda, B_pairs=50, w=NULL, cores=1,
                               seed0=7, adaptive=TRUE, ...) {
  X = as.matrix(X)
  n = nrow(X)
  p = ncol(X)
  if (is.null(w)) w = rep(1, n)

  one.pair = function(rep) {
    set.seed(seed0 + rep)
    ind = sample.int(n)
    half = floor(n/2)
    train.1 = ind[1:half]
    train.2 = ind[(half + 1):n]
    fit.half = function(train) {
      X.train = standardize_split(X[train, , drop=FALSE])$train
      fit0 = dcqr_fit(X.train, y[train], taus, w=w[train], lambda=0, seed=rep, ...)
      weights = if (adaptive) adaptive_weights(fit0) else rep(1, p)
      fit = dcqr_fit(X.train, y[train], taus, w=w[train], lambda=lambda, omega=weights,
                     par_init=fit0$par, seed=rep, ...)
      fit$active
    }
    rbind(fit.half(train.1), fit.half(train.2))
  }
  if (cores > 1) {
    var.list = parallel::mclapply(seq_len(B_pairs), one.pair, mc.cores=cores)
  } else {
    var.list = vector("list", B_pairs)
    for (rep in seq_len(B_pairs)) var.list[[rep]] = one.pair(rep)
  }
  var.mat = do.call(rbind, var.list)
  list(freq=colMeans(var.mat), matrix=var.mat, q_avg=mean(rowSums(var.mat)))
}
