################################################################################
# Examples 1 and 2. Mean and quantile function estimation
################################################################################
# In RStudio
# source("fn_simulation_ex1_ex2.R")
# err.dist = "T"                    # N, T, Mix, Exp, LN or chi
# result = run_ex1_ex2(error=err.dist)
# Rscript fn_simulation_ex1_ex2.R --error=T --out=results
# For a short base R run, add --quick --kcqr-solver=admm
.driver_files = unlist(lapply(sys.frames(), function(f) f$ofile), use.names = FALSE)
.driver_path = if (length(.driver_files)) tail(.driver_files, 1L) else
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L])
if (is.na(.driver_path) || !file.exists(.driver_path)) .driver_path = "fn_simulation_ex1_ex2.R"
.driver_dir = dirname(normalizePath(.driver_path, mustWork = TRUE))
source(file.path(.driver_dir, "dcqr_analysis_common.R"), local = TRUE)

################################################################################
# Simulation
################################################################################
run_ex1_ex2 = function(error="T", cfg=dcqr_control(), reps=cfg$reps) {
  # Fixed parameter setting
  Replication = length(replicate_id(reps))
  cfg$reps = Replication
  tau.set = cfg$taus
  err.dist = error_model(error)
  methods = c("KRR", "KCQR", "DLSE", "DMQR", "DCQR")
  mean.methods = c("KRR", "KCQR", "DLSE", "DCQR")
  quant.methods = c("KCQR", "DMQR", "DCQR")

  # Storage
  MSE = metric_matrix(Replication, mean.methods)
  MAE = metric_matrix(Replication, quant.methods)
  MAE.tau = method_array(Replication, tau.set, quant.methods)
  time_mat = metric_matrix(Replication, methods)
  setting.list = vector("list", Replication)
  grid.list = vector("list", Replication)
  first = NULL

  for (rep in seq_len(Replication)) {
    # 1. Training, validation and test data
    dat = sim_split("M1", err.dist, cfg, rep)
    X.train = dat$X$train
    X.val = dat$X$val
    X.test = dat$X$test
    Y.train = dat$train$y
    Y.val = dat$val$y
    M.test = dat$test$mu
    Y.test = dat$q_true

    # 2. KRR (kernel on the original x scale)
    fit.krr = tune_kernel("KRR", dat, cfg)
    K.test = gaussian_kernel(dat$test$X, dat$train$X, fit.krr$setting$sigma2)
    pred.krr = drop(K.test %*% fit.krr$fit$beta) + fit.krr$fit$b
    MSE[rep, "KRR"] = mean((M.test - pred.krr)^2)

    # 3. KCQR
    fit.kcqr = tune_kernel("KCQR", dat, cfg)
    K.test = gaussian_kernel(dat$test$X, dat$train$X, fit.kcqr$setting$sigma2)
    f.kcqr = drop(K.test %*% fit.kcqr$fit$beta)
    pred.mean = f.kcqr + fit.kcqr$fit$mean_intercept
    pred.kcqr = outer(f.kcqr, fit.kcqr$fit$b, "+")
    MSE[rep, "KCQR"] = mean((M.test - pred.mean)^2)

    # 4. Deep least squares
    fit.dlse = tune_network("DLSE", X.train, Y.train, X.val, Y.val, cfg, rep)
    pred.dlse = predict_network("DLSE", fit.dlse$fit, X.test)
    MSE[rep, "DLSE"] = mean((M.test - pred.dlse)^2)

    # 5. DMQR, one network at each quantile
    fit.dmqr = tune_network("DMQR", X.train, Y.train, X.val, Y.val, cfg, rep)
    pred.dmqr = predict_network("DMQR", fit.dmqr$fit, X.test)

    # 6. DCQR, one network for all quantiles
    fit.dcqr = tune_network("DCQR", X.train, Y.train, X.val, Y.val, cfg, rep)
    pred.dcqr = predict_network("DCQR", fit.dcqr$fit, X.test)
    MSE[rep, "DCQR"] = mean((M.test - rowMeans(pred.dcqr))^2)

    # 7. Test errors and tuning results
    pred.list = list(KCQR=pred.kcqr, DMQR=pred.dmqr, DCQR=pred.dcqr)
    for (m in quant.methods) {
      MAE.tau[rep, , m] = colMeans(abs(Y.test - pred.list[[m]]))
      MAE[rep, m] = mean(MAE.tau[rep, , m])
    }
    fit.list = list(KRR=fit.krr, KCQR=fit.kcqr, DLSE=fit.dlse, DMQR=fit.dmqr, DCQR=fit.dcqr)
    for (m in methods) {
      time_mat[rep, m] = fit.list[[m]]$seconds
      setting.list[[rep]][[m]] = fit.list[[m]]$setting
      grid.list[[rep]][[m]] = fit.list[[m]]$grid
    }
    if (rep == 1L) first = list(x=dat$test$X[, 1L], truth=Y.test, predictions=pred.list)
  }

  # Summary over replications
  tab = data.frame(method=methods, MSE=NA_real_, MSE.SE=NA_real_,
                   Ave.MAE=NA_real_, MAE.SE=NA_real_, seconds=colMeans(time_mat),
                   row.names=methods)
  tab[mean.methods, "MSE"] = colMeans(MSE)
  tab[mean.methods, "MSE.SE"] = apply(MSE, 2, standard_error)
  tab[quant.methods, "Ave.MAE"] = colMeans(MAE)
  tab[quant.methods, "MAE.SE"] = apply(MAE, 2, standard_error)

  # Matrices used by the original table scripts
  old.cols = c("KRR", "CKQR", "  ", "D-LSE", "D-MQR", "D-CQR", "", "", "", "")
  MSE.old = MAE.old = time.old = metric_matrix(Replication, old.cols)
  MSE.old[, c(1, 2, 4, 6)] = MSE
  MAE.old[, c(2, 5, 6)] = MAE
  time.old[, c(1, 2, 4, 5, 6)] = time_mat
  best.kernel = function(m) as.matrix(do.call(rbind, lapply(setting.list, "[[", m)))
  mae.level = function(m) {
    matrix(MAE.tau[, , m], nrow=Replication, dimnames=list(seq_len(Replication), tau.set))
  }
  list(summary=tab, MAE_summary_by_tau=summarize_levels(MAE.tau),
       MSE_by_method=MSE, Ave_MAE_by_method=MAE, fit_seconds=time_mat, MAE_by_tau=MAE.tau,
       MSE=MSE.old, MAE=MAE.old, time_mat=time.old, MAE.ckqr=mae.level("KCQR"),
       MAE.dmqr=mae.level("DMQR"), MAE.dcqr=mae.level("DCQR"),
       Best.krr=best.kernel("KRR"), Best.ckqr=best.kernel("KCQR"),
       settings=setting.list, tuning=grid.list, config=cfg, error=error, first=first)
}

################################################################################
# Kernel functions for KRR and KCQR
################################################################################
gaussian_kernel = function(A, B, sigma2) {
  d2 = outer(rowSums(A^2), rowSums(B^2), "+") - 2 * tcrossprod(A, B)
  exp(-pmax(d2, 0) / (2 * sigma2))
}

# Sum of check losses + lambda * beta'G beta, with the same 1e-6 diagonal term in both fits.
kcqr_qp = function(G, y, taus, lambda, eps = 1e-6) {
  if (!requireNamespace("quadprog", quietly = TRUE))
    stop("KCQR requires quadprog. Install it, or set cfg$kcqr_solver to 'admm'.")
  n = length(y)
  k = length(taus)
  nk = n * k
  # Coefficients, then intercepts and the two residual parts.
  D = diag(eps, n + k + 2L * nk)
  D[seq_len(n), seq_len(n)] = D[seq_len(n), seq_len(n)] + 2 * lambda * G
  A = cbind(G[rep(seq_len(n), k), , drop = FALSE],
             diag(k)[rep(seq_len(k), each = n), , drop = FALSE], diag(nk), -diag(nk))
  positive = cbind(matrix(0, 2L * nk, n + k), diag(2L * nk))
  d = -c(rep(0, n + k), rep(taus, each = n), rep(1 - taus, each = n))
  sol = quadprog::solve.QP(D, d, t(rbind(A, positive)),
                             c(rep(y, k), rep(0, 2L * nk)), meq = nk)$solution
  list(beta = head(sol, n), b = sol[n + seq_len(k)])
}

# ADMM fit of the same KCQR problem using base R.
kcqr_admm = function(G, y, taus, lambda, eps = 1e-6, tolerance = 1e-5,
                      iterations = 15000L) {
  n = length(y)
  k = length(taus)
  rho = 1
  Y = matrix(y, n, k)
  tau.mat = matrix(taus, n, k, byrow = TRUE)
  cross = matrix(colSums(G), n, k)
  gram = rbind(cbind(k * crossprod(G), cross),
                cbind(t(cross), n * diag(k)))
  penalty = diag(eps, n + k)
  penalty[seq_len(n), seq_len(n)] = penalty[seq_len(n), seq_len(n)] + 2 * lambda * G
  design_t = function(A) c(crossprod(G, rowSums(A)), colSums(A))
  make_factor = function(rho) chol(penalty + rho * gram)
  L = make_factor(rho)
  b = as.numeric(quantile(y, taus, type = 1))
  residual = sweep(Y, 2L, b, "-")
  dual = matrix(0, n, k)
  for (it in seq_len(iterations)) {
    rhs = rho * design_t(Y - residual - dual)
    par = backsolve(L, forwardsolve(t(L), rhs))
    beta = par[seq_len(n)]
    b = par[n + seq_len(k)]
    fitted = outer(drop(G %*% beta), b, "+")
    previous = residual
    v = Y - fitted - dual
    residual = (pmax(v - tau.mat / rho, 0) + pmin(v + (1 - tau.mat) / rho, 0)) / (1 + eps / rho)
    primal = fitted + residual - Y
    dual = dual + primal
    rnorm = max(abs(primal))
    snorm = rho * max(abs(design_t(residual - previous)))
    if (rnorm <= tolerance * (1 + max(abs(Y))) &&
        snorm <= tolerance * (1 + rho * max(abs(design_t(dual)))))
      return(list(beta = beta, b = b))
    if (it %% 50L == 0L) {
      new_rho = if (rnorm > 10 * snorm) 2 * rho else if (snorm > 10 * rnorm) rho / 2 else rho
      if (new_rho != rho) {
        dual = dual * rho / new_rho
        rho = new_rho
        L = make_factor(rho)
      }
    }
  }
  stop("KCQR ADMM did not meet its convergence tolerances. Use the quadprog backend.")
}

fit_kernel = function(method, X, y, taus, setting, solver) {
  G = gaussian_kernel(X, X, setting$sigma2)
  n = length(y)
  if (method == "KRR") {
    system = rbind(c(0, rep(1, n)), cbind(1, G + diag(setting$lambda, n)))
    sol = solve(system, c(0, y))
    return(list(beta = sol[-1L], b = sol[1L]))
  }
  fun = switch(solver, quadprog = kcqr_qp, admm = kcqr_admm,
                stop("kcqr_solver must be 'quadprog' or 'admm'."))
  fit = fun(G, y, taus, setting$lambda)
  # The original Example 1 averages its two KCQR mean-intercept estimates.
  fit$mean_intercept = mean(c(mean(fit$b), mean(y - drop(G %*% fit$beta))))
  fit
}

tune_kernel = function(method, dat, cfg) {
  grid = expand.grid(lambda = cfg$kernel_lambda, sigma2 = cfg$kernel_sigma2)
  grid$score = NA_real_
  best = NULL
  best.loss = Inf
  for (i in seq_len(nrow(grid))) {
    setting = grid[i, c("lambda", "sigma2"), drop = FALSE]
    start.time = proc.time()[["elapsed"]]
    fit = fit_kernel(method, dat$train$X, dat$train$y, cfg$taus, setting, cfg$kcqr_solver)
    fit.time = proc.time()[["elapsed"]] - start.time
    gram = gaussian_kernel(dat$val$X, dat$train$X, setting$sigma2)
    q = outer(drop(gram %*% fit$beta), fit$b, "+")
    score = if (method == "KRR") mean((dat$val$y - q)^2) else
      validation_loss(dat$val$y, q, cfg$taus, cfg$h)
    grid$score[i] = score
    if (is.finite(score) && score < best.loss) {
      best.loss = score
      best = list(fit = fit, setting=setting, seconds=fit.time)
    }
  }
  if (is.null(best)) stop("No finite validation score for ", method)
  best$grid = grid
  best
}

################################################################################
# Plot. First replication
################################################################################
plot_ex2 = function(result, path) {
  X.test = result$first$x
  pred.list = result$first$predictions
  ord = order(X.test)
  cols = grDevices::hcl.colors(ncol(result$first$truth), "Dark 3")
  ylim = range(result$first$truth, unlist(pred.list, use.names=FALSE))

  grDevices::pdf(path, width=7, height=5)
  on.exit(grDevices::dev.off(), add=TRUE)
  for (m in names(pred.list)) {
    pred = pred.list[[m]][ord, , drop=FALSE]
    graphics::plot(X.test[ord], pred[, 1L], type="l", lty=1L, col=cols[1L], lwd=1.5,
                   ylim=ylim, xlab="x", ylab="Conditional quantile", main=m)
    for (k in seq_len(ncol(pred))[-1L]) {
      graphics::lines(X.test[ord], pred[, k], lty=1L, col=cols[k], lwd=1.5)
    }
    graphics::legend("topright", legend=result$config$taus, title="Quantile",
                     col=cols, lty=1L, bty="n", cex=0.8, ncol=3)
  }
}

# Run the simulation
if (sys.nframe() == 0L) {
  opt = cli_options()
  result = run_ex1_ex2(error=opt$error, cfg=opt$cfg)
  file.name = paste0("ex1_ex2_", opt$error)
  save_analysis(result, opt$out, file.name)
  plot_ex2(result, file.path(opt$out, paste0(file.name, "_quantiles.pdf")))
  print(result$summary, row.names=FALSE)
}
