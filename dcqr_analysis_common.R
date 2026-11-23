################################################################################
# Common settings for Examples 1-3 and Vancouver precipitation
################################################################################
.dcqr = new.env(parent=environment())
sys.source(file.path(.driver_dir, "fn_new_DCQR_lib.R"), envir=.dcqr)

dcqr_control = function(quick=FALSE) {
  # Fixed parameter setting
  tau.set = (1:9)/10
  node = c(16L, 16L)
  activation = "sigmoid"
  h = 1e-4
  gamma = 0.001
  zeta = 1e-6
  group.tol = 1e-4

  # Simulation size and starting values
  Replication = 100L
  #Replication = 3L
  n.train = 100L
  n.val = 1000L
  n.test = 10000L
  n.start = 5L
  nepoch = c(DLSE=10000L, DMQR=5000L, DCQR=7000L, "G-DCQR"=5000L, "AG-DCQR"=5000L)

  # Tuning grids
  rate.set = 10^seq(-4, -2, 0.5)
  lambda.set = 10^seq(-5, -1, length.out=10)
  Lambda.krr = exp(seq(-2, 0, 0.2))
  Sigma2.set = exp(seq(-1, 1, 0.3))

  if (quick) {
    Replication = 2L
    n.train = 24L
    n.val = 36L
    n.test = 60L
    n.start = 1L
    nepoch = 5L
    rate.set = c(0.003, 0.01)
    lambda.set = c(0.001, 0.1)
    Lambda.krr = c(0.3, 1)
    Sigma2.set = 1
  }

  list(taus=tau.set, hidden=node, act=activation, h=h, gamma=gamma, zeta=zeta,
       threshold=group.tol, lr=rate.set, lambda=lambda.set, starts=n.start,
       epochs=nepoch, patience=100L, tol=1e-5, reps=Replication,
       n_train=n.train, n_val=n.val, n_test=n.test, kernel_lambda=Lambda.krr,
       kernel_sigma2=Sigma2.set, kcqr_solver="quadprog")
}

################################################################################
# Fitting and prediction
################################################################################
replicate_id = function(reps) {
  if (length(reps) != 1L || !is.finite(reps) || reps < 1 || reps != floor(reps))
    stop("reps must be a positive integer.")
  seq_len(reps)
}

epoch_limit = function(cfg, method) {
  if (length(cfg$epochs) == 1L) return(as.integer(cfg$epochs))
  if (method == "Oracle") method = "DCQR"
  as.integer(cfg$epochs[[method]])
}

# dcqr_fit resets the RNG to seed * 100003 + epoch internally.
fit_seed = function(rep_id, start = 1L, level = 0L) {
  as.integer(1 + (193 * as.double(rep_id) + 83 * start + 337 * level) %% 20000)
}

residual_matrix = function(y, pred) {
  if (!is.matrix(pred) || nrow(pred) != length(y)) stop("Prediction dimensions differ from y.")
  -sweep(pred, 1L, as.numeric(y), "-")
}

validation_loss = function(y, pred, taus, h) {
  stopifnot(ncol(pred) == length(taus))
  .dcqr$composite_loss(t(residual_matrix(y, pred)), taus, h)
}

# The library normalizes its adaptive weights to mean one. Restore the
# manuscript's inverse-norm scale so that the prespecified lambda grid applies.
pilot_weights = function(pilot, zeta) {
  .dcqr$adaptive_weights(pilot, gamma = 1, eps = zeta) *
    mean(1 / (pilot$group_norm + zeta))
}

upper_penalty = function(par) sum(par$W2[, -1, drop = FALSE]^2) + sum(par$W0^2)

################################################################################
# DCQR fit from several starting values
################################################################################
fit_quantiles = function(X, y, taus, cfg, rate, lambda, omega, epochs, run_id, level = 0L) {
  best = NULL
  best.loss = Inf
  for (start in seq_len(cfg$starts)) {
    fit.start = .dcqr$dcqr_fit(
      X, y, taus, hidden = cfg$hidden, act = cfg$act, h = cfg$h,
      lambda = lambda, omega = omega, wd = 2 * cfg$gamma,
      lr = rate, batch = nrow(X), momentum = 0, lr_decay = 1,
      epochs = epochs, min_epochs = cfg$patience, patience = cfg$patience,
      tol = cfg$tol, full_batch = 0, monitor = "objective",
      seed = fit_seed(run_id, start, level))
    # Score the parameters actually returned, including their refitted intercepts.
    value = validation_loss(y, .dcqr$predict_q(fit.start, X), taus, cfg$h) +
      lambda * sum(fit.start$omega * fit.start$group_norm) +
      cfg$gamma * upper_penalty(fit.start$par)
    if (is.finite(value) && value < best.loss) {
      best = fit.start
      best.loss = value
    }
  }
  if (is.null(best)) stop("No finite training objective at learning rate ", rate)
  best
}

################################################################################
# Deep least squares
################################################################################
fit_dlse = function(X, y, cfg, rate, run_id) {
  best = NULL
  best.loss = Inf
  epochs = epoch_limit(cfg, "DLSE")
  for (start in seq_len(cfg$starts)) {
    par = .dcqr$init_params(ncol(X), 1L, cfg$hidden, fit_seed(run_id, start))
    par$b = mean(y)
    loss.train = numeric(epochs)
    for (e in seq_len(epochs)) {
      fw = .dcqr$forward(par, X, cfg$act)
      residual = fw$f + par$b - y
      loss.train[e] = mean(residual^2) + cfg$gamma * upper_penalty(par)
      if (e >= 2L * cfg$patience) {
        recent = mean(loss.train[(e - cfg$patience + 1L):e])
        previous = mean(loss.train[(e - 2L * cfg$patience + 1L):(e - cfg$patience)])
        if (previous - recent < cfg$tol * abs(recent)) break
      }
      df = matrix(2 * residual / nrow(X), nrow = 1L)
      d2 = (t(par$W0) %*% df) * .dcqr$.act_deriv(fw$A2, fw$Z2, cfg$act)
      d1 = (crossprod(par$W2[, -1, drop = FALSE], d2)) *
        .dcqr$.act_deriv(fw$A1, fw$Z1, cfg$act)
      g2 = d2 %*% t(rbind(1, fw$A1))
      g2[, -1] = g2[, -1, drop = FALSE] + 2 * cfg$gamma * par$W2[, -1, drop = FALSE]
      par$W0 = par$W0 - rate * (tcrossprod(df, fw$A2) + 2 * cfg$gamma * par$W0)
      par$W2 = par$W2 - rate * g2
      par$W1 = par$W1 - rate * (d1 %*% cbind(1, X))
      par$b = par$b - rate * sum(df)
    }
    value = mean((.dcqr$forward(par, X, cfg$act)$f + par$b - y)^2) +
      cfg$gamma * upper_penalty(par)
    if (is.finite(value) && value < best.loss) {
      best = list(par = par, act = cfg$act)
      best.loss = value
    }
  }
  if (is.null(best)) stop("No finite DLSE fit.")
  best
}

fit_network = function(method, X, y, cfg, setting, run_id, omega=NULL) {
  lr.rate = setting$lr[[1L]]
  if (method == "DLSE") return(fit_dlse(X, y, cfg, lr.rate, run_id))

  nepoch = epoch_limit(cfg, method)
  if (method == "DMQR") {
    K = length(cfg$taus)
    fit.list = vector("list", K)
    for (k in seq_len(K)) {
      fit.list[[k]] = fit_quantiles(X, y, cfg$taus[k], cfg, lr.rate, 0, NULL,
                                    nepoch, run_id, k)
    }
    return(fit.list)
  }

  if (method == "AG-DCQR" && is.null(omega)) stop("AG-DCQR needs training-pilot weights.")
  fit_quantiles(X, y, cfg$taus, cfg, lr.rate, setting$lambda[[1L]], omega, nepoch, run_id)
}

predict_network = function(method, fit, X) {
  if (method == "DLSE") return(.dcqr$forward(fit$par, X, fit$act)$f + fit$par$b)
  if (method == "DMQR") {
    pred.list = vector("list", length(fit))
    for (k in seq_along(fit)) pred.list[[k]] = .dcqr$predict_q(fit[[k]], X)
    return(do.call(cbind, pred.list))
  }
  .dcqr$predict_q(fit, X)
}

################################################################################
# Validation
################################################################################
# Quantile models use the smoothed validation loss. DLSE uses squared error.
# Elapsed time covers all starts at the selected setting.
tune_network = function(method, X, y, Xval, yval, cfg, run_id, omega=NULL) {
  lambda.set = 0
  if (method %in% c("G-DCQR", "AG-DCQR")) lambda.set = cfg$lambda
  grid = expand.grid(lr=cfg$lr, lambda=lambda.set)
  grid$score = NA_real_
  best.loss = Inf
  best.fit = NULL

  for (i in seq_len(nrow(grid))) {
    setting = grid[i, c("lr", "lambda"), drop=FALSE]
    start.time = proc.time()[["elapsed"]]
    fit = fit_network(method, X, y, cfg, setting, run_id, omega)
    fit.time = proc.time()[["elapsed"]] - start.time

    pred.val = predict_network(method, fit, Xval)
    if (method == "DLSE") {
      loss.val = mean((yval - pred.val)^2)
    } else {
      loss.val = validation_loss(yval, pred.val, cfg$taus, cfg$h)
    }
    grid$score[i] = loss.val
    if (is.finite(loss.val) && loss.val < best.loss) {
      best.loss = loss.val
      best.fit = list(fit=fit, setting=setting, seconds=fit.time)
    }
  }
  if (is.null(best.fit)) stop("No finite validation score for ", method)
  best.fit$grid = grid
  best.fit
}

################################################################################
# Error distributions and simulation data
################################################################################
error_model = function(error) {
  err.dist = match.arg(error, c("N", "T", "Mix", "MN", "Exp", "LN", "chi"))
  if (err.dist == "MN") err.dist = "Mix"

  if (err.dist == "N") return(list(draw=rnorm, q=qnorm))
  if (err.dist == "T") {
    draw = function(n) rt(n, 3)/sqrt(3)
    q = function(p) qt(p, 3)/sqrt(3)
  }
  if (err.dist == "Exp") {
    draw = function(n) rexp(n) - 1
    q = function(p) qexp(p) - 1
  }
  if (err.dist == "LN") {
    draw = function(n) (rlnorm(n) - exp(0.5))/sqrt(exp(1) * expm1(1))
    q = function(p) (qlnorm(p) - exp(0.5))/sqrt(exp(1) * expm1(1))
  }
  if (err.dist == "chi") {
    draw = function(n) (rchisq(n, 3) - 3)/sqrt(6)
    q = function(p) (qchisq(p, 3) - 3)/sqrt(6)
  }
  if (err.dist == "Mix") {
    draw = function(n) rnorm(n, sd=ifelse(runif(n) < 0.1, 5, 1))/sqrt(3.4)
    q.one = function(p) {
      if (p == 0.5) return(0)
      lim = sort(qnorm(p) * c(1, 5))
      fun = function(z) 0.9 * pnorm(z) + 0.1 * pnorm(z/5) - p
      uniroot(fun, lim, tol=1e-11, extendInt="yes")$root
    }
    q = function(p) vapply(p, q.one, numeric(1))/sqrt(3.4)
  }
  list(draw=draw, q=q)
}

sim_data = function(model, n, error, seed) {
  set.seed(seed)
  if (model == "M1") {
    X = matrix(runif(n, -5, 5), ncol=1L, dimnames=list(NULL, "x1"))
    x = X[, 1L]
    M = 1 + 2 * (1 - x + 2*x^2) * exp(-x^2/2)
  } else {
    X = matrix(runif(n * 12L, -1, 1), ncol=12L, dimnames=list(NULL, paste0("x", 1:12)))
    M = 0.1 + 0.1*X[, 1]^3 + 0.5*X[, 2]^2 + rowSums(X[, 1:3, drop=FALSE])
  }
  Esp = error$draw(n)
  list(X=X, y=M + Esp, mu=M)
}

sim_split = function(model, error, cfg, rep_id) {
  seed.set = if (model == "M1") c(1234, 4567, 7891) else c(123, 456, 789)
  train = sim_data(model, cfg$n_train, error, seed.set[1] + rep_id)
  val = sim_data(model, cfg$n_val, error, seed.set[2] + rep_id)
  test = sim_data(model, cfg$n_test, error, seed.set[3] + rep_id)
  X = .dcqr$standardize_split(train$X, val=val$X, test=test$X)
  Y.test = outer(test$mu, error$q(cfg$taus), "+")
  list(train=train, val=val, test=test, X=X, q_true=Y.test)
}

################################################################################
# Storage and summary tables
################################################################################
metric_matrix = function(reps, cols) {
  matrix(NA_real_, reps, length(cols), dimnames = list(seq_len(reps), cols))
}

method_array = function(reps, variables, methods, fill = NA_real_) {
  array(fill, c(reps, length(variables), length(methods)),
        dimnames = list(split = seq_len(reps), variable = variables, method = methods))
}

standard_error = function(x) {
  if (length(x) < 2L) return(NA_real_)
  sd(x) / sqrt(length(x))
}

summarize_levels = function(metrics) {
  tau.set = as.numeric(dimnames(metrics)[[2L]])
  methods = dimnames(metrics)[[3L]]
  summary.list = vector("list", length(methods))
  for (j in seq_along(methods)) {
    m = methods[j]
    value = matrix(metrics[, , m], nrow=dim(metrics)[1L])
    summary.list[[j]] = data.frame(method=m, tau=tau.set, mean=colMeans(value),
                                   SE=apply(value, 2L, standard_error), row.names=NULL)
  }
  do.call(rbind, summary.list)
}

selection_summary = function(selected, signal) {
  methods = dimnames(selected)[[3L]]
  variables = dimnames(selected)[[2L]]
  nc = nic = exact = metric_matrix(dim(selected)[1L], methods)
  counts = matrix(0L, length(methods), length(variables), dimnames = list(methods, variables))
  for (m in methods) {
    var.mat = matrix(selected[, , m], ncol = length(variables))
    nc[, m] = rowSums(var.mat[, signal, drop = FALSE])
    nic[, m] = rowSums(var.mat[, !signal, drop = FALSE])
    exact[, m] = nc[, m] == sum(signal) & nic[, m] == 0
    counts[m, ] = colSums(var.mat)
  }
  list(NC.val = nc, NIC.val = nic, NT.val = exact,
       counts = counts, frequency = counts / nrow(nc),
       summary = data.frame(method = methods, NC = colMeans(nc), NIC = colMeans(nic),
                            NT = colSums(exact), row.names = NULL))
}

################################################################################
# Command line and result files
################################################################################
cli_options = function() {
  a = commandArgs(trailingOnly = TRUE)
  allowed = a == "--quick" | grepl("^--(reps|error|out|data|kcqr-solver)=", a)
  if (any(!allowed)) stop("Unknown argument ", a[which(!allowed)[1]])
  opt = function(name, default) {
    prefix = paste0("--", name, "=")
    hit = a[startsWith(a, prefix)]
    if (length(hit)) substring(tail(hit, 1), nchar(prefix) + 1L) else default
  }
  cfg = dcqr_control("--quick" %in% a)
  cfg$reps = as.integer(opt("reps", cfg$reps))
  cfg$kcqr_solver = opt("kcqr-solver", cfg$kcqr_solver)
  list(cfg = cfg, error = opt("error", "T"), out = opt("out", "dcqr_results"),
       data = opt("data", NULL))
}

save_analysis = function(result, dir, stem) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  path = file.path(dir, paste0(stem, ".rds"))
  temp = tempfile(pattern = paste0(".", stem), tmpdir = dir)
  on.exit(unlink(temp), add = TRUE)
  saveRDS(result, temp)
  if (!file.rename(temp, path)) stop("Could not write ", path)
  message("Saved ", normalizePath(path))
  invisible(path)
}
