################################################################################
# Example 3. Variable selection
################################################################################
.driver_files = unlist(lapply(sys.frames(), function(f) f$ofile), use.names = FALSE)
.driver_path = if (length(.driver_files)) tail(.driver_files, 1L) else
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L])
if (is.na(.driver_path) || !file.exists(.driver_path)) .driver_path = "fn_simulation_ex3.R"
.driver_dir = dirname(normalizePath(.driver_path, mustWork = TRUE))
source(file.path(.driver_dir, "dcqr_analysis_common.R"), local = TRUE)

run_ex3 = function(error="T", cfg=dcqr_control(), reps=cfg$reps) {
  # Fixed parameter setting
  Replication = length(replicate_id(reps))
  cfg$reps = Replication
  tau.set = cfg$taus
  err.dist = error_model(error)
  var.name = paste0("x", 1:12)
  signal = seq_along(var.name) <= 3L
  methods = c("Oracle", "DCQR", "G-DCQR", "AG-DCQR")
  penalized = c("G-DCQR", "AG-DCQR")

  # Storage
  MAE = metric_matrix(Replication, methods)
  time_mat = metric_matrix(Replication, methods)
  MAE.tau = method_array(Replication, tau.set, methods)
  var.mat = method_array(Replication, var.name, methods, FALSE)
  norm.mat = method_array(Replication, var.name, penalized)
  setting.list = grid.list = weights.list = par.list = vector("list", Replication)

  for (rep in seq_len(Replication)) {
    # 1. Training, validation and test data
    dat = sim_split("M2", err.dist, cfg, rep)
    X.train = dat$X$train
    X.val = dat$X$val
    X.test = dat$X$test
    Y.train = dat$train$y
    Y.val = dat$val$y
    Y.test = dat$q_true

    # 2. Oracle DCQR using x1, x2 and x3
    fit.oracle = tune_network("Oracle", X.train[, signal, drop=FALSE], Y.train,
                              X.val[, signal, drop=FALSE], Y.val, cfg, rep)
    pred.oracle = predict_network("Oracle", fit.oracle$fit, X.test[, signal, drop=FALSE])

    # 3. DCQR using all predictors
    fit.dcqr = tune_network("DCQR", X.train, Y.train, X.val, Y.val, cfg, rep)
    pred.dcqr = predict_network("DCQR", fit.dcqr$fit, X.test)

    # 4. Group Lasso
    fit.g = tune_network("G-DCQR", X.train, Y.train, X.val, Y.val, cfg, rep)
    pred.g = predict_network("G-DCQR", fit.g$fit, X.test)

    # 5. Adaptive group Lasso. DCQR supplies the pilot for this replication.
    weights.list[[rep]] = pilot_weights(fit.dcqr$fit, cfg$zeta)
    fit.ag = tune_network("AG-DCQR", X.train, Y.train, X.val, Y.val, cfg, rep,
                          omega=weights.list[[rep]])
    pred.ag = predict_network("AG-DCQR", fit.ag$fit, X.test)

    # 6. Quantile errors and selected variables
    fit.list = list(Oracle=fit.oracle, DCQR=fit.dcqr, "G-DCQR"=fit.g, "AG-DCQR"=fit.ag)
    pred.list = list(Oracle=pred.oracle, DCQR=pred.dcqr, "G-DCQR"=pred.g, "AG-DCQR"=pred.ag)
    for (m in methods) {
      MAE.tau[rep, , m] = colMeans(abs(pred.list[[m]] - Y.test))
      MAE[rep, m] = mean(MAE.tau[rep, , m])
      time_mat[rep, m] = fit.list[[m]]$seconds
      setting.list[[rep]][[m]] = fit.list[[m]]$setting
      grid.list[[rep]][[m]] = fit.list[[m]]$grid
    }

    var.mat[rep, signal, "Oracle"] = TRUE
    var.mat[rep, , "DCQR"] = TRUE
    for (m in penalized) {
      fit = fit.list[[m]]$fit
      norm.mat[rep, , m] = fit$group_norm
      var.mat[rep, , m] = fit$group_norm > cfg$threshold
      par.list[[rep]][[m]] = fit$par
    }
  }

  # Summary over replications
  sel = selection_summary(var.mat, signal)
  tab = data.frame(method=methods, Ave.MAE=colMeans(MAE), SE=apply(MAE, 2, standard_error),
                   sel$counts[methods, , drop=FALSE], sel$summary[, c("NC", "NIC", "NT")],
                   row.names=NULL, check.names=FALSE)
  group.mat = function(m) matrix(var.mat[, , m], nrow=Replication,
                                 dimnames=list(seq_len(Replication), var.name))
  weights.by.rep = function(m, layer) lapply(par.list, function(s) s[[m]][[layer]])
  list(summary=tab, MAE_summary_by_tau=summarize_levels(MAE.tau), MAE=MAE, MAE_by_tau=MAE.tau,
       time_mat=time_mat, NC.val=sel$NC.val, NIC.val=sel$NIC.val, NT.val=sel$NT.val,
       selected=var.mat, group_norm=norm.mat, selection=sel,
       var.val.PDCQR.mat=group.mat("G-DCQR"), var.val.APDCQR.mat=group.mat("AG-DCQR"),
       W1.hat.PDCQR=weights.by.rep("G-DCQR", "W1"), W0.hat.PDCQR=weights.by.rep("G-DCQR", "W0"),
       W1.hat.APDCQR=weights.by.rep("AG-DCQR", "W1"), W0.hat.APDCQR=weights.by.rep("AG-DCQR", "W0"),
       parameters=par.list, settings=setting.list, tuning=grid.list, pilot_weights=weights.list,
       config=cfg, error=error)
}

# Run the simulation
if (sys.nframe() == 0L) {
  opt = cli_options()
  result = run_ex3(error=opt$error, cfg=opt$cfg)
  save_analysis(result, opt$out, paste0("ex3_", opt$error))
  print(result$summary, row.names=FALSE)
}
