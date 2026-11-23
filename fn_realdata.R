################################################################################
# Real data. Vancouver daily precipitation
################################################################################
# In RStudio
# source("fn_realdata.R")
# result = run_realdata()
# Rscript fn_realdata.R --out=results
# Load qrnn::YVRprecip or pass --data=/path/YVRprecip.rds
.driver_files = unlist(lapply(sys.frames(), function(f) f$ofile), use.names = FALSE)
.driver_path = if (length(.driver_files)) tail(.driver_files, 1L) else
  sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1L])
if (is.na(.driver_path) || !file.exists(.driver_path)) .driver_path = "fn_realdata.R"
.driver_dir = dirname(normalizePath(.driver_path, mustWork = TRUE))
source(file.path(.driver_dir, "dcqr_analysis_common.R"), local = TRUE)

################################################################################
# Reading data
################################################################################
prepare_yvr = function(data = NULL, offset = 0.1, noise_seed = 731L) {
  if (is.null(data)) {
    if (!requireNamespace("qrnn", quietly = TRUE))
      stop("Install qrnn for its YVRprecip dataset or supply a saved YVRprecip list.")
    env = new.env(parent = emptyenv())
    suppressWarnings(utils::data("YVRprecip", package = "qrnn", envir = env))
    if (!exists("YVRprecip", envir = env, inherits = FALSE))
      stop("Install qrnn for its YVRprecip dataset or supply a saved YVRprecip list.")
    data = env$YVRprecip
  }
  ncep = as.matrix(data$ncep)
  if (is.null(data$date) || nrow(ncep) != length(data$date) ||
      length(data$precip) != length(data$date)) stop("Invalid YVRprecip input.")
  precip = as.numeric(data$precip)
  if (length(offset) != 1L || !is.finite(offset) || offset <= 0)
    stop("offset must be positive so zero-rainfall days are included.")
  year = as.integer(format(as.Date(data$date), "%Y"))
  # qrnn documents the three columns as sea-level pressure, sh700 and z500.
  nm = tolower(gsub("[^[:alnum:]]", "", colnames(ncep)))
  humidity = which(nm %in% c("sh700", "shum700", "hus700", "q700", "specifichumidity700"))
  height = which(nm %in% c("z500", "zg500", "hgt500", "geopotentialheight500"))
  if (length(humidity) > 1L || length(height) > 1L) stop("Ambiguous NCEP predictor names.")
  pos = c(humidity[1L], height[1L])
  if (anyNA(pos)) {
    if (ncol(ncep) != 3L) stop("Expected NCEP covariates sh700 and z500.")
    pos[is.na(pos)] = c(2L, 3L)[is.na(pos)]
    if (anyDuplicated(pos))
      stop("Expected NCEP covariates sh700 and z500.")
  }
  X.weather = ncep[, pos, drop = FALSE]
  colnames(X.weather) = c("sh700", "z500")
  rows = which(year %in% 1996:2000 & is.finite(precip) & complete.cases(X.weather) &
                  apply(is.finite(X.weather), 1L, all))
  if (length(rows) < 24L || any(precip[rows] < 0))
    stop("Need complete, nonnegative precipitation observations in 1996-2000.")
  X.weather = X.weather[rows, , drop = FALSE]
  set.seed(noise_seed)
  noise.mat = matrix(runif(length(rows) * 5L, -5, 5), ncol = 5L,
                  dimnames = list(NULL, paste0("x", 1:5)))
  list(X = cbind(X.weather, noise.mat), y = log(precip[rows] + offset),
       date = as.Date(data$date)[rows], source_rows = rows, offset = offset)
}

################################################################################
# Analysis
################################################################################
run_realdata = function(data=NULL, cfg=dcqr_control(), reps=cfg$reps,
                        offset=0.1, noise_seed=731L, split_seed=2026L) {
  # Fixed parameter setting
  Replication = length(replicate_id(reps))
  cfg$reps = Replication
  tau.set = cfg$taus
  methods = c("DMQR", "DCQR", "G-DCQR", "AG-DCQR")
  penalized = c("G-DCQR", "AG-DCQR")

  dat = prepare_yvr(data, offset, noise_seed)
  X = dat$X
  Y = dat$y
  var.name = colnames(X)
  signal = var.name %in% c("sh700", "z500")
  n = nrow(X)

  # 1. Reserve 25% for validation and take the initial training draw from the rest.
  set.seed(split_seed)
  val = sample.int(n, floor(0.25*n))
  pool = setdiff(seq_len(n), val)
  n.train = floor(0.7*length(pool))
  set.seed(split_seed + 1L)
  train.pilot = pool[sample.int(length(pool), n.train)]
  X.scaled = .dcqr$standardize_split(X[train.pilot, , drop=FALSE], val=X[val, , drop=FALSE])
  X.train = X.scaled$train
  X.val = X.scaled$val
  Y.train = Y[train.pilot]
  Y.val = Y[val]

  # 2. Tune once on the full validation set
  fit.dmqr = tune_network("DMQR", X.train, Y.train, X.val, Y.val, cfg, 0L)
  fit.dcqr = tune_network("DCQR", X.train, Y.train, X.val, Y.val, cfg, 0L)
  fit.g = tune_network("G-DCQR", X.train, Y.train, X.val, Y.val, cfg, 0L)
  w.pilot = pilot_weights(fit.dcqr$fit, cfg$zeta)
  fit.ag = tune_network("AG-DCQR", X.train, Y.train, X.val, Y.val, cfg, 0L, omega=w.pilot)

  tune.list = list(DMQR=fit.dmqr, DCQR=fit.dcqr, "G-DCQR"=fit.g, "AG-DCQR"=fit.ag)
  fixed = grid.list = list()
  for (m in methods) {
    fixed[[m]] = tune.list[[m]]$setting
    grid.list[[m]] = tune.list[[m]]$grid
  }

  # Storage
  MPE = coverage = method_array(Replication, tau.set, methods)
  Ave.MPE = time_mat = metric_matrix(Replication, methods)
  var.mat = method_array(Replication, var.name, methods, FALSE)
  norm.mat = method_array(Replication, var.name, penalized)
  split.list = weights.list = par.list = vector("list", Replication)
  first = NULL

  for (rep in seq_len(Replication)) {
    # 3. New 70:30 training/test split from the same pool
    set.seed(split_seed + 1L + rep)
    train = pool[sample.int(length(pool), n.train)]
    test = setdiff(pool, train)
    split.list[[rep]] = list(train=train, test=test)
    X.scaled = .dcqr$standardize_split(X[train, , drop=FALSE], test=X[test, , drop=FALSE])
    X.train = X.scaled$train
    X.test = X.scaled$test
    Y.train = Y[train]
    Y.test = Y[test]

    # 4. Fit at the fixed settings. Only AG-DCQR uses the pilot weights.
    if (rep == 1L) {
      first = list(X=X[test, , drop=FALSE], observed=Y.test,
                   date=dat$date[test], predictions=list())
    }
    for (m in methods) {
      w = NULL
      if (m == "AG-DCQR") w = weights.list[[rep]]
      start.time = proc.time()[["elapsed"]]
      fit = fit_network(m, X.train, Y.train, cfg, fixed[[m]], rep, omega=w)
      time_mat[rep, m] = proc.time()[["elapsed"]] - start.time
      if (m == "DCQR") weights.list[[rep]] = pilot_weights(fit, cfg$zeta)

      # Test MPE and empirical coverage at each quantile
      pred.test = predict_network(m, fit, X.test)
      resid.test = residual_matrix(Y.test, pred.test)
      MPE[rep, , m] = rowMeans(.dcqr$rho_h_mat(t(resid.test), tau.set, h=0))
      Ave.MPE[rep, m] = mean(MPE[rep, , m])
      coverage[rep, , m] = colMeans(resid.test <= 0)
      if (rep == 1L) first$predictions[[m]] = pred.test

      # Variable selection, one indicator per input
      if (m %in% penalized) {
        norm.mat[rep, , m] = fit$group_norm
        var.mat[rep, , m] = fit$group_norm > cfg$threshold
        par.list[[rep]][[m]] = fit$par
      } else {
        var.mat[rep, , m] = TRUE
      }
    }
  }

  # Summary over splits
  sel = selection_summary(var.mat, signal)
  tab = data.frame(method=methods, Ave.MPE=colMeans(Ave.MPE), SE=apply(Ave.MPE, 2, standard_error),
                   sel$counts[methods, , drop=FALSE], NC=colMeans(sel$NC.val),
                   NIC=colMeans(sel$NIC.val), row.names=NULL, check.names=FALSE)
  group.mat = function(m) matrix(var.mat[, , m], nrow=Replication,
                                 dimnames=list(seq_len(Replication), var.name))
  weights.by.rep = function(m, layer) lapply(par.list, function(s) s[[m]][[layer]])
  list(summary=tab, MPE_summary_by_tau=summarize_levels(MPE), MPE=MPE, Ave.MPE=Ave.MPE,
       coverage=coverage, selected=var.mat, group_norm=norm.mat, selection=sel, time_mat=time_mat,
       NC.val=sel$NC.val, NIC.val=sel$NIC.val, NT.val=sel$NT.val,
       var.val.GDCQR.mat=group.mat("G-DCQR"), var.val.AGDCQR.mat=group.mat("AG-DCQR"),
       W1.hat.GDCQR=weights.by.rep("G-DCQR", "W1"), W0.hat.GDCQR=weights.by.rep("G-DCQR", "W0"),
       W1.hat.AGDCQR=weights.by.rep("AG-DCQR", "W1"), W0.hat.AGDCQR=weights.by.rep("AG-DCQR", "W0"),
       parameters=par.list, settings=fixed, tuning=grid.list, pilot_weights=weights.list, config=cfg,
       reserve=val, tuning_train=train.pilot, tuning_validation=val, splits=split.list,
       date=dat$date, source_rows=dat$source_rows, offset=offset,
       seeds=c(noise=noise_seed, split=split_seed), first=first)
}

################################################################################
# Plot. First split
################################################################################
plot_yvr = function(result, path, methods = c("DCQR", "AG-DCQR")) {
  plot.data = result$first
  taus = result$config$taus
  show.tau = rep(FALSE, length(taus))
  for (k in seq_along(taus)) show.tau[k] = any(abs(taus[k] - c(0.3, 0.5, 0.7)) < 1e-8)
  k = which(show.tau)
  if (!length(k)) k = seq_along(taus)
  cols = rep(c("black", "royalblue3", "firebrick3"), length.out = length(k))
  pch = rep(c(16L, 17L, 18L), length.out = length(k))
  sh700 = plot.data$X[, "sh700"]
  z500 = plot.data$X[, "z500"]
  zlim = range(lapply(plot.data$predictions[methods], function(q) q[, k, drop = FALSE]))
  limits = function(v) {
    lim = range(v)
    if (diff(lim) == 0) lim = lim + c(-0.5, 0.5)
    grDevices::extendrange(lim, f = 0.03)
  }
  # Constant predictions are possible for a fully shrunk network.
  if (diff(zlim) == 0) zlim = zlim + c(-0.5, 0.5)
  grDevices::pdf(path, width = 7.3, height = 6.4)
  on.exit(grDevices::dev.off(), add = TRUE)
  graphics::par(mar = c(3.8, 4.2, 4.5, 2.0))
  xl = limits(sh700)
  yl = limits(z500)
  zl = limits(zlim)
  for (m in methods) {
    p = graphics::persp(xl, yl, matrix(zl[1], 2L, 2L), zlim = zl,
      theta = 36, phi = 18, expand = 0.8, col = NA, border = NA, axes = FALSE, box = FALSE)
    usr = graphics::par("usr")
    sx = diff(usr[1:2])
    sy = diff(usr[3:4])
    projection = function(x, y, z) grDevices::trans3d(x, y, z, p)
    ticks = function(lim) {
      v = pretty(lim, n = 4L)
      v[v >= lim[1] & v <= lim[2]]
    }
    # Draw tick labels as ordinary base-graphics text on each straight axis.
    axis3 = function(lim, at, offset, label, title_offset, rotation = 0) {
      graphics::lines(at(lim))
      values = ticks(lim)
      xy = at(values)
      dx = offset[1] * sx
      dy = offset[2] * sy
      graphics::segments(xy$x, xy$y, xy$x + dx, xy$y + dy)
      graphics::text(xy$x + 3 * dx, xy$y + 3 * dy, format(values, trim = TRUE),
                     cex = 0.85, xpd = NA)
      mid = at(mean(lim))
      graphics::text(mid$x + title_offset[1] * sx, mid$y + title_offset[2] * sy,
                     label, srt = rotation, xpd = NA)
    }
    graphics::lines(projection(c(xl[1], xl[1], xl[2]), c(yl[1], yl[2], yl[2]), zl[1]),
                    col = "grey80")
    axis3(xl, function(v) projection(v, yl[1], zl[1]), c(-0.002, -0.012),
          "sh700", c(-0.008, -0.12))
    axis3(yl, function(v) projection(xl[2], v, zl[1]), c(0.010, -0.007),
          "z500", c(0.11, -0.07))
    axis3(zl, function(v) projection(xl[1], yl[1], v), c(-0.011, 0),
          "log(Precipitation)", c(-0.13, 0), rotation = 90)
    for (j in seq_along(k)) {
      coord = grDevices::trans3d(sh700, z500, plot.data$predictions[[m]][, k[j]], p)
      graphics::points(coord, col = cols[j], pch = pch[j], cex = 0.6)
    }
    graphics::title(main = m, line = 2.6)
    graphics::legend("top", inset = c(0, -0.08), horiz = TRUE,
                     legend = paste("tau =", taus[k]), col = cols,
                     pch = pch, bty = "n", xpd = NA)
  }
}

# Run the analysis
if (sys.nframe() == 0L) {
  opt = cli_options()
  data.yvr = NULL
  if (!is.null(opt$data)) data.yvr = readRDS(opt$data)
  result = run_realdata(data=data.yvr, cfg=opt$cfg)
  save_analysis(result, opt$out, "vancouver")
  plot_yvr(result, file.path(opt$out, "vancouver_quantiles.pdf"))
  print(result$summary, row.names=FALSE)
}
