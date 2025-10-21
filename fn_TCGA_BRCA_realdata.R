##################################################################
#Disclaimer: this code applies mclapply for parallel computation 
# may not work properly under window envirionment.               #
##################################################################

rm(list=ls())

#setwd("set to your working directory.")
#load("BRCA_DCQR_BRCA_LumAB.RData")
#source("Deep_CQR_BANG.r")

library(mvtnorm)
library(VGAM)
library(nor1mix)
library(qrnn)
library(dplyr)
library(tidyr)
library(scatterplot3d)
library(parallel)

Replication <- 100
nepoch <- 10000
BRCA_data <- final_data
tau_groups <- c(0.2, 0.5, 0.8)
p <- ncol(BRCA_data) - 2

MPE_results <- list()
var_selection_gdcqr <- list()
var_selection_agdcqr <- list()
Weight1_gdcqr <- list()
Weight1_agdcqr <- list()

for (tau_val in tau_groups) {
  tau_name <- paste0("tau_", tau_val)
  MPE_results[[tau_name]] <- matrix(NA, nrow = Replication, ncol = 5,
                                    dimnames = list(NULL, c("KCQR", "D-MQR", "D-CQR", "G-DCQR", "AG-DCQR")))
  var_selection_gdcqr[[tau_name]] <- matrix(NA, nrow = Replication, ncol = p)
  var_selection_agdcqr[[tau_name]] <- matrix(NA, nrow = Replication, ncol = p)
  Weight1_gdcqr[[tau_name]] <- vector("list", Replication)
  Weight1_agdcqr[[tau_name]] <- vector("list", Replication)
}

time_mat <- matrix(NA, nrow = Replication, ncol = 5,
                   dimnames = list(NULL, c("KCQR", "D-MQR", "D-CQR", "G-DCQR", "AG-DCQR")))


total_start_time <- Sys.time()

for (rep in 1:Replication) {
  rep_start_time <- Sys.time()
  set.seed(rep)
  
  y <- BRCA_data$os_time_log
  x <- BRCA_data[, !(names(BRCA_data) %in% c("os_time_log", "diagnosis_age"))]
  
  n <- nrow(x)
  train_size <- floor(0.6 * n)
  val_size   <- floor(0.2 * n)
  train_indices <- sample(1:n, train_size)
  remaining_indices <- setdiff(1:n, train_indices)
  val_indices <- sample(remaining_indices, val_size)
  test_indices <- setdiff(remaining_indices, val_indices)
  
  X.train <- x[train_indices, ]; Y.train <- y[train_indices]
  X.val   <- x[val_indices, ];   Y.val   <- y[val_indices]
  X.test  <- x[test_indices, ];  Y.test  <- y[test_indices]
  
  cat(sprintf("\n========== Replication %d started ==========\n", rep))
  
  num_cores <- max(1, detectCores() - 1)
  cat(sprintf("🧠 Using %d cores (mclapply/fork)\n", num_cores))
  
  tau_results <- mclapply(tau_groups, function(tau_val) {
  
    cat(sprintf("\n--- Starting task for Rep: %d, Quantile: %.1f ---\n", rep, tau_val))
    
    node_1 <- 32; node_2 <- 32
    p <- ncol(X.train)
    
    W1_ini <- matrix(runif(node_1 * (p + 1), -1, 1), nrow = node_1)
    W2_ini <- matrix(runif(node_2 * (node_1 + 1), -1, 1), nrow = node_2)
    W0_ini_dcqr <- matrix(runif(1 * node_2, -1, 1), nrow = 1)
    B0_ini_dcqr <- matrix(runif(1, -1, 1), nrow = 1)
    
    ## =============================
    ## (1) KCQR (lambda, sigma tuning)
    ## =============================
    lambda_grid <- 10^seq(-3, 2, length.out = 6)   # 0.001 ~ 100
    sigma_grid  <- 10^seq(-1, 1, length.out = 5)   # 0.1 ~ 10

    tune_results <- expand.grid(lambda = lambda_grid, sigma = sigma_grid)
    tune_results$val_mpe <- NA

    for (i in seq_len(nrow(tune_results))) {
      lambda_try <- tune_results$lambda[i]
      sigma_try  <- tune_results$sigma[i]

      model_tmp <- tryCatch({
        Val.CKQR2(
          X.T = X.train, Y.T = Y.train,
          X.V = X.val,   Y.V = Y.val,
          TAUS = tau_val, L.S = lambda_try, S2.S = sigma_try
        )
      }, error = function(e) return(NULL))

      if (!is.null(model_tmp)) {
        y.val.tmp <- ckqr.predict.Q(X.train, X.val, Model = model_tmp$Best.model)
        tune_results$val_mpe[i] <- mean(rho(Y.val - y.val.tmp, tau_val))
      }
    }

    best_idx <- which.min(tune_results$val_mpe)
    best_lambda <- tune_results$lambda[best_idx]
    best_sigma  <- tune_results$sigma[best_idx]

    t0 <- Sys.time()
    model.sel.kcqr <- Val.CKQR2(X.T = X.train, Y.T = Y.train,
                                X.V = X.val,   Y.V = Y.val,TAUS = tau_val, L.S = best_lambda, S2.S = best_sigma
    )
    t1 <- Sys.time()
    y.test.kcqr <- ckqr.predict.Q(X.train, X.test, Model = model.sel.kcqr$Best.model)
    mpe.kcqr <- mean(rho(Y.test - y.test.kcqr, tau_val))
    time_kcqr <- as.numeric(difftime(t1, t0, units = "secs"))
    
    ## =============================
    # (2) D-MQR
    ## =============================
    t0 <- Sys.time()
    W0_ini_dmqr <- matrix(runif(1 * (node_2 + 1), -1, 1), nrow = 1)
    mod1 <- check2_mqr_val(W1=W1_ini, W2=W2_ini, W0=W0_ini_dmqr, X=cbind(1,X.train), y=Y.train,
                           Epoch=nepoch, Bsize=20, rate=0.005, X_val=cbind(1,X.val), y_val=Y.val, Taus=tau_val)
    t1 <- Sys.time()
    pred_model1 <- mCheck2_predict(W1=mod1$Weight1, W2=mod1$Weight2, W0=mod1$Weight0, X=cbind(1, X.test),
                                   y=Y.test, Taus=tau_val)
    time_dmqr <- as.numeric(difftime(t1, t0, units="secs"))
    
    ## =============================
    # (3) D-CQR
    ## =============================
    t0 <- Sys.time()
    mod2 <- check2_cqr_val(W1=W1_ini, W2=W2_ini, W0=W0_ini_dcqr, B0=B0_ini_dcqr, X=cbind(1,X.train), y=Y.train, 
                           Epoch=nepoch, Bsize=20, rate=0.005, X_val=cbind(1, X.val), y_val=Y.val, Taus=tau_val)
    t1 <- Sys.time()
    pred_model2 <- cCheck2_predict(W1=mod2$Weight1, W2=mod2$Weight2, W0=mod2$Weight0, B0=mod2$WeightB0,
                                   X=cbind(1, X.test), y=Y.test, Taus=tau_val)
    time_dcqr <- as.numeric(difftime(t1, t0, units="secs"))
    
    ## =============================
    # (4) G-DCQR
    ## =============================
    
    gamma_candidates <- 10^seq(-3, -1, length = 10)
    Weis_gdcqr <- c(0.2, rep(0.25, p))
  
    
    results_g <- lapply(gamma_candidates, function(gamma_try) {
      mod_tmp <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini_dcqr, B0=B0_ini_dcqr, X=cbind(1,X.train), y=Y.train,
                                   Epoch=nepoch, Bsize=20, rate=0.01, X_val=cbind(1, X.val), y_val=Y.val, Taus=tau_val,
                                   gamma=gamma_try, Weights=Weis_gdcqr)
      pred_tmp <- cCheck2_predict_grlasso(W1=mod_tmp$Weight1, W2=mod_tmp$Weight2, W0=mod_tmp$Weight0, B0=mod_tmp$B0,
                                          X=cbind(1, X.val), y=Y.val, Taus=tau_val, gamma=gamma_try, Weights=Weis_gdcqr)
      list(gamma=gamma_try, mpe=pred_tmp$MPE)
    })
    
    val_mpe_g <- sapply(results_g, `[[`, "mpe")
    best_gamma_gdcqr <- sapply(results_g, `[[`, "gamma")[which.min(val_mpe_g)]

    
    t0 <- Sys.time()
    mod3 <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini_dcqr, B0=B0_ini_dcqr, X=cbind(1, X.train), y=Y.train, 
                              Epoch=nepoch, Bsize=20, rate=0.005, X_val=cbind(1, X.val), y_val=Y.val, Taus=tau_val,
                              gamma=best_gamma_gdcqr, Weights=Weis_gdcqr)
    t1 <- Sys.time()
    pred_model3 <- cCheck2_predict_grlasso(W1=mod3$Weight1, W2=mod3$Weight2, W0=mod3$Weight0, B0=mod3$B0,
                                           X=cbind(1, X.test), y=Y.test, Taus=tau_val, gamma=best_gamma_gdcqr, Weights=Weis_gdcqr)
    time_gdcqr <- as.numeric(difftime(t1, t0, units="secs"))
    
    
    
    ## =============================
    # (5) AG-DCQR
    ## =============================
    
    cst_grid <- c("0.2"=0.7, "0.5"=1.0, "0.8"=0.7)
    cst_val <- cst_grid[as.character(tau_val)]
    agdcqr_gamma_candidates <- seq(0.05, 0.18, length=10)
    Weis_adaptive <- glasso_weight(W_ini=mod2$Weight1, cst=cst_val)
    
    results_ag <- lapply(agdcqr_gamma_candidates, function(gamma_try) {
      mod_tmp <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini_dcqr, B0=B0_ini_dcqr, X=cbind(1, X.train), y=Y.train,
                                   Epoch=nepoch, Bsize=20, rate=0.05, X_val=cbind(1, X.val), y_val=Y.val, Taus=tau_val,
                                   gamma=gamma_try, Weights=Weis_adaptive)
      pred_tmp <- cCheck2_predict_grlasso(W1=mod_tmp$Weight1, W2=mod_tmp$Weight2, W0=mod_tmp$Weight0, B0=mod_tmp$B0,
                                          X=cbind(1, X.val), y=Y.val, Taus=tau_val, gamma=gamma_try, Weights=Weis_adaptive)
      list(gamma=gamma_try, mpe=pred_tmp$MPE)
    })
    val_mpe_ag <- sapply(results_ag, `[[`, "mpe")
    best_gamma_agdcqr <- sapply(results_ag, `[[`, "gamma")[which.min(val_mpe_ag)]
    
    
    t0 <- Sys.time()
    mod4 <- check2_cqr_glasso(W1=W1_ini, W2=W2_ini, W0=W0_ini_dcqr, B0=B0_ini_dcqr, X=cbind(1, X.train), y=Y.train, 
                              Epoch=nepoch, Bsize=20, rate=0.005, X_val=cbind(1, X.val), y_val=Y.val, Taus=tau_val,
                              gamma=best_gamma_agdcqr, Weights=Weis_adaptive)
    t1 <- Sys.time()
    pred_model4 <- cCheck2_predict_grlasso(W1=mod4$Weight1, W2=mod4$Weight2, W0=mod4$Weight0, B0=mod4$B0,
                                           X=cbind(1, X.test), y=Y.test, Taus=tau_val, gamma=best_gamma_agdcqr, Weights=Weis_adaptive)
    time_agdcqr <- as.numeric(difftime(t1, t0, units="secs"))
    
    cat(sprintf("\n--- Finished task for Rep: %d, Quantile: %.1f ---\n", rep, tau_val))
    
    return(list(
      tau = tau_val, MPE_kcqr = mpe.kcqr, MPE_dmqr = pred_model1$MPE, MPE_dcqr = pred_model2$MPE, MPE_gdcqr = pred_model3$MPE, MPE_agdcqr = pred_model4$MPE,
      varsel_gdcqr = as.integer(apply(mod3$Weight1[, -1], 2, function(x) sqrt(sum(x^2))) > 0.0001),
      varsel_agdcqr = as.integer(apply(mod4$Weight1[, -1], 2, function(x) sqrt(sum(x^2))) > 0.0001),
      Weight1_gdcqr = mod3$Weight1[, -1], Weight1_agdcqr = mod4$Weight1[, -1],
      time_kcqr = time_kcqr, time_dmqr = time_dmqr, time_dcqr = time_dcqr, time_gdcqr = time_gdcqr, time_agdcqr = time_agdcqr
    ))
  }, mc.cores = num_cores)
  
  for (res in tau_results) {
    if (is.null(res) || !("MPE_dmqr" %in% names(res))) {
      cat(sprintf("Skipping failed result for tau = %s\n", res$tau))
      next
    }
    tau_name <- paste0("tau_", res$tau)
    MPE_results[[tau_name]][rep, ] <- c(res$MPE_kcqr, res$MPE_dmqr, res$MPE_dcqr, res$MPE_gdcqr, res$MPE_agdcqr)
    var_selection_gdcqr[[tau_name]][rep, ] <- res$varsel_gdcqr
    var_selection_agdcqr[[tau_name]][rep, ] <- res$varsel_agdcqr
    Weight1_gdcqr[[tau_name]][[rep]] <- res$Weight1_gdcqr
    Weight1_agdcqr[[tau_name]][[rep]] <- res$Weight1_agdcqr
    time_mat[rep, ] <- c(res$time_kcqr, res$time_dmqr, res$time_dcqr, res$time_gdcqr, res$time_agdcqr)
  }
  
  rep_end_time <- Sys.time()
  rep_time <- round(as.numeric(rep_end_time - rep_start_time, units="mins"), 2)
  cat(sprintf("\n✅ Replication %d completed (%.2f min elapsed)\n", rep, rep_time))
}

total_end_time <- Sys.time()
total_time <- round(as.numeric(total_end_time - total_start_time, units="mins"), 2)
cat(sprintf("\n🎉 Total simulation finished in %.2f minutes.\n", total_time))


# ##################### ####################
# # Visualize the results (optional)
# ##################### ####################
# library(dplyr)
# library(tidyr)
# library(purrr)
# library(ggplot2)
# library(scales)
# library(viridis)
# 
# tau_groups <- c(0.2, 0.5, 0.8)
# var_names <- colnames(BRCA_data)[!(names(BRCA_data) %in% c("os_time_log", "diagnosis_age"))]
# 
# MPE_summary <- data.frame(
#   Tau = tau_groups,
#   KCQR = NA,
#   DMQR = NA,
#   DCQR = NA,
#   G_DCQR = NA,
#   AG_DCQR = NA
# )
# 
# Time_summary <- data.frame(
#   Tau = tau_groups,
#   KCQR = NA,
#   DMQR = NA,
#   DCQR = NA,
#   G_DCQR = NA,
#   AG_DCQR = NA
# )
# 
# VarSel_summary_gdcqr <- list()
# VarSel_summary_agdcqr <- list()
# 
# 
# for (tau_val in tau_groups) {
#   tau_name <- paste0("tau_", tau_val)
#   
# 
#   MPE_mat <- MPE_results[[tau_name]]
#   MPE_mean <- colMeans(MPE_mat, na.rm = TRUE)
#   MPE_sd <- colMeans(MPE_mat, na.rm = TRUE)
#   MPE_summary[MPE_summary$Tau == tau_val, 2:6] <- MPE_mean
#   
#   
#   Time_summary[Time_summary$Tau == tau_val, 2:6] <- colMeans(time_mat, na.rm = TRUE)
#   
#   
#   VarSel_summary_gdcqr[[tau_name]] <- setNames(
#     colMeans(var_selection_gdcqr[[tau_name]], na.rm = TRUE),
#     var_names
#   )
#   VarSel_summary_agdcqr[[tau_name]] <- setNames(
#     colMeans(var_selection_agdcqr[[tau_name]], na.rm = TRUE),
#     var_names
#   )
# }
# 
# 
# cat("\n==== Mean Prediction Error (MPE) by τ ====\n")
# print(round(MPE_summary, 4))
# 
# cat("\n==== Mean Computation Time (seconds) by τ ====\n")
# print(round(Time_summary, 2))
# 
# 
# #################################################################
# # For MPE boxplot
# #################################################################
# 
# mpe_long <- bind_rows(
#   lapply(names(MPE_results), function(tau_name) {
#     tau_val <- sub("tau_", "", tau_name)
#     df <- as.data.frame(MPE_results[[tau_name]])
#     df$Tau <- tau_val
#     df
#   }),
#   .id = "Tau_ID"
# ) %>%
#   select(-Tau_ID) %>%
#   pivot_longer(cols = -Tau, names_to = "Method", values_to = "MPE") %>%
#   mutate(
#     Tau = factor(Tau, levels = c("0.2", "0.5", "0.8")),
#     Method = factor(Method, levels = c("KCQR", "D-MQR", "D-CQR", "G-DCQR", "AG-DCQR")),
#     Strata = recode(Tau,
#                     "0.2" = "Poor",
#                     "0.5" = "Intermediate",
#                     "0.8" = "Favorable"
#     ),
#     Strata = factor(Strata, levels = c("Poor", "Intermediate", "Favorable"))
#   )
# 
# 
# 
# 
# 
# method_colors <- c(
#   "KCQR" = "#E69F00",
#   "D-MQR" = "#56B4E9",
#   "D-CQR" = "#009E73",
#   "G-DCQR" = "#F0E442",
#   "AG-DCQR" = "#CC79A7"
# )
# 
# # ---- Boxplot with expression-based x-axis labels ----
# p <- ggplot(mpe_long, aes(x = Strata, y = MPE, fill = Method)) +
#   geom_boxplot(
#     position = position_dodge(width = 0.8),
#     width = 0.65,
#     outlier.shape = 21, outlier.size = 1.2, color = "black"
#   ) +
#   scale_fill_manual(values = method_colors) +
#   scale_x_discrete(
#     labels = c(
#       "Poor" = expression(bold("Poor") ~ "(" * tau == 0.2 * ")"),
#       "Intermediate" = expression(bold("Intermediate") ~ "(" * tau == 0.5 * ")"),
#       "Favorable" = expression(bold("Favorable") ~ "(" * tau == 0.8 * ")")
#     ),
#     expand = expansion(mult = c(0.05, 0.25))
#   ) +
#   labs(
#     title = "MPE by Prognostic Strata and Method",
#     x = "Prognostic strata",
#     y = "Mean Prediction Error"
#   ) +
#   theme_minimal(base_size = 13) +
#   theme(
#     plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
#     axis.title.x = element_text(size = 15, face = "bold", margin = margin(t = 10)),
#     axis.title.y = element_text(size = 15, face = "bold", margin = margin(r = 10)),
#     axis.text.x = element_text(size = 13),
#     axis.text.y = element_text(size = 11),
#     legend.position = "none",
#     panel.grid.major = element_line(color = "gray85", linetype = "dotted"),
#     panel.grid.minor = element_blank()
#   )
# 
# # Method labels below boxes
# label_df <- mpe_long %>%
#   distinct(Strata, Method) %>%
#   mutate(
#     x_pos = as.numeric(factor(Strata)) +
#       (as.numeric(factor(Method, levels = levels(mpe_long$Method))) - 3) * 0.15,
#     y_pos = 0.17
#   )
# 
# p +
#   geom_text(
#     data = label_df,
#     aes(x = x_pos, y = y_pos, label = Method, color = Method),
#     angle = 45, hjust = 0, size = 3.3, fontface = "bold", color = "black"
#   ) +
#   scale_color_manual(values = method_colors) +
#   coord_cartesian(ylim = c(0.15, 0.68))
# 
# #################################################################
# # For whole variable selection heatmap
# #################################################################
# genomic_instability_vars <- c(
#   "aneuploidy_score.x","tmb_nonsynonymous","fraction_genome_altered","log_aneuploidy"
# )
# 
# proliferation_vars <- c("proliferation","wound_healing")
# 
# immune_response_vars <- c("ifn_gamma_response","tgf_beta_response")
# 
# immune_cell_vars <- c(
#   "t_cells_cd8","t_cells_cd4","b_cells","nk_cells","macrophage",
#   "dendritic_cells","plasma_cells","t_cells_gamma_delta","mast_cells","eosinophils_41"
# )
# 
# Tumor_microenvironment_var <- c("stromal_fraction","buffa_hypoxia_score","winter_hypoxia_score")
# 
# clinical_vars <- c("diagnosis_age_sq")
# variable_order_biological <- c(
#   clinical_vars,
#   genomic_instability_vars,
#   proliferation_vars,
#   immune_response_vars,
#   immune_cell_vars,
#   Tumor_microenvironment_var
# )
# 
# variable_order_biological <- intersect(variable_order_biological, unique(var_freq_agdcqr_df$Variable))
# 
# 
# var_freq_agdcqr_df <- do.call(rbind, lapply(names(VarSel_summary_agdcqr), function(tau_name) {
#   data.frame(
#     Quantile = tau_name,
#     Variable = names(VarSel_summary_agdcqr[[tau_name]]),
#     Frequency = VarSel_summary_agdcqr[[tau_name]]
#   )
# })) %>%
#   mutate(Quantile = factor(Quantile, levels = paste0("tau_", tau_groups)))
# 
# 
# # === 5️⃣ Prepare color grouping ===
# plot_data_agdcqr <- var_freq_agdcqr_df %>%
#   mutate(
#     Variable = factor(Variable, levels = rev(variable_order_biological)),
#     FreqGroup = case_when(
#       Frequency < 0.5 ~ "< 50%",
#       Frequency >= 0.5 & Frequency < 0.7 ~ "50–70%",
#       Frequency >= 0.7 ~ "> 70%"
#     ),
#     FreqGroup = factor(FreqGroup, levels = c("< 50%", "50–70%", "> 70%"))
#   )
# 
# # 🎨 pastel-tone color palette (soft red, light yellow, pale green)
# freq_colors <- c(
#   "< 50%" = "#F4A7A1",   # pastel red
#   "50–70%" = "#F9E79F",  # pastel yellow
#   "> 70%"  = "#ABEBC6"   # pastel green
# )
# 
# # === 7️⃣ AG-DCQR heatmap ===
# ggplot(plot_data_agdcqr, aes(x = Quantile, y = Variable, fill = FreqGroup)) +
#   geom_tile(color = "white", linewidth = 0.5) +
#   geom_text(aes(label = percent(Frequency, accuracy = 1)), size = 3, color = "black") +
#   scale_fill_manual(values = freq_colors) +
#   scale_x_discrete(
#     labels = c(
#       "tau_0.2" = expression("Poor (" * tau == 0.2 * ")"),
#       "tau_0.5" = expression("Intermediate (" * tau == 0.5 * ")"),
#       "tau_0.8" = expression("Favorable (" * tau == 0.8 * ")")
#     )
#   ) +
#   labs(
#     title = "AG-DCQR Variable Selection Frequency by Prognostic Strata",
#     x = "Prognostic strata",
#     y = "Predictor Variables",
#     fill = "Selection\nFrequency"
#   ) +
#   theme_minimal(base_size = 12) +
#   theme(
#     axis.title.x = element_text(size = 14, face = "bold", margin = margin(t = 10)),
#     axis.title.y = element_text(size = 13, face = "bold", margin = margin(r = 10)),
#     axis.text.x = element_text(size = 12, face = "bold"),
#     axis.text.y = element_text(face = "bold"),
#     plot.title = element_text(hjust = 0.5, face = "bold"),
#     legend.position = "right"
#   )
# 
# 
# 
# #################################################################
# # For increasing & decreasing figures
# #################################################################
# selected_vars <- c("diagnosis_age_sq", "t_cells_cd8", "t_cells_cd4",
#                    "dendritic_cells")
# 
# var_summary <- var_freq_agdcqr_df %>%
#   filter(Variable %in% selected_vars) %>%
#   group_by(Variable, Quantile) %>%
#   summarise(Mean_Freq = mean(Frequency, na.rm = TRUE), .groups = "drop") %>%
#   mutate(
#     Quantile = factor(Quantile, levels = c("tau_0.2", "tau_0.5", "tau_0.8")),
#     Variable = factor(Variable, levels = selected_vars)
#   )
# 
# pretty_labels <- c(
#   "diagnosis_age_sq" = "Diagnosis age",
#   "t_cells_cd8"      = "T cells CD8",
#   "t_cells_cd4"      = "T cells CD4",
#   "dendritic_cells"  = "Dendritic cells"
# )
# 
# strata_labels <- c(
#   "tau_0.2" = expression("Poor (" * tau == 0.2 * ")"),
#   "tau_0.5" = expression("Intermediate (" * tau == 0.5 * ")"),
#   "tau_0.8" = expression("Favorable (" * tau == 0.8 * ")")
# )
# 
# 
# var_colors <- c(
#   "diagnosis_age_sq" = "#E41A1C",   # vivid red
#   "t_cells_cd8"      = "#377EB8",   # vivid blue
#   "t_cells_cd4"      = "#4DAF4A",   # vivid green
#   "dendritic_cells"  = "#FF7F00",   # vivid orange
#   "mast_cells"       = "#984EA3"    # vivid purple
# )
# 
# 
# line_styles <- rep("solid", length(selected_vars))
# names(line_styles) <- selected_vars
# 
# label_df <- var_summary %>%
#   filter(Quantile == "tau_0.8")
# 
# ggplot(var_summary,
#        aes(x = Quantile, y = Mean_Freq,
#            group = Variable, color = Variable, linetype = Variable)) +
#   geom_line(linewidth = 1.3) +
#   geom_point(size = 2.8, shape = 21, fill = "white", stroke = 1.1) +
#   
#   geom_text(
#     data = label_df,
#     aes(
#       x = as.numeric(Quantile) + 0.05,
#       y = Mean_Freq,
#       label = pretty_labels[as.character(Variable)]
#     ),
#     hjust = 0, size = 4, fontface = "bold", family = "Helvetica"
#   ) +
#   
#   scale_x_discrete(
#     labels = strata_labels,
#     expand = expansion(mult = c(0.05, 0.25))
#   ) +
#   scale_y_continuous(limits = c(0.4, 0.95), breaks = seq(0, 1, 0.1)) +
#   scale_color_manual(values = var_colors, labels = pretty_labels) +
#   scale_linetype_manual(values = line_styles, labels = pretty_labels) +
#   
#   labs(
#     title = "Increasing Selection Frequency across Prognostic Strata",
#     x = "Prognostic strata",
#     y = "Selection Frequency"
#   ) +
#   
#   theme_classic(base_family = "Helvetica", base_size = 13) +
#   theme(
#     plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
#     axis.title.x = element_text(face = "bold", size = 14, margin = margin(t = 10)),
#     axis.title.y = element_text(face = "bold", size = 14, margin = margin(r = 10)),
#     axis.text = element_text(size = 12, color = "black"),
#     panel.grid.major = element_line(color = "gray85", linetype = "dotted"),
#     panel.grid.minor = element_blank(),
#     legend.position = "none",
#     plot.margin = margin(10, 60, 10, 10)
#   )
# 
# 
# 
# #################################################################
# selected_vars <- c("aneuploidy_score.x", "tmb_nonsynonymous",
#                    "stromal_fraction", "buffa_hypoxia_score")
# 
# var_summary <- var_freq_agdcqr_df %>%
#   filter(Variable %in% selected_vars) %>%
#   group_by(Variable, Quantile) %>%
#   summarise(Mean_Freq = mean(Frequency, na.rm = TRUE), .groups = "drop") %>%
#   mutate(
#     Quantile = factor(Quantile, levels = c("tau_0.2", "tau_0.5", "tau_0.8")),
#     Variable = factor(Variable, levels = selected_vars)
#   )
# 
# 
# pretty_labels <- c(
#   "aneuploidy_score.x"  = "Aneuploidy score",
#   "tmb_nonsynonymous"   = "Tumor mutation burden",
#   "stromal_fraction"    = "Stromal fraction",
#   "buffa_hypoxia_score" = "Buffa hypoxia score"
# )
# 
# 
# strata_labels <- c(
#   "tau_0.2" = expression("Poor (" * tau == 0.2 * ")"),
#   "tau_0.5" = expression("Intermediate (" * tau == 0.5 * ")"),
#   "tau_0.8" = expression("Favorable (" * tau == 0.8 * ")")
# )
# 
# 
# var_colors <- c(
#   "aneuploidy_score.x"  = "#E41A1C",  # red
#   "tmb_nonsynonymous"   = "#377EB8",  # blue
#   "stromal_fraction"    = "#4DAF4A",  # green
#   "buffa_hypoxia_score" = "#FF7F00"   # orange
# )
# 
# line_styles <- rep("solid", length(selected_vars))
# names(line_styles) <- selected_vars
# 
# label_df <- var_summary %>%
#   filter(Quantile == "tau_0.8")
# 
# label_df <- label_df %>%
#   mutate(
#     nudge_y = case_when(
#       Variable == "aneuploidy_score.x"  ~ +0.015,
#       Variable == "tmb_nonsynonymous"   ~ +0.050,
#       Variable == "stromal_fraction"    ~ +0.015,
#       Variable == "buffa_hypoxia_score" ~ -0.020,
#       TRUE ~ 0
#     ),
#     nudge_x = case_when(
#       Variable == "aneuploidy_score.x"  ~ -0.020,
#       Variable == "tmb_nonsynonymous"   ~ -0.170,
#       Variable == "stromal_fraction"    ~ -0.040,
#       Variable == "buffa_hypoxia_score" ~ -0.060,
#       TRUE ~ 0
#     )
#   )
# 
# 
# ggplot(var_summary,
#        aes(x = Quantile, y = Mean_Freq,
#            group = Variable, color = Variable, linetype = Variable)) +
#   geom_line(linewidth = 1.3) +
#   geom_point(size = 2.8, shape = 21, fill = "white", stroke = 1.1) +
#   
# 
#   geom_text(
#     data = label_df,
#     aes(
#       x = as.numeric(Quantile) + nudge_x,
#       y = Mean_Freq + nudge_y,
#       label = pretty_labels[as.character(Variable)]
#     ),
#     hjust = 0, size = 4, fontface = "bold", family = "Helvetica"
#   ) +
#   
#   
#   scale_x_discrete(
#     labels = strata_labels,
#     expand = expansion(mult = c(0.05, 0.25))
#   ) +
#   scale_y_continuous(limits = c(0.33, 0.70), breaks = seq(0, 1, 0.1)) +
#   scale_color_manual(values = var_colors, labels = pretty_labels) +
#   scale_linetype_manual(values = line_styles, labels = pretty_labels) +
#   
#   labs(
#     title = "Decreasing Selection Frequency across Prognostic Strata",
#     x = "Prognostic strata",
#     y = "Selection Frequency"
#   ) +
#   
#   theme_classic(base_family = "Helvetica", base_size = 13) +
#   theme(
#     plot.title = element_text(hjust = 0.5, face = "bold", size = 15),
#     axis.title.x = element_text(face = "bold", size = 14, margin = margin(t = 10)),
#     axis.title.y = element_text(face = "bold", size = 14, margin = margin(r = 10)),
#     axis.text = element_text(size = 12, color = "black"),
#     panel.grid.major = element_line(color = "gray85", linetype = "dotted"),
#     panel.grid.minor = element_blank(),
#     legend.position = "none",
#     plot.margin = margin(10, 80, 10, 10)  
#   )

