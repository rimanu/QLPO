#### Simulation sampled from a population ####
# Import the data files  & calculate AUC values and the points needed for ROC curves.
result_folder <- "./Simulation/Results"
sample_sizes <- c(30) # c("30","100")
pos_fracts <- c("0.5") # c("0.1", "0.5")
theta_values <- c("0.5") # c("0.0", "0.25","0.5", "0.75", "1.0") 
for (ss in sample_sizes) {
  for (pf in pos_fracts) {
    for (theta in theta_values) {
      file_name_common_part <- paste0("_mp0.25_ns",ss,"_pf", pf,"_theta",theta)
      assign(paste0("auc",file_name_common_part), 
             AUC_averages_etc(paste0(result_folder,"results",file_name_common_part,".csv")))
      assign(paste0("auc_test",file_name_common_part), 
             AUC_averages_etc_test_set(paste0(result_folder,"test_results",file_name_common_part,".csv")))
      assign(paste0("tpr",file_name_common_part), 
             TPR_at_FPR(paste0(result_folder,"results",file_name_common_part,".csv")))
      assign(paste0("tpr_test",file_name_common_part), 
             TPR_at_FPR_test_set(paste0(result_folder,"test_results",file_name_common_part,".csv")))
    }
  }  
}

# Create a data frame that contains all AUC values calculated in the simulation study.
all_aucs <- data.frame()
for (ss in sample_sizes) {
  for (pf in pos_fracts) {
    for (theta in theta_values) {
      file_name_common_part <- paste0("_mp0.25_ns",ss,"_pf", pf,"_theta",theta)
      data_file <- get(paste0("auc", file_name_common_part))
      test_data_file <- get(paste0("auc_test", file_name_common_part))
      test_data_file$method <- "test"
      both_data_files <- rbind.fill(data_file, test_data_file)
      both_data_files$n_sample <- ss
      both_data_files$pos_fract <- pf
      both_data_files$theta <- theta
      
      all_aucs <- rbind(all_aucs, both_data_files)
    }
  }
}
all_aucs$method <- factor(all_aucs$method, levels = c("QLPO", "TLPO", "P10F", "LOO", "test"))
all_aucs$n_sample <- factor(all_aucs$n_sample, levels = c("30", "100"))

# Create a data frame that contains all TPR values for drawing the ROC curves.
all_tprs <- data.frame()
for (ss in sample_sizes) {
  for (pf in pos_fracts) {
    for (theta in theta_values) {
      file_name_common_part <- paste0("_mp0.25_ns",ss,"_pf", pf,"_theta",theta)
      data_file <- get(paste0("tpr", file_name_common_part))
      test_data_file <- get(paste0("tpr_test", file_name_common_part))
      test_data_file$method <- "test"
      both_data_files <- rbind.fill(data_file, test_data_file)
      both_data_files$n_sample <- ss
      both_data_files$pos_fract <- pf
      both_data_files$theta <- theta
      
      all_tprs <- rbind(all_tprs, both_data_files)
    }
  }
}
all_tprs$method <- factor(all_tprs$method, levels = c("QLPO", "TLPO", "P10F", "LOO", "test"))
all_tprs$n_sample <- factor(all_tprs$n_sample, levels = c("30", "100"))

#### Real data ####
# Import the data files & calculate AUC values and the points needed for ROC curves.
result_folder_real_data <- "./Real data/" 

auc_30 <- AUC_averages_etc(paste0(result_folder_real_data,"results_30.csv"))
auc_100 <- AUC_averages_etc(paste0(result_folder_real_data,"results_100.csv"))
test_auc_30 <- AUC_averages_etc_test_set(paste0(result_folder_real_data,"test_results_30.csv"))
test_auc_100 <- AUC_averages_etc_test_set(paste0(result_folder_real_data,"test_results_100.csv"))