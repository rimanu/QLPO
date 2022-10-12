#### Read the data ####
# Function to read the data and change the names of the columns
read_data <- function(file){
  D <- read.csv(file)
  colnames(D)[1:2] <- c("RoundId", "VarType")
  return(D)
}

#### AUC ####
calculate_AUCs <- function(D){
  aucs <- bind_rows(lapply(unique(D$RoundId), function(r){
    data_subset <- t(as.data.frame(subset(D, RoundId == r)[,-1]))
    colnames(data_subset) <- data_subset[1,]
    data_subset <- as.data.frame(data_subset[-1,])
    data_subset$y <- as_reliable_num(data_subset$y)
    data_subset$RLS_QLPO <- as_reliable_num(data_subset$RLS_QLPO)
    data_subset$RLS_TLPO <- as_reliable_num(data_subset$RLS_TLPO)
    data_subset$RLS_P10F <- as_reliable_num(data_subset$RLS_P10F)
    data_subset$RLS_LOO <- as_reliable_num(data_subset$RLS_LOO)
    
    data_subset$RF_QLPO <- as_reliable_num(data_subset$RF_QLPO)
    data_subset$RF_TLPO <- as_reliable_num(data_subset$RF_TLPO)
    data_subset$RF_P10F <- as_reliable_num(data_subset$RF_P10F)
    data_subset$RF_LOO <- as_reliable_num(data_subset$RF_LOO)
    
    data_subset$LR_QLPO <- as_reliable_num(data_subset$LR_QLPO)
    data_subset$LR_TLPO <- as_reliable_num(data_subset$LR_TLPO)
    data_subset$LR_P10F <- as_reliable_num(data_subset$LR_P10F)
    data_subset$LR_LOO <- as_reliable_num(data_subset$LR_LOO)
    
    RLS_QLPO = as.numeric(auc(data_subset$y, data_subset$RLS_QLPO, direction ="<", quiet = TRUE))
    RLS_TLPO = as.numeric(auc(data_subset$y, data_subset$RLS_TLPO, direction = "<", quiet = TRUE))
    RLS_P10F = as.numeric(auc(data_subset$y, data_subset$RLS_P10F, direction = "<", quiet = TRUE))
    RLS_LOO = as.numeric(auc(data_subset$y, data_subset$RLS_LOO, direction = "<", quiet = TRUE))
    
    RF_QLPO = as.numeric(auc(data_subset$y, data_subset$RF_QLPO, direction ="<", quiet = TRUE))
    RF_TLPO = as.numeric(auc(data_subset$y, data_subset$RF_TLPO, direction = "<", quiet = TRUE))
    RF_P10F = as.numeric(auc(data_subset$y, data_subset$RF_P10F, direction = "<", quiet = TRUE))
    RF_LOO = as.numeric(auc(data_subset$y, data_subset$RF_LOO, direction = "<", quiet = TRUE))
    
    LR_QLPO = as.numeric(auc(data_subset$y, data_subset$LR_QLPO, direction ="<", quiet = TRUE))
    LR_TLPO = as.numeric(auc(data_subset$y, data_subset$LR_TLPO, direction = "<", quiet = TRUE))
    LR_P10F = as.numeric(auc(data_subset$y, data_subset$LR_P10F, direction = "<", quiet = TRUE))
    LR_LOO = as.numeric(auc(data_subset$y, data_subset$LR_LOO, direction = "<", quiet = TRUE))
    
    result <- as.data.frame(rbind(cbind("QLPO" = RLS_QLPO, "TLPO" = RLS_TLPO, "P10F" = RLS_P10F, "LOO" = RLS_LOO, "algorithm" = "RLS", "RoundID" = r),
                                  cbind("QLPO" = RF_QLPO, "TLPO" = RF_TLPO, "P10F" = RF_P10F, "LOO" = RF_LOO, "algorithm" = "RF", "RoundID" = r),
                                  cbind("QLPO" = LR_QLPO, "TLPO" = LR_TLPO, "P10F" = LR_P10F, "LOO" = LR_LOO, "algorithm" = "LR", "RoundID" = r)))
    colnames(result) <- c("QLPO", "TLPO", "P10F", "LOO", "algorithm", "RoundID")
    result
  }))
  return(aucs)
}

#### ROC ####
# Function to return a reversed list structure
reverseListStructure <- function(list_of_lists){
  RLS_QLPO <- list()
  RLS_TLPO <- list()
  RLS_P10F <- list()
  RLS_LOO <- list()
  
  RF_QLPO <- list()
  RF_TLPO <- list()
  RF_P10F <- list()
  RF_LOO <- list()
  
  LR_QLPO <- list()
  LR_TLPO <- list()
  LR_P10F <- list()
  LR_LOO <- list()
  
  for (i in 1:length(list_of_lists)) {
    RLS_QLPO <- append(RLS_QLPO, list(list_of_lists[[i]][["RLS_QLPO"]]))
    RLS_TLPO <- append(RLS_TLPO, list(list_of_lists[[i]][["RLS_TLPO"]]))
    RLS_P10F <- append(RLS_P10F, list(list_of_lists[[i]][["RLS_P10F"]]))
    RLS_LOO <- append(RLS_LOO, list(list_of_lists[[i]][["RLS_LOO"]]))
    
    RF_QLPO <- append(RF_QLPO, list(list_of_lists[[i]][["RF_QLPO"]]))
    RF_TLPO <- append(RF_TLPO, list(list_of_lists[[i]][["RF_TLPO"]]))
    RF_P10F <- append(RF_P10F, list(list_of_lists[[i]][["RF_P10F"]]))
    RF_LOO <- append(RF_LOO, list(list_of_lists[[i]][["RF_LOO"]]))
    
    LR_QLPO <- append(LR_QLPO, list(list_of_lists[[i]][["LR_QLPO"]]))
    LR_TLPO <- append(LR_TLPO, list(list_of_lists[[i]][["LR_TLPO"]]))
    LR_P10F <- append(LR_P10F, list(list_of_lists[[i]][["LR_P10F"]]))
    LR_LOO <- append(LR_LOO, list(list_of_lists[[i]][["LR_LOO"]]))
  }
  
  return(list(RLS_QLPO = RLS_QLPO, RLS_TLPO = RLS_TLPO, RLS_P10F = RLS_P10F, RLS_LOO = RLS_LOO, 
              RF_QLPO = RF_QLPO, RF_TLPO = RF_TLPO, RF_P10F = RF_P10F, RF_LOO = RF_LOO,
              LR_QLPO = LR_QLPO, LR_TLPO = LR_TLPO, LR_P10F = LR_P10F, LR_LOO = LR_LOO))
}

# Function to calculate the ROC curves
calculate_ROCs <- function(D){
  rocs <- lapply(unique(D$RoundId), function(r){
    data_subset <- t(as.data.frame(subset(D, RoundId == r)[,-1]))
    colnames(data_subset) <- data_subset[1,]
    data_subset <- as.data.frame(data_subset[-1,])
    data_subset$y <- as_reliable_num(data_subset$y)
    data_subset$RLS_QLPO <- as_reliable_num(data_subset$RLS_QLPO)
    data_subset$RLS_TLPO <- as_reliable_num(data_subset$RLS_TLPO)
    data_subset$RLS_P10F <- as_reliable_num(data_subset$RLS_P10F)
    data_subset$RLS_LOO <- as_reliable_num(data_subset$RLS_LOO)
    
    data_subset$RF_QLPO <- as_reliable_num(data_subset$RF_QLPO)
    data_subset$RF_TLPO <- as_reliable_num(data_subset$RF_TLPO)
    data_subset$RF_P10F <- as_reliable_num(data_subset$RF_P10F)
    data_subset$RF_LOO <- as_reliable_num(data_subset$RF_LOO)
    
    data_subset$LR_QLPO <- as_reliable_num(data_subset$LR_QLPO)
    data_subset$LR_TLPO <- as_reliable_num(data_subset$LR_TLPO)
    data_subset$LR_P10F <- as_reliable_num(data_subset$LR_P10F)
    data_subset$LR_LOO <- as_reliable_num(data_subset$LR_LOO)
    
    roc.list <- roc(y ~ RLS_QLPO + RLS_TLPO + RLS_P10F + RLS_LOO + RF_QLPO + RF_TLPO + RF_P10F + RF_LOO + LR_QLPO + LR_TLPO + LR_P10F + LR_LOO, data = data_subset,
                    direction = "<", quiet = TRUE)
  })
  return(reverseListStructure(rocs))
}

#### Sensitivity at given specificity ####
# Function to find out the sensitivity values
get_sensitivities <- function(roc_object){
  
  n_control <- length(roc_object$controls)
  FPR <- sort(rep(seq(0, 1, by = 1/n_control), 2))
  sp <- 1-FPR
  sens <- coords(roc_object, x = sp, input = "specificity", transpose = T, ret = "sensitivity")
  
  for (value_multiple_times in as.numeric(names(which(table(roc_object$specificities) > 1)))) {
    ind <- which(abs(sp-value_multiple_times) < 10^-8)
    spec_ind <- which(abs(roc_object$specificities-value_multiple_times) < 10^-8)
    sens[ind] <- c(min(roc_object$sensitivities[spec_ind]), max(roc_object$sensitivities[spec_ind]))
    # print(spec_ind)
  }
  
  # return(data.frame(TPR = sens, FPR))
  return(sens)
  
}

# Function to calculate true positive rate at given false positive rate
# D_list is a list of lists of ROC objects
sens_at_spec <- function(D_list, alpha){
  sensitivities_methods <- lapply(D_list, function(method_data){
    sensitivities_rounds <- t(sapply(method_data, function(round_data){
      get_sensitivities(round_data)
    }))
    
    result_specificities <- apply(sensitivities_rounds, 2, function(data_specificity){
      
      ave <- mean(data_specificity)
      sd <- sd(data_specificity)
      
      lower_conf <- max(0, ave - qnorm(1-alpha/2)*sd)
      upper_conf <- min(1, ave + qnorm(1-alpha/2)*sd)
      
      lower_cred <- as.numeric(quantile(data_specificity, probs = alpha/2))
      upper_cred <- as.numeric(quantile(data_specificity, probs = 1-alpha/2))
      
      result <- c(ave = ave, sd = sd, lower_conf = lower_conf, upper_conf = upper_conf, lower_cred = lower_cred, upper_cred = upper_cred)
      
    })
    
  })
  # It is assumed that the number of controls is a constant in a list of lists of ROC objects.
  n_control <- length(D_list[[1]][[1]]$controls)
  FPR <- sort(rep(seq(0, 1, by = 1/n_control), 2))
  
  RLS_QLPO <- cbind(t(sensitivities_methods[["RLS_QLPO"]]), method = "QLPO", algorithm = "RLS", FPR)
  RLS_TLPO <- cbind(t(sensitivities_methods[["RLS_TLPO"]]), method = "TLPO", algorithm = "RLS", FPR)
  RLS_P10F <- cbind(t(sensitivities_methods[["RLS_P10F"]]), method = "P10F", algorithm = "RLS", FPR)
  RLS_LOO <- cbind(t(sensitivities_methods[["RLS_LOO"]]), method = "LOO", algorithm = "RLS", FPR)
  
  RF_QLPO <- cbind(t(sensitivities_methods[["RF_QLPO"]]), method = "QLPO", algorithm = "RF", FPR)
  RF_TLPO <- cbind(t(sensitivities_methods[["RF_TLPO"]]), method = "TLPO", algorithm = "RF", FPR)
  RF_P10F <- cbind(t(sensitivities_methods[["RF_P10F"]]), method = "P10F", algorithm = "RF", FPR)
  RF_LOO <- cbind(t(sensitivities_methods[["RF_LOO"]]), method = "LOO", algorithm = "RF", FPR)
  
  LR_QLPO <- cbind(t(sensitivities_methods[["LR_QLPO"]]), method = "QLPO", algorithm = "LR", FPR)
  LR_TLPO <- cbind(t(sensitivities_methods[["LR_TLPO"]]), method = "TLPO", algorithm = "LR", FPR)
  LR_P10F <- cbind(t(sensitivities_methods[["LR_P10F"]]), method = "P10F", algorithm = "LR", FPR)
  LR_LOO <- cbind(t(sensitivities_methods[["LR_LOO"]]), method = "LOO", algorithm = "LR", FPR)
  
  result <- as.data.frame(rbind(RLS_QLPO, RLS_TLPO, RLS_P10F, RLS_LOO,RF_QLPO, RF_TLPO, RF_P10F, RF_LOO, LR_QLPO, LR_TLPO, LR_P10F, LR_LOO))
  
  result$ave <- as_reliable_num(result$ave)
  result$sd <- as_reliable_num(result$sd)
  result$lower_conf <- as_reliable_num(result$lower_conf)
  result$upper_conf <- as_reliable_num(result$upper_conf)
  result$lower_cred <- as_reliable_num(result$lower_cred)
  result$upper_cred <- as_reliable_num(result$upper_cred)
  result$FPR <- as_reliable_num(result$FPR)
  result$method <- factor(result$method, levels = c("QLPO", "TLPO", "P10F", "LOO"))
  
  return(result)
}



#### Simulation data ####
TPR_at_FPR <- function(file, alpha = 0.05){
  tpr_at_fpr <- read_data(file) %>%
    calculate_ROCs() %>%
    sens_at_spec(alpha)
  return(tpr_at_fpr)
}

AUC_averages_etc <- function(file, alpha = 0.05){
  aucs <- read_data(file) %>%
    calculate_AUCs() %>%
    melt(id = c("RoundID", "algorithm")) %>%
    group_by(algorithm, variable) %>%
    summarise(ave = mean(as_reliable_num(value)),
              sd = sd(as_reliable_num(value)),
              lower_conf = max(0, ave - qnorm(1-alpha/2)*sd),
              upper_conf = min(1, ave + qnorm(1-alpha/2)*sd),
              lower_cred = as.numeric(quantile(as_reliable_num(value), probs = alpha/2)), 
              upper_cred = as.numeric(quantile(as_reliable_num(value), probs = 1-alpha/2)))
  colnames(aucs)[2] <- "method"
  return(aucs)
}
#### Test data ####
# Callback function to be used together with function read_chunked_csv.
ftest <- function(x,pos){
  sens_chunk <- lapply(unique(x$`0`), function(r){
    # print(r)
    data_subset <- subset(x, `0` == r)
    column_names <- t(data_subset[,2])
    round_data <- as.data.frame(t(data_subset[,-c(1,2)]))
    colnames(round_data) <- column_names
    roc.list <-  roc(y_test ~ RLS_test + RF_test + LR_test, data = round_data,
                     direction = "<", quiet = TRUE)
    FPR <- seq(0,1,by = 1/300)
    RLS_sensitivities <- coords(roc.list[["RLS_test"]], x = 1-FPR, input = "specificity", transpose = T, ret = "sensitivity")
    RF_sensitivities <- coords(roc.list[["RF_test"]], x = 1-FPR, input = "specificity", transpose = T, ret = "sensitivity")
    LR_sensitivities <- coords(roc.list[["LR_test"]], x = 1-FPR, input = "specificity", transpose = T, ret = "sensitivity")
    result <- data.frame(RLS = RLS_sensitivities, RF = RF_sensitivities, LR = LR_sensitivities, FPR)
  })
  return(sens_chunk)
}

TPR_at_FPR_test_set <- function(file, chunk_size = 80, alpha = 0.05){
  tpr_at_fpr <- read_csv_chunked(file = file, callback = ListCallback$new(ftest), chunk_size = chunk_size) %>%
    melt(id = c("FPR")) %>%
    group_by(FPR, variable) %>%
    summarise(ave = mean(value),
              sd = sd(value),
              lower_conf = max(0, ave - qnorm(1-alpha/2)*sd),
              upper_conf = min(1, ave + qnorm(1-alpha/2)*sd),
              lower_cred = as.numeric(quantile(value, probs = alpha/2)), 
              upper_cred = as.numeric(quantile(value, probs = 1-alpha/2)))
  colnames(tpr_at_fpr)[2] = "algorithm"
  return(tpr_at_fpr)
}

# Function to calculate test data AUC values
f_AUC <- function(x,pos){
  
  aucs_chunk <- lapply(unique(x$`0`), function(r){
    data_subset <- subset(x, `0` == r)
    column_names <- t(data_subset[,2])
    round_data <- as.data.frame(t(data_subset[,-c(1,2)]))
    colnames(round_data) <- column_names
    roc_objects <-  roc(y_test ~ RLS_test + RF_test + LR_test, data = round_data,
                       direction = "<", quiet = TRUE)
    RLS_auc <- roc_objects[["RLS_test"]]$auc
    RF_auc <- roc_objects[["RF_test"]]$auc
    LR_auc <- roc_objects[["LR_test"]]$auc
    result <- data.frame(RLS = RLS_auc, RF = RF_auc, LR = LR_auc)
  })
  return(aucs_chunk)
}

AUC_averages_etc_test_set <- function(file, chunk_size = 80, alpha = 0.05){
  aucs <- read_csv_chunked(file, callback = ListCallback$new(f_AUC), chunk_size = chunk_size) %>%
    melt() %>%
    group_by(variable) %>%
    summarise(ave = mean(value),
              sd = sd(value),
              lower_conf = max(0, ave - qnorm(1-alpha/2)*sd),
              upper_conf = min(1, ave + qnorm(1-alpha/2)*sd),
              lower_cred = as.numeric(quantile(value, probs = alpha/2)), 
              upper_cred = as.numeric(quantile(value, probs = 1-alpha/2)))
  colnames(aucs)[1] <- "algorithm"
  return(aucs)
}
