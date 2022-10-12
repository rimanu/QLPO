#### Libraries ####
library(ggplot2)
library(patchwork)

#### Simulation AUC plots ####
sample_sizes <- c("30","100")
for (ss in sample_sizes) {
    assign(paste0("g_ss",ss),
           ggplot() + 
             geom_point(
               data = subset(all_aucs, n_sample == ss),
               aes(x = pos_fract, y = ave, group = method, col = method, shape = method),
               position = position_dodge(width = 0.5), 
               show.legend = TRUE) +
             geom_errorbar(
               data = subset(all_aucs, n_sample == ss),
               aes(x = pos_fract, ymin = lower_cred, ymax = upper_cred, group = method, col = method),
               position = position_dodge(width = 0.5), 
               size = 0.4,
               show.legend = TRUE) +
             ylim(c(0,1)) + 
             scale_colour_manual(name = "",
                                 values = c("#F8766D", "#00BFC4", "#7CAE00", "#C77CFF", "#000000"),
                                 labels = c("QLPO", "TLPO", "P10F", "LOO", "Approx. true AUC")) +
             scale_shape_manual(name = "",
                                values = c(16, 17, 15, 8, 3),
                                labels = c("QLPO", "TLPO", "P10F", "LOO", "Approx. true AUC")) + 
             labs(y = "AUC", x = "Fraction of positives",
                  col = "Method", shape = "Method") +
             facet_grid(algorithm~theta, labeller = label_bquote(cols = theta == .(theta))) + 
             theme(text = element_text(size = 8),
             ) +
             NULL)
}

(guide_area() / g_ss30 / plot_spacer() / g_ss100) + 
  plot_layout(guides = "collect", widths = c(50, 50), heights = c(5, 50, 0, 50)) + 
  plot_annotation(tag_levels = list(c(30, 100)), tag_prefix = "Sample size = ") & 
  theme(legend.position = "top",
        legend.justification = "right",
        plot.tag.position = c(0, 1),
        plot.tag = element_text(hjust = 0, vjust = -1),)
ggsave("Fig4.eps", device = "eps", width = 174, height = 174, units = "mm")

#### Simulation ROC plots ####
# Plot the ROC curves
params <- list(c("30","0.5","0.0"), c("30","0.5","1.0"), c("100","0.1","0.0"), c("100","0.1","1.0"))
for (p in params) {
  ss <- p[1]
  pf <- p[2]
  tv <- p[3]
  
  assign(paste0("g_ss",ss,"_pf",pf, "_theta", tv),
         ggplot() +
          geom_path(data = subset(all_tprs, n_sample == ss & pos_fract == pf & method != "test" & theta == tv), 
                    aes(x = FPR, y = ave, col = method), size = 0.3, show.legend = FALSE) +
          geom_ribbon(data = subset(all_tprs, n_sample == ss & pos_fract == pf & method != "test" & theta == tv), 
                      aes(x = FPR, 
                          ymin = lower_cred, ymax = upper_cred,
                          col = method, fill = method, size = 0.3), 
                      alpha = 0.1,
                      show.legend = FALSE
                      ) +
          geom_line(data = subset(all_tprs, n_sample == ss & pos_fract == pf & method == "test" & theta == tv)[,-7], 
                    aes(x = FPR, y = ave, size = 0.3),
                    linetype = "dashed") +
          scale_size_identity(name = "Approx. true ROC curve", labels = NULL, guide = "legend") +
          scale_colour_manual(values = c("#F8766D", "#00BFC4", "#7CAE00", "#C77CFF"),
                              labels = c("QLPO", "TLPO", "P10F", "LOO")) +
          scale_fill_manual(values = c("#F8766D", "#00BFC4", "#7CAE00", "#C77CFF"),
                            labels = c("QLPO", "TLPO", "P10F", "LOO")) +
          labs( 
               y = "TPR",
               col = "CV method", fill = "CV method"
               ) +
          theme(axis.text.x = element_text(angle = 90, vjust = 0),
                text = element_text(size = 8)) +
          facet_grid(algorithm~method))
}
(guide_area() / (g_ss30_pf0.1_theta0.0 + g_ss30_pf0.1_theta1.0) /
    (g_ss100_pf0.5_theta0.0 + g_ss100_pf0.5_theta1.0)) + 
  plot_layout(guides = "collect", tag_level = "new", widths = c(50, 50), heights = c(10, 50, 50)) +
  plot_annotation(tag_levels = "a", tag_suffix = ")") &
  theme(legend.position = "top",
        legend.justification = "right",
        plot.tag.position = c(0, 1),
        plot.tag = element_text(hjust = 0, vjust = -1),
        text = element_text(size = 8))
ggsave("Fig3.eps", device = cairo_ps, width = 174, height = 150, units = "mm")

#### Real data plots ####
g_real_ss100 <- ggplot(results_100) + 
  geom_path(aes(x = FPR, y = ave, col = method), show.legend = FALSE) +
  geom_ribbon(aes(x = FPR, ymin = lower_cred, ymax = upper_cred, col = method, fill = method, size = 0.3), 
              alpha = 0.1,
              show.legend = FALSE) +
  geom_line(aes(x = FPR, y = ave, size = 0.3), data = test_results_100,
            linetype = "dashed") +
  scale_size_identity(name = "Approx. true ROC curve", labels = NULL, guide = "legend") +
  scale_colour_manual(values = c("#F8766D", "#00BFC4", "#7CAE00", "#C77CFF"),
                      labels = c("QLPO", "TLPO", "P10F", "LOO")) +
  scale_fill_manual(values = c("#F8766D", "#00BFC4", "#7CAE00", "#C77CFF"),
                    labels = c("QLPO", "TLPO", "P10F", "LOO")) +
  labs( 
    y = "TPR",
    col = "CV method", fill = "CV method"
  ) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0)) +
  facet_grid(algorithm~method)

g_real_ss30 <- ggplot(results_30) + 
  geom_path(aes(x = FPR, y = ave, col = method), show.legend = FALSE) +
  geom_ribbon(aes(x = FPR, ymin = lower_cred, ymax = upper_cred, col = method, fill = method, size = 0.3), 
              alpha = 0.1,
              show.legend = FALSE) +
  geom_line(aes(x = FPR, y = ave, size = 0.3), data = test_results_30,
            linetype = "dashed") +
  scale_size_identity(name = "Approx. true ROC curve", labels = NULL, guide = "legend") +
  scale_colour_manual(values = c("#F8766D", "#00BFC4", "#7CAE00", "#C77CFF"),
                      labels = c("QLPO", "TLPO", "P10F", "LOO")) +
  scale_fill_manual(values = c("#F8766D", "#00BFC4", "#7CAE00", "#C77CFF"),
                    labels = c("QLPO", "TLPO", "P10F", "LOO")) +
  labs( 
    y = "TPR",
    col = "CV method", fill = "CV method"
  ) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0)) +
  facet_grid(algorithm~method)

# AUC
auc_30$n_sample <- 30
auc_100$n_sample <- 100
test_auc_30$method <- "test"
test_auc_30$n_sample <- 30
test_auc_100$method <- "test"
test_auc_100$n_sample <- 100
simulation_aucs_all <- rbind(auc_30, auc_100, test_auc_30, test_auc_100)
simulation_aucs_all$method <- factor(simulation_aucs_all$method, levels = c("QLPO", "TLPO", "P10F", "LOO", "test"))

g_real_auc <- ggplot(simulation_aucs_all) + 
  geom_point(aes(x = method, y = ave, col = method, shape = method),
             ) + 
  geom_errorbar(aes(x = method, ymin = lower_cred, ymax = upper_cred,
                    col = method),
                size = 0.4
                ) +
  ylim(c(0,1)) + 
  scale_colour_manual(name = "",
                      values = c("#F8766D", "#00BFC4", "#7CAE00", "#C77CFF", "#000000"),
                      labels = c("QLPO", "TLPO", "P10F", "LOO", "Approx. true AUC")) +
  scale_shape_manual(name = "",
                     values = c(16, 17, 15, 8, 3),
                     labels = c("QLPO", "TLPO", "P10F", "LOO", "Approx. true AUC")) + 
  labs(y = "AUC", x = "",
       col = "Method", shape = "Method") +
  facet_grid(algorithm~n_sample)

# Plot
(g_real_ss30 + g_real_ss100 + plot_layout(guides = "collect")) / 
  (g_real_auc + plot_layout(tag_level = "new") & 
     theme(axis.text.x = element_blank(),
           axis.ticks.x = element_blank())) +
  plot_layout(widths = c(50, 50), heights = c(50, 80)) +
  plot_annotation(tag_levels = list(c(30,100)), tag_prefix = "Sample size = ") &
  theme(legend.position = "top",
        legend.justification = "right",
        plot.tag.position = c(0.05, 0.8),
        plot.tag = element_text(hjust = 0, vjust = -1),
        text = element_text(size = 8))
ggsave("Fig5.eps", device = cairo_ps, width = 174, height = 174, units = "mm")

#### Simulation data distribution ####
populations <- data.frame()
for (th in theta_values) {
  populations <- rbind(populations, cbind(get(paste0("population_data_mp0.25_theta",th,"_popsizeE6.0.csv"))[,2:4], theta = th))
}

g_f1f2 <- ggplot(populations) + 
  geom_density2d(aes(x = Feature1, 
                     y = Feature2, 
                     group = Class, 
                     col = factor(Class, levels = c(1,-1)),
                     linetype = factor(Class, levels = c(1,-1))
                     )
                 ) + 
  labs(col = "Class", linetype = "Class", x = expression(X[1]), y = expression(X[2])) + 
  facet_wrap(~theta, labeller = label_bquote(cols = theta == .(theta)), ncol = 5) + 
  theme(legend.position = "top",
        legend.justification = "right",
        text = element_text(size = 8))
ggsave("Fig2.eps", plot = g_f1f2, device = cairo_ps, width = 174, height = 60, units = "mm")
