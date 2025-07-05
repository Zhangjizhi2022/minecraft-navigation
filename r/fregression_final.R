library(fda)
library(tidyverse)

# Load the data
updated_data <- read.csv('Updated_z_data.csv')
curve_data <- read.csv('avg_cluster01_curves.csv', row.names = 1)

# Rename columns for clarity (adjust based on new file structure)
colnames(updated_data) <- c(
  'df_name', 'age', 'gender', 'ethnicity', 'other_ethnicity', 'gaming_experience',
  'game_type', 'gaming_hours', 'minecraft_skill', 'curve_count',  'avg_original_length', 'avg_last_value', 'Env',  'removed_curves_count', 'cluster0', 'cluster1', 'cluster2', 'cluster3', 'time'
)

# Remove rows with missing relevant data
clean_data <- updated_data %>% 
  select(df_name, curve_count, avg_original_length, avg_last_value, age, gender, 
         gaming_experience, game_type, gaming_hours, minecraft_skill, removed_curves_count, 
         Env, cluster0, cluster1, cluster2, cluster3, time) %>%
  drop_na()

# Filter to match rows with df_name in curve data
clean_data <- clean_data %>%
  filter(df_name %in% rownames(curve_data)) %>%
  mutate(
    gender = as.factor(case_when(
      gender %in% c('Male', 'Female') ~ gender,
      TRUE ~ 'Other'
    )),
    gaming_experience = as.factor(gaming_experience),
    game_type = as.factor(game_type),
    minecraft_skill = as.factor(minecraft_skill),
    Env = as.factor(Env)
)

# Reorder to ensure matching order
curve_data <- curve_data[match(clean_data$df_name, rownames(curve_data)), ]

# Convert curve data to functional data object
x_basis <- create.bspline.basis(rangeval = c(1, ncol(curve_data)), nbasis = 5)
x_fd <- Data2fd(argvals = seq(1, ncol(curve_data)), y = as.matrix(t(curve_data)), basisobj = x_basis)

# Response variable and scalar covariates
y <- as.numeric(clean_data$curve_count)
scalar_covariates <- clean_data %>% 
  select(gender, gaming_experience, game_type, minecraft_skill, gaming_hours, 
         removed_curves_count, age, Env, cluster0, cluster1, cluster2, cluster3)

# Convert scalar variables to model matrix
scalar_matrix <- model.matrix(~ ., data = scalar_covariates)[, -1]

# 创建惩罚参数对象
beta_fdPar <- fdPar(fdobj = x_basis, Lfdobj = int2Lfd(2), lambda = 0.1)

# 用于 fRegress(, Z=cbind(scalar_matrix))
fregress_result <- fRegress(
  y = y,
  xfdlist = list(x_fd),
  betalist = list(beta_fdPar),
  zmat = scalar_matrix
)


# Extract functional regression coefficients beta(t)
beta_fd <- fregress_result$betaestlist[[1]]$fd
time_points <- seq(1, ncol(curve_data))

set.seed(2025)
B <- 1000  # Bootstrap iterations
beta_bootstrap <- matrix(NA, nrow = B, ncol = length(time_points))

for (b in 1:B) {
  sample_indices <- sample(1:nrow(clean_data), replace = TRUE)
  y_boot <- y[sample_indices]
  scalar_boot <- scalar_matrix[sample_indices, ]
  y_data_boot <- curve_data[sample_indices, ]
  x_fd_boot <- Data2fd(argvals = seq(1, ncol(curve_data)), y = as.matrix(t(y_data_boot)), basisobj = x_basis)
  fregress_boot <- fRegress(y_boot, list(x_fd_boot), list(fdPar(x_basis)), zmat = scalar_boot)
  beta_bootstrap[b, ] <- eval.fd(time_points, fregress_boot$betaestlist[[1]]$fd)
}

# Calculate confidence intervals and mean
beta_ci_lower <- apply(beta_bootstrap, 2, quantile, probs = 0.025)
beta_ci_upper <- apply(beta_bootstrap, 2, quantile, probs = 0.975)
beta_mean <- colMeans(beta_bootstrap)

# Set y-axis limits
y_min <- min(beta_ci_lower) * 1.1
y_max <- max(beta_ci_upper) * 1.1

# Plot beta(t) with confidence intervals
plot(time_points, beta_mean, type = "l", col = "blue", lwd = 2,
     xlab = "Time", ylab = "Beta(t)",
     main = "Beta(t) With 95% CI",
     ylim = c(y_min, y_max))
lines(time_points, beta_ci_lower, col = "red", lty = 2)
lines(time_points, beta_ci_upper, col = "red", lty = 2)
abline(h = 0, col = "black", lwd = 2)
legend("topleft", legend = c("Beta(t)", "95% CI"), col = c("blue", "red"),
       lty = c(1, 2), lwd = 2)



set.seed(2025)
B <- 1000  # Bootstrap iterations
beta_bootstrap <- matrix(NA, nrow = B, ncol = length(time_points))

# 惩罚参数对象
beta_fdPar <- fdPar(fdobj = x_basis, Lfdobj = int2Lfd(2), lambda = 0.1)

for (b in 1:B) {
  sample_indices <- sample(1:nrow(clean_data), replace = TRUE)

  # Bootstrap sampling
  y_boot <- y[sample_indices]
  scalar_boot <- scalar_matrix[sample_indices, ]
  y_data_boot <- curve_data[sample_indices, ]

  # Convert to functional data
  x_fd_boot <- Data2fd(argvals = seq(1, ncol(curve_data)), 
                       y = as.matrix(t(y_data_boot)), 
                       basisobj = x_basis)

  # Fit penalized functional regression
  fregress_boot <- fRegress(y_boot, list(x_fd_boot), list(beta_fdPar), zmat = scalar_boot)

  # Evaluate beta(t) at time points
  beta_bootstrap[b, ] <- eval.fd(time_points, fregress_boot$betaestlist[[1]]$fd)
}

# Compute CI and plot
beta_ci_lower <- apply(beta_bootstrap, 2, quantile, probs = 0.025)
beta_ci_upper <- apply(beta_bootstrap, 2, quantile, probs = 0.975)
beta_mean <- colMeans(beta_bootstrap)

y_min <- min(beta_ci_lower) * 1.1
y_max <- max(beta_ci_upper) * 1.1

# Plot
plot(time_points, beta_mean, type = "l", col = "blue", lwd = 2,
     xlab = "Time", ylab = "Beta(t)",
     main = "Penalized Beta(t) with 95% Bootstrap CI",
     ylim = c(y_min, y_max))
lines(time_points, beta_ci_lower, col = "red", lty = 2)
lines(time_points, beta_ci_upper, col = "red", lty = 2)
abline(h = 0, col = "black", lwd = 2)
legend("topleft", legend = c("Beta(t)", "95% CI"), col = c("blue", "red"),
       lty = c(1, 2), lwd = 2)




y <- as.numeric(clean_data$avg_last_value)

scalar_covariates <- clean_data %>% 
  select(gender, game_type, minecraft_skill, gaming_hours, 
         removed_curves_count, age, Env, cluster1, cluster2, cluster3, time)
# Linear regression for scalar covariates
#scalar_covariates <- scalar_covariates %>% 
#  mutate(cluster1_2 = cluster1 + cluster2) %>% 
#  select(-cluster1, -cluster2, -cluster3)

colnames(scalar_covariates) <- sub("^scalar_matrix", "", colnames(scalar_covariates))
scalar_matrix <- model.matrix(~ ., data = scalar_covariates)[, -1]

lm_model <- lm(y ~ scalar_matrix)

# Generate summary
summary_df <- summary(lm_model)

# Print summary
print(summary_df)



# 创建列联表
table_data <- table(clean_data$gaming_experience, clean_data$minecraft_skill)

# 查看列联表
print(table_data)

# 计算卡方检验
chi_test <- chisq.test(table_data)
print(chi_test)



library(fda)
library(tidyverse)
library(refund)

# Load data
updated_data <- read.csv('Updated_z_data.csv')
curve_data <- read.csv('avg_cluster01_curves.csv', row.names = 1)

curve_data2 <- read.csv('avg_cluster23_curves.csv', row.names = 1)

# Rename columns for clarity
colnames(updated_data) <- c(
  'df_name', 'age', 'gender', 'ethnicity', 'other_ethnicity', 'gaming_experience',
  'game_type', 'gaming_hours', 'minecraft_skill', 'curve_count',  'avg_original_length', 'avg_last_value', 'Env',  'removed_curves_count', 'cluster0', 'cluster1', 'cluster2', 'cluster3'
)

# Clean + align
clean_data <- updated_data %>% 
  select(df_name, curve_count, avg_original_length, avg_last_value, age, gender, 
         gaming_experience, game_type, gaming_hours, minecraft_skill, removed_curves_count, 
         Env, cluster0, cluster1, cluster2, cluster3) %>%
  drop_na() %>%
  filter(df_name %in% rownames(curve_data2)) %>%
  mutate(
    gender = factor(ifelse(gender %in% c("Male", "Female"), gender, "Other")),
    gaming_experience = factor(gaming_experience),
    game_type = factor(game_type),
    minecraft_skill = factor(minecraft_skill),
    Env = factor(Env)
  )

# Reorder curve_data to match clean_data
curve_data <- curve_data[match(clean_data$df_name, rownames(curve_data)), ]
x_mat <- as.matrix(curve_data)
time_grid <- seq(0, 1, length.out = ncol(x_mat))

clean_data$xmat <- I(x_mat)

# Reorder curve_data to match clean_data
curve_data2 <- curve_data2[match(clean_data$df_name, rownames(curve_data2)), ]
x_mat2 <- as.matrix(curve_data2)
time_grid <- seq(0, 1, length.out = ncol(x_mat2))

clean_data$xmat2 <- I(x_mat2)

# 使用 fpc()，注意：X 参数使用变量名，而不是对象
fit <- pfr(
  formula = avg_last_value ~ fpc(x_mat) + fpc(x_mat2),
  data = clean_data,
  method = "REML"
)


plot(fit)


data(DTI)
DTI1 <- DTI[DTI$visit==1 & complete.cases(DTI),]
par(mfrow=c(1,2))
# Fit model with linear functional term for CCA
fit.lf <- pfr(pasat ~ lf(cca, k=30, bs="ps"), data=DTI1)
plot(fit.lf, ylab=expression(paste(beta(t))), xlab="t")
## Not run:
# Alternative way to plot
bhat.lf <- coef(fit.lf, n=101)
bhat.lf$upper <- bhat.lf$value + 1.96*bhat.lf$se
bhat.lf$lower <- bhat.lf$value - 1.96*bhat.lf$se
matplot(bhat.lf$cca.argvals, bhat.lf[,c("value", "upper", "lower")],
type="l", lty=c(1,2,2), col=1,
ylab=expression(paste(beta(t))), xlab="t")
# Fit model with additive functional term for CCA, using tensor product basis
fit.af <- pfr(pasat ~ af(cca, Qtransform=TRUE, k=c(7,7)), data=DTI1)
plot(fit.af, scheme=2, xlab="t", ylab="cca(t)", main="Tensor Product")
plot(fit.af, scheme=2, Qtransform=TRUE,
xlab="t", ylab="cca(t)", main="Tensor Product")
# Change basistype to thin-plate regression splines
fit.af.s <- pfr(pasat ~ af(cca, basistype="s", Qtransform=TRUE, k=50),
data=DTI1)
plot(fit.af.s, scheme=2, xlab="t", ylab="cca(t)", main="TPRS", rug=FALSE)
plot(fit.af.s, scheme=2, Qtransform=TRUE,
xlab="t", ylab="cca(t)", main="TPRS", rug=FALSE)
# Visualize bivariate function at various values of x
par(mfrow=c(2,2))
vis.pfr(fit.af, xval=.2)
vis.pfr(fit.af, xval=.4)
vis.pfr(fit.af, xval=.6)
vis.pfr(fit.af, xval=.8)
