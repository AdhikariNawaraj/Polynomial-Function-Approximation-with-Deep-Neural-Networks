# ===== 0) Data Setup =====
# Required libraries
library(keras3)
library(ggplot2)
library(dplyr)
library(tidyr)

set.seed(6298) # For the reproducibility
##=============================================================================================================================================
# ===== 1) Data Generation =====
n <- 200
x <- runif(n, min = -2, max = 2)

# Polynomial of degree 4 (≥ 3), plus small noise
true_f <- function(x) {
  2 - 0.5*x + 0.8*x^2 - 0.3*x^3 + 0.1*x^4
}
y <- true_f(x) + rnorm(n, sd = 0.4)

dat <- tibble(x = x, y = y)
#==============================================================================================================================================
# ===== 2) Initial Data Exploration =====
# First 10 observations of x and y
head(dat, 10)

#==============================================================================================================================================
# ===== 3) Visualization the Function (simulated data) =====
ggplot(dat, aes(x, y)) +
  geom_point(alpha = 0.6) +
  stat_function(fun = true_f, linewidth = 1) +
  labs(title = "Simulated Polynomial Data", x = "x", y = "y")

#=============================================================================================================================================
# ===== 4) Building the Model (DNN: 2 hidden layers, 3 neurons each) =====
model <- keras_model_sequential() %>%
  layer_dense(units = 3, activation = "tanh", input_shape = 1) %>%
  layer_dense(units = 3, activation = "tanh") %>%
  layer_dense(units = 1)

model %>% compile(optimizer = "adam", loss = "mse")

history <- model %>% fit(
  x = as.matrix(dat$x),
  y = as.matrix(dat$y),
  epochs = 800,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 0,
  callbacks = list(callback_early_stopping(patience = 50, restore_best_weights = TRUE))
)

#============================================================================================================================================
# ===== 5) Visualization of the Model Fit (fitted curve vs. true function) =====
grid <- seq(-2.5, 2.5, length.out = 400)
pred <- as.numeric(predict(model, as.matrix(grid)))
plot_df <- tibble(
  x = grid,
  True = true_f(grid),
  DNN = pred
) |> pivot_longer(cols = c(True, DNN), names_to = "series", values_to = "y")

ggplot() +
  geom_point(data = dat, aes(x, y), alpha = 0.3) +
  geom_line(data = plot_df, aes(x, y, color = series), linewidth = 1) +
  labs(title = "DNN Fit vs. True Polynomial", x = "x", y = "y")

#===========================================================================================================================================
# ===== 6) Model Evaluation (10 new observations + table) =====
x_new <- seq(-2, 2, length.out = 10)
pred_new <- as.numeric(predict(model, as.matrix(x_new)))
actual_new <- true_f(x_new)

eval_tbl <- tibble(
  Attribute_x   = round(x_new, 3),
  DNN_Prediction = round(pred_new, 3),
  Actual_Value   = round(actual_new, 3)
)
eval_tbl  # <- the requested table

#==== Calculating ME, RMSE, and MAE for evaluation ========
eval_tbl |>
  mutate(error = DNN_Prediction - Actual_Value,
         abs_error = abs(error)) |>
  summarise(ME = mean(error),
            RMSE = sqrt(mean(error^2)),
            MAE = mean(abs_error))

#=============================================================================================================================================
# ===== 7) Prediction vs. Actual Comparison (plot) =====
cmp_df <- tibble(x = x_new, Actual = actual_new, Predicted = pred_new) |>
  pivot_longer(cols = c(Actual, Predicted), names_to = "series", values_to = "y")

ggplot(cmp_df, aes(x, y, color = series)) +
  geom_point(size = 2) +
  geom_line() +
  labs(title = "Predicted vs Actual (10 Hold-out Points)", x = "x", y = "y")

