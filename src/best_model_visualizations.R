library(tidyverse)

# Read in Data
opt1_results = read.csv("src/opt_1_all_models_performance.csv")[, -1] #First column is index right now, might change later
opt2_results = read.csv("src/opt_2_all_models_performance.csv")[, -1]


wrangle <- function(df){
  wrangled_df <- df |> 
    mutate(data = str_extract(df$full_df_name, ".*(?= )")) |> 
    mutate(model = case_when(model == "Linear Regressor" ~ "Lin.", 
                             model == "Support Vector Regressor" ~ "SVR", 
                             model == "Decision Tree Regressor" ~ "Decision\nTree", 
                             model == "K-Neighbors Regressor" ~ "K-Neigh", 
                             model == "Stochastic Gradient Descent Regressor" ~ "Stoch.\nGrad. Desc.", 
                             model == "Voting Regressor" ~ "Voting", 
                             model == "Stacking Regressor" ~ "Stacking", 
                             model == "Gradient Boosting Regressor" ~ "Grad.\nBoost", 
                             model == "Random Forest Regressor" ~ "Rand.\nForest", 
                             model == "AdaBoostRegressor" ~ "Ada\nBoost", 
                             model == "BaggingRegressor" ~ "Bagging", 
                             model == "ExtraTreesRegressor" ~ "Extra\nTrees", 
                             model == "XGBoost Regressor" ~ "XGBoost", 
                             model == "Multilayer Perceptron" ~ "MLP", 
                             model == "Long Short-Term Memory Network" ~ "LSTM", 
                             model == "Convolutional neural network" ~ "CNN")) |> 
    mutate(train_percent = str_extract(df$full_df_name, "(?<=Overall_).*(?=:)")) |> 
    mutate(normalization_method = str_extract(df$full_df_name, "(MinMax)|(Log)")) |> 
    mutate(DR_method = str_extract(df$full_df_name, "(PCA|ICA|Isomap|KPCA)_\\d{2}")) |> 
    mutate(norm_split = paste(wrangled_df$normalization_method, wrangled_df$train_percent, sep = "_")) |> 
    select(full_df_name, model, MSE:data, DR_method:norm_split) |> 
    group_by(full_df_name) |> 
    arrange(SMAPE, .by_group = TRUE) |> 
    distinct(full_df_name, .keep_all = TRUE)
  return(wrangled_df)
}








ggplot(wrangled_df, aes(x = norm_split, y = DR_method, fill = MSE)) +
  geom_tile(color = "white") +
  geom_text(aes(label = model), color = "black", size = 3) +
  scale_fill_gradient(low = "blue", high = "red") +
  scale_shape_manual(values = c(0:25)) +  # Adjust the shape scale according to the `model` levels
  #labs(title = paste("Data Level:", data_level), fill = "SMAPE") +
  facet_wrap(~data)+
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggplot(wrangled_df, aes(x = norm_split, y = DR_method, fill = MAE)) +
  geom_tile(color = "white") +
  geom_text(aes(label = model), color = "black", size = 3) +
  scale_fill_gradient(low = "blue", high = "red") +
  scale_shape_manual(values = c(0:25)) +  # Adjust the shape scale according to the `model` levels
  #labs(title = paste("Data Level:", data_level), fill = "SMAPE") +
  facet_wrap(~data)+
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggplot(wrangled_df, aes(x = norm_split, y = DR_method, fill = MAPE)) +
  geom_tile(color = "white") +
  geom_text(aes(label = model), color = "black", size = 3) +
  scale_fill_gradient(low = "blue", high = "red") +
  scale_shape_manual(values = c(0:25)) +  # Adjust the shape scale according to the `model` levels
  #labs(title = paste("Data Level:", data_level), fill = "SMAPE") +
  facet_wrap(~data)+
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggplot(wrangled_df, aes(x = norm_split, y = DR_method, fill = RMSE)) +
  geom_tile(color = "white") +
  geom_text(aes(label = model), color = "black", size = 3) +
  scale_fill_gradient(low = "blue", high = "red") +
  scale_shape_manual(values = c(0:25)) +  # Adjust the shape scale according to the `model` levels
  #labs(title = paste("Data Level:", data_level), fill = "SMAPE") +
  facet_wrap(~data)+
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )

ggplot(wrangled_df, aes(x = norm_split, y = DR_method, fill = SMAPE)) +
  geom_tile(color = "white") +
  geom_text(aes(label = model), color = "black", size = 3) +
  scale_fill_gradient(low = "blue", high = "red") +
  scale_shape_manual(values = c(0:25)) +  # Adjust the shape scale according to the `model` levels
  #labs(title = paste("Data Level:", data_level), fill = "SMAPE") +
  facet_wrap(~data)+
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  )
