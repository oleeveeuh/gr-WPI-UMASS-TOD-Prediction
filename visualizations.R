library(tidyverse)

total_error_df = read.csv("/Users/tillieslosser/Downloads/total_error_df")
wrangled_df <- total_error_df[, -1] |> 
  mutate(data = str_extract(total_error_df$full_df_name, ".*(?= )")) |> 
  mutate(model = case_when(model == "Stacking Regressor" ~ "Stacking", 
                           model == "K-Neighbors Regressor" ~ "K-Neigh", 
                           model == "Long Short-Term Memory Network" ~ "LSTM", 
                           model == "Support Vector Regressor" ~ "SVR", 
                           model == "Random Forest Regressor" ~ "Rand.\nForest", 
                           model == "Voting Regressor" ~ "Voting", 
                           model == "Linear Regressor" ~ "Lin.", 
                           model == "ExtraTreesRegressor" ~ "ExTrees", 
                           model == "Convolutional neural network" ~ "CNN", 
                           model == "BaggingRegressor" ~ "Bagging", 
                           model == "XGBoost Regressor" ~ "XGBoost", 
  )) |> 
  mutate(train_percent = str_extract(total_error_df$full_df_name, "(?<=Overall_).*(?=:)")) |> 
  mutate(normalization_method = str_extract(total_error_df$full_df_name, "(MinMax)|(Log)")) |> 
  mutate(DR_method = str_extract(total_error_df$full_df_name, "(PCA|ICA|Isomap|KPCA)_\\d{2}")) |> 
  mutate(norm_split = paste(wrangled_df$normalization_method, wrangled_df$train_percent, sep = "_")) |> 
  select(full_df_name, model, MSE:data, DR_method:norm_split) |> 
  group_by(full_df_name) |> 
  arrange(SMAPE, .by_group = TRUE) |> 
  distinct(full_df_name, .keep_all = TRUE)





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
