library(fastICA)

load("data/all_dfs.RData")

all_names <- names(all_dfs)
for (df in all_dfs) {
  selected_columns <- df[, -c(1, 4)]
  
}




# Select columns for ICA
selected_columns <- df[, c('A', 'B', 'C')]

# Perform ICA
ica_result <- fastICA(selected_columns, n.comp = 3)

# Access the independent components
ica_components <- as.data.frame(ica_result$S)

print(ica_components)
