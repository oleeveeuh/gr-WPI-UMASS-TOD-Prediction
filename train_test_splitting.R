# Load in data
full_data <- read.csv("data/wrangled_data/full_data_6_17_2024.csv")
BA_11 <- read.csv("data/wrangled_data/BA11_data_6_17_2024.csv")
BA_47 <- read.csv("data/wrangled_data/BA47_data_6_17_2024.csv")


# Splits a dataframe into a list of dataframes based on the TOD. Each list-item contains the values
# for 1 hour window
split_hourly <- function(data) {
  # Initialize output list
  output_list <- list()
  # Starting at index = 1 (R indices start at 1 not 0), append the rows in the dataframe for which
  # TOD is in a 2 hour window. Loop through all possible hour values
  bin_cutoffs <- c(0:12) * 2
  for (i in 1:(length(bin_cutoffs) - 1)) {
    output_list[i] <- list(data[(data$TOD_pos >= bin_cutoffs[[i]] &
                                   data$TOD_pos < bin_cutoffs[[i +
                                                                 1]]), ])
  }
  return(output_list)
}

# Makes a list with 2 items: the data frame with the training data, and the dataframe with testing
# data
split_train_test <- function(data_list, train_percent, name) {
  #Make sure that the training percent in is appropriate decimal form
  stopifnot(train_percent <= 1 && train_percent >= 0)
  #Initialize data frames
  train_data <- data.frame()
  test_data <- data.frame()
  # For each data frame in list of data passed to function
  for (hour_data in data_list) {
    # calculate number of training data points (round up)
    num_train <- ceiling(train_percent * nrow(hour_data))
    # calculate number of testing data points
    num_test <- nrow(hour_data) - num_train
    # select the first num_train rows from the current hour_data frame
    train_data <- rbind(train_data, hour_data[1:num_train, ])
    # negatively select (remove) the first num_train rows from the hour_data frame. This results in
    # the last num_test rows to be kept
    test_data <- rbind(test_data, hour_data[-c(1:num_train), ])
    
    #Keeping for later in case we go back to the random sampling method
    # # Generate random logical vector of T/F with `num_train` TRUE values and `num_test` FALSE values
    # is_training <- sample(c(rep(TRUE, num_train), rep(FALSE, num_test)), nrow(hour_data) , replace = F)
    # # Subset current dataframe in for loop using logical from above, add to the train_data frame
    # train_data <- rbind(train_data, hour_data[is_training, ])
    # # Subset current dataframe in for loop using OPPOSITE (!) values as logical from above, add to
    # # the train_data frame
    # test_data <- rbind(test_data, hour_data[!is_training, ])
  }
  # Print some info to verify things worked / see final percent breakdowns
  cat("Total Observations:", nrow(train_data) + nrow(test_data))
  cat("\nReal % Testing Data:", nrow(test_data) / (nrow(train_data) + nrow(test_data)))
  return_list <- list(train_data, test_data)
  names(return_list) <- c(
    paste0(name, '_', train_percent * 100, "_train"),
    paste0(name, '_', train_percent * 100, "_test")
  )
  return(return_list)
}

min_max_normalize <- function(data_list) {
  for (col in 1:238) {
    min <- min(data_list[[1]][col])
    max <- max(data_list[[1]][col])
    data_list[[1]][col] <- (data_list[[1]][col] - min) / (max - min)
    data_list[[2]][col] <- (data_list[[2]][col] - min) / (max - min)
  }
  current_names <- names(data_list)
  names_front <- stringr::str_extract(current_names[[1]], "[BA1147full]{4,6}_\\d{2}_")
  names(data_list) <- c(paste0(names_front, "MM_train"), (paste0(names_front, "MM_test")))
  return(data_list)
}

log_normalize <- function(data_list) {
  for (col in c(3:238)) {
    data_list[[1]][col] <- log(data_list[[1]][col])
    data_list[[2]][col] <- log(data_list[[2]][col])
  }
  min <- min(data_list[[1]][1])
  max <- max(data_list[[1]][1])
  data_list[[1]][1] <- (data_list[[1]][1] - min )/ (max - min)
  data_list[[2]][1] <- (data_list[[2]][1] - min )/ (max - min)
  current_names <- names(data_list)
  names_front <- stringr::str_extract(current_names[[1]], "[BA1147full]{4,6}_\\d{2}_")
  names(data_list) <- c(paste0(names_front, "log_train"), paste0(names_front, "log_test"))
  return(data_list)
}

# Loop for all data

dataframes <- list(full_data, BA_11, BA_47)
df_names <- c("full", "BA11", "BA47")
train_percents <- c(0.8, 0.7, 0.6)

output <- list()
for (i in 1:3) {
  current_df <- dataframes[[i]]
  df_name <- df_names[[i]]
  df_hourly <- split_hourly(current_df)
  for (percent in train_percents) {
    temp_list <- split_train_test(df_hourly, percent, df_name)
    non_normalized <- temp_list
    names_front <- stringr::str_extract(names(non_normalized)[[1]], "[BA1147full]{4,6}_\\d{2}_")
    names(non_normalized) <- c(paste0(names_front, "nonnormalized_train"), (paste0(names_front, "nonnormalized_test")))
    output[[length(output) + 1]] <- list(min_max_normalize(temp_list),
                                         log_normalize(temp_list),
                                         non_normalized)
  }
}

all_dfs <- unlist(unlist(output, recursive = FALSE), recursive = FALSE)

for (df in names(all_dfs)) {
  names(all_dfs[[df]])[names(all_dfs[[df]]) == 'TOD_pos'] <- 'TOD'
}



save(all_dfs, file = "data/all_dfs.RData")

for (name in names(all_dfs)) {
  # Get the dataset
  dataset <- all_dfs[[name]]
  # Create the file name
  file_name <- paste0("data/train_test_split_data/", name, ".csv")
  # Write the dataset to a CSV file
  write.csv(dataset, file = file_name, row.names = FALSE)
}




# code for examining data and stuff ____________________________________________________________
for (df_name in names(all_dfs)) {
  df <- all_dfs[[df_name]]
  
  # Get the subset of the dataframe from the 4th to the 238th column
  subset_df <- df[, 4:238]
  
  # Find the minimum value and its column
  min_val <- min(subset_df)
  min_col <- which(subset_df == min_val, arr.ind = TRUE)[2]
  
  # Find the maximum value and its column
  max_val <- max(subset_df)
  max_col <- which(subset_df == max_val, arr.ind = TRUE)[2]
  
  # Print the information
  cat("Dataframe Name:", df_name, "\n")
  cat("Minimum Value:", min_val, "in Column:", min_col + 3, "\n") # +3 to adjust for original column index
  cat("Maximum Value:", max_val, "in Column:", max_col + 3, "\n") # +3 to adjust for original column index
  cat("\n")
}

for (df in names(all_dfs)) {
  if (grepl("train", df)) {
    for (col in colnames(all_dfs[[df]])) {
      if (min(all_dfs[[df]][col]) < 0) {
        cat("WARNING: ",df, col, " has a minimum value of ", min(all_dfs[[df]][col]), " not 0\n")
      }
    }
  }
}
