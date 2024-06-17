# Load in data
full_data <- read.csv("data/wrangled data/full_data_6_17_2024.csv")
BA_11 <- read.csv("data/wrangled data/BA11_data_6_17_2024.csv")
BA_47 <- read.csv("data/wrangled data/BA47_data_6_17_2024.csv")


# Splits a dataframe into a list of dataframes based on the TOD. Each list-item contains the values
# for 1 hour window
split_hourly <- function(data) {
  # Initialize output list
  output_list <- list()
  # Starting at index = 1 (R indices start at 1 not 0), append the rows in the dataframe for which
  # TOD is in a 2 hour window. Loop through all possible hour values
  bin_cutoffs <- c(0:12) * 2
  for (i in 1:(length(bin_cutoffs)-1)) {
    output_list[i] <- list(data[(data$TOD_pos >= bin_cutoffs[[i]] &
                                          data$TOD_pos < bin_cutoffs[[i+1]]), ])
  }
  return(output_list)
}

# Makes a list with 2 items: the data frame with the training data, and the dataframe with testing
# data
split_train_test <- function(data_list, train_percent) {
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
  return(list(train_data, test_data))
}

min_max_normalize <- function(data_list) {
  for (i in 1:2) {
    for (col in c(2, 4:239)) {
      data_list[[i]][col] <- (data_list[[i]][col] - min(data_list[[i]][col])) / (max(data_list[[i]][col]) - min(data_list[[i]][col]))
    }
  }
  return(data_list)
}

log_normalize <- function(data_list) {
  for (i in 1:2) {
    for (col in c(2, 4:239)) {
      data_list[[i]][col] <- log(data_list[[i]][col])
    }
  }
  return(data_list)
}





# BA11 data
BA11_hourly <- split_hourly(BA_11)
BA11_80_list <- split_train_test(BA11_hourly, 0.8)
BA11_70_list <- split_train_test(BA11_hourly, 0.7)
BA11_60_list <- split_train_test(BA11_hourly, 0.6)

BA11_80_MM_list <- min_max_normalize(BA11_80_list)
BA11_80_log_list <- log_normalize(BA11_80_list)

BA11_70_MM_list <- min_max_normalize(BA11_70_list)
BA11_70_log_list <- log_normalize(BA11_70_list)

BA11_60_MM_list <- min_max_normalize(BA11_60_list)
BA11_60_log_list <- log_normalize(BA11_60_list)


write.csv(BA11_80_MM_list[[1]], "data/train test split data/BA11_80_MM_train.csv")
write.csv(BA11_80_MM_list[[2]], "data/train test split data/BA11_80_MM_test.csv")
write.csv(BA11_80_log_list[[1]], "data/train test split data/BA11_80_log_train.csv")
write.csv(BA11_80_log_list[[2]], "data/train test split data/BA11_80_log_test.csv")

write.csv(BA11_70_MM_list[[1]], "data/train test split data/BA11_70_MM_train.csv")
write.csv(BA11_70_MM_list[[2]], "data/train test split data/BA11_70_MM_test.csv")
write.csv(BA11_70_log_list[[1]], "data/train test split data/BA11_70_log_train.csv")
write.csv(BA11_70_log_list[[2]], "data/train test split data/BA11_70_log_test.csv")

write.csv(BA11_60_MM_list[[1]], "data/train test split data/BA11_60_MM_train.csv")
write.csv(BA11_60_MM_list[[2]], "data/train test split data/BA11_60_MM_test.csv")
write.csv(BA11_60_log_list[[1]], "data/train test split data/BA11_60_log_train.csv")
write.csv(BA11_60_log_list[[2]], "data/train test split data/BA11_60_log_test.csv")



# BA47 data
BA47_hourly <- split_hourly(BA_47)
BA47_80_list <- split_train_test(BA47_hourly, 0.8)
BA47_70_list <- split_train_test(BA47_hourly, 0.7)
BA47_60_list <- split_train_test(BA47_hourly, 0.6)

BA47_80_MM_list <- min_max_normalize(BA47_80_list)
BA47_80_log_list <- log_normalize(BA47_80_list)

BA47_70_MM_list <- min_max_normalize(BA47_70_list)
BA47_70_log_list <- log_normalize(BA47_70_list)

BA47_60_MM_list <- min_max_normalize(BA47_60_list)
BA47_60_log_list <- log_normalize(BA47_60_list)


write.csv(BA47_80_MM_list[[1]], "data/train test split data/BA47_80_MM_train.csv")
write.csv(BA47_80_MM_list[[2]], "data/train test split data/BA47_80_MM_test.csv")
write.csv(BA47_80_log_list[[1]], "data/train test split data/BA47_80_log_train.csv")
write.csv(BA47_80_log_list[[2]], "data/train test split data/BA47_80_log_test.csv")

write.csv(BA47_70_MM_list[[1]], "data/train test split data/BA47_70_MM_train.csv")
write.csv(BA47_70_MM_list[[2]], "data/train test split data/BA47_70_MM_test.csv")
write.csv(BA47_70_log_list[[1]], "data/train test split data/BA47_70_log_train.csv")
write.csv(BA47_70_log_list[[2]], "data/train test split data/BA47_70_log_test.csv")

write.csv(BA47_60_MM_list[[1]], "data/train test split data/BA47_60_MM_train.csv")
write.csv(BA47_60_MM_list[[2]], "data/train test split data/BA47_60_MM_test.csv")
write.csv(BA47_60_log_list[[1]], "data/train test split data/BA47_60_log_train.csv")
write.csv(BA47_60_log_list[[2]], "data/train test split data/BA47_60_log_test.csv")


# full data
full_hourly <- split_hourly(full_data)
full_80_list <- split_train_test(full_hourly, 0.8)
full_70_list <- split_train_test(full_hourly, 0.7)
full_60_list <- split_train_test(full_hourly, 0.6)

full_80_MM_list <- min_max_normalize(full_80_list)
full_80_log_list <- log_normalize(full_80_list)

full_70_MM_list <- min_max_normalize(full_70_list)
full_70_log_list <- log_normalize(full_70_list)

full_60_MM_list <- min_max_normalize(full_60_list)
full_60_log_list <- log_normalize(full_60_list)


write.csv(full_80_MM_list[[1]], "data/train test split data/full_80_MM_train.csv")
write.csv(full_80_MM_list[[2]], "data/train test split data/full_80_MM_test.csv")
write.csv(full_80_log_list[[1]], "data/train test split data/full_80_log_train.csv")
write.csv(full_80_log_list[[2]], "data/train test split data/full_80_log_test.csv")

write.csv(full_70_MM_list[[1]], "data/train test split data/full_70_MM_train.csv")
write.csv(full_70_MM_list[[2]], "data/train test split data/full_70_MM_test.csv")
write.csv(full_70_log_list[[1]], "data/train test split data/full_70_log_train.csv")
write.csv(full_70_log_list[[2]], "data/train test split data/full_70_log_test.csv")

write.csv(full_60_MM_list[[1]], "data/train test split data/full_60_MM_train.csv")
write.csv(full_60_MM_list[[2]], "data/train test split data/full_60_MM_test.csv")
write.csv(full_60_log_list[[1]], "data/train test split data/full_60_log_train.csv")
write.csv(full_60_log_list[[2]], "data/train test split data/full_60_log_test.csv")