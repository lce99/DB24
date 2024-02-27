library(reticulate)
library(caret)
library(dplyr)
library(nnet)
library(afex)
library(car)
library(MLmetrics)
# py_install("pandas")
pd <- import("pandas")
# install.packages("remotes")
# remotes::install_github("LqNoob/Machine-Learning-Evaluation-Metrics")
최종딕셔너리 <-pd$read_pickle('./final_gu_vars.pkl') 
set_sum_contrasts()
# Load other data frames
rbl <- read.csv('region_bubble_labels.csv')
gbl <- read.csv('bubble_labels.csv')

# Preparing the data
gu <- c('종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구',
        '성북구', '강북구', '도봉구', '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구',
        '금천구', '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구')
gu_label <- paste(gu, "_label", sep = "")
seoul_gu <- c('종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구',
        '성북구', '강북구', '도봉구', '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구',
        '금천구', '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구', '서울')
gu_data <- gbl[gu_label]
seoul_data <- rbl %>% select(날짜, 서울_label)
total_data <- bind_cols(seoul_data, gu_data)

fit_logistic <- function(df1, df2, idx) {
  df2$날짜 <- as.Date(df2$날짜)
  df1$날짜 <- as.Date(df1$날짜)
  label <- select(df2, 날짜, paste(idx, "_label", sep = ""))
  data <- merge(df1, label, by = "날짜")
  predict_set <- data[245:247,]
  data <- data[1:244,]
  data <- na.omit(data)
  
  X <- select(data, -c(paste(idx, "_label", sep = ""), 날짜))
  y <- as.factor(data[[paste(idx, "_label", sep = "")]])
  predict_X <- select(predict_set, -c(paste(idx, "_label", sep = ""), 날짜))
  split_point <- floor(nrow(data) * 0.8)
  X_train <- X[1:split_point, ]
  y_train <- y[1:split_point]
  X_test <- X[(split_point + 1):nrow(X), ]
  y_test <- y[(split_point + 1):length(y)]
  X_train_scaled <- scale(X_train)
  X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), scale = attr(X_train_scaled, "scaled:scale"))
  predict_X_scaled <- scale(predict_X, center = attr(X_train_scaled, "scaled:center"), scale = attr(X_train_scaled, "scaled:scale"))
  y_train <- relevel(y_train, ref = '1')
  model <- multinom(y_train ~ ., data = as.data.frame(X_train_scaled))
  
  # Predict on test set for F1 score
  pred_y <- predict(model, newdata=as.data.frame(X_test_scaled), type='class')
  pred<- predict(model, newdata = as.data.frame(predict_X_scaled), type = 'class')
  F1 <- F1_Score_macro_weighted(y_pred = pred_y, y_true = y_test, labels = unique(y_test))
  p_values <- round(as.data.frame(Anova(model,type="III")),5)
  coef_df <- as.data.frame(t(round(summary(model)$coefficients,4))[-1, ], check.names = FALSE) # Drop intercept and transpose
  p_values_df <- as.data.frame(t(as.data.frame(t(p_values), check.names = FALSE)))
  coef_df$Var <- rownames(coef_df)
  p_values_df$Var <- rownames(p_values_df)
  merged_df <- merge(coef_df, p_values_df, by = "Var")
  
  # If you wish to order the merged data frame by Variable names for better readability
  merged_df <- merged_df[order(merged_df$Var), ]
  
  return(list(df = merged_df, F1 = F1, p_value = p_values , coef=coef_df, pred = pred))
}


# Loop through 구 and fit models
results <- list()
models <- list()
for (idx in seoul_gu) {
  df_idx <- paste(idx, "_df", sep = "")
  df1 <- as.data.frame(최종딕셔너리[[df_idx]])
  df2 <- total_data
  
  models[[idx]] <- fit_logistic(df1, df2, idx)
}

