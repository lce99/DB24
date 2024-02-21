library(reticulate)
py_install("pandas")
pd <- import("pandas")
install.packages("remotes")
remotes::install_github("LqNoob/Machine-Learning-Evaluation-Metrics")
library(MLmetrics)
최종딕셔너리 <-pd$read_pickle('./final_gu_vars.pkl') 
str(최종딕셔너리)

# Load other data frames
rbl <- read.csv('region_bubble_labels.csv')
gbl <- read.csv('bubble_labels.csv')
library(caret)
library(dplyr)
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

# Fit logistic model
fit_logistic <- function(df1, df2, idx) {
  df2$날짜 <- as.Date(df2$날짜)
  df1$날짜 <- as.Date(df1$날짜)
  label <- select(df2, 날짜, paste(idx, "_label", sep = ""))
  data <- merge(df1, label, by = "날짜")
  data <- na.omit(data)
  
  X <- select(data, -c(paste(idx, "_label", sep = ""), 날짜))
  y <- as.factor(data[[paste(idx, "_label", sep = "")]])
  
  split_point <- floor(nrow(data) * 0.8)
  X_train <- X[1:split_point,]
  y_train <- y[1:split_point]
  X_test <- X[(split_point + 1):nrow(X), ]
  y_test <- y[(split_point + 1):length(y)]
  X_train_scaled <- scale(X_train)
  X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), scale = attr(X_train_scaled, "scaled:scale"))

  # Fitting multinomial logistic regression
  model <- multinom(y_train ~ ., data = as.data.frame(X_train_scaled))
  pred_x <- predict(model, data=y_test, type='probs')
  F1 <- F1_Score_macro_weighted(y_test, pred_labels, labels = NULL)
  return(list(model = model, F1 = F1))}
library(nnet)

# Loop through 구 and fit models
results <- list()
models <- list()
for (idx in seoul_gu) {
  df_idx <- paste(idx, "_df", sep = "")
  df1 <- as.data.frame(최종딕셔너리[[df_idx]])
  df2 <- total_data
  
  models[[idx]] <- fit_logistic(df1, df2, idx)
}


models[['노원구']]$model

##################
idx <- '서울'
df_idx <- paste(idx, "_df", sep = "")
df1 <- as.data.frame(최종딕셔너리[[df_idx]])
df2 <- total_data
  
df2$날짜 <- as.Date(df2$날짜)
df1$날짜 <- as.Date(df1$날짜)
label <- select(df2, 날짜, paste(idx, "_label", sep = ""))
data <- merge(df1, label, by = "날짜")
data <- na.omit(data)

X <- select(data, -c(paste(idx, "_label", sep = ""), 날짜))
y <- as.factor(data[[paste(idx, "_label", sep = "")]])


split_point <- floor(nrow(data) * 0.8)
X_train <- X[1:split_point,]
y_train <- y[1:split_point]
X_test <- X[(split_point + 1):nrow(X), ]
y_test <- y[(split_point + 1):length(y)]
X_train_scaled <- scale(X_train)
X_test_scaled <- scale(X_test, center = attr(X_train_scaled, "scaled:center"), scale = attr(X_train_scaled, "scaled:scale"))

# Fitting multinomial logistic regression
model <- multinom(y_train ~ ., data = as.data.frame(X_train_scaled))
pred_labels <- predict(model, newdata=X_test_scaled, type="class")
y_test
pred_labels
F1_Score_macro_weighted(y_test, pred_labels, labels = NULL)
cm <- confusionMatrix(as.factor(pred_labels), y_test)
# Calculate weighted F1 score
precision <- cm$byClass['Pos Pred Value']
recall <- cm$byClass['Sensitivity']
cm$table
f1_scores <- 2 * ((precision * recall) / (precision + recall))
weighted_f1_score <- sum((f1_scores * cm$table[,'Reference']) / sum(cm$table[,'Reference']))

print(weighted_f1_score)
cm$table[,3]
