###Project: Predicting price of the Airbnb's and identifying important factors
library(dplyr)
library(tidyverse)
library(caret)     #partition data
library(lubridate) #edit dates
library(car)       #check VIF
library(forecast)  #to make predictions
library(neuralnet) #NN
#setwd("/Users/ltbrand/Desktop/Course Materials/DNSC 4280/Group Project")


listings2_full <- read.csv("listings2.csv")

### Data Pre-processing -----------------------------------------------------
names(listings2_full)
str(listings2_full)

#Remove columns that aren't going to be used
listings2 = listings2_full[-c(1:14,15,17,18,20,21:25,28,30:33,36,38,44:56,69,72:74)]
sum(is.na(listings2))

#Change amenities listing into total amenities
listings2 <- listings2 %>%
  mutate(
    amenities = sapply(strsplit(gsub("[\\[\\]\"]", "", amenities), ", "), length)
  )
#Use first and latest review dates to get host length time
listings2 <- listings2 %>%
  mutate(
    first_review = as.Date(first_review),
    last_review = as.Date(last_review),
    years_hosted = round(as.numeric(difftime(last_review, first_review, units = "days")) / 365.25, 2)
  )
#Make some columns usable and numeric
listings2 <- listings2 %>%
  mutate(
    bathrooms = round(as.numeric(gsub("[^0-9.]", "", bathrooms_text)), 0),
    price = as.numeric(gsub("[^0-9.]", "", price))
  )
#Convert binary columns to numeric
#Also change some character columns to factor.
listings2 <- listings2 %>%
  mutate(
    host_is_superhost = as.integer(host_is_superhost == "t"),
    host_has_profile_pic = as.integer(host_has_profile_pic == "t"),
    host_identity_verified = as.integer(host_identity_verified == "t"),
    instant_bookable = as.integer(instant_bookable == "t"),
    neighbourhood_cleansed = as.factor(neighbourhood_cleansed),
    room_type = as.factor(room_type),
    host_response_time = as.factor(host_response_time)
  )

listings2 = listings2[-c(8,15:18,20:25)]

# Remove outliers for Price
mean(air.df$price)
hist(listings2$price, breaks = 40, col = "skyblue", main = "Histogram of Prices before",
     xlab = "Price", ylab = "Frequency")
{
Q1 <- quantile(listings2$price, 0.25)
Q3 <- quantile(listings2
               $price, 0.75)
IQR <- Q3 - Q1
lower_threshold <- Q1 - 1.5 * IQR
upper_threshold <- Q3 + 1.5 * IQR
}
listings2 <- listings2 %>%
  filter(price >= lower_threshold, price <= upper_threshold)

# Plot Historam for Prices (outliers removed)
hist(listings2$price, breaks = 40, col = "skyblue",
     main = "Histogram of Prices (outliers removed)", xlab = "Price", ylab = "Frequency")
mean(air.df$price)


### Multiple Linear Regression --------------------------------------------------------
air.df = na.omit(listings2)

# select variables for regression
selected.var = c(2,3,4,7:9, 10:19)
# partition data [applied to all models from here on out]
set.seed(1)  # set seed for reproducing the partition
train.index = createDataPartition(1:nrow(air.df), p = 0.6, list = FALSE)
train.df = air.df[train.index, selected.var] #further split the data into train&valid
valid.df = air.df[-train.index, selected.var]

#Create model
air.lm = lm(price ~ ., data = train.df)
options(scipen = 999) #remove scientific notation
summary(air.lm)
#Check VIF
vif(air.lm)

# use predict() to make predictions on a new set. 
air.lm.pred <- predict(air.lm, valid.df)

# use accuracy() to compute common accuracy measures.
accuracy(air.lm.pred, valid.df$price)

ggplot() +
  geom_point(mapping = aes(x=air.lm.pred, y=valid.df$price), color = "blue") +
  geom_abline(intercept = 0, slope = 1,
              linetype = "solid", lwd=0.5) +
  ggtitle("Full MLR Predictions vs Actual Price")



### STEPWISE REGRESSION -----------------------------------------------------
#### Backward
# use step() to run stepwise regression.
air.lm.Bstep <- step(air.lm, direction = "backward")
summary(air.lm.Bstep)  # Which variables were dropped?
air.lm.Bstep.pred <- predict(air.lm.Bstep, valid.df)
accuracy(air.lm.Bstep.pred, valid.df$price)

#### Forward
# create model with no predictors
air.lm.null <- lm(price~1, data = train.df)
# use step() to run forward regression.
air.lm.Fstep <- step(air.lm.null, scope=list(lower=air.lm.null, upper=air.lm), direction = "forward")
summary(air.lm.Fstep)  # Which variables were added?
air.lm.Fstep.pred <- predict(air.lm.Fstep, valid.df)
accuracy(air.lm.Fstep.pred, valid.df$price)

#### All (Both)
# use step() to run stepwise regression.
air.lm.Astep <- step(air.lm, direction = "both")
summary(air.lm.Astep)  # Which variables were dropped/added?
air.lm.Astep.pred <- predict(air.lm.Astep, valid.df)
accuracy(air.lm.Astep.pred, valid.df$price)





### NEURAL NETWORKS ---------------------------------------------------------

# normalize
norm.values <- preProcess(train.df, method="range")
train.norm.df <- predict(norm.values, train.df)
valid.norm.df <- predict(norm.values, valid.df)

#### NN with a single layer of 2 hidden nodes ####
nn.1 <- neuralnet(price ~ ., 
                  data = train.norm.df, linear.output = T, hidden = 2)

# visualize the neural net
plot(nn.1)

# Training & Validation Predictions
train.preds.norm.nn.1 <- predict(nn.1, newdata=train.norm.df)
valid.preds.norm.nn.1 <- predict(nn.1, newdata=valid.norm.df)

# check the accuracy
# note: these results are in the transformed scale
accuracy(train.norm.df$price, train.preds.norm.nn.1)
accuracy(valid.norm.df$price, valid.preds.norm.nn.1)


# plot predictions vs actuals (in original scale)
#price.scale <- norm.values$ranges[,"price"][2] - norm.values$ranges[,"price"][1]
#price.offset <- norm.values$ranges[,"price"][1]
#valid.preds.raw.nn.1 <- price.offset + 
#  valid.preds.norm.nn.1 * price.scale

#ggplot() +
#  geom_point(mapping = aes(x=valid.preds.raw.nn.1, y=valid.df$price), color = "red") +
#  geom_abline(intercept = 0, slope = 1,
#              linetype = "solid", lwd=0.5) +
#  ggtitle("NN (1L,2N) vs Actual Price")


#### NN with two layers of 2 hidden nodes each ####
nn.2 <- neuralnet(price ~ ., 
                  data = train.norm.df, linear.output = T, hidden = c(2,2))

# visualize the neural net
plot(nn.2)

# Training & Validation Predictions
train.preds.norm.nn.2 <- predict(nn.2, newdata=train.norm.df)
valid.preds.norm.nn.2 <- predict(nn.2, newdata=valid.norm.df)

# check the accuracy
# note: these results are in the transformed scale
accuracy(train.norm.df$price, train.preds.norm.nn.2)
accuracy(valid.norm.df$price, valid.preds.norm.nn.2)

# plot predictions vs actuals (in original scale)
price.scale <- norm.values$ranges[,"price"][2] - norm.values$ranges[,"price"][1]
price.offset <- norm.values$ranges[,"price"][1]
valid.preds.raw.nn.2 <- price.offset + 
  valid.preds.norm.nn.2 * price.scale

ggplot() +
  geom_point(mapping = aes(x=valid.preds.raw.nn.2, y=valid.df$price), color = "red") +
  geom_abline(intercept = 0, slope = 1,
              linetype = "solid", lwd=0.5) +
  ggtitle("NN (2 Layers, 2 Nodes) vs Actual Price")


#### NN with two layers of 4 hidden nodes each ####
nn.3 <- neuralnet(price ~ ., 
                  data = train.norm.df, linear.output = T, hidden = c(2,4))

# Training & Validation Predictions
train.preds.norm.nn.3 <- predict(nn.3, newdata=train.norm.df)
valid.preds.norm.nn.3 <- predict(nn.3, newdata=valid.norm.df)

# check the accuracy
# note: these results are in the transformed scale
accuracy(train.norm.df$price, train.preds.norm.nn.3)
accuracy(valid.norm.df$price, valid.preds.norm.nn.3)






