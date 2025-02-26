# King County Real Estate Price Prediction Model

install.packages('corrplot')
install.packages('gridExtra')
install.packages('Metrics')

# import data
house.data <- read.csv("C:\\Users\\Public\\kc_house_data.csv", header = TRUE)


# Drop Columns not part of the Multiple Regression from data frame
house.data <- subset(house.data, select = -c(id, date, zipcode, lat, long))
head(house.data)

# Initial regression modelling showed a high degree of heteroscedasticity
# To reduce the variation, the log of price was used instead of the original price
# Transforming Price using Log
house.data$price = log(house.data$price) 
head(house.data)


# 1. Data Exploration
# View the structure of the data
str(house.data)


# View a summary of the data
summary(house.data)

# Split the dataset into 4:1 ratio as training dataset and testing dataset
sample_size = floor(0.80*nrow(house.data))
set.seed(123)
train_ind = sample(seq_len(nrow(house.data)), size = sample_size)
train = house.data[train_ind,]
test = house.data[-train_ind,]

# Checking for Multicollinearity (Independence of Variables)
correlation = cor(train)
corrplot(correlation, method = "number")


# The correlation plot shows correlations between each variable and all the others. 
# Many of the variables related to the size of homes are highly correlated with each other.
# Generally, we do not want to include two x variables whose correlation with each other exceeds 0.8.
# However, we do want to include variables that are correlated with the y variable (price).
# For example, sqft_living and sqft_above are both highly correlated with price, but we will only include 
# one of them in our analysis because they are also highly correlated with each other. 
# Based on these plots, we chose independent variables: grade, number of bedrooms, number of bathrooms, view, waterfront, square footage of home,
# and square footage of homes for the nearest 15 neighbors. 



# Checking for Linearity (Scatter plots):
p1 = ggplot(data = train, aes(x = grade, y = price)) + geom_jitter() + geom_smooth(method = "lm", se = FALSE) + labs(title = "Scatterplot of Grade and Price", x = "Grade", y = "Price")
p2 = ggplot(data = train, aes(x = bedrooms, y = price)) + geom_jitter() + geom_smooth(method = "lm", se = FALSE) + labs(title = "Scatterplot of Bedrooms and Price", x = "Bedrooms", y = "Price")
p3 = ggplot(data = train, aes(x = bathrooms, y = price)) + geom_jitter() + geom_smooth(method = "lm", se = FALSE) + labs(title = "Scatterplot of Bathrooms and Price", x = "Bathrooms", y = "Price")
p4 = ggplot(data = train, aes(x = sqft_living, y = price)) + geom_jitter() + geom_smooth(method = "lm", se = FALSE) + labs(title = "Scatterplot of Sqft and Price", x = "Square Footage", y = "Price")
p5 = ggplot(data = train, aes(x = sqft_living15, y = price)) + geom_jitter() + geom_smooth(method = "lm", se = FALSE) + labs(title = "Scatterplot of Neighborhood Sqft and Price", x = "Neighborhood Sqft", y = "Price")
grid.arrange(p1, p2, p3, p4, p5, nrow = 3)


# Box plots for categorical data:
par(mfrow = c(1,2))
boxplot(price~grade, data = train, main = "Price by Grade", xlab = "Grade", ylab = "Price",  col = "skyblue")
boxplot(price~view, data = train, main = "Price by View", xlab = "View", ylab = "Price",  col = "skyblue")

# Box plot for Response Variable (Outliers):
p6 = ggplot(data = train) + geom_boxplot(aes(x = price), col = "black", fill = "skyblue")
p7 = ggplot(data = train) +geom_density(aes(x = price, col = "black", fill = "lightblue"))
grid.arrange(p6, p7, nrow = 1)

# We can see from the box plot that we have a significantly large number of outliers. 
# Altering extreme outliers in genuine observations is not a standard operating procedure. 
# However, it is essential to understand their impact on predictive models. 
# In our case,the outliers seem to be genuine values so we decided not to remove them


# Create a baseline Multiple Regression Model
house.model <- lm(price ~ grade + sqft_living + sqft_living15 + view, data = train)
summary(house.model)

# Run the prediction on the test data set and check the performance
predicted <- predict(house.model, test)
head(predicted)
head (test)

# Checking the accuracy 
accuracy = rmse(test$price, predicted)
print(accuracy)


#Checking for Homoscedasticity and Normal Distribution of Residuals:
par(mfrow = c(2,2))
plot(house.model)

# Since almost all data points fall along a straight line in this Q-Q plot, we can consider the normality condition satisfied
# The homoscedasticity assumption states that for any value of x, the variance of the residuals is the same. 
# For this condition to be satisfied, the residuals need to be symmetrical along the line y = 0. We can see that the points have a roughly symmetrical  shape around the x-axis. 
# Thus, the model satisfies the last assumption of homoscedasticity.
