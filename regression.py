import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Read the CSV file
data = pd.read_csv('C:/Users/HP/Desktop/Colorado State/Capstone/regression.csv')

# Separate the input features and target variable
input_cols = ['data.stats.keyboardMouse.overall.winRate']                  #data.stats.keyboardMouse.overall.winRate    |  data.stats.gamepad.overall.winRate'
target_col = 'overallKD'
X = data[input_cols]
y = data[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values using mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train_imputed, y_train)

# Evaluate the model
r2_score = model.score(X_test_imputed, y_test)
print("R-squared score:", r2_score)
