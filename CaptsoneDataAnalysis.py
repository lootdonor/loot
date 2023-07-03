import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.weightstats import ztest
import statsmodels.api as sm
from scipy.stats import shapiro

# Read the CSV file into a DataFrame
df = pd.read_csv('C:/Users/HP/Desktop/Colorado State/Capstone/data.csv')

# Select relevant columns
df = df[['data.stats.gamepad.overall.kd', 'data.stats.gamepad.overall.winRate', 'data.stats.keyboardMouse.overall.kd', 'data.stats.keyboardMouse.overall.winRate', 'overallKD']]

# Remove NaN values
df = df.dropna()



# Function to remove outliers using Z-score
def remove_outliers_zscore(data):
    z_scores = stats.zscore(data)
    threshold = 3  # Set the threshold for outlier detection
    filtered_data = data[(np.abs(z_scores) < threshold)]
    return filtered_data

# Apply Shapiro-Wilk test to your data
win_rate_gamepad_stat, win_rate_gamepad_pvalue = stats.shapiro(df['data.stats.gamepad.overall.winRate'])
win_rate_keyboardMouse_stat, win_rate_keyboardMouse_pvalue = stats.shapiro(df['data.stats.keyboardMouse.overall.winRate'])
kd_gamepad_stat, kd_gamepad_pvalue = stats.shapiro(df['data.stats.gamepad.overall.kd'])
kd_keyboardMouse_stat, kd_keyboardMouse_pvalue = stats.shapiro(df['data.stats.keyboardMouse.overall.kd'])

# Remove outliers from win rate data (Gamepad)
win_rate_gamepad_filtered = remove_outliers_zscore(df['data.stats.gamepad.overall.winRate'])

# Remove outliers from win rate data (KeyboardMouse)
win_rate_keyboardMouse_filtered = remove_outliers_zscore(df['data.stats.keyboardMouse.overall.winRate'])

# Remove outliers from KD ratio data (Gamepad)
kd_gamepad_filtered = remove_outliers_zscore(df['data.stats.gamepad.overall.kd'])

# Remove outliers from KD ratio data (KeyboardMouse)
kd_keyboardMouse_filtered = remove_outliers_zscore(df['data.stats.keyboardMouse.overall.kd'])

# Print the results
print('Shapiro-Wilk Test Results:')
print(f'Win Rate (Gamepad): Test Statistic = {win_rate_gamepad_stat}, p-value = {win_rate_gamepad_pvalue}')
print(f'Win Rate (KeyboardMouse): Test Statistic = {win_rate_keyboardMouse_stat}, p-value = {win_rate_keyboardMouse_pvalue}')
print(f'KD Ratio (Gamepad): Test Statistic = {kd_gamepad_stat}, p-value = {kd_gamepad_pvalue}')
print(f'KD Ratio (KeyboardMouse): Test Statistic = {kd_keyboardMouse_stat}, p-value = {kd_keyboardMouse_pvalue}')

# Calculate descriptive statistics for win rates
print('Descriptive statistics for gamepad win rate:')
print(df['data.stats.gamepad.overall.winRate'].describe())
print('\nDescriptive statistics for keyboardMouse win rate:')
print(df['data.stats.keyboardMouse.overall.winRate'].describe())

# Apply Box-Cox transformation to win rate data
df['data.stats.gamepad.overall.winRate'], _ = stats.boxcox(df['data.stats.gamepad.overall.winRate'] + 0.01)
df['data.stats.keyboardMouse.overall.winRate'], _ = stats.boxcox(df['data.stats.keyboardMouse.overall.winRate'] + 0.01)

# Apply Box-Cox transformation to KD ratio data
df['data.stats.gamepad.overall.kd'], _ = stats.boxcox(df['data.stats.gamepad.overall.kd'] + 0.01)
df['data.stats.keyboardMouse.overall.kd'], _ = stats.boxcox(df['data.stats.keyboardMouse.overall.kd'] + 0.01)

# Plot the win rate distribution for each input type
plt.figure(figsize=(10, 6))
sns.histplot(df['data.stats.gamepad.overall.winRate'], bins=20, kde=True)
plt.title('Distribution of Win Rates (Gamepad)')
plt.xlabel('Win Rates')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['data.stats.keyboardMouse.overall.winRate'], bins=20, kde=True)
plt.title('Distribution of Win Rates (KeyboardMouse)')
plt.xlabel('Win Rates')
plt.ylabel('Frequency')
plt.show()

# Plot the KD ratio distribution for each input type
plt.figure(figsize=(10, 6))
sns.histplot(df['data.stats.gamepad.overall.kd'], bins=20, kde=True)
plt.title('Distribution of KD Ratios (Gamepad)')
plt.xlabel('KD Ratios')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['data.stats.keyboardMouse.overall.kd'], bins=20, kde=True)
plt.title('Distribution of KD Ratios (KeyboardMouse)')
plt.xlabel('KD Ratios')
plt.ylabel('Frequency')
plt.show()

# Perform a z-test to check if the difference in win rate is statistically significant between input types
gamepad_data = df['data.stats.gamepad.overall.winRate'].dropna()
keyboardMouse_data = df['data.stats.keyboardMouse.overall.winRate'].dropna()
if len(gamepad_data) > 0 and len(keyboardMouse_data) > 0:
    z_stat, p_value = ztest(gamepad_data, keyboardMouse_data)
    print(f'Win rate difference: z-statistic = {z_stat}, p-value = {p_value}')
else:
    print('Insufficient data for z-test.')

# Perform a z-test to check if the difference in KD ratio is statistically significant between input types
gamepad_data = df['data.stats.gamepad.overall.kd'].dropna()
keyboardMouse_data = df['data.stats.keyboardMouse.overall.kd'].dropna()
if len(gamepad_data) > 0 and len(keyboardMouse_data) > 0:
    z_stat, p_value = ztest(gamepad_data, keyboardMouse_data)
    print(f'KD ratio difference: z-statistic = {z_stat}, p-value = {p_value}')
else:
    print('Insufficient data for z-test.')

# Perform one-way ANOVA for win rates
result_win_rate = stats.f_oneway(df['data.stats.gamepad.overall.winRate'],
                                 df['data.stats.keyboardMouse.overall.winRate'])

# Perform one-way ANOVA for KD rates
result_kd_rate = stats.f_oneway(df['data.stats.gamepad.overall.kd'],
                                df['data.stats.keyboardMouse.overall.kd'])

# Extract the F-statistics and p-values from the results
f_statistic_win_rate = result_win_rate.statistic
p_value_win_rate = result_win_rate.pvalue
f_statistic_kd_rate = result_kd_rate.statistic
p_value_kd_rate = result_kd_rate.pvalue

# Print the results
print('One-Way ANOVA Results:')
print('Win Rate:')
print(f'F-Statistic: {f_statistic_win_rate}')
print(f'p-value: {p_value_win_rate}')
print('KD Rate:')
print(f'F-Statistic: {f_statistic_kd_rate}')
print(f'p-value: {p_value_kd_rate}')


#RUN REGRESSION

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Separate the input features and target variable
input_cols = ['data.stats.gamepad.overall.kd', 'data.stats.keyboardMouse.overall.kd']
target_col = 'overallKD'  # Replace with the correct target column name
X = df[input_cols]
y = df[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values using mean
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Add constant to the input features
X_train_imputed = sm.add_constant(X_train_imputed)
X_test_imputed = sm.add_constant(X_test_imputed)

# Train the linear regression model
model = sm.OLS(y_train, X_train_imputed)
results = model.fit()

# Extract the estimated parameters and p-values
estimated_params = results.params
p_values = results.pvalues

# Evaluate the model
r2_score = results.rsquared

# Print the regression results
print("Regression Result")
print("R-squared score:", r2_score)
print("Estimated parameters:")
print("Intercept:", estimated_params[0])
for i, col in enumerate(input_cols):
    print(col + " coefficient:", estimated_params[i+1])
    print(col + " p-value:", p_values[i+1])
