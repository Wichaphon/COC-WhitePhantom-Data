import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('../data/dataset_2024-12-21.csv')

print(df.info())
print(df.describe())

correlation = df.corr()
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

df['Performance_Group'] = pd.cut(
    df['league_point'], 
    bins=[0, 1500, 3000, 5000], 
    labels=['Low', 'Medium', 'High']
)

print(df['Performance_Group'].value_counts())

plt.figure(figsize=(12, 6))
sns.boxplot(x='Performance_Group', y='WarStarWon', data=df, palette='Set2')
plt.title('War Stars Won by Performance Group')
plt.xlabel('Performance Group')
plt.ylabel('War Stars Won')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

X = df[['townhall']]
y = df['league_point']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Coefficient (Slope):", model.coef_[0])
print("Model Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(10, 6))
sns.scatterplot(x='townhall', y='league_point', data=df, label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Linear Regression: Townhall vs League Points')
plt.xlabel('Townhall Level')
plt.ylabel('League Points')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

summary = df.groupby('Performance_Group').agg(
    Avg_League_Points=('league_point', 'mean'),
    Avg_WarStars=('WarStarWon', 'mean'),
    Max_League_Points=('league_point', 'max'),
    Max_WarStars=('WarStarWon', 'max'),
    Clan_Count=('Name', 'count')
).reset_index()

print(summary)
