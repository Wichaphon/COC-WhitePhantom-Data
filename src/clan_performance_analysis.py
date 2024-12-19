import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

townhall = [13, 12, 13, 14, 8, 11, 9, 10, 11, 11, 8, 8, 8, 9, 8, 9, 9, 12, 9, 8, 9, 8, 7, 8, 6, 7, 8, 8, 6, 7, 6, 5]
league_point = [3950, 3608, 3237, 3083, 2131, 2106, 1918, 1832, 1699, 1689, 1395, 1300, 1247, 1240, 1170, 1121, 1105, 1089, 1004, 992, 978, 943, 870, 853, 817, 757, 746, 720, 709, 694, 579, 514]
df = pd.DataFrame({'townhall': townhall, 'league_point': league_point})

def describe_data(df):
    print(df.describe())

#Townhall และ League Points
def calculate_correlation(df):
    correlation = df.corr()
    print("\nCorrelation Townhall กับ League Points:")
    print(correlation)

def plot_graphs(df):
    #Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='townhall', y='league_point', data=df)
    plt.title('Scatter Plot of Townhall vs League Points')
    plt.xlabel('Townhall Level')
    plt.ylabel('League Points')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    #Regression Plot
    plt.figure(figsize=(10, 6))
    sns.regplot(x='townhall', y='league_point', data=df, line_kws={'color': 'red'}, scatter_kws={'s': 50})
    plt.title('Regression Plot of Townhall vs League Points')
    plt.xlabel('Townhall Level')
    plt.ylabel('League Points')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    #Box Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='townhall', y='league_point', data=df)
    plt.title('Box Plot of League Points by Townhall Level')
    plt.xlabel('Townhall Level')
    plt.ylabel('League Points')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    #Violin Plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='townhall', y='league_point', data=df)
    plt.title('Violin Plot of League Points by Townhall Level')
    plt.xlabel('Townhall Level')
    plt.ylabel('League Points')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    #Pair Plot
    sns.pairplot(df)
    plt.show()

    #Joint Plot
    sns.jointplot(x='townhall', y='league_point', data=df, kind='scatter', height=8)
    plt.show()

    #KDE Plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x='townhall', y='league_point', data=df, cmap="Reds", shade=True)
    plt.title('KDE Plot of Townhall vs League Points')
    plt.xlabel('Townhall Level')
    plt.ylabel('League Points')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


describe_data(df)
calculate_correlation(df)
plot_graphs(df)
