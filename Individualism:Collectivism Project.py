import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import t
import plotly.express as px
sns.set_theme(style='darkgrid')

#importing world happiness report dataset
WHR_df = pd.read_csv('/Users/erikrice/Downloads/World Happiness Report 2022.csv')
print(WHR_df.head())
print(WHR_df.info())
print(WHR_df.describe())

#a quick visual of the distribution
print(WHR_df.hist(column='Happiness score', bins=5))
plt.show()

#importing Geert Hofstede cultural dimensions dataset
GH_df = pd.read_csv('/Users/erikrice/Downloads/Individualism_Collectivism Dataset - Sheet1.csv')
print(GH_df.head())
print(GH_df.info())
print(GH_df.describe())

#quick visual of the distribution
print(GH_df.hist(column='IDV', bins=5))
plt.show()

#merging tables
df = WHR_df.merge(GH_df, how='outer', on='Country')

#cleaning merged dataframe
print(df.columns)
df['Happiness_Ranking'] = df['RANK']
df['Happiness_Ranking'] = df['Happiness_Ranking'].astype(float)
df['Happiness_Score'] = df['Happiness score']
df['Happiness_Score'] = df['Happiness_Score'].astype(float)
df['Individualism_Score'] = df['IDV']
df['Individualism_Score'] = df['Individualism_Score'].astype(float)
df = df[['Happiness_Ranking', 'Country', 'Happiness_Score', 'Individualism_Score']]
print(df.isna().sum())
df = df.dropna()
df = pd.DataFrame(df)
print(df.info())
print(df)

#adding country codes to make graph
country_code_df = pd.read_csv('/Users/erikrice/Documents/name,alpha-2,alpha-3,country-code,iso_31.csv')
print(country_code_df.head())

#cleaning/preparring table
country_code_df['name'] = country_code_df['name'].str.replace('United States of America', 'United States')
country_code_df['name'] = country_code_df['name'].str.replace('United Kingdom of Great Britain and Northern Ireland', 'United Kingdom')
country_code_df['name'] = country_code_df['name'].str.replace('Korea, Republic of', 'South Korea')

#merging tables
world_df = df.merge(country_code_df, how='outer', left_on='Country', right_on='name')
test4 = world_df[['Country', 'name', 'Individualism_Score']]

#world visual of individualism score
individualism_visual = px.data.gapminder().query("year==2022")
fig = px.choropleth(world_df, locations="alpha-3",
                    color="Individualism_Score", 
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()

#and the happiness scores
happiness_visual = px.data.gapminder().query("year==2022")
fig = px.choropleth(world_df, locations="alpha-3",
                    color="Happiness_Score", 
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()

#adjusting graph for bubbleplot
df['Individualism_Score_Scaled'] = df['Individualism_Score'] * 25
df_sorted = df.sort_values('Country')

#bubbleplot
plt.figure(figsize=(8, 8))
plt.style.use('ggplot')
plt.scatter('Country', 'Happiness_Score', alpha=0.5, s='Individualism_Score_Scaled', c='Individualism_Score', edgecolors='white', data=df_sorted)
plt.xlabel("Country", size=16)
plt.ylabel("Happiness Score", size=16)
plt.title('Happiness Measured Against Individualism', size=16)
plt.xticks(rotation=90)
plt.show()

#visualizing potential correlation 
scatter1 = sns.relplot(data=df_sorted, y='Happiness_Score', x='Individualism_Score', kind='scatter')
plt.xticks(rotation=90)
scatter1.fig.suptitle('Happiness Measured Against Individualism')
scatter1.fig.subplots_adjust(top=0.9)
scatter1.set_xlabels('Individualism Score')
scatter1.set_ylabels('Happiness Score')
plt.show()

#looking at it with the country names and individualism as the hue
scatter2 = sns.relplot(data=df_sorted, x='Country', y='Happiness_Score', hue='Individualism_Score', kind='scatter')
plt.xticks(rotation=90)
scatter2.fig.suptitle('Happiness/Individualism by Country')
scatter2.fig.subplots_adjust(top=0.9)
scatter2.set_xlabels('Country')
scatter2.set_ylabels('Happiness Score')
plt.show()

#looking at correlation
df_numerical = df[['Happiness_Score', 'Individualism_Score']]
print(df_numerical.corr())

#creating relative categorical variables for hypothesis testing
print(df['Individualism_Score'].median())
IND_COL = []
for x in df['Individualism_Score']:
    if x >= 38:
        IND_COL.append('Individualistic')
    else:
        IND_COL.append('Collectivist')
df['IND_COL'] = IND_COL
df_sorted['IND_COL'] = IND_COL
print(df['IND_COL'].value_counts())

#a quick visualization with the categorical variable 
scatter3 = sns.relplot(data=df, x='Individualism_Score', y='Happiness_Score', hue='IND_COL', style='IND_COL', kind='scatter')
scatter3.fig.suptitle('Happiness Score by Cultural Categories')
scatter3.fig.subplots_adjust(top=0.9)
scatter3.set_xlabels('Individualism Score')
scatter3.set_ylabels('Happiness Score')
plt.show()

#calculating some summary statistics 
print(df.agg({'Happiness_Score':['mean', 'median', 'std'], 'Individualism_Score':['mean', 'median', 'std']}))

#summary statistics by category
summary_stats_by_group = df.groupby('IND_COL').agg(
    Mean_Happiness=('Happiness_Score', 'mean'),
    Median_Happiness=('Happiness_Score', 'median'),
    STD_Happiness=('Happiness_Score', 'std'),
    Mean_Individualism=('Individualism_Score', 'mean'),
    Median_Individualism=('Individualism_Score', 'median'),
    STD_Individualism=('Individualism_Score', 'std')
)

print(summary_stats_by_group)

#finally, a look at the top 20 is informative
print(df.to_string())

#calculating the test statistic
alpha = .05
xbar_ind = 6.810600
xbar_col = 5.014367
s_ind = .511095
s_col = .872190
n_ind = 30
n_col = 30
numerator = xbar_ind - xbar_col
denominator = np.sqrt(s_ind ** 2 / n_ind + s_col ** 2 / n_col)
t_stat = numerator / denominator
degrees_of_freedom = n_ind + n_col - 2
p = 1 - t.cdf(t_stat, df=degrees_of_freedom)
print(p)

#null hypothesis is rejected 

print(df.sort_values('Individualism_Score', ascending=False).to_string())