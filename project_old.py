import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #visualize
import seaborn as sns #also visualize
import warnings
warnings.filterwarnings('ignore')
sns.set_theme()

raw_df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#inspect data
raw_df.head(10)
raw_df.columns

# supprimer les colonnes inutiles
films_df = raw_df.drop(columns=['side_genre'])

films_df.head()

films_df.info()

films_df.describe()

# ajuster les types de données
films_df['Movie_Title'] = films_df['Movie_Title'].astype('string')
films_df['Year'] = films_df['Year'].astype('int')
films_df['Director'] = films_df['Director'].astype('string')
films_df['Actors'] = films_df['Actors'].astype('string')
films_df['Rating'] = films_df['Rating'].astype('float')
films_df['main_genre'] = films_df['main_genre'].astype('string')
films_df['Runtime(Mins)'] = films_df['Runtime(Mins)'].astype(int)
films_df['Censor'] = films_df['Censor'].astype('string')
films_df['Total_Gross'] = films_df['Total_Gross'].astype('string')

films_df.info()

sns.histplot(films_df['Year'], bins=30, kde=True)
plt.title('Nb de titres par année')

sns.histplot(films_df['Rating'], bins=15, kde=True)
plt.title('Nb de titres par Classement')

genres = films_df['main_genre'].str.split(", ").explode().unique()
print("Il y a", genres.size, "genres uniques.")

# One-hot encodage les genres
genre_dummies = films_df['main_genre'].str.split(',').apply(lambda x: '|'.join(i.strip() for i in x)).str.get_dummies(sep='|')

films_df = pd.concat([films_df, genre_dummies], axis=1)

genre_counts = genre_dummies.sum()
print("Numero de films tagués dans chaque genre:")
print(genre_counts)

plt.figure(figsize=(12, 6))
genre_counts.sort_values(ascending=False, inplace=True)
genre_counts.plot(kind='bar', width=1)
plt.xticks(rotation=45)
plt.title('Nb de Titres dans chaque genre')

comedy_films = films_df[films_df['Comedy'] == 1]
comedy_films.sort_values(by='Rating', ascending=False, inplace=True)
print('Top 25 des comédies françaises par classement')
plt.figure(figsize=(10,7))
sns.barplot(data=comedy_films.head(25), x='Rating', y='Movie_Title')
plt.title('Top 25 des comédies françaises par classement')
plt.xlim(7,9)