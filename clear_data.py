import pandas as pd

# clear buildings file
buildings = pd.read_csv('data/buildings_raw.csv', usecols=['szerokosc', 'dlugosc'])
buildings = buildings.dropna()
buildings = buildings.rename(columns={'szerokosc': 'latitude', 'dlugosc': 'longitude'})
buildings.to_csv('data/buildings.csv', index=False)

# clear fires file
fires = pd.read_csv('data/fires_raw.csv', usecols=[1, 4, 5])
fires.dropna()
fires.columns = ['date', 'latitude', 'longitude']
fires['latitude'] = fires['latitude'].str.replace(',', '.').astype(float)
fires['longitude'] = fires['longitude'].str.replace(',', '.').astype(float)
fires.to_csv('data/fires.csv', index=False)