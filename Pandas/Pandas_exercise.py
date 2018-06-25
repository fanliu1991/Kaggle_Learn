
# coding: utf-8

# In[2]:


import pandas as pd


# In[11]:


'''
Creating, reading, and writing workbook
'''
# create DataFrame
data = {
    "Apples": [30],
    "Bananas": [21]
       }
df = pd.DataFrame(data, columns=["Apples", "Bananas"])
print("#1:")
print(df)
print("\n")

data = {
    "Apples": [35, 41],
    "Bananas": [21, 34]
       }
df = pd.DataFrame(data, columns=["Apples", "Bananas"], index=["2017 Sales", "2018 Sales"])
print("#2:")
print(df)
print("\n")


# In[13]:


# create Series
s = pd.Series(["4 cups", "1 cup", "2 large", "1 can"],               index=["Flour", "Milk", "Eggs", "Spam"],               name="Dinner")
print("#3:")
print(s)
print("\n")


# In[ ]:


# read CSV file
df = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv",                  usecols=["country", "description", "designation", "points", "price", "province", "region_1", "region_2", "variety", "winery"])
# use the first column in the dataset as index
# reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

# read Excel xls file, and add multiple empty columns
df = pd.read_excel("../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls")
df[['Unnamed:1', 'Unnamed:2', 'Unnamed:3']] = pd.DataFrame([[np.nan, np.nan, np.nan]], index=df.index)

# Save DataFrame as a csv file, with the name "cows_and_goats.csv"
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
q6_df.to_csv("cows_and_goats.csv")

# read SQL data into a DataFrame
import sqlite3
 
filepath = sqlite3.connect('../input/pitchfork-data/database.sqlite')
query = "SELECT * FROM artists;"
df = pd.read_sql_query(query, filepath)


# In[ ]:


'''
Indexing, Selecting & Assigning
'''

import seaborn as sns
# Select the "description" column from "reviews" DataFrame
description = reviews["description"]

# Select the first value from the "description" column of "reviews" DataFrame
value = description[0]

# Select the first row of data (the first record) from "reviews" DataFrame
# "iloc" uses the Python stdlib indexing scheme, where the first element of the range is included and the last one excluded. So "0:10" will select entries "0,...,9".
# "loc", meanwhile, indexes inclusively. So "0:10" will select entries "0,...,10".
record = reviews.loc[0] 
# or record = reviews.iloc[0]

# Select the first 10 values from the "description" column in "reviews" DataFrame
values = reviews.loc[:9, "description"]  # not 0:10 !!!
# the above return a pandas Series
values = reviews.loc[:9, ["description"]]
# the above return a pandas DataFrame

# Select the records with the "1", "2", "3", "5", and "8" row index positions
record = reviews.loc[[1,2,3,5,8]]

# Select the "country", "province", "region_1", and "region_2" columns of the records with the "0", "1", "10", and "100" index positions
record = reviews.loc[[0,1,10,100], ["country", "province", "region_1", "region_2"]]

# Select the "country" and "variety" columns of the first 100 records
record = reviews.loc[0:100, ["country", "variety"]]

# Select wines made in "Italy"
italy_wine = reviews.loc[reviews["country"] == "Italy"]

# Select wines whose "region_2" is not "NaN"
nan_region2 = reviews.loc[reviews["region_2"].notnull()]
# Select wines whose "region_2" is "NaN"
nan_region2 = reviews.loc[reviews["region_2"].isnull()]

# Select the "points" column
points = reviews.loc[:, "points"]

# Select the "points" column for the first 1000 wines
points = reviews.loc[:1000, "points"]

# Select the "points" column for the last 1000 wines
points = reviews[-1000:]["points"]

# Select the "points" column, but only for wines made in Italy
italy_points = reviews.loc[reviews["country"] == "Italy", "points"]

# Select the "country" column, but only when said "country" is France or Italy, and the "points" column is greater than or equal to 90
good_wine = reviews.loc[reviews["points"] >= 90, ["country"]]
good_wine_country = good_wine.loc[(reviews["country"] == "Italy") | (reviews["country"] == "France"), "country"]


# In[ ]:


'''
Summary functions and maps workbook
'''
# What is the median of the "points" column?
points_median = reviews["points"].median()

# What countries are represented in the dataset?
countries = reviews["country"].unique()

# What countries appear in the dataset most often?
countries_count = reviews["country"].value_counts()
'''
US        54504
France    22093
          ...  
China         1
Egypt         1
Name: country, Length: 43, dtype: int64
'''
most_ofter_country = countries_count[0]
'''
54504
'''
most_ofter_country = countries_count.index.values[0]
'''
US
'''

# Remap the "price" column by subtracting the median price. Use the "Series.map" method.
price_median = reviews["price"].median()
reviews["price_diff_median"] = reviews["price"].map(lambda p: p - price_median)

# Which wine in is the "best bargain", e.g., which wine has the highest points-to-price ratio in the dataset?
reviews["points_price_ratio"] = reviews["points"].divide(reviews["price"])
highest_points_price_ratio_wine = reviews.loc[reviews["points_price_ratio"].argmax()]["title"]
# reviews.loc[(reviews.points / reviews.price).argmax()].title

# Create a "Series" counting how many times words "tropical" and "fruity" appears in the "description" column in the dataset.
tropical_wine = reviews["description"].map(lambda r: "tropical" in r).value_counts()
'''
False    126364
True       3607
Name: description, dtype: int64
'''
fruity_wine = reviews.description.map(lambda r: "fruity" in r).value_counts()
tropical_fruity_count = pd.Series([tropical_wine[True], fruity_wine[True]], index=["tropical", "fruity"])

# What combination of countries and varieties are most common?
country_variety_notNull = reviews.loc[(reviews["country"].notnull()) & (reviews["variety"].notnull())]
country_variety_combination = country_variety_notNull.apply(lambda row: row["country"] + " - " + row["variety"], axis="columns")
'''
0               Italy - White Blend
1         Portugal - Portuguese Red
                   ...           
129969          France - Pinot Gris
129970      France - Gewürztraminer
Length: 129907, dtype: object
'''
common = country_variety_combination.value_counts()


# In[ ]:


'''
Grouping and Sorting
'''
# Create a "Series" whose index is the "taster_twitter_handle" category from the dataset, and whose values count how many reviews each person wrote.
common_wine_reviewers = reviews.groupby("taster_twitter_handle").taster_twitter_handle.count()
# === reviews["taster_twitter_handle"].value_counts()

# Create a "Series" whose index is wine prices and whose values is the maximum number of points. Sort the valeus by price, ascending.
best_wine = reviews.groupby("price").points.max().sort_index()

# Create a "DataFrame" whose index is the "variety" category and whose values are the count, "min" and "max" prices.
wine_price_extremes = reviews.groupby("variety").price.agg([len, min, max])

# Create a "DataFrame" whose index are country and province and whose values is the wine with highest points.
best_wine = reviews.groupby(["country", "province"]).apply(lambda df: df.loc[df.points.argmax()])
# MultiIndex !!! 

# Create a "Series" whose index is reviewers and whose values is the average review score given out by that reviewer.
reviewer_mean_ratings = reviews.groupby("taster_name").points.mean()

# Create a "DataFrame" whose index is wine varieties and whose values are columns with the "min" and the "max" price of wines of this variety.
# Sort in descending order based on "min" first, "max" second.
wine_price_range = reviews.groupby("variety").price.agg([min, max]).sort_values(by=["min", "max"], ascending=False)
# to sort by index values, use the companion method sort_index. This method has the same arguments and default order.
wine_price_range = reviews.groupby("variety").price.agg([min, max]).sort_index()

# Create a "Series" whose index is a "MultiIndex"of "{country, variety}" pairs. 
# Sort the values in the "Series" in descending order based on wine count.
country_variety_pairs_series = reviews.groupby(["country", "variety"]).country.count().sort_values(ascending=False)
# Create a "DataFrame" under same requirement
country_variety_pairs_df = reviews.groupby(["country", "variety"]).country.agg([len]).sort_values(by="len", ascending=False)

# Converting MultiIndex back to a regular index.
country_variety_pairs_df.reset_index()


# In[ ]:


'''
Data types, Missing data and Replacing
'''
# What is the data type of the index in the dataset?
index_type = reviews.index.dtype

# What is the data type of the "points" column in the dataset?
points_type = reviews["points"].dtypes

# What is the data type of every column in the dataset?
columns_type = reviews.dtypes

# Create a "Series" from entries in the "price" column, but convert the entries to strings
price_series = reviews["price"].astype("str")

# Create a "Series"that, for each review in the dataset, states whether the wine reviewed has a null "price".
null_price = reviews[reviews["price"].isnull()]
# === reviews.loc[reviews["price"].isnull()]

# Create a "Series" counting the number of times each value occurs in the "region_1" field. 
# Replace missing values with "Unknown". Sort in descending order.
region1_count = reviews["region_1"].fillna("Unknown").value_counts().sort_values(ascending=False)

# Create a "pandas" "Series" showing how many times each of the columns in the dataset contains null values.
null_count = reviews.isnull().sum()
# Boolean data types has a property that "False" gets treated as 0 and "True" as 1 when performing math on the values.
# Thus, the "sum()" of a list of boolean values will return how many times "True" appears in that list.

# Create a "Series" replacing values "Invalid" with "Unknown" in the "region_1" field.
region1_replace = reviews["region_1"].replace("Invalid", "Unknown")


# In[ ]:


'''
Renaming and Combining workbook
'''
# Rename `region_1` and `region_2` columns to `region` and `locale`.
reviews.rename(columns={'region_1': 'region', 'region_2': 'locale'})

# Set the index name in the dataset to `wines`.
reviews.index.names = ['wines']

# Create a `DataFrame` of products mentioned on "gaming_products" and "movie_products"
products = pd.concat([gaming_products, movie_products])

# Both tables "powerlifting_meets" and "powerlifting_competitors" include column `MeetID`, a unique key for each meet (competition) included in the database.
# Generate a dataset combining the two tables into one.
combine_tables = powerlifting_meets.set_index("MeetID").join(powerlifting_competitors.set_index("MeetID"))


# In[ ]:


'''
Method chaining workbook
'''
'''
chess_games = ['id', 'rated', 'created_at', 'last_move_at', 'turns', 'victory_status',
               'winner', 'increment_code', 'white_id', 'white_rating', 'black_id',
               'black_rating', 'moves', 'opening_eco', 'opening_name', 'opening_ply']
'''
# Use the `winner` column to create a `Series` showing ratio of white wins, black wins and tie.
result_ratio = chess_games["winner"].value_counts() / len(chess_games)

# The `opening_name` field has information such as `Queen's Pawn Game` and `Queen's Pawn Game: Zukertort Variation`.
# Parse the `opening_name` field and generate a `Series` counting each of the "opening archetypes" used times.
striped_opening = chess_games["opening_name"].map(lambda n: n.split(":")[0].split("|")[0].split("#")[0].strip())
opening_count = striped_opening.value_counts()

# Group the games by `{white_id, victory_status}` and count the times of each ended result for each white player.
white_victory_status = chess_games.groupby(["white_id", "victory_status"]).victory_status.agg([len])
white_victory_status = white_victory_status.reset_index().rename(columns={"len": "n"})
# === chess_games.assign(n=0).groupby(['white_id', 'victory_status']).n.apply(len).reset_index()

# Create a `DataFrame` like the one in the previous exercise, but only include users who are in the top 20 users by number of games played. 
top_20 = chess_games["white_id"].value_counts().head(20).index
'''
Index(['taranga', 'chess-brahs', 'a_p_t_e_m_u_u', 'bleda', 'ssf7',
       'hassan1365416', 'khelil', 'saviter', 'anakgreget', '1240100948',
       'ozguragarr', 'ivanbus', 'vladimir-kramnik-1', 'vovkakuz',
       'thegrim123321', 'king5891', 'mastersalomon', 'islam01', 'ozil17',
       'artem555'],
      dtype='object')
'''
white_victory_status_top20 = white_victory_status.loc[white_victory_status["white_id"].isin(top_20)]

# Generate a `Series` whose index is a `MultiIndex` based on the `{koi_pdisposition, koi_disposition}` fields,
# and whose values is a count of how many times each possible combination occurred.
combination_count = kepler.groupby(["koi_pdisposition", "koi_disposition"]).koi_pdisposition.count()

# The `points` column in the `wine_reviews` dataset is measured on a 20-point scale between 80 and 100.
# Create a `Series` which normalizes the ratings to fit on a 1-to-5 scale, i.e., 80 -> 1, 100 -> 5
# Set the `Series` name to "Wine Ratings", and sort by index value (ascending).
normalized_points = wine_reviews["points"].dropna().map(lambda p: (p-80) / 4.0).sort_index()
normalized_points.index.names = ['Wine Ratings']

# Create a `Series` counting how many ramens earned each of the possible scores in the dataset.
# Convert the `Series` to the `float64` dtype and drop ramen whose rating is "Unrated".
# Set the name of the `Series` to "Ramen Ratings". Sort by index value (ascending).
ramen_ratings_count = ramen_reviews.loc[ramen_reviews["Stars"] != "Unrated", "Stars"].astype("float64")
# ramen_reviews["Stars"].replace('Unrated', None).dropna().astype("float64")
ramen_ratings_count = ramen_ratings_count.value_counts().sort_index()
ramen_ratings_count.index.names = ["Ramen Ratings"]

# Modify answer to the previous exercise by rounding review scores to the nearest half-point (so 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, or 5).
ramen_ratings = ramen_reviews["Stars"].replace('Unrated', None).dropna().astype("float64")
ramen_ratings_rounding = ramen_ratings.map(lambda v: int(v) if v - int(v) < 0.5 else int(v) + 0.5)
ramen_ratings_rounding_count = ramen_ratings_rounding.value_counts().sort_index()
ramen_ratings_rounding_count.index.names = ["Ramen Ratings"]

