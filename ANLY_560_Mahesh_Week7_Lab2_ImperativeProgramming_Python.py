# -*- coding: utf-8 -*-
"""
Mahesh Mitikiri

Week7_Lab2_ImperativeProgramming_Python

Editor: Spyder (Python 3.6)
Other versions can cause library issues

"""

## 1. Import pandas under the name pd

import pandas as pd

## 2. Print the version of pandas that has been imported.

print(pd.__version__)

## 3. Print out all the version information of the libraries that are required by the pandas library.
print(pd.show_versions())

## 4. Create a DataFrame df from this dictionary data which has the index labels.

import numpy as np
data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(data, index=labels)
print('\n')


## 5. Display a summary of the basic information about this DataFrame and its data.
print(df.info(verbose=True))
print('\n')

## 6. Return the first 3 rows of the DataFrame df.
print(df.head(n=3)) # head is a common funciton to retrieve top n values
print('\n')
## 7. Select just the 'animal' and 'age' columns from the DataFrame df.

print(df[['animal', 'age']]) # subset using variable names
print('\n')
## 8. Select the data in rows [3, 4, 8] and in columns ['animal', 'age'].
df.loc[df.index[[3, 4, 8]], ['animal', 'age']] # indexing to retrieve data

## 9. Select only the rows where the number of visits is greater than 3.
print(df['visits'] > 3)
## 10. Select the rows where the age is missing, i.e. is NaN.

print(df[df['age'].isnull()]) # isnull() is the funciton to get NaN

## 11. Select the rows where the animal is a cat and the age is less than 3.
df[(df['animal'] == 'cat') & (df['age'] < 3)] # simple boolean

## 12. Select the rows the age is between 2 and 4 (inclusive).
df[df['age'].between(2, 4)] # between is a custom function in pandas/numpy
#print(df[df['age'] > 2][df['age'] < 4])

## 13. Change the age in row 'f' to 1.5.
df.loc['f', 'age'] = 1.5 # position based selection

## 14. Calculate the sum of all visits (the total number of visits).

df['visits'].sum()

## 15. Calculate the mean age for each different animal in df.

df.groupby('animal')['age'].mean()  # group and then use mean

## 16. Append a new row 'k' to df with your choice of values for each column. Then delete that row to return the original DataFrame.
df.loc['m'] = df.loc['a']
df = df.drop('m')

## 17. Count the number of each type of animal in df.

df['animal'].value_counts() #in-built function

## 18. Sort df first by the values in the 'age' in decending order, then by the value in the 'visit' column in ascending order.

df.sort_values(by=['age', 'visits'], ascending=[False, True])

## 19. The 'priority' column contains the values 'yes' and 'no'. Replace this column with a column of boolean values: 'yes' should be True and 'no' should be False.

df['priority'] = df['priority'].map({'yes': True, 'no': False}) # pandas.series.map

## 20. In the 'animal' column, change the 'snake' entries to 'python'.

df['animal'] = df['animal'].replace('snake', 'python')

## 21. For each animal type and each number of visits, find the mean age. In other words, each row is an animal, each column is a number of visits and the values are the mean ages (hint: use a pivot table).

df.pivot_table(index='animal', columns='visits', values='age', aggfunc='mean')


## 22. You have a DataFrame df with a column 'A' of integers. For example:

df = pd.DataFrame({'A': [1, 2, 2, 3, 4, 5, 5, 5, 6, 7, 7]})
## How do you filter out rows which contain the same integer as the row immediately above?

df.loc[df['A'].shift() != df['A']] # pandas series functions

## 23. Given a DataFrame of numeric values, say

df = pd.DataFrame(np.random.random(size=(5, 3))) # a 5x3 frame of float values
# how do you subtract the row mean from each element in the row?

df.rsub(df.mean(axis=1), axis=0)*-1

## 24. Suppose you have DataFrame with 10 columns of real numbers, for example:
df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
##Which column of numbers has the smallest sum? (Find that column's label.)

df.sum().idxmin() # idxmin is a funciton to get index of pandas series

## 25. How do you count how many unique rows a DataFrame has

len(df.drop_duplicates(keep=False))

## 26. You have a DataFrame that consists of 10 columns of floating--point numbers. Suppose that exactly 5 entries in each row are NaN values.
## For each row of the DataFrame, find the column which contains the third NaN value.

(df.isnull().cumsum(axis=1) == 3).idxmax(axis=1) # combining numpy cumulative sum on columns (axis == 1)

## 27. A DataFrame has a column of groups 'grps' and and column of numbers 'vals'. For example:

df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})
# For each group, find the sum of the three greatest values.

df.groupby('grps')['vals'].nlargest(3).sum(level=0)


## 28. A DataFrame has two integer columns 'A' and 'B'. The values in 'A' are between 1 and 100 (inclusive). For each group of 
## 10 consecutive integers in 'A' (i.e. (0, 10], (10, 20], ...), calculate the sum of the 
## corresponding values in column 'B'.

# df.groupby(pd.cut(df['A'], np.arange(0, 101, 10)))['B'].sum()

## 29

df = pd.DataFrame({'X': [7, 2, 0, 3, 4, 2, 5, 0, 3, 4]})
y = (df['X'] != 0).cumsum() != (df['X'] != 0).cumsum().shift()
df['Y'] = y.groupby((y != y.shift()).cumsum()).cumsum()

## 30
df.unstack().sort_values()[-3:].index.tolist()


## 33. Create a DatetimeIndex that contains each business day of 2015 and use it to index a Series of random numbers. 
# Let's call this Series s.

DateTime = pd.date_range('1/1/2015', periods=261, freq='B')
s = pd.Series(np.random.rand(len(DateTime)), index=DateTime)

## 34. Find the sum of the values in s for every Wednesday.

s[s.index.weekday == 2].sum()

## 35. For each calendar month in s, find the mean of values.
s.resample('M').mean() # frequency conversion and resample


## 36. For each group of four consecutive calendar months in s, find the date on which the highest value occurred.
s.groupby(pd.TimeGrouper('4M')).idxmax() # using pandas dataframe object to convert from series for index

## 37. Create a DateTimeIndex consisting of the third Thursday in each month for the years 2015 and 2016.

pd.date_range('1/1/2015', periods=24, freq='WOM-3THU')

## Cleaning data

df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm', 
                               'Budapest_PaRis', 'Brussels_londOn'],
              'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
              'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
                   'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )', 
                               '12. Air France', '"Swiss Air"']})
## 38. Some values in the the FlightNumber column are missing. These numbers are meant to increase by 
# 10 with each row so 10055 and 10075 need to be put in place. Fill in these missing numbers and make the 
# column an integer column (instead of a float column).

df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)

# 39 The From_To column would be better as two separate columns! Split each string on the underscore delimiter _ 
# to give a new temporary DataFrame with the correct values. Assign the correct column names to this temporary DataFrame.

FooBar = df.From_To.str.split('_', expand=True)
FooBar.columns = ['From', 'To']

## 40 Standardise the strings so that only the first letter in city is uppercase
FooBar['From'] = FooBar['From'].str.capitalize()

FooBar['To'] = FooBar['To'].str.capitalize()

## 41. Delete the From_To column from df and attach the temporary DataFrame from the previous questions.
df = df.drop('From_To', axis=1)
df = pd.concat([df , FooBar],axis=1,join='inner')

## 42. In the Airline column, you can see some extra puctuation and symbols have appeared around the airline names. 
## Pull out just the airline name. E.g. '(British Airways. )' should become 'British Airways'.

df['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)', expand=False).str.strip()

## 44

letters = ['A', 'B', 'C']
numbers = list(range(10))

s = pd.Series(np.random.rand(30), index=pd.MultiIndex.from_product([letters, numbers]))

## 45
s.index.is_lexsorted()

## 46
s.loc[:,[1, 3, 6]]

## 47
s.loc[pd.IndexSlice[: 'B', 5: ]]

## 48
s.sum(level=0)

## 49
s.unstack(level=0).sum(axis=1)


## 56
import matplotlib.pyplot as plt
plt.style.use('ggplot')
df = pd.DataFrame({"xs":[1,5,2,8,1], "ys":[4,2,1,9,6]})

df.plot.scatter('xs', 'ys', marker = 'o', color = 'black')

## 57

df = pd.DataFrame({"productivity":[5,2,3,1,4,5,6,7,8,3,4,8,9],
                   "hours_in"    :[1,9,6,5,3,9,2,9,1,7,4,2,2],
                   "happiness"   :[2,1,3,2,3,1,2,3,1,2,2,1,3],
                   "caffienated" :[0,0,1,1,0,0,0,0,1,1,0,1,0]})

df.plot.scatter("hours_in", "happiness", s = df.hours_in*10, c = df.caffienated, edgecolors = 'face')

## 58

df = pd.DataFrame({"revenue":[57,68,63,71,72,90,80,62,59,51,47,52],
                   "advertising":[2.1,1.9,2.7,3.0,3.6,3.2,2.7,2.4,1.8,1.6,1.3,1.9],
                   "month":range(12)
                  })
bp = df.plot.bar("month", "revenue", color = "violet")
df.plot.line("month", "advertising", secondary_y = True, ax = bp)

import numpy as np
def float_to_time(x):
    return str(int(x)) + ":" + str(int(x%1 * 60)).zfill(2) + ":" + str(int(x*60 % 1 * 60)).zfill(2)

def day_stock_data():
    #NYSE is open from 9:30 to 4:00
    time = 9.5
    price = 100
    results = [(float_to_time(time), price)]
    while time < 16:
        elapsed = np.random.exponential(.001)
        time += elapsed
        if time > 16:
            break
        price_diff = np.random.uniform(.999, 1.001)
        price *= price_diff
        results.append((float_to_time(time), price))
    
    
    df = pd.DataFrame(results, columns = ['time','price'])
    df.time = pd.to_datetime(df.time)
    return df

def plot_candlestick(agg):
    fig, ax = plt.subplots()
    for time in agg.index:
        ax.plot([time.hour] * 2, agg.loc[time, ["high","low"]].values, color = "black")
        ax.plot([time.hour] * 2, agg.loc[time, ["open","close"]].values, color = agg.loc[time, "color"], linewidth = 10)

    ax.set_xlim((8,16))
    ax.set_ylabel("Price")
    ax.set_xlabel("Hour")
    ax.set_title("OHLC of Stock Value During Trading Day")
    plt.show()
    
## 59
df = day_stock_data()
df.set_index("time", inplace = True)
hour = df.resample("H").ohlc()
hour.columns.droplevel()
hour["color"] = (hour.close > hour.open).map({True:"green",False:"red"})
