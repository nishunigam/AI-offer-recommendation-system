import pandas as pd

# Load datasets
users = pd.read_csv('../data/users.csv')
transactions = pd.read_csv('../data/transactions.csv')

# Convert date column to datetime
transactions['date'] = pd.to_datetime(transactions['date'])

# -------------------------------
# BASIC DATA CLEANING
# -------------------------------

# Handle missing values (if any)
users.fillna({'age': users['age'].mean(),
              'income': users['income'].mean()}, inplace=True)

transactions.fillna({'amount': 0}, inplace=True)

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------

# 1. Average spend per user
avg_spend = transactions.groupby('user_id')['amount'].mean().reset_index()
avg_spend.rename(columns={'amount': 'avg_spend'}, inplace=True)

# 2. Transaction frequency per user
freq = transactions.groupby('user_id').size().reset_index(name='transaction_count')

# 3. Most frequent category (favorite category)
fav_category = transactions.groupby(['user_id', 'category']) \
                           .size().reset_index(name='count')

fav_category = fav_category.sort_values(['user_id', 'count'], ascending=[True, False])
fav_category = fav_category.drop_duplicates('user_id')[['user_id', 'category']]
fav_category.rename(columns={'category': 'favorite_category'}, inplace=True)

# -------------------------------
# MERGE ALL FEATURES
# -------------------------------

features = users.merge(avg_spend, on='user_id', how='left') \
                .merge(freq, on='user_id', how='left') \
                .merge(fav_category, on='user_id', how='left')

# Fill missing values after merge
features.fillna({
    'avg_spend': 0,
    'transaction_count': 0,
    'favorite_category': 'unknown'
}, inplace=True)

# -------------------------------
# SAVE FEATURES
# -------------------------------

features.to_csv('../features/features.csv', index=False)

print("Feature engineering completed. Saved to features/features.csv")
print(features.head())