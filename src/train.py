import pandas as pd
import numpy as np
import pickle

# Try importing LightGBM
try:
    import lightgbm as lgb
except ImportError:
    print("LightGBM not installed. Run: pip install lightgbm")
    exit()

# -------------------------------
# LOAD FEATURES
# -------------------------------
features = pd.read_csv('../features/features.csv')

# -------------------------------
# CREATE DUMMY OFFERS
# -------------------------------
offers = pd.DataFrame({
    'offer_id': [1, 2, 3],
    'category': ['food', 'shopping', 'travel'],
    'discount': [10, 20, 15]
})

# -------------------------------
# CREATE USER-OFFER PAIRS
# -------------------------------
features['key'] = 1
offers['key'] = 1

data = features.merge(offers, on='key').drop('key', axis=1)

# -------------------------------
# CREATE LABEL (SIMULATED LOGIC)
# -------------------------------
# Better scoring logic
# Stronger and more varied labels
data['label'] = 0

# Strong preference
data.loc[data['favorite_category'] == data['category'], 'label'] += 2

# High spend users
data.loc[data['avg_spend'] > 700, 'label'] += 1

# Frequent users
data.loc[data['transaction_count'] > 1, 'label'] += 1

# High discount offers
data.loc[data['discount'] > 15, 'label'] += 1

data['label'] = data['label'] + np.random.randint(0, 2, size=len(data))
# -------------------------------
# ENCODE CATEGORICALS
# -------------------------------
data['user_offer_interaction'] = data['avg_spend'] * data['discount']
data = pd.get_dummies(data, columns=['city', 'favorite_category', 'category'])

# -------------------------------
# PREPARE DATA
# -------------------------------
X = data.drop(['user_id', 'offer_id', 'label'], axis=1)
y = data['label']

# Group for ranking
group = data.groupby('user_id').size().to_list()

# -------------------------------
# TRAIN MODEL
# -------------------------------
train_data = lgb.Dataset(X, label=y, group=group)

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'verbosity': -1,
    'learning_rate': 0.1,
    'num_leaves': 31,
    'min_data_in_leaf': 1
}
model = lgb.train(params, train_data, num_boost_round=300)

# -------------------------------
# SAVE MODEL
# -------------------------------
import os
os.makedirs('../models', exist_ok=True)

with open('../models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved to models/model.pkl")