from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import os

# -------------------------------
# INIT APP (ONLY ONCE)
# -------------------------------
app = Flask(__name__, template_folder='templates')
CORS(app)

# -------------------------------
# LOAD DATA
# -------------------------------
# (Model kept but not used in API to avoid timeout)
model = pickle.load(open('models/model.pkl', 'rb'))

features = pd.read_csv('features/features.csv')

# Reduce memory (IMPORTANT for Render)
features = features[['user_id', 'avg_spend', 'transaction_count', 'city', 'favorite_category']]

# Dummy offers
offers = pd.DataFrame({
    'offer_id': [1, 2, 3],
    'category': ['food', 'shopping', 'travel'],
    'discount': [10, 20, 15]
})

# -------------------------------
# LIGHTWEIGHT RECOMMENDATION ENGINE
# -------------------------------
# Load feature columns
model_features = pickle.load(open('models/model_features.pkl', 'rb'))

# Precompute user features (ONE ROW PER USER)
user_features_map = {}

for _, row in encoded_features.iterrows():
    user_features_map[row['user_id']] = row.to_dict()

def recommend(user_id):
    if user_id not in user_features_map:
        return []

    user = user_features_map[user_id]

    results = []

    for _, offer in offers.iterrows():
        row = user.copy()

        # Add offer info
        row['discount'] = offer['discount']

        # Encode offer category manually
        for cat in ['food', 'shopping', 'travel']:
            row[f'category_{cat}'] = 1 if offer['category'] == cat else 0

        results.append(row)

    data = pd.DataFrame(results)

    # Align columns
    for col in model_features:
        if col not in data.columns:
            data[col] = 0

    data = data[model_features]

    scores = model.predict(data)

    data['score'] = scores
    data['offer_id'] = offers['offer_id']
    data['category'] = offers['category']
    data['discount'] = offers['discount']

    return data[['offer_id', 'category', 'discount', 'score']].to_dict(orient='records')
# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['GET'])
def get_recommendations():
    try:
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        user_id = int(user_id)

        results = recommend(user_id)
        return jsonify(results)

    except Exception as e:
        import traceback
        print("FULL ERROR:\n", traceback.format_exc())  # 🔥 IMPORTANT
        return jsonify({"error": str(e)}), 500

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
