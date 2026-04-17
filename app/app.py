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
def recommend(user_id):
    user = features[features['user_id'] == user_id]

    if user.empty:
        return []

    user = user.iloc[0]

    results = []

    for _, offer in offers.iterrows():
        score = 0

        # Rule-based scoring (FAST)
        if user['favorite_category'] == offer['category']:
            score += 5

        score += offer['discount'] / 5
        score += user['avg_spend'] / 500  # simulate ML signal

        results.append({
            "offer_id": int(offer['offer_id']),
            "category": offer['category'],
            "discount": int(offer['discount']),
            "score": float(score)
        })

    # Sort by score
    results = sorted(results, key=lambda x: x['score'], reverse=True)

    return results


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
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
