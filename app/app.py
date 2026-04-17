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

def recommend(user_id):
    user = features[features['user_id'] == user_id].head(1)

    if user.empty:
        return []

    user = user.copy()

    # Cross join
    user['key'] = 1
    offers['key'] = 1
    data = user.merge(offers, on='key').drop('key', axis=1)

    # Encode
    data = pd.get_dummies(data, columns=['city', 'favorite_category', 'category'])

    # Align with training columns
    for col in model_features:
        if col not in data.columns:
            data[col] = 0

    data = data[model_features]

    # Predict
    scores = model.predict(data)
    data['score'] = scores

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
