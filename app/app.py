from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)
# -------------------------------
# LOAD MODEL & FEATURES
# -------------------------------
model = pickle.load(open('models/model.pkl', 'rb'))
features = pd.read_csv('features/features.csv')
# Pre-encode user features ONCE (startup time)
encoded_features = pd.get_dummies(
    features,
    columns=['city', 'favorite_category']
)

# Dummy offers (same as training)
offers = pd.DataFrame({
    'offer_id': [1, 2, 3],
    'category': ['food', 'shopping', 'travel'],
    'discount': [10, 20, 15]
})

# -------------------------------
# RECOMMENDATION FUNCTION
# -------------------------------
def recommend(user_id):
    user = encoded_features[encoded_features['user_id'] == user_id]

    if user.empty:
        return []

    user = user.copy()

    # Cross join
    user['key'] = 1
    offers['key'] = 1
    data = user.merge(offers, on='key').drop('key', axis=1)

    # Encode only offer category (small)
    data = pd.get_dummies(data, columns=['category'])

    # Align columns
    model_features = model.feature_name()

    for col in model_features:
        if col not in data.columns:
            data[col] = 0

    data = data[model_features]

    scores = model.predict(data)
    data['score'] = scores

    return data[['offer_id', 'discount', 'score']].to_dict(orient='records')


# -------------------------------
# API ENDPOINT
# -------------------------------
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates')

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
        print("ERROR:", str(e))  # logs in Render
        return jsonify({"error": str(e)}), 500


# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
