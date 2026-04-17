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
    user = features[features['user_id'] == user_id]

    if user.empty:
        return ["User not found"]

    # Cross join user with offers
    user = user.copy()
    user['key'] = 1
    offers['key'] = 1
    data = user.merge(offers, on='key').drop('key', axis=1)

    # Save original columns before encoding
    original_data = data[['offer_id', 'category', 'discount']].copy()

    # Encode
    data = pd.get_dummies(data, columns=['city', 'favorite_category', 'category'])

    # Align columns with training model
    model_features = model.feature_name()
    for col in model_features:
        if col not in data.columns:
            data[col] = 0

    data = data[model_features]

    # Predict scores
    scores = model.predict(data)
    data['score'] = scores

    # Rank offers
    top_offers = data.sort_values('score', ascending=False)

    # Merge scores back with original data
    top_offers = data.copy()
    top_offers['score'] = scores

    result = original_data.copy()
    result['score'] = scores

    result = result.sort_values('score', ascending=False)

    return result.to_dict(orient='records')


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
    import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
