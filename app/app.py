from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os
from flask_cors import CORS

# -------------------------------
# INIT APP
# -------------------------------
app = Flask(__name__, template_folder='templates')
CORS(app)

# -------------------------------
# LOAD MODEL & DATA
# -------------------------------
model = pickle.load(open('models/model.pkl', 'rb'))
model_features = pickle.load(open('models/model_features.pkl', 'rb'))

features = pd.read_csv('features/features.csv')

# -------------------------------
# OFFERS (same as training)
# -------------------------------
offers = pd.DataFrame({
    'offer_id': [1, 2, 3],
    'category': ['food', 'shopping', 'travel'],
    'discount': [10, 20, 15]
})

# -------------------------------
# RECOMMEND FUNCTION (ULTRA LIGHT)
# -------------------------------
def recommend(user_id):
    # Get only ONE user
    user_df = features[features['user_id'] == user_id]

    if user_df.empty:
        return []

    user = user_df.iloc[0].to_dict()
    rows = []

    for _, offer in offers.iterrows():
        row = {}

        # --- Numeric / base features ---
        for key in user:
            if key not in ['city', 'favorite_category']:
                row[key] = user[key]

        # --- Encode user categorical features ---
        for col in model_features:
            if col.startswith('city_'):
                row[col] = 1 if col == f"city_{user.get('city')}" else 0
            elif col.startswith('favorite_category_'):
                row[col] = 1 if col == f"favorite_category_{user.get('favorite_category')}" else 0

        # --- Offer features ---
        row['discount'] = offer['discount']

        for cat in ['food', 'shopping', 'travel']:
            row[f'category_{cat}'] = 1 if offer['category'] == cat else 0

        rows.append(row)

    data = pd.DataFrame(rows)

    # Align with training features
    for col in model_features:
        if col not in data.columns:
            data[col] = 0

    data = data[model_features]

    # Predict
    scores = model.predict(data)

    # Build response
    result = []
    for i, offer in offers.iterrows():
        result.append({
            "offer_id": int(offer['offer_id']),
            "category": offer['category'],
            "discount": int(offer['discount']),
            "score": float(scores[i])
        })

    return result

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
        print("FULL ERROR:\n", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# -------------------------------
# RUN SERVER
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
