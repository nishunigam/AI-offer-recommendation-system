from flask import Flask, request, jsonify, render_template
import pickle
import json
import os
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load lightweight JSON
with open(os.path.join(BASE_DIR, '../features/features.json')) as f:
    features = json.load(f)

model = None
model_features = None

def load_model():
    global model, model_features
    if model is None:
        model = pickle.load(open(os.path.join(BASE_DIR, '../models/model.pkl'), 'rb'))
        model_features = pickle.load(open(os.path.join(BASE_DIR, '../models/model_features.pkl'), 'rb'))

offers = [
    {"offer_id": 1, "category": "food", "discount": 10},
    {"offer_id": 2, "category": "shopping", "discount": 20},
    {"offer_id": 3, "category": "travel", "discount": 15},
]

def recommend(user_id):
    load_model()

    user_id = str(user_id)

    if user_id not in features:
        return []

    user = features[user_id]
    results = []

    for offer in offers:
        row = {}

        # numeric features
        for key, value in user.items():
            if key not in ['city', 'favorite_category']:
                row[key] = value

        # encode user categorical
        for col in model_features:
            if col.startswith('city_'):
                row[col] = 1 if col == f"city_{user.get('city')}" else 0
            elif col.startswith('favorite_category_'):
                row[col] = 1 if col == f"favorite_category_{user.get('favorite_category')}" else 0

        # offer features
        row['discount'] = offer['discount']

        for cat in ['food', 'shopping', 'travel']:
            row[f'category_{cat}'] = 1 if offer['category'] == cat else 0

        # align
        input_row = [row.get(col, 0) for col in model_features]

        score = model.predict([input_row])[0]

        results.append({
            "offer_id": offer['offer_id'],
            "category": offer['category'],
            "discount": offer['discount'],
            "score": float(score)
        })

    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend')
def get_recommendations():
    try:
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({"error": "user_id required"}), 400

        results = recommend(int(user_id))
        return jsonify(results)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)