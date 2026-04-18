# AI Offer Recommendation System
> Phase 1 Complete — Production-ready ML system deployed

An end-to-end machine learning system for generating **personalized offer recommendations** using user behavior and transactional data.


## Live Demo
https://ai-offer-recommendation-system-7ju8.onrender.com/


## Demo

### Home UI
![UI](https://raw.githubusercontent.com/nishunigam/AI-offer-recommendation-system/main/UI_Image.png)

### Recommendations Output
![Results](https://raw.githubusercontent.com/nishunigam/AI-offer-recommendation-system/main/UI_Output.png)


## Project Overview

This project simulates a real-world **fintech recommendation system** where personalized offers are ranked for users based on behavioral and transactional data.

The system includes:

* Data preprocessing & feature engineering  
* Machine learning ranking model (LightGBM)  
* REST API using Flask  
* Interactive frontend UI using Bootstrap  
* Cloud deployment for real-time inference  

## Architecture

```
User Data + Transactions
↓
Feature Engineering (Pandas)
↓
Training Dataset (User × Offer)
↓
LightGBM Ranking Model
↓
Flask API (/recommend)
↓
Frontend UI (HTML + Bootstrap)
```

## Tech Stack

* **Language:** Python  
* **Data Processing:** Pandas, NumPy  
* **Model:** LightGBM (Learning-to-Rank)  
* **Backend:** Flask  
* **Frontend:** HTML, Bootstrap, JavaScript  
* **Deployment:** Render  


## Features Implemented

### Feature Engineering

* Average spend per user  
* Transaction frequency  
* Favorite category detection  
* Demographic enrichment (age, income, city)

### Recommendation Logic

* Cross join users with available offers  
* Relevance labeling based on behavior  
* Ranking using LightGBM (`lambdarank` objective)  
* Lightweight scoring layer for real-time inference  

### Production Optimization

- Optimized inference to avoid memory issues in cloud deployment
- Reduced heavy pandas operations during runtime
- Implemented lightweight feature handling for fast API response
- Fixed worker timeout issues in deployment
  

## Model Details

- Algorithm: LightGBM Ranker (LambdaRank)  
- Objective: Optimize ranking quality using NDCG  
- Input: User features + offer features  
- Output: Relevance score for each user-offer pair


## API

- Endpoint: `/recommend?user_id=<id>`  
- Method: GET  
- Response: Ranked list of offers with relevance scores  

Example:
http://127.0.0.1:5000/recommend?user_id=1


## Frontend UI

* User input for ID  
* Dynamic recommendations display  
* Top recommendation highlighting  
* Responsive Bootstrap design  
* Loading indicator  


## Project Structure

```
AI-offer-recommendation-system/
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── src/
│   ├── preprocess.py
│   └── train.py
│
├── data/
├── features/
│   └── features.csv
│
├── models/
│   └── model.pkl
│
├── requirements.txt
├── Procfile
```

## How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/nishunigam/AI-offer-recommendation-system.git
cd AI-offer-recommendation-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run preprocessing
```bash
python src/preprocess.py
```

### 4. Train model
```bash
python src/train.py
```

### 5. Start Flask server
```bash
python app/app.py
```

### 6. Open in browser
http://127.0.0.1:5000/

### Example Request
http://127.0.0.1:5000/recommend?user_id=1


## Sample Output

```json
[
  {
    "category": "food",
    "discount": 10,
    "offer_id": 1,
    "score": 4.85
  },
  {
    "category": "shopping",
    "discount": 20,
    "offer_id": 2,
    "score": 3.27
  }
]
```


## Deployment

The application is deployed on a cloud platform and supports real-time inference.

### Deployment Highlights

- Deployed using Flask + Gunicorn
- Handles real-time API requests
- Optimized to prevent memory overflow and worker timeouts
- End-to-end integration of ML model with frontend UI


## Production Insight

To ensure **low-latency performance in a resource-constrained environment**, a lightweight scoring layer is used during inference while the ML model is trained offline.


## Key Learning Outcomes

* Built an end-to-end ML pipeline from scratch  
* Implemented Learning-to-Rank using LightGBM  
* Designed feature engineering from transactional data  
* Integrated ML model into a production-like API  
* Connected backend with a dynamic frontend UI  


## Resume Highlight

Built and deployed a production-ready personalized recommendation system using LightGBM ranking, serving real-time predictions via a Flask API, with optimized inference to handle cloud deployment constraints.


## Author

**Nishchala Nigam**  
https://github.com/nishunigam
