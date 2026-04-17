# AI Offer Recommendation System (Phase 1)

An end-to-end machine learning system for generating **personalized offer recommendations** using user behavior and transactional data.


## Live Demo
[https://ai-offer-recommendation-system.onrender.com](https://ai-offer-recommendation-system.onrender.com/)

## Demo

### Home UI
UI_Image.png

### Recommendations Output
UI_Output.png

## Project Overview

This project simulates a real-world **fintech recommendation system** where personalized offers are ranked for users based on behavioral and transactional data.

The system includes:

* Data preprocessing & feature engineering  
* Machine learning ranking model (LightGBM)  
* REST API using Flask  
* Interactive frontend UI using Bootstrap  
* Cloud deployment for real-time inference  

## Architecture

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

AI-offer-recommendation-system/
│
├── app/
│ ├── app.py
│ └── templates/
│ └── index.html
│
├── src/
│ ├── preprocess.py
│ └── train.py
│
├── data/
├── features/
│ └── features.csv
│
├── models/
│ └── model.pkl
│
├── requirements.txt
├── Procfile



## How to Run Locally

### 1. Clone the repo



