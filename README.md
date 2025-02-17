
# **Fraud Detection for E-commerce & Bank Transactions**
**10 Academy AI Mastery – Week 8 & 9 Challenge**  
**Date:** 05 Feb - 18 Feb 2025  
**Prepared by:** <Mulsew M. Tesfaye>  

## **🔹 Project Overview**
Fraudulent activities in e-commerce and banking transactions cause significant financial losses. This project aims to build a **fraud detection system** using **machine learning** and **geolocation analysis**. By analyzing transaction patterns and deploying predictive models, we enhance fraud detection accuracy, minimize false positives, and ensure transaction security.

---

## **📌 Objectives**
✔ **Analyze and preprocess transaction data**  
✔ **Engineer new fraud detection features**  
✔ **Build and train machine learning models**  
✔ **Evaluate model performance using metrics**  
✔ **Deploy the models using Flask API & Docker**  
✔ **Develop a fraud detection dashboard using Dash**  

---

## **📂 Dataset Description**
We use **two datasets**:  
1. **Fraud_Data.csv** (E-commerce transactions)  
2. **creditcard.csv** (Bank credit transactions)  
3. **IpAddress_to_Country.csv** (IP-based geolocation mapping)  

### **Fraud_Data.csv**
| Column Name | Description |
|-------------|------------|
| user_id | Unique identifier for the user |
| signup_time | Timestamp of user registration |
| purchase_time | Timestamp of purchase |
| purchase_value | Transaction amount in dollars |
| device_id | Unique device identifier |
| source | Traffic source (SEO, Ads, etc.) |
| browser | Browser used for the transaction |
| sex | Gender of the user (M/F) |
| age | Age of the user |
| ip_address | IP address from where the transaction was made |
| class | Target variable (1 = Fraud, 0 = Not Fraud) |

### **creditcard.csv**
| Column Name | Description |
|-------------|------------|
| Time | Seconds since the first transaction |
| V1-V28 | Anonymized features (PCA-transformed) |
| Amount | Transaction amount in dollars |
| Class | Target variable (1 = Fraud, 0 = Not Fraud) |

### **IpAddress_to_Country.csv**
| Column Name | Description |
|-------------|------------|
| lower_bound_ip_address | Start of IP range |
| upper_bound_ip_address | End of IP range |
| country | Mapped country |

---

## **🛠 Tech Stack & Tools**
### **Languages & Frameworks**
- **Python**: Data processing, model training, and API development
- **Flask**: Model deployment as a REST API
- **Dash (Plotly)**: Fraud detection dashboard
- **Docker**: Containerization for deployment

### **Machine Learning Libraries**
- **Pandas, NumPy**: Data preprocessing
- **Scikit-learn**: Model training & evaluation
- **SHAP, LIME**: Model explainability
- **MLflow**: Experiment tracking

### **Other Tools**
- **GitHub**: Version control
- **PostgreSQL**: Database for storing fraud data
- **Matplotlib, Seaborn**: Data visualization

---

## **🚀 Project Structure**
```
fraud_detection_project/
│── data/                     # Raw datasets
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   ├── creditcard.csv
│── notebooks/                 # Jupyter Notebooks
│   ├── EDA.ipynb              # Exploratory Data Analysis
│   ├── Model_Training.ipynb   # Model Training & Evaluation
│   ├── Model_Explainability.ipynb # SHAP & LIME analysis
│── models/                    # Trained models
│   ├── fraud_model.pkl
│   ├── creditcard_model.pkl
│── api/                       # Flask API for fraud detection
│   ├── serve_model.py
│   ├── requirements.txt
│   ├── Dockerfile
│── dashboard/                 # Dash application for fraud insights
│   ├── app.py
│── reports/                   # Reports & documentation
│   ├── interim_1_report.md
│   ├── interim_2_report.md
│── logs/                      # Logs for debugging
│   ├── app.log
│── README.md                  # Project overview
│── .gitignore                 # Ignored files
```

---

## **📌 Key Deliverables**
### **✅ Task 1: Data Analysis & Preprocessing**
✔ Handle missing values & clean data  
✔ Perform **Exploratory Data Analysis (EDA)**  
✔ Merge IP-based geolocation data  
✔ Feature engineering (**transaction frequency, velocity, time-based features**)  
✔ Normalize and encode categorical variables  

### **✅ Task 2: Model Building & Training**
✔ Train multiple models:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - Multi-Layer Perceptron (MLP)
   - CNN, RNN, LSTM (for advanced fraud patterns)
✔ Use **cross-validation** to improve accuracy  
✔ Evaluate models using:
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC-AUC Score  

### **✅ Task 3: Model Explainability**
✔ **SHAP & LIME Analysis**
   - SHAP Summary Plot (Feature importance)
   - Force Plot (Local explanation)
   - Dependence Plot (Feature impact on fraud prediction)  
✔ **Interpret LIME explanations for individual fraud cases**  

### **✅ Task 4: Model Deployment (Flask API)**
✔ Create a REST API to **detect fraud in real-time**  
✔ **Containerize API using Docker**  
✔ Implement **logging & monitoring**  

### **✅ Task 5: Fraud Detection Dashboard (Dash)**
✔ Build an interactive **dashboard using Dash (Plotly)**  
✔ Show **fraud statistics, trends, and insights**  
✔ Provide **geolocation-based fraud detection analysis**  

---

## **📦 Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

### **2️⃣ Install Dependencies**
```bash
pip install -r api/requirements.txt
```

### **3️⃣ Run Flask API**
```bash
cd api
python serve_model.py
```
- The API will start at `http://127.0.0.1:5000/`.

### **4️⃣ Run the Dashboard**
```bash
cd dashboard
python app.py
```
- Open the dashboard in a browser at `http://127.0.0.1:8050/`.

### **5️⃣ Run Dockerized API**
```bash
docker build -t fraud-detection-model .
docker run -p 5000:5000 fraud-detection-model
```

---

## **🧪 Testing the API**
### **Example Request**
```python
import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "purchase_value": 200,
    "browser": "Chrome",
    "source": "SEO",
    "age": 45,
    "ip_address": 19216811,
}
response = requests.post(url, json=data)
print(response.json())
```
### **Expected Response**
```json
{
  "fraud_prediction": 1,
  "confidence": 0.92
}
```

---

## **📌 Contributors**
- **[Your Name]** - Data Science & Machine Learning  
- **Team Members** - Contributions to API, dashboard, and model training  

---

## **📌 References**
- Kaggle Fraud Detection Datasets: [Link](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Scikit-learn Documentation: [Link](https://scikit-learn.org/stable/)  
- Flask Documentation: [Link](https://flask.palletsprojects.com/)  

---

## **📜 License**
This project is licensed under the **MIT License**.  

---

## **🔹 Next Steps**
- Improve model with **advanced deep learning techniques**  
- Optimize API for **faster response times**  
- Deploy API and dashboard **on cloud (AWS, Render, or Heroku)**  
