# BigMart Sales Prediction Dashboard

## 📌 Project Overview
The **BigMart Sales Prediction Dashboard** is a **machine learning-powered web application** that predicts item sales using historical data. Built with **Python, Streamlit, and Machine Learning models (XGBoost, RandomForest, Linear Regression, and Ridge Regression)**, this dashboard provides real-time insights and predictions based on key features such as item visibility, MRP, outlet type, and more.

---

## 🏗️ Features
✅ **Predict Sales** using trained machine learning models  
✅ **Interactive Dashboard** with Streamlit for real-time analysis  
✅ **Feature Importance Visualization** to understand model behavior  
✅ **Data Preprocessing & Cleaning** including missing value treatment  
✅ **Model Comparisons** between Linear Regression, Ridge, RandomForest & XGBoost  
✅ **GitHub Integration & Deployment** for easy updates & version control  

---

## 🚀 Installation & Setup
### 1️⃣ Clone the repository
```sh
git clone https://github.com/ngoubimaximillian12/BigMart-Sales-Dashboard.git
cd BigMart-Sales-Dashboard
```

### 2️⃣ Create a virtual environment (Recommended)
```sh
python -m venv .venv
source .venv/bin/activate  # On Mac/Linux
.venv\Scripts\activate    # On Windows
```

### 3️⃣ Install dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit Dashboard
```sh
streamlit run app.py
```

The dashboard will be available at **http://localhost:8501** in your browser.

---

## 🔎 Usage Guide
- **Upload Data** (or use the default dataset)
- **Choose Model** (Linear Regression, Ridge, RandomForest, XGBoost)
- **Adjust Hyperparameters** (for tuning models)
- **View Predictions & Feature Importance**
- **Export Results** (Download Predictions & Visualizations)

---

## 📊 Model Performance
| Model                  | R2 Score |
|------------------------|----------|
| Linear Regression      | 0.5605   |
| Ridge Regression       | 0.5605   |
| RandomForest Regressor | 0.9791   |
| XGBoost (Tuned)       | 0.9934   |

🚀 **XGBoost is the best-performing model with 99.3% accuracy!**

---

## 🔗 Deployment Options
### 🌐 Deploy on Streamlit Cloud
1. Push all your files to GitHub
2. Create an account at [Streamlit Cloud](https://share.streamlit.io/)
3. Deploy directly from GitHub

### ☁️ Deploy on Heroku
1. Install Heroku CLI
2. Create a `Procfile`: `web: streamlit run app.py`
3. Push to Heroku: `git push heroku main`

### 🚀 Deploy on AWS (EC2 or Lambda)
- Use **AWS EC2** for hosting the app
- Use **AWS Lambda + API Gateway** for serverless execution

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Streamlit, Matplotlib, Seaborn  
- **Version Control:** Git & GitHub  
- **Deployment:** Streamlit Cloud, Heroku, AWS  

---

## 🎯 Future Improvements
🚀 **Add more ML models (LGBM, CatBoost, Neural Networks)**  
📊 **Improve Feature Engineering & Data Cleaning**  
🛠 **Enhance UI with better visualizations & customization**  
☁ **Deploy API for batch predictions**  

---

## 📝 License
This project is **open-source** under the MIT License.

---

## 💡 Contributors & Contact
👤 **Ngoubi Maximillian Diangha**  
📧 Email: ngoubimaximillian12@example.com  
🔗 GitHub: [@ngoubimaximillian12](https://github.com/ngoubimaximillian12)  

📢 Feel free to open an **issue** or **pull request** for improvements!

---

⭐ **If you found this useful, don't forget to star the repository!** ⭐

