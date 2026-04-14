# 🎬 CineMatch — Movie Recommendation System

🚀 **Live App:**
👉 https://movie-recommendation-system-vprwkubyz92sfbvcw5443p.streamlit.app/

---

## 📌 Overview

**CineMatch** is a content-based movie recommendation system that suggests movies based on their similarity in content — not popularity.

It analyzes movie metadata like **overview, genres, and keywords** to recommend films you are most likely to enjoy.

---

## 🎯 Problem Statement

With thousands of movies available, users often struggle to find content that truly matches their taste.
Most recommendation systems rely on popularity or user ratings rather than actual content similarity.

---

## 💡 Solution

CineMatch uses **Natural Language Processing (NLP)** techniques to understand movie content and recommend similar movies using:

* **TF-IDF Vectorization**
* **Cosine Similarity**

This ensures recommendations are based on *what the movie is about*, not just how popular it is.

---

## 🛠 Tech Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **Streamlit**
* **Pickle (Model Serialization)**

---

## ⚙️ Features

✅ Content-based movie recommendations
✅ Clean and modern UI
✅ Real-time results
✅ Fully deployed web application
✅ Scalable ML pipeline

---

## 🧠 How It Works

1. Movie data is preprocessed and combined into a single "tags" column
2. Text data is converted into numerical vectors using **TF-IDF**
3. **Cosine similarity** is calculated between movies
4. Top similar movies are recommended based on similarity score

---

## 📂 Project Structure

```
Movie-Recommendation-System/
│── main.py
│── requirements.txt
│── movies_metadata.csv
│── indices.pkl
│── tfidf_matrix.pkl
│── tfidf_vectorizer.pkl
```

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/affanali-meth/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run main.py
```

---

## 📚 Learnings

This project helped me:

* Build end-to-end ML applications
* Handle real-world deployment challenges
* Work with large data and model files
* Improve UI/UX for ML products

---

## 🚧 Challenges Faced

* Managing large `.pkl` files during deployment
* Resolving Git merge conflicts
* Ensuring compatibility across environments

---

## 🤝 Connect with Me

* 🔗 LinkedIn: https://www.linkedin.com/in/mohdaffanali/
* 💻 GitHub: https://github.com/affanali-meth

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!

---

### 🎥 Built with passion for movies & machine learning
