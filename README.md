# ⚽ GOAT Analysis: Messi vs Ronaldo

A data science project analyzing 20+ seasons of career data (1,415+ goals) for Lionel Messi and Cristiano Ronaldo — from exploratory comparison to ML-based playing-style clustering with a deployed interactive app.

## 📁 Project Structure

- **`notebooks/`** — Exploratory data analysis: career stats comparison, season-wise trends, goal types, competition performance
- **`ml_clustering/`** — K-Means clustering model that identifies playing-style profiles per season, plus a Streamlit app for interactive exploration and prediction
- **`visuals/`** — Saved chart outputs (goals per season, goal types, competition breakdown, etc.)
- **`data.csv`** — Raw goal-by-goal dataset

## 🔍 Key Findings
- Messi averages **37.0 goals/season** vs Ronaldo's **33.8**, across 19 and 21 seasons respectively
- Ronaldo leads total career goals (710 vs 703) due to longer career span
- Clustering revealed **3 distinct playing-style profiles**:
  - **Prolific Right-Footed Finisher** — high output, right-foot dominant, strong away form
  - **Left-Foot Dominant Playmaker** — high output, left-foot dominant (matches Messi's signature style)
  - **Lower-Output Season** — transitional or lower-scoring seasons

## 🛠️ Tech Stack
Python, Pandas, NumPy, Matplotlib, Scikit-learn (StandardScaler, KMeans), Streamlit

## 🚀 Live Demo
[Try the interactive app here](#) *(add your Streamlit link once deployed)*

## 👤 Author
**Mihika Jain**
[LinkedIn](https://linkedin.com/in/mihika-jain-27b093271) | [GitHub](https://github.com/Mihi2724)