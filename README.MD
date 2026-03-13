# Movie Recommendation Systems – ML with Big Data Assignment

## Overview
This project implements multiple recommendation system techniques using the MovieLens dataset.  
The goal is to explore different approaches to building recommender systems and analyze their performance, interpretability, and behavior.

The notebook includes implementations of:

- Content-Based Filtering
- User Profile Based Recommendation
- Collaborative Filtering
- Matrix Factorization (SVD)
- Hybrid Recommendation Model
- Neural Network Based Recommender
- Reinforcement Learning Based Recommender
- Explainability in Recommender Systems

These methods are applied to the **MovieLens dataset** to recommend movies based on user preferences and movie features.

---

## Dataset
The project uses the **MovieLens dataset(ml-latest-small)**, which contains:

- Movie metadata
- User ratings
- Movie genres

Dataset files used in the notebook include:

- `movies.csv`
- `ratings.csv`
- `links.csv`
- `tags.csv`

These files should be placed in the same directory as the notebook.

---

## Dependencies
The notebook requires the following Python libraries:

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scikit-surprise
- tensorflow
- shap

Dependencies can be installed using pip:

```bash
pip install "numpy<2"
pip install pandas scikit-learn matplotlib seaborn
pip install scikit-surprise
pip install tensorflow
pip install shap
```

---

## How to Run the Notebook

1. Clone or download the repository.
2. Ensure the MovieLens dataset files are present in the project directory.
3. Install the required dependencies.
4. Open the notebook `MLBD_assignment3.ipynb`.
5. Run the notebook **from top to bottom**.

If NumPy is downgraded during installation, restart the runtime before executing the notebook again.

---

## Project Structure

```
MLBD_assignment3.ipynb   - Main notebook containing all implementations
README.md                - Instructions and project description
dataset/                 - MovieLens dataset files
```

---

## Key Components Implemented

### 1. Content-Based Filtering
Uses TF-IDF features derived from movie genres and cosine similarity to recommend similar movies.

### 2. Collaborative Filtering
Implements user-based collaborative filtering by computing similarity between users based on their rating behavior.

### 3. Matrix Factorization
Uses Singular Value Decomposition (SVD) to learn latent relationships between users and movies.

### 4. Hybrid Recommendation Model
Combines both content-based and collaborative filtering scores to generate improved recommendations.

### 5. Neural Network Recommender
Uses embedding layers for users and movies to learn complex interactions and predict ratings.

### 6. Reinforcement Learning Recommender
Implements a Multi-Armed Bandit approach to balance exploration and exploitation in recommendation selection.

### 7. Explainability
Provides explanations for recommendations using:

- Feature-based explanations based on movie genres
- Neighborhood-based explanations using similar users
- Model-agnostic explanations using SHAP

---

## Notes

- The notebook is organized into modular functions for better readability and maintainability.
- Major code sections include comments explaining the logic and workflow.
- Visualizations are included to analyze model behavior and training performance.
- Explainability techniques are used to improve transparency of the recommendation models.

---

## Author

Sushantak Parashar Jha  
M25CSA035