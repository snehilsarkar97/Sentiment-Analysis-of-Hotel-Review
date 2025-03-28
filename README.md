# Sentiment-Analysis-of-Hotel-Review
CIS 5850: Communication and Information Systems
# Hotel Review Sentiment & Topic Analysis

This project analyzes hotel reviews from TripAdvisor to extract valuable insights using Natural Language Processing (NLP) and machine learning techniques. It focuses on sentiment classification, rating prediction, topic modeling, and clustering of negative reviews.

## Project Overview

This project demonstrates the power of machine learning techniques in analyzing hotel reviews. The key components include:

* **Sentiment Analysis:** Classifying reviews into positive, negative, or neutral categories using NLTK.
* **Rating Prediction Models:** Building models to predict review ratings.
* **Topic Modeling (LDA):** Identifying key themes within the reviews.
* **Clustering Negative Reviews (K-means):** Grouping negative reviews to uncover common issues.

## Key Results

* **Sentiment Analysis Accuracy:** Achieved 78% accuracy using NLTK for sentiment classification.
* **Rating Prediction Accuracy:** Reached up to 84.83% accuracy in predicting review ratings using machine learning models.
* **Identified 5 Key Themes:** Topic modeling revealed five distinct themes: General Hotel Features, Resort & Leisure Experience, Issues/Service Concerns, Guest Experiences and Amenities & Comfort.
* **Negative Review Clustering (ARI):** Clustering of negative reviews resulted in an Adjusted Rand Index (ARI) score of 0.019, indicating limited alignment with original sentiment labels.

## Technologies Used

* **Python:** Core programming language.
* **NLTK (Natural Language Toolkit):** For text preprocessing and sentiment analysis.
* **Scikit-learn (sklearn):** For machine learning models, clustering, and vectorization.
* **Gensim:** For Latent Dirichlet Allocation (LDA) topic modeling.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib/Seaborn:** For data visualization.
* **WordCloud:** For visualizing word frequencies.

## Dataset

The dataset used is `tripadvisor_hotel_reviews.csv`, available on Kaggle: [Insert Kaggle dataset link here].

## Installation

1.  Clone the repository:

    ```bash
    git clone [repository URL]
    ```

2.  Install the required Python libraries:

    ```bash
    pip install numpy pandas nltk scikit-learn gensim matplotlib seaborn wordcloud xgboost
    ```

3.  Download the NLTK data:

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('punkt_tab')
    ```

## Usage

1.  Place `tripadvisor_hotel_reviews.csv` in the project directory.
2.  Run the Python script:

    ```bash
    python [your_script_name.py]
    ```

## Project Structure
Markdown

# Hotel Review Sentiment & Topic Analysis

This project analyzes hotel reviews from TripAdvisor to extract valuable insights using Natural Language Processing (NLP) and machine learning techniques. It focuses on sentiment classification, rating prediction, topic modeling, and clustering of negative reviews.

## Project Overview

This project demonstrates the power of machine learning techniques in analyzing hotel reviews. The key components include:

* **Sentiment Analysis:** Classifying reviews into positive, negative, or neutral categories using NLTK.
* **Rating Prediction Models:** Building models to predict review ratings.
* **Topic Modeling (LDA):** Identifying key themes within the reviews.
* **Clustering Negative Reviews (K-means):** Grouping negative reviews to uncover common issues.

## Key Results

* **Sentiment Analysis Accuracy:** Achieved 78% accuracy using NLTK for sentiment classification.
* **Rating Prediction Accuracy:** Reached up to 84.83% accuracy in predicting review ratings using machine learning models.
* **Identified 5 Key Themes:** Topic modeling revealed five distinct themes: General Hotel Features, Resort & Leisure Experience, Issues/Service Concerns, Guest Experiences and Amenities & Comfort.
* **Negative Review Clustering (ARI):** Clustering of negative reviews resulted in an Adjusted Rand Index (ARI) score of 0.019, indicating limited alignment with original sentiment labels.

## Technologies Used

* **Python:** Core programming language.
* **NLTK (Natural Language Toolkit):** For text preprocessing and sentiment analysis.
* **Scikit-learn (sklearn):** For machine learning models, clustering, and vectorization.
* **Gensim:** For Latent Dirichlet Allocation (LDA) topic modeling.
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Matplotlib/Seaborn:** For data visualization.
* **WordCloud:** For visualizing word frequencies.

## Dataset

The dataset used is `tripadvisor_hotel_reviews.csv`, available on Kaggle: [Insert Kaggle dataset link here].

## Installation

1.  Clone the repository:

    ```bash
    git clone [repository URL]
    ```

2.  Install the required Python libraries:

    ```bash
    pip install numpy pandas nltk scikit-learn gensim matplotlib seaborn wordcloud xgboost
    ```

3.  Download the NLTK data:

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('punkt_tab')
    ```

## Usage

1.  Place `tripadvisor_hotel_reviews.csv` in the project directory.
2.  Run the Python script:

    ```bash
    python [your_script_name.py]
    ```

## Project Structure

Hotel-Review-Analysis/
├── tripadvisor_hotel_reviews.csv
├── [your_script_name.py]
├── README.md

## Future Improvements

* Explore advanced machine learning models for improved sentiment and rating prediction.
* Fine-tune topic modeling for more coherent topics.
* Investigate alternative clustering algorithms for better negative review analysis.
* Develop an interactive dashboard to visualize results.

## Author

[Snehil Sarkar]
