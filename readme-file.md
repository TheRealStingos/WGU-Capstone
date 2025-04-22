# Movie Recommendation System

A content-based movie recommendation system that suggests films based on similarities in genre, directors, and actors. This project uses natural language processing techniques to analyze movie features and find meaningful patterns.

## Overview

This application analyzes the IMDb Top 1000 Movies dataset to provide personalized movie recommendations. It uses Term Frequency-Inverse Document Frequency (TF-IDF) vectorization to convert text descriptions into numerical representations and cosine similarity to measure relationships between movies.

### Features

- Content-based movie recommendations based on genre, director, and actor similarity
- Interactive command-line interface for getting personalized suggestions
- Data visualizations to explore rating distributions and movie relationships
- Clustering visualization to understand how movies group by content features

## Installation

### Prerequisites

- Python 3.8 or higher
- Pip package manager

### Setup Instructions

1. Install Python 3.6 or higher
2. Install the required libraries using the command: pip install pandas numpy scikit-learn matplotlib seaborn
3. Download the movie_recommendation_system,ipynb and imdb_top_1000.csv files to the same directory
4. Open Jupyter Notebook using command prompt utilizing the command: jupyter notebook
5. Navigate to the directory with the previously downloaded files
6. Open movie_recommendation_system.ipynb
7. Run all cells sequentially by pressing Shift+Enter
8. At the interactive prompt, enter a movie from the IMDb Top 1000 list
9. View the 5 recommended similar movies
10. Enter ‘exit’ to quit the program

### Example

```
Welcome to the Movie Recommendation System!
This system uses machine learning to recommend movies based on your preferences.
The dataset contains the IMDb Top 1000 movies.

Enter a movie title (or 'exit' to quit): Inception

Because you liked 'Inception', you might also enjoy:
- Interstellar (2014) - Adventure, Drama, Sci-Fi - Rating: 8.6
- The Prestige (2006) - Drama, Mystery, Sci-Fi - Rating: 8.5
- Shutter Island (2010) - Mystery, Thriller - Rating: 8.2
- The Dark Knight (2008) - Action, Crime, Drama - Rating: 9.0
- Memento (2000) - Mystery, Thriller - Rating: 8.4
```

## Dependencies

- pandas - Data manipulation
- numpy - Numerical operations
- scikit-learn - Machine learning algorithms
- matplotlib - Data visualization
- seaborn - Enhanced visualizations
