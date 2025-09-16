# 🎬 Movie Recommendation System

A sophisticated content-based movie recommendation system that leverages machine learning to suggest films based on genre, director, and cast similarities. Built with Python and scikit-learn, this system analyzes the IMDb Top 1000 movies to provide personalized recommendations.

## 🌟 Features

- **Content-Based Filtering**: Uses TF-IDF vectorization and cosine similarity for intelligent recommendations
- **Interactive CLI**: User-friendly command-line interface for real-time movie suggestions
- **Data Visualizations**: Comprehensive charts showing rating distributions and movie relationships
- **Model Validation**: Built-in metrics to evaluate recommendation quality
- **Genre Analysis**: Recommendations based on genre overlap and thematic similarity
- **Director & Cast Matching**: Considers director and actor preferences in suggestions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the IMDb dataset:
   - Download `imdb_top_1000.csv` and place it in the project directory
   - Dataset should contain columns: Series_Title, Genre, Director, Star1-4, IMDB_Rating, etc.

4. Run the system:
```bash
python movie_recommender_code.py
```

## 📊 How It Works

### Machine Learning Approach

1. **Feature Engineering**: Combines genre, director, and cast information into a unified text feature
2. **TF-IDF Vectorization**: Converts text features into numerical representations
3. **Similarity Calculation**: Uses cosine similarity to measure movie relationships
4. **Recommendation Generation**: Returns top 5 most similar movies based on content features

### Algorithm Details

```python
# Feature combination
soup = Genre + Director + Star1 + Star2 + Star3 + Star4

# TF-IDF transformation
tfidf_matrix = TfidfVectorizer(stop_words='english').fit_transform(soup)

# Cosine similarity calculation
similarity_matrix = cosine_similarity(tfidf_matrix)
```

## 🎯 Usage Examples

### Basic Usage

```python
# Get recommendations for a specific movie
recommendations = get_recommendations('The Shawshank Redemption')
print(recommendations)
```

**Output:**
```
Because you liked 'The Shawshank Redemption', you might also enjoy:
- The Green Mile (1999) - Crime, Drama, Fantasy - Rating: 8.6
- 12 Angry Men (1957) - Crime, Drama - Rating: 9.0
- One Flew Over the Cuckoo's Nest (1975) - Drama - Rating: 8.7
- Forrest Gump (1994) - Drama, Romance - Rating: 8.8
- Goodfellas (1990) - Biography, Crime, Drama - Rating: 8.7
```

### Interactive Mode

```bash
Enter a movie title (or 'exit' to quit): Inception

Because you liked 'Inception', you might also enjoy:
- Interstellar (2014) - Adventure, Drama, Sci-Fi - Rating: 8.6
- The Prestige (2006) - Drama, Mystery, Sci-Fi - Rating: 8.5
- Shutter Island (2010) - Mystery, Thriller - Rating: 8.2
- The Dark Knight (2008) - Action, Crime, Drama - Rating: 9.0
- Memento (2000) - Mystery, Thriller - Rating: 8.4
```

## 📈 Visualizations

The system includes several data visualizations:

- **Rating Distribution**: Histogram of IMDb ratings across the dataset
- **Votes vs Rating**: Scatter plot showing the relationship between popularity and quality
- **Clustering Analysis**: t-SNE visualization of movie similarities (if implemented)

## 🔍 Model Validation

The system includes comprehensive validation metrics:

- **Genre Overlap Rate**: Percentage of recommendations sharing genres with the input movie
- **Era Similarity**: Rate of recommendations from the same decade
- **Director Match Rate**: Percentage of recommendations from the same director

Example validation output:
```
Validation Results for TF-IDF/Cosine Similarity Method:
- Genre match rate: 78.2% of recommended movies shared at least one genre
- Era similarity: 34.5% of recommendations came from the same decade
- Director match rate: 12.1% of recommendations featured the same director
```

## 📁 Project Structure

```
movie-recommendation-system/
├── movie_recommender_code.py    # Main recommendation engine
├── requirements.txt             # Python dependencies
├── README.md                   # Project documentation
├── imdb_top_1000.csv          # Dataset (download separately)
└── visualizations/            # Generated plots and charts
```

## 🛠️ Technical Stack

- **Python 3.8+**: Core programming language
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms (TF-IDF, cosine similarity)
- **numpy**: Numerical computations
- **matplotlib & seaborn**: Data visualization
- **IMDb Top 1000 Dataset**: Movie information and ratings

## 🎨 Customization

### Adjusting Recommendation Count

```python
# Modify the number of recommendations returned
sim_scores = sim_scores[0:11]  # Returns top 10 instead of 5
movie_indices = [i[0] for i in sim_scores[1:11]]
```

### Feature Weighting

```python
# Customize feature importance
def create_soup(x):
    return (x['Genre'] * 2 + ' ' +  # Give more weight to genre
            x['Director'] + ' ' +
            x['Star1'] + ' ' + x['Star2'])
```
- **Coverage**: 1000 top-rated movies from IMDb

---
