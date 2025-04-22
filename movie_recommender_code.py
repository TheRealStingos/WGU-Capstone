# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('imdb_top_1000.csv')

# Display the first few rows to understand the data
print("Dataset Overview:")
df.head()

# Check for missing values
print("\nMissing Values:")
df.isnull().sum()

# Convert runtime to numeric (minutes)
df['Runtime'] = df['Runtime'].str.replace(' min', '').astype(int)

# Create a combined feature for content-based filtering
def create_soup(x):
    return (x['Genre'] + ' ' + x['Director'] + ' ' +
            x['Star1'] + ' ' + x['Star2'] + ' ' +
            x['Star3'] + ' ' + x['Star4'])
df['soup'] = df.apply(create_soup, axis=1)

# Visualization 1: Histogram of IMDB Ratings
plt.figure(figsize=(10, 6))
plt.hist(df['IMDB_Rating'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of IMDb Ratings', fontsize=16)
plt.xlabel('IMDb Rating', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.grid(axis='y', alpha=0.75)
plt.show()

# Visualization 2: Scatterplot of Rating vs Votes
plt.figure(figsize=(10, 6))
plt.scatter(df['No_of_Votes'], df['IMDB_Rating'], alpha=0.6, color='teal')
plt.title('Relationship Between Number of Votes and IMDb Rating', fontsize=16)
plt.xlabel('Number of Votes', fontsize=12)
plt.ylabel('IMDb Rating', fontsize=12)
plt.xscale('log')  # Using log scale for better visualization
plt.grid(True, alpha=0.3)
plt.show()

# TRAINING PHASE: Create and train the TF-IDF model
print("\nTraining the machine learning model...")

# Initialize the TF-IDF vectorizer (our ML algorithm)
tfidf = TfidfVectorizer(stop_words='english')

# Train the model on our movie feature data
# The fit_transform method is where the actual training happens
tfidf_matrix = tfidf.fit_transform(df['soup'])
print("Model training complete!")

# Use the trained model to compute similarity between all movies
print("Computing similarity matrix using trained model...")
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reset index and create a reverse mapping
df = df.reset_index()
indices = pd.Series(df.index, index=df['Series_Title'])

# Function to recommend similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    try:
        idx = indices[title]
    except KeyError:
        print(f"Movie '{title}' not found in the dataset.")
        print("Available movies include:", ", ".join(df['Series_Title'].head(5).tolist()), "...")
        return None

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 6 most similar movies (including the movie itself)
    sim_scores = sim_scores[0:6]

    # Return the top 5 most similar movies (excluding the movie itself)
    movie_indices = [i[0] for i in sim_scores[1:6]]

    recommendations = df.iloc[movie_indices][['Series_Title', 'Genre', 'IMDB_Rating', 'Released_Year', 'Director']]
    return recommendations
    recommendations

# Example usage
# Get recommendations for "The Shawshank Redemption"
print("\nExample Recommendations:")
recommendations = get_recommendations('The Shawshank Redemption')

# Model validation
print("\nValidating recommendation quality...")


def validate_recommendations(movie_title, num_recommendations=5):
    # Get recommendations
    recommendations = get_recommendations(movie_title)
    if recommendations is None:
        print(f"No recommendations found for '{movie_title}'")
        return None

    try:
        # Get original movie details
        original_movie = df[df['Series_Title'] == movie_title].iloc[0]

        # Calculate genre overlap
        genre_overlap = []
        for _, rec in recommendations.iterrows():
            try:
                original_genres = set(original_movie['Genre'].split(', '))
                rec_genres = set(rec['Genre'].split(', '))
                overlap = len(original_genres.intersection(rec_genres)) / len(original_genres)
                genre_overlap.append(overlap)
            except Exception as e:
                print(f"Error in genre calculation: {e}")
                genre_overlap.append(0)

        # Calculate era similarity (same decade)
        era_similarity = []
        for _, rec in recommendations.iterrows():
            try:
                # Make sure we're working with integers
                orig_year = int(original_movie['Released_Year'])
                rec_year = int(rec['Released_Year'])
                same_decade = (orig_year // 10) == (rec_year // 10)
                era_similarity.append(1 if same_decade else 0)
            except Exception as e:
                print(f"Error in era calculation: {e}")
                era_similarity.append(0)

        # Calculate director match
        director_match = []
        for _, rec in recommendations.iterrows():
            try:
                # The issue might be here if 'Director' exists but there's
                # something wrong with the comparison
                same_director = original_movie['Director'] == rec['Director']
                director_match.append(1 if same_director else 0)
            except Exception as e:
                print(f"Error in director calculation: {e}")
                director_match.append(0)

        results = {
            'avg_genre_overlap': sum(genre_overlap) / len(genre_overlap) if genre_overlap else 0,
            'same_decade_rate': sum(era_similarity) / len(era_similarity) if era_similarity else 0,
            'director_match_rate': sum(director_match) / len(director_match) if director_match else 0
        }

        return results
    except Exception as e:
        print(f"Error in validate_recommendations for '{movie_title}': {e}")
        return None


# Let's use a smaller sample for testing
print("Selecting sample movies for validation...")
sample_movies = df['Series_Title'].sample(5).tolist()
validation_results = []

for movie in sample_movies:
    print(f"Validating: {movie}")
    result = validate_recommendations(movie)
    if result is not None:
        validation_results.append(result)
        print(f"  Results: {result}")

if validation_results:
    # Calculate average metrics
    avg_genre_overlap = sum(r['avg_genre_overlap'] for r in validation_results) / len(validation_results)
    avg_decade_match = sum(r['same_decade_rate'] for r in validation_results) / len(validation_results)
    avg_director_match = sum(r['director_match_rate'] for r in validation_results) / len(validation_results)

    print(f"\nValidation Results for TF-IDF/Cosine Similarity Method:")
    print(f"- Genre match rate: {avg_genre_overlap * 100:.1f}% of recommended movies shared at least one genre")
    print(f"- Era similarity: {avg_decade_match * 100:.1f}% of recommendations came from the same decade")
    print(f"- Director match rate: {avg_director_match * 100:.1f}% of recommendations featured the same director")
else:
    print("No validation results were collected. All validations failed.")

# Interactive recommendation function
def interactive_recommendation():
    while True:
        movie_title = input("\nEnter a movie title (or 'exit' to quit): ")
        if movie_title.lower() == 'exit':
            break

        recommendations = get_recommendations(movie_title)
        if recommendations is not None:
            print(f"\nBecause you liked '{movie_title}', you might also enjoy:")
            for i, row in recommendations.iterrows():
                print(f"- {row['Series_Title']} ({row['Released_Year']}) - {row['Genre']} - Rating: {row['IMDB_Rating']}")

# Run the interactive recommendation system
print("\nWelcome to the Movie Recommendation System!")
print("This system uses machine learning to recommend movies based on your preferences.")
print("The dataset contains the IMDb Top 1000 movies.")
interactive_recommendation()