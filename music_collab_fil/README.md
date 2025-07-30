# Music Recommendation System (Collaborative Filtering)

A sophisticated music recommendation system that uses collaborative filtering algorithms to suggest artists to users based on their listening history and the preferences of similar users.

## Features

- **Collaborative Filtering**: Recommends music artists based on user listening patterns
- **Implicit Feedback Processing**: Works with listening counts rather than explicit ratings
- **Alternating Least Squares (ALS)**: Advanced matrix factorization algorithm for recommendations
- **Sparse Matrix Optimization**: Efficiently handles large, sparse datasets
- Handles 1,892 users and 17,632 artists efficiently
- Uses sparse matrices to minimize memory usage

## Dataset

This system uses the **Last.fm dataset** containing:
- **1,892 users** with listening history
- **17,632 artists** in the database
- **92,834 user-artist listening relationships**
- **Social network data** (user friendships)
- **Tagging data** (user generated tags for artists)

### Dataset Files
- `user_artists.dat`: User artist listening relationships with weights
- `artists.dat`: Artist information (ID, name, URL)
- `user_friends.dat`: Social network connections between users
- `tags.dat`: Available tags for categorizing artists
- `user_taggedartists.dat`: User generated tags for artists
- `user_taggedartists-timestamps.dat`: Timestamps for tag assignments

### Installation
```bash
# Clone the repository
git clone https://github.com/shifs999/Recommender-Systems.git
cd music_collab_fil

# Install dependencies
pip install -r requirements.txt
```

### Alternative: Poetry Installation
```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

## Dependencies
- **pandas**
- **numpy**
- **scipy**
- **implicit** 

## Usage

### Basic Usage

1. **Run the Recommendation System**
   ```bash
   python -m music_collab_fil.recommender
   ```

2. **Test Data Loading**
   ```bash
   python -m music_collab_fil.data
   ```

### Expected Output

#### Example Output
```
User-item matrix shape: (1892, 17632)
User index for user ID 2: 0
Ondubground: 1.2793043851852417
Fady Maalouf: 1.2022783756256104
Jay Park: 1.1696767807006836
Deaf Center: 1.1675994396209717
Rachel Stevens: 1.1532193422317505
```

#### Data Module Output
```
User-artists matrix shape: (1892, 17632)
Number of users: 1892
Number of artists: 17632
Artist ID 1: MALICE MIZER
```
### What Does the Output Mean?

When you run the recommender system, you will see output like this:

```
User-item matrix shape: (1892, 17632)
User index for user ID 2: 0
Denise Rosenthal: 1.546019196510315
Queen: 1.428629875183105
Sam the Kid: 1.399303913116455
Revis: 1.370678186416626
Deaf Center: 1.282248616218567
```

**What does each line mean?**

- **User item matrix shape: (1892, 17632)**
  - This tells you the size of the data being used. There are 1,892 users and 17,632 artists in the system.
  - Imagine a big table where each row is a user and each column is an artist.

- **User index for user ID 2: 0**
  - The system uses its own way to keep track of users, starting from 0. Here, user ID 2 is at position 0 in the system's list.

- **Artist Name: Score** (for example, `Denise Rosenthal: 1.54`)
  - These are the recommended artists for the user.
  - The name before the colon is the artist the system thinks the user will like.
  - The number after the colon is the recommendation score. Higher scores mean the system is more confident the user will like that artist.

## Understanding the Output

### Matrix Shape
- `(1892, 17632)`: 1,892 users × 17,632 artists matrix
- Each cell contains the listening count (weight) for user artist pairs

### Recommendation Scores
- **Higher scores** indicate stronger recommendations
- Scores are based on the ALS algorithm's learned user and artist latent factors
- Range typically: 0.5 - 2.0 

### User Index Mapping
- User IDs from the dataset are mapped to zero-based indices
- Example: User ID 2 --> Index 0

## How It Works

### 1. Data Processing (`data.py`)
```python
# Load user-artist listening data
user_artists, user_id_map, artist_id_map = load_user_artists(data_file)

# Create sparse matrix for efficient computation
# Shape: (users, artists) with listening counts as values
```

### 2. Artist Name Resolution (`data.py`)
```python
# Convert artist IDs to readable names
artist_retriever = ArtistRetriever()
artist_retriever.load_artists(artists_file)
artist_name = artist_retriever.get_artist_name_from_id(artist_id)
```

### 3. Recommendation Engine (`recommender.py`)
```python
# Train ALS model
model = implicit.als.AlternatingLeastSquares(
    factors=50,      # Latent factors
    iterations=10,   # Training iterations
    regularization=0.01  # Regularization parameter
)

# Generate recommendations
recommendations = recommender.recommend(user_index, matrix, n=5)
```

## Contributions 

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

## Acknowledgments

- **Last.fm** for providing the dataset
- **HetRec 2011** workshop for dataset curation

## Contact

For any queries or collaborations, feel free to reach me out at **saizen777999@gmail.com**
