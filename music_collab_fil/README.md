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
