# Music Collaborative Filtering Recommendation System

A sophisticated music recommendation system that uses collaborative filtering algorithms to suggest artists to users based on their listening history and the preferences of similar users. Built with the Last.fm dataset and powered by the `implicit` library.

## 🎵 Features

### Core Functionality
- **Collaborative Filtering**: Recommends music artists based on user listening patterns
- **Implicit Feedback Processing**: Works with listening counts rather than explicit ratings
- **Alternating Least Squares (ALS)**: Advanced matrix factorization algorithm for recommendations
- **Artist Name Resolution**: Converts artist IDs to human-readable names
- **Sparse Matrix Optimization**: Efficiently handles large, sparse datasets

### Technical Features
- **Scalable Architecture**: Handles 1,892 users and 17,632 artists efficiently
- **Memory Optimized**: Uses sparse matrices to minimize memory usage
- **Type Safety**: Full type hints and error handling
- **Modular Design**: Separated data processing and recommendation logic

## 📊 Dataset

This system uses the **Last.fm dataset** containing:
- **1,892 users** with listening history
- **17,632 artists** in the database
- **92,834 user-artist listening relationships**
- **Social network data** (user friendships)
- **Tagging data** (user-generated tags for artists)

### Dataset Files
- `user_artists.dat`: User-artist listening relationships with weights
- `artists.dat`: Artist information (ID, name, URL, picture)
- `user_friends.dat`: Social network connections between users
- `tags.dat`: Available tags for categorizing artists
- `user_taggedartists.dat`: User-generated tags for artists
- `user_taggedartists-timestamps.dat`: Timestamps for tag assignments

## 🚀 Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Quick Installation
```bash
# Clone the repository
git clone <your-repo-url>
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

## 📦 Dependencies

### Core Dependencies
- **pandas (≥2.1.0)**: Data manipulation and analysis
- **numpy (≥1.24.0)**: Numerical computing
- **scipy (≥1.11.0)**: Scientific computing and sparse matrices
- **implicit (≥0.7.0)**: Implicit feedback recommendation algorithms

### Development Dependencies
- **black (≥23.0.0)**: Code formatting

## 🎯 Usage

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

#### Recommender Output
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

- **User-item matrix shape: (1892, 17632)**
  - This tells you the size of the data being used. There are 1,892 users and 17,632 artists in the system.
  - Imagine a big table where each row is a user and each column is an artist.

- **User index for user ID 2: 0**
  - The system uses its own way to keep track of users, starting from 0. Here, user ID 2 is at position 0 in the system's list.

- **Artist Name: Score** (for example, `Denise Rosenthal: 1.54`)
  - These are the recommended artists for the user.
  - The name before the colon is the artist the system thinks the user will like.
  - The number after the colon is the recommendation score. Higher scores mean the system is more confident the user will like that artist.

**In simple words:**
- The system looks at what music a user has listened to and finds other artists they might enjoy.
- It shows a list of artists, sorted by how much it thinks the user will like them.
- The higher the score, the stronger the recommendation.

*This makes it easy for anyone to discover new music based on their listening habits!*

## 📈 Understanding the Output

### Matrix Shape
- `(1892, 17632)`: 1,892 users × 17,632 artists matrix
- Each cell contains the listening count (weight) for user-artist pairs

### Recommendation Scores
- **Higher scores** indicate stronger recommendations
- Scores are based on the ALS algorithm's learned user and artist latent factors
- Range typically: 0.5 - 2.0 (higher = better recommendation)

### User Index Mapping
- User IDs from the dataset are mapped to zero-based indices
- Example: User ID 2 → Index 0

## 🔧 How It Works

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

## 🧠 Algorithm Details

### Alternating Least Squares (ALS)
- **Purpose**: Matrix factorization for implicit feedback
- **Latent Factors**: 50 (configurable)
- **Training**: 10 iterations (configurable)
- **Regularization**: 0.01 (prevents overfitting)

### Collaborative Filtering Process
1. **User-Artist Matrix**: Sparse matrix of listening counts
2. **Factorization**: Decompose into user and artist latent factors
3. **Prediction**: Use learned factors to predict missing values
4. **Recommendation**: Rank artists by predicted scores

## 🛠️ Customization

### Modify Recommendation Parameters
```python
# In recommender.py, change these parameters:
model = implicit.als.AlternatingLeastSquares(
    factors=100,        # More factors = more complex model
    iterations=20,      # More iterations = better training
    regularization=0.1  # Higher = more regularization
)
```

### Change Target User
```python
# In recommender.py, modify:
user_id = 123  # Change to any user ID in the dataset
```

### Adjust Number of Recommendations
```python
# In recommender.py, modify:
artists, scores = recommender.recommend(user_index, user_artists, n=10)
```

## 📁 Project Structure

```
music_collab_fil/
├── music_collab_fil/
│   ├── data.py              # Data loading and processing
│   └── recommender.py       # Recommendation engine
├── dataset/
│   ├── user_artists.dat     # User-artist listening data
│   ├── artists.dat          # Artist information
│   ├── user_friends.dat     # Social network data
│   ├── tags.dat             # Available tags
│   ├── user_taggedartists.dat      # User tags for artists
│   ├── user_taggedartists-timestamps.dat  # Tag timestamps
│   └── readme.txt           # Dataset documentation
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Poetry configuration
└── README.md               # This file
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd music_collab_fil
   python -m music_collab_fil.recommender
   ```

2. **Memory Issues**
   - The system uses sparse matrices to minimize memory usage
   - If you encounter memory problems, reduce the number of factors

3. **File Not Found Errors**
   - Ensure the dataset files are in the `dataset/` directory
   - Check file permissions

4. **OpenBLAS Warning**
   - This is a performance warning, not an error
   - The system will still work correctly

### Performance Optimization
```bash
# Set environment variable to reduce OpenBLAS threads
export OPENBLAS_NUM_THREADS=1
python -m music_collab_fil.recommender
```

## 📊 Performance Metrics

### Dataset Statistics
- **Users**: 1,892
- **Artists**: 17,632
- **Listening Relationships**: 92,834
- **Average Artists per User**: 49.067
- **Average Users per Artist**: 5.265

### Computational Performance
- **Training Time**: ~1-2 seconds (10 iterations)
- **Memory Usage**: ~50-100MB (sparse matrix)
- **Recommendation Speed**: <1 second per user

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. The Last.fm dataset is provided for non-commercial use.

## 🙏 Acknowledgments

- **Last.fm** for providing the dataset
- **Implicit** library developers for the recommendation algorithms
- **HetRec 2011** workshop for dataset curation

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

---

**Happy Music Discovery! 🎵** 