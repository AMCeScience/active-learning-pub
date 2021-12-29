import Database.db_queries as db
import Libs.preprocess as preprocess
import Libs.feature_extraction as feature_extraction

# Fetch all articles from the local database
documents = db.get_articles()

# Run the articles through the preprocessing steps
document_collection = preprocess.clean(documents)

# Build the feature matrix
matrix_collection = feature_extraction.vectorize(document_collection)

# Determine cosine similarity for all items in the feature matrix
feature_extraction.get_cosine_similarity(matrix_collection)
