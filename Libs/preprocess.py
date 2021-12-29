import Libs.Preprocessing.stemmer as stemmer
import Libs.Preprocessing.remove_characters as string_cleaning
import Libs.Preprocessing.remove_stopwords as stopwords
import Libs.Preprocessing.token_length as token_length
import Libs.Preprocessing.tokenizer as tokenizer
import Libs.Preprocessing.create_objects as collection_creator
import Libs.file_storage as fs

def clean(documents):
  existing_collection = fs.load_documents()

  if existing_collection is not None:
    print('Using pre-existing document collection')
    return existing_collection

  # Merge titles and abstracts and change to lower case
  print('Converting to plain text.')
  documents = tokenizer.convert_to_plain_text(documents)

  # Filter HTML, dashes, special characters, numbers, and double spaces
  print('Removing characters.')
  documents = string_cleaning.remove_all(documents)

  # Tokenize the text on whitespace
  print('Tokenizing text.')
  documents = tokenizer.tokenize(documents)

  # Remove any stopwords
  print('Removing stopwords.')
  documents = stopwords.remove_all(documents)

  # Remove any short or long tokens
  print('Removing based on token length.')
  documents = token_length.remove_all(documents)

  # Stem the remaining tokens
  print('Stemming tokens')
  documents = stemmer.stem(documents)

  # Join the tokens back together into strings
  documents = [' '.join(x) for x in documents]

  # Create and store a document collection class
  document_collection = collection_creator.create_document_collection(documents)

  return document_collection