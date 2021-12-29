import pandas
import Libs.qrel_parser as qrel_parser
import Database.db as db, config

if __name__ == "__main__":
  train_parser = qrel_parser.QrelParser(config.TRAIN_QREL_LOCATION)
  test_parser = qrel_parser.QrelParser(config.TEST_QREL_LOCATION)

  train_clef_data = train_parser.qrel_data
  test_clef_data = test_parser.qrel_data

  # Merge the qrel data into one dataframe
  frames = [train_clef_data, test_clef_data]
  clef_data = pandas.concat(frames)

  # Fetch all the data from the local database
  pubmed_data = db.Article.select()

  # Loop over the qrel lines
  for idx in range(0, len(clef_data)):
    # Select the items from both data sources
    clef_item = clef_data.iloc[idx]
    pubmed_item = pubmed_data[idx]

    # Compare the three columns from both data sources
    # Ouput errors when they are encountered
    if str(clef_item['review_id']) != str(pubmed_item.review_id):
      print('review_id error, id: %i, clef: %s, db: %s.' % (idx, clef_item['review_id'], pubmed_item.review_id))

    if str(clef_item['pmid']) != str(pubmed_item.pubmed_id):
      print('pubmed_id error, id: %i, clef: %i, db: %s.' % (idx, clef_item['pmid'], pubmed_item.pubmed_id))

    if bool(clef_item['included']) != bool(pubmed_item.included):
      print('included error, id: %i' % (idx))
      print(clef_item)
      print(pubmed_item.id)