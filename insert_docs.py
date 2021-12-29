import pickle

from PmedConnect import PubmedAPI as api
import config
import Libs.qrel_parser as qrel_parser
import Database.db_inserts as db

def run_qrel_file(filename, start_batch_num = 1):
  connector = db.Connector()
  parser = qrel_parser.QrelParser(filename)

  # Keep a batch number to read each separate batch pickle file
  batch_num = start_batch_num

  # Loop the lines in the qrel file
  while not parser.isEnd():
    # Fetch a batch of qrel lines
    batch = parser.getBatch()

    with open(config.STORE_LOCATION + 'pubmed_article_batch_%i.pickle' % batch_num, 'rb') as handle:
      articles = pickle.load(handle)

      # Insert articles into the local database
      connector.insert_fetched_articles(articles, batch)

    batch_num += 1

  return batch_num

if __name__ == '__main__':
  # Run both the train and test qrel files to fetch
  # all (i.e. currently 50) systematic review results
  batch_num = run_qrel_file(config.TRAIN_QREL_LOCATION)
  run_qrel_file(config.TEST_QREL_LOCATION, batch_num)