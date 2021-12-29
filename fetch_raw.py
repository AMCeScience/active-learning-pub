import pickle

from PmedConnect import PubmedAPI as api
import config
import Libs.qrel_parser as qrel_parser

def run_qrel_file(filename, start_batch_num = 1):
  searcher = api.PubmedAPI(config.PUBMED_EMAIL)
  parser = qrel_parser.QrelParser(filename)

  # Keep a batch number to store each batch as a separate file
  batch_num = start_batch_num

  # Loop the lines in the qrel file
  while not parser.isEnd():
    # Fetch a batch of qrel lines
    batch = parser.getBatch()

    # Fetch the Pubmed IDs for the qrel lines
    pubmed_ids = parser.getBatchPMIDs(batch)

    # Fetch the data for the Pubmed IDs
    articles = searcher.fetch(pubmed_ids)

    # Store the fetched article data in a batch file
    with open(config.STORE_LOCATION + 'pubmed_article_batch_%i.pickle' % batch_num, 'wb') as handle:
      pickle.dump(articles, handle, protocol = pickle.HIGHEST_PROTOCOL)

    batch_num += 1

  return batch_num

if __name__ == '__main__':
  # Run both the train and test qrel files to fetch
  # all (i.e. currently 50) systematic review results
  batch_num = run_qrel_file(config.TRAIN_QREL_LOCATION)
  run_qrel_file(config.TEST_QREL_LOCATION, batch_num)