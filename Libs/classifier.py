from multiprocessing import Pool
from functools import partial
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import Libs.file_storage as fs
import numpy as np, config
import Libs.performance_calculation as perf

class ModelBuilder(object):

  def __init__(self, matrix_collection, review_names, label_list, grid_parameters, run_similarity):
    self.matrix_collection = matrix_collection
    self.review_names = review_names
    self.label_list = label_list
    self.grid_parameters = grid_parameters
    self.run_similarity = run_similarity
    self.performance_calculation = perf.PerformanceObj()

    if self.run_similarity is True:
      self.cosine_similarity = fs.load_matrix('cosine_sim_matrix')


  def get_similar_review_names(self, test_review_name, size):
    """
    Get the cosine similarities ordered and remove the review itself.
    """

    similarities = self.cosine_similarity.loc[[test_review_name]]

    similarities = similarities.drop(test_review_name, axis = 1)
    similarities = similarities.sort_values(by = test_review_name, ascending = False, axis = 1)

    similar_reviews = similarities.iloc[:, 0:size]

    return similar_reviews.columns.values.tolist()


  def split_dataset(self, test_review_name):
    """
    Split dataset into a train and test set
    """

    training_review_names = self.review_names.copy()
    training_review_names.remove(test_review_name)

    X_train, y_train = self.matrix_collection.get_training_set(training_review_names, self.label_list)
    X_test, y_test = self.matrix_collection.get_test_set(test_review_name, self.label_list)

    dataset = {
      'X_train': X_train,
      'y_train': y_train,
      'X_test': X_test,
      'y_test': y_test
    }

    return dataset


  def query_docs(self, clf, X_test):
    # Obtain (or update) the distance (including both positive values and negative values) to the hyperplane
    distance = clf.decision_function(X_test)

    # Get indices of selected documents nearest the hyperplane (implement uncertainty sampling)
    # Select top-2 most uncertain
    query_indices = np.argsort(np.absolute(distance))[0:2]

    return query_indices


  def delete_row_csr(self, mat, indices):
    """
    Remove the rows listed in the indices parameter then form the CSR sparse matrix mat.
    """

    indices = list(indices)

    mask = np.ones(mat.shape[0], dtype = bool)

    mask[indices] = False

    return mat[mask]


  def oracle_read(self, X_test, y_test, query_indices):
    """
    Use the oracle to label documents
    """

    # List of Documents that are to be reviewed by the oracle (reviewer)
    self.reviewer_list = self.reviewer_list + list(query_indices)

    # list of "Relevant" Documents that are screened by the oracle (reviewer)
    for query_index in query_indices:
      if y_test[query_index] == 1:
        self.tp_labelled = self.tp_labelled + 1

    # Remove X and y of query documents from Unlabelled pool
    X_test = self.delete_row_csr(X_test, query_indices)
    y_test = np.delete(y_test, query_indices)

    return X_test, y_test


  def find_first_relevant(self):
    """
    Iterate the test set until the first relevant article is found.
    """

    y_test = self.dataset['y_test']

    # Find the first positive sample.
    first_positive = np.where(y_test == 1)[0][0]
    first_position = first_positive + 1

    # If the first positive is found at the top of the list
    # we need to add the first negative, otherwise there is
    # only one class in the final y_training set.
    if first_positive == 0:
      # Find the first negative sample.
      first_negative = np.where(y_test[0:100] == 0)[0][0]

      first_position = first_negative + 1

    # Subset the X set to include only the first positive
    # sample, but with a minimum of two classes in the list.
    X_train = self.dataset['X_test'][0:first_position, :]
    y_train = y_test[0:first_position]

    # Update the list of documents that the reviewer has read.
    self.reviewer_list = list(range(0, first_position))

    return X_train, y_train


  def get_training_for_similar(self, review_name, similarity_size):
    """
    Build a training set using cosine similarity
    """

    if similarity_size is 0:
      X_train, y_train = self.find_first_relevant()
    else:
      similar_review_names = self.get_similar_review_names(review_name, similarity_size)
      X_train, y_train = self.matrix_collection.get_training_set(similar_review_names, self.label_list)

    return X_train, y_train


  def get_training_for_all(self):
    """
    Fetch both the training data and training labels
    """

    return self.dataset['X_train'], self.dataset['y_train']


  def get_random_sample(self, X_train, y_train):
    """
    Create a random undersampled set from the provided training set
    """

    sampler = RandomUnderSampler()
    undersampled_X_train, undersampled_y_train = sampler.fit_resample(X_train, y_train)

    return undersampled_X_train, undersampled_y_train


class ParallelProcess:

  @staticmethod
  def run_iteration(iteration, matrix_collection, review_names, label_list, grid_parameters, run_similarity, review_name, similarity_size = None):
    """
    Execute a active learning iteration for the provided test review.
    The similarity size determines how many reviews are included in
    the training set based on cosine similarity.
    """

    model = ModelBuilder(matrix_collection, review_names, label_list, grid_parameters, run_similarity)

    model.dataset = model.split_dataset(review_name)

    if model.run_similarity is True:
      print('%s, iteration %i, similarity size %i' % (review_name, iteration, similarity_size))

      X_train, y_train = model.get_training_for_similar(review_name, similarity_size)
    else:
      print('%s, iteration %i, all data' % (review_name, iteration))

      X_train, y_train = model.get_training_for_all()

    undersampled_X_train, undersampled_y_train = model.get_random_sample(X_train, y_train)

    # Number of relevant documents in the training set
    num_positive = len(np.where(undersampled_y_train == 1)[0])

    ########################
    # Bootstrap
    ########################

    # Bootstrap the active learning process with a selected training
    # set. This may be either: no training set (the first relevant sample
    # from the test set is used as training set), a cosine similarity
    # selected set, or all available data.
    # Perform gridsearch with folds. Some datasets for similarity size
    # 0 have only one positive sample. CV gridsearch is not possible
    # because we can't crossfold on  one sample.
    if similarity_size is 0 or num_positive < 3:
      svmSGD = SGDClassifier(loss = "hinge", penalty = "l2", max_iter = 1000, tol = 1e-3)

      clf = svmSGD.fit(undersampled_X_train, undersampled_y_train)
    else:
      svmSGD = SGDClassifier(loss = "hinge", penalty = "l2")

      clf = GridSearchCV(svmSGD, model.grid_parameters, cv = 3)
      clf.fit(undersampled_X_train, undersampled_y_train)

      clf = clf.best_estimator_

    ########################
    # Apply active learning
    ########################

    # Initiate variables
    model.tp_labelled = 0

    # Indices of documents that were queried and read by the reviewer
    model.reviewer_list = []

    X_test = model.dataset['X_test']
    y_test = model.dataset['y_test']

    total_documents = len(y_test)

    # Count all includes
    includes_count = y_test.sum()

    # Initialise the performance metric lists
    model.performance_calculation.new_iteration(total_documents, includes_count)

    while len(y_test) != 0:
      # Record metrics: yield and burden for each iteration
      model.performance_calculation.calculate_performance(clf, X_test, y_test, model.tp_labelled, model.reviewer_list)

      # Fetch the next documents to send to the oracle
      query_indices = model.query_docs(clf, X_test)

      # The X and Y of the queried document
      # stored as a scipy.sparse.csc.csc_matrix
      X_update = X_test[query_indices]
      y_update = y_test[query_indices]

      # Have the oracle label the queried documents
      X_test, y_test = model.oracle_read(X_test, y_test, query_indices)

      # Update the Classifier by partial fit
      clf = clf.partial_fit(X_update, y_update, classes = [0, 1])

    # Fetch the results object
    result_obj = model.performance_calculation.get_performance_object()

    return result_obj


  def run_reviews(self, run_similarity, similarity_size = None):
    print('Loading data objects')
    documents = fs.load_documents()
    matrix_collection = fs.load_matrix_collection()

    label_list = documents.get_labels()

    # Full set of reviews
    review_names = matrix_collection.get_review_names()

    alpha_range = list(10.0 ** -np.arange(-1, 7))
    tolerance_range = list(10.0 ** -np.arange(-1, 7))

    grid_parameters = {'alpha': alpha_range, 'tol': tolerance_range}

    for review_name in review_names:
      if fs.load_results(review_name, run_similarity, similarity_size) is not None:
        continue

      print('Running %s, %i out of %i' % (review_name, review_names.index(review_name) + 1, len(review_names)))

      # Set up the partial functions ran through pool.map
      parallel_func = partial(self.run_iteration, matrix_collection = matrix_collection, review_names = review_names, label_list = label_list, grid_parameters = grid_parameters, run_similarity = run_similarity, review_name = review_name, similarity_size = similarity_size)

      with Pool(processes = config.POOL_PROCESSES) as pool:
        if config.DEBUG is True:
          results = pool.map(parallel_func, range(0, 1))
        else:
          results = pool.map(parallel_func, range(0, config.NUM_ITERATIONS))

        for result in results:
          fs.store_results(review_name, result, run_similarity, similarity_size)


  def run(self, run_similarity = False):
    if run_similarity is True:
      print('Similarity')
      for i in range(0, len(config.SIMILARITY_STEPS)):
        similarity_size = config.SIMILARITY_STEPS[i]

        self.run_reviews(run_similarity, similarity_size)
    else:
      print('Baseline')
      self.run_reviews(run_similarity)
