import numpy as np

import Libs.file_storage as fs
import config

class ResultRetriever(object):

  def get_all_performance(self, remove_singles):
    self.remove_singles = remove_singles

    existing_results = self.get_existing_results()

    if existing_results is not None:
      return existing_results

    self.burden = []

    # Get similarity performance
    for size in config.SIMILARITY_STEPS:
      size_burden = self.loop_reviews(True, size)

      self.burden.append(size_burden)

    # Get baseline performance
    baseline_burden = self.loop_reviews(False)

    self.burden.append(baseline_burden)

    self.store(self.burden)

    return self.burden


  def loop_reviews(self, similarity = False, similarity_size = None):
    burden_performance = []

    for review in self.get_review_names():
      # Skip the reviews that do not have a similar review in the dataset
      if self.remove_singles and review in config.SINGLE_REVIEWS:
        continue

      # Load the results for this specific review. The results are a list
      # with the size of the number of repeats we do for each review. Each
      # result item consists of: list of burden and list of yield.
      # Each list is the metric measured for a specific active learning
      # iteration.
      results = fs.load_results(review, similarity, similarity_size)

      if results is None:
        continue

      # Loop over every repeat for this review.
      for result in results:
        burden_performance.append(self.get_burden_at_95_yield(result))

    return burden_performance


  def get_burden_at_95_yield(self, result):
    # load Yield and Burden list
    burden_al = result['burden']
    yield_al = result['yield']

    # Reason for if-else:
    # As we did not record Yield and Burden for the last iteration,
    # but review 'CD009020' and  'CD009185' retrieved one relevant doc at their last active learning iteration.
    if max(yield_al) < 1:
      burden = burden_al[-1]
    else:
      # Index of the time point when the Yield first time reach required recall
      index = np.where(np.array(yield_al) >= 0.95)[0]
      burden = burden_al[index[0]]

    # Calculate One Minus Burden (OMB) at that time point
    performance = 1 - burden

    return performance


  def get_review_names(self):
    matrix_collection = fs.load_matrix_collection()

    removed_reviews = []

    if self.remove_singles:
      if config.DEBUG is True:
        removed_reviews = REMOVED_REVIEWS_DEBUG
      else:
        # Determined by their characteristics
        removed_reviews = config.REMOVED_REVIEWS

    # Full set of reviews
    review_names = matrix_collection.get_review_names()
    # Remove the reviews that were excluded based on their characteristics
    review_names = list(set(review_names) - set(removed_reviews))

    return review_names


  def get_existing_results(self):
    stratified = ''

    if self.remove_singles:
      stratified = 'stratified_'

    return fs.load_result_item('%sone_minus_burden_results' % stratified)


  def store(self, burden):
    stratified = ''

    if self.remove_singles:
      stratified = 'stratified_'

    fs.store_result_item('%sone_minus_burden_results' % stratified, burden)