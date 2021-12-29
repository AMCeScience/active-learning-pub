import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as multi
import Libs.result_retriever as rr, config

font = {'size': 16}

matplotlib.rc('font', **font)

class Analysis:

  def __init__(self):
    retriever_obj = rr.ResultRetriever()

    # Get the results stratified to training set size, with the single reviews removed
    self.stratified_burden_results = retriever_obj.get_all_performance(True)


  def plot_stratified_performance(self):
    ticks = list(range(1, len(config.SIMILARITY_STEPS) + 2))
    labels = config.SIMILARITY_STEPS + ['37']

    # Fit polynomial to the medians
    medians = [np.median(i) for i in self.stratified_burden_results]

    poly_x = np.array(ticks)
    fit, res, _, _, _ = np.polyfit(poly_x, medians, 2, full = True)
    poly = np.poly1d(fit)

    print('The residual sum of squares on the poly fit is: ', res[0])

    y = np.exp(fit[1]) * np.exp(fit[0] * np.array(ticks))

    plt.figure()
    plt.boxplot(self.stratified_burden_results)
    plt.xticks(ticks, labels)

    plt.plot(poly_x, poly(poly_x))

    plt.ylabel('Inverse Burden at 95% Yield')
    plt.xlabel('Initial pool size')

    plt.tight_layout()
    plt.savefig(config.PLOT_LOCATION + 'stratified_burden_outcome.pdf')


  def create_burden_table(self):
    min_values = [np.min(i) for i in self.stratified_burden_results]
    medians = [np.median(i) for i in self.stratified_burden_results]
    max_values = [np.max(i) for i in self.stratified_burden_results]

    labels = config.SIMILARITY_STEPS + ['37']

    baseline_median = medians[-1]
    median_comparison = medians - baseline_median

    df = pd.DataFrame({'# reviews': labels, 'Minimum': min_values, 'Median': medians, 'Maximum': max_values, 'Difference median\n versus baseline': median_comparison})

    df.to_excel('Plots/inverse_burden_table.xlsx')


  # def statistical_tests(self):
  #   burden = self.stratified_burden_results

  #   arr = []
  #   cols = config.SIMILARITY_STEPS + [37]

  #   # Loop over every training set size
  #   for i in range(0, len(burden)):
  #     # Loop over every training set size
  #     for j in range(0, len(burden)):
  #       if i == j:
  #         arr.append(1)

  #         continue

  #       # Calculate Wilcoxon rank sum test on every training set
  #       # size against every other training set size
  #       p_val = stats.wilcoxon(burden[i], burden[j]).pvalue

  #       # Create one big flat list
  #       arr.append(p_val)

  #   # Kruskal-Wallis check
  #   print(stats.kruskal(*burden))

  #   # Adjust p-values using Holm-Sidak method
  #   adjusted = multi.multipletests(arr, method = 'bonferroni')
  #   # Create list of lists with adjusted p-values,
  #   # one list per initial pool size
  #   restruct_arr = list(self.chunks(adjusted[1], len(cols)))

  #   df = pd.DataFrame(restruct_arr, columns = cols, index = cols)

  #   df.to_excel('Plots/significance.xlsx', engine = 'openpyxl')

    # df.to_csv('Plots/significance.csv')


  # def chunks(self, l, n):
  #   for i in range(0, len(l), n):
  #     yield l[i:i + n]



if __name__ == '__main__':
  obj = Analysis()

  # obj.statistical_tests()
  obj.create_burden_table()
  obj.plot_stratified_performance()
