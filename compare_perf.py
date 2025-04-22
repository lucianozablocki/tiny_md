import pandas as pd

def compare_perf(baseline, improved):
  df1 = pd.read_csv(baseline)
  df2 = pd.read_csv(improved)

  # max particulas/s for each (N, opt, compiler)
  max_df1 = df1.groupby(['N', 'opt', 'compiler'])['particulas/s'].max().reset_index()
  max_df2 = df2.groupby(['N', 'opt', 'compiler'])['particulas/s'].max().reset_index()

  # Merge the two DataFrames to align the same (N, opt, compiler) pairs
  merged = pd.merge(
      max_df1, 
      max_df2, 
      on=['N', 'opt', 'compiler'], 
      suffixes=('_1', '_2')
  )

  # Compute the ratio (N2 / N1)
  merged['ratio'] = merged['particulas/s_2'] / merged['particulas/s_1']

  # Print the results
  merged[['N', 'opt', 'compiler', 'particulas/s_1', 'particulas/s_2', 'ratio']]
  return merged

compare_df = compare_perf(
    'https://raw.githubusercontent.com/lucianozablocki/tiny_md/refs/heads/main/results/atom.csv',
    'https://raw.githubusercontent.com/lucianozablocki/tiny_md/refs/heads/main/results/atom-native.csv'
)

compare_df[(compare_df['opt']=='-Ofast') & (compare_df['compiler']=='gcc')]