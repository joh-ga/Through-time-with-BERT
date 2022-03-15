import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# choose csv with clusterdistances between centroids for creating a heatmap
path_to_csv= "C:/Users/Laura/Downloads/documents-export-2022-01-12/3cAdverbsdist.xlsx"
df=pd.read_excel(path_to_csv ,index_col=0)

#plot heatmap
ax = sb.heatmap(df, cmap="YlGnBu", linewidths=.5, annot=True, fmt='.2g', vmin=0, vmax=12)
plt.show()