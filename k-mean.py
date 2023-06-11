from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv('student.csv')
df.head()
from matplotlib import pyplot as plt
plt.scatter(df.cgpa,df['iq'])
plt.xlabel('cgpa')
plt.ylabel('iq')
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['cgpa','iq']])
y_predicted
df['cluster'] = y_predicted
df.head()
km.cluster_centers_
import matplotlib.pyplot as plt

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.cgpa, df1['iq'], color='green')
plt.scatter(df2.cgpa, df2['iq'], color='red')
plt.scatter(df3.cgpa, df3['iq'], color='yellow')

plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
plt.legend()

plt.show()