import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
# List of 30 cryptocurrencies
crypto_list = [
    'BTC-USD', 'ETH-USD', 'XRP-USD', 'LTC-USD', 'BCH-USD', 'ADA-USD', 'DOT-USD',
    'BNB-USD', 'LINK-USD', 'XLM-USD', 'DOGE-USD', 'UNI-USD', 'AAVE-USD', 'ATOM-USD',
    'AVAX-USD', 'PEPE24478-USD', 'SOL-USD', 'CHR-USD', 'ALGO-USD', 'FTT-USD', 'VET-USD',
    'USDT-USD', 'TRX-USD', 'ETC-USD', 'XMR-USD', 'EOS-USD', 'THETA-USD', 'NEO-USD',
    'DASH-USD', 'WIF-USD'
]
def download_data(cryptos, duration):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(duration-1))

    #initialize empty data frame
    data=pd.DataFrame()

    data = yf.download(cryptos, start=start_date, end=end_date)
    return data

df = download_data(crypto_list, 365) 

#Convert downloaded data to csv format
df.to_csv(r'All_Data.csv',index=True, header = True)

#Clean up data by deleting irrelevant columns. Only want to look at Closing price.
df = df.drop(['High', 'Low', 'Open', 'Volume'], axis=1)
#save new data frame to csv file
df.to_csv(r'All_Data.csv')

#Drop level of data frame to change axis
df = df.droplevel('Price', axis=1) #remove comment on this line when re-running
df.to_csv(r'All_Data.csv', index=True, header=True)
df.head()

#Transpose columns and rows so that we have 30 rows and 365 columns
df_transposed = df.transpose()

#Display transposed data frame
df_transposed


#PCA Analysis

#PCA Analysis: reduce dimensionality so that dataset can be processed by k-means algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# Handle missing values (if any) using Forward fill
df_transposed.fillna(method='ffill', inplace=True)

# Scale the data because some crypto prices are much higher than others
x = StandardScaler().fit_transform(df_transposed)

# Apply PCA with 2 components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

# Create a new DataFrame with the principal components
# preserve the ticker index
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1','principal component 2'])
principalDf['coin'] = df_transposed.reset_index()['Ticker']

#Display the new data frame
principalDf 

#Print the explained variance ratio
print("Explained Variance Ratio:",pca.explained_variance_ratio_)


#K-Means Clustering

from sklearn.cluster import KMeans
#Apply KMeans clustering with 4 clusters, drop Coin column to make it more readable by algorithm
kmeans = KMeans(n_clusters=4, random_state=0)
principalDf['cluster'] = kmeans.fit_predict(principalDf.drop('coin', axis=1))

#Display the DataFrame with cluster assignments 
principalDf.sort_values(by='cluster')

#Plot the clusters
import plotly.express as px

#Convert the 'cluster' column to string to ensure discrete coloring
principalDf['cluster']=principalDf['cluster'].astype(str)

#Create an interactive scatter plot
fig = px.scatter(
    principalDf,
    x='principal component 1',
    y='principal component 2',
    color='cluster',
    hover_data=['coin'],
    title='K-Means Clustering of Cryptocurrencies (2D PCA)'
)
fig.show()