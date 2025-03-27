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


#Correlation Analysis

data = pd.read_csv('All_Data.csv',sep=',', encoding='utf-8', index_col=0, parse_dates=True)

print(data.head(3).to_markdown())

# Calculate the correlation matrix for the 'Close' prices of all cryptocurrencies
correlation_matrix = df.corr()

# Create a correlation matrix heatmap plot for all 30 coins
import plotly.express as px

# Create the heatmap
fig = px.imshow(
    correlation_matrix,  # The correlation matrix
    labels=dict(x="Cryptocurrency", y="Cryptocurrency", color="Correlation"),  # Axis labels
    x=correlation_matrix.columns,  # X-axis labels (cryptocurrencies)
    y=correlation_matrix.index,    # Y-axis labels (cryptocurrencies)
    #text_auto=True,  # Display correlation values on the heatmap
    color_continuous_scale='RdBu',  # Color scale
    title="Cryptocurrency Correlation Matrix"  # Title of the plot
)
# Update layout for better readability
fig.update_layout(
    xaxis_title="Cryptocurrency",
    yaxis_title="Cryptocurrency",
    width=800,  # Width of the plot
    height=800,  # Height of the plot
)

# Show the plot
fig.show()

#Create a heatmap for all 30 coins against 4 selected coins
selected_coins = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'WIF-USD']
selected_correlation_matrix = correlation_matrix[selected_coins]

fig = px.imshow(
    selected_correlation_matrix,
    labels=dict(color="Correlation"),
    color_continuous_scale='RdBu',
    text_auto=True,
    aspect="auto",
    height = 800,
    title="Selected Cryptocurrency Correlation Matrix"
)
fig.show()
import plotly.graph_objects as go
#Create function to get top positive correlations for each chosen coin
#def get_top_positive_correlations(coin):
  # Extract the correlation values for BTC-USD
  # 
coin = 'BTC-USD'
btc_correlations = correlation_matrix[coin]

  # Sort the correlations in descending order and exclude BTC-USD itself
sorted_correlations = btc_correlations.drop(coin).sort_values(ascending=False)

  # Select the top 4 positively correlated coins
  top_4_correlated = sorted_correlations.head(4)
  # Convert the top_4_correlated Series to a DataFrame for better formatting
  top_4_table = pd.DataFrame(top_4_correlated)

  # Add a column for the cryptocurrency names
  top_4_table['Cryptocurrency'] = top_4_table.index

  # Reset the index to make the table cleaner
  top_4_table = top_4_table.reset_index(drop=True)
  top_4_table.rename(columns={coin: f'Correlation with {coin}'}, inplace=True)

  #plot the information into a presentable table
  formatted_correlations = top_4_table[f'Correlation with {coin}'].apply(lambda x: f'{x:.6f}')

  # Create the table
  fig = go.Figure(data=[go.Table(
      header=dict(
          values=['Cryptocurrency', f'Correlation with {coin}'],  # Column headers
          fill_color='paleturquoise',  # Header background color
          align='center',  # Align text to the left
          font=dict(size=14, color='black'), # Header font style
      ),
      cells=dict(
          values=[top_4_table['Cryptocurrency'], formatted_correlations],  # Cell values
          fill_color='lavender',  # Cell background color
          align='center',  # Align text to the left
          font=dict(size=12, color='black')  # Cell font style
      ))
  ])
# Update layout for better readability
  fig.update_layout(
      title=f'Top 4 Cryptocurrencies Positively Correlated with {coin}',
      title_x=0.5,  # Center the title
      margin=dict(l=80, r=80, t=80, b=20),  # Adjust margins
      width=700,
      height=300
  )

  # Show the table
  return fig.show()
get_top_positive_correlations('BTC-USD')