import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

retail_data = pd.read_csv('retailPulse.csv')

st.title("Retail Analytics Dashboard")

#sidebar 
st.sidebar.header("Filters")
selected_store = st.sidebar.selectbox("Select Store", retail_data['City'].unique())
selected_category = st.sidebar.selectbox("Select Product Category", retail_data['Product Category'].unique())
selected_location = st.sidebar.selectbox("Select Location", retail_data['State'].unique())

filtered_data = retail_data[
    (retail_data['City'] == selected_store) &
    (retail_data['Product Category'] == selected_category) &
    (retail_data['State'] == selected_location)]


#Get the profit and profit margins
retail_data['Profit'] = retail_data['Order Total'] - (retail_data['Cost Price'] * retail_data['Order Quantity'])
retail_data['Profit Margin'] = retail_data.apply(
    lambda row: (row['Profit'] / row['Order Total']) * 100 if row['Order Total'] > 0 else 0, axis=1)

#sales trends
st.header("Sales Trends")
sales_trends = filtered_data.groupby('Order Date')['Order Total'].sum().reset_index()
fig_sales_trends = px.line(sales_trends, x='Order Date', y='Order Total', title='Sales Over Time')
st.plotly_chart(fig_sales_trends)

#Sydney vs Melbourne
st.header("Demographic Distribution of Customers")
city_distribution = retail_data['City'].value_counts().reset_index()
city_distribution.columns = ['City', 'Customer Count']
fig_city_dist = px.bar(city_distribution, x='City', y='Customer Count', title='Customer Distribution by City')
st.plotly_chart(fig_city_dist)

#Customer Segmentation
scaler = StandardScaler()
clustering_data = retail_data.groupby('Customer Name').agg({
    'Order Total': 'sum',
    'Order Quantity': 'count'}).reset_index()
clustering_data.columns = ['Customer Name', 'Total Spend', 'Frequency']
scaled_data = scaler.fit_transform(clustering_data[['Total Spend', 'Frequency']])

kmeans = KMeans(n_clusters=4, random_state=4400, n_init=10)
clustering_data['Cluster'] = kmeans.fit_predict(scaled_data)

#look at spending patterns for each segment
st.header("Spending Patterns by Customer Segment")
cluster_summary = clustering_data.groupby('Cluster').agg({
    'Total Spend': 'mean',
    'Frequency': 'mean',
    'Customer Name': 'count'}).reset_index()
cluster_summary.rename(columns={'Customer Name': 'Customer Count'}, inplace=True)

st.subheader("Customer Segment Summary")
st.write(cluster_summary)

#cluster spending patterns
fig_clusters = px.bar(
    cluster_summary.melt(id_vars='Cluster', value_vars=['Total Spend', 'Customer Count', 'Frequency']),
    x='Cluster',
    y='value',
    color='variable',
    barmode='group',
    title='Cluster Metrics: Spending, Customer Count, and Frequency',
    labels={'value': 'Value', 'variable': 'Metric'})

st.plotly_chart(fig_clusters)


#store performance
store_performance = retail_data.groupby('City').agg({
    'Order Total': 'sum',
    'Profit': 'sum'}).reset_index()

fig_scatter = px.scatter(
    store_performance,
    x='Order Total',
    y='Profit',
    size='Profit',
    color='Profit',
    title='Store Performance: Sales vs. Profit',
    labels={'Order Total': 'Total Sales ($)', 'Profit': 'Total Profit ($)'},
    hover_name='City')

st.plotly_chart(fig_scatter)


# Product Performance During Holidays
product_holiday_sales = retail_data.groupby(['Product Category', 'Is Holiday Period'])['Order Total'].sum().reset_index()
fig_product_holiday = px.bar(
    product_holiday_sales,
    x='Product Category',
    y='Order Total',
    color='Is Holiday Period',
    title='Product Category Performance: Holiday vs Non-Holiday Periods',
    labels={'Order Total': 'Total Sales ($)', 'Product Category': 'Product Category'},
    barmode='stack',
    text='Order Total')

st.plotly_chart(fig_product_holiday)

# Correlation Between Holidays and Metrics
holiday_correlation_data = retail_data[['Is Holiday Period', 'Order Total', 'Order Quantity', 'Profit Margin']]
holiday_correlation_data.loc[:, 'Is Holiday Period'] = holiday_correlation_data['Is Holiday Period'].astype(int)
correlation_matrix = holiday_correlation_data.corr()
fig_correlation = px.imshow(
    correlation_matrix,
    text_auto=True,
    color_continuous_scale='viridis',
    title='Correlation Between Holidays and Sales Metrics')

st.plotly_chart(fig_correlation)
