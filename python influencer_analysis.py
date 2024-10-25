import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import Flask
import os

# Load the dataset
df = pd.read_csv('top_insta_influencers_data.csv')
print(df.head())

# 1. Initial Data Inspection and Cleanup

# Check for missing values
print(df.info())
print("Missing values:\n", df.isnull().sum())

# Fill missing country values with 'Unknown'
df['country'] = df['country'].fillna('Unknown')

# Convert columns with 'k', 'm', 'b' notations to numeric values
def convert_to_number(x):
    try:
        if 'k' in str(x):
            return float(x.replace('k', '')) * 1e3
        elif 'm' in str(x):
            return float(x.replace('m', '')) * 1e6
        elif 'b' in str(x):
            return float(x.replace('b', '')) * 1e9
        return float(x)
    except ValueError:
        return np.nan

# Apply conversion function to relevant columns
for col in ['posts', 'followers', 'avg_likes', 'new_post_avg_like', 'total_likes']:
    df[col] = df[col].apply(convert_to_number)

# Convert engagement rate from percentage to float
def convert_percentage(x):
    try:
        if isinstance(x, str) and x.endswith('%'):
            return float(x.replace('%', '')) / 100  # Convert to decimal
        return float(x)
    except ValueError:
        return np.nan

df['60_day_eng_rate'] = df['60_day_eng_rate'].apply(convert_percentage)

# Filter out any rows where 'followers' or 'posts' are zero (if any)
df = df[(df['followers'] > 0) & (df['posts'] > 0)]

# 2. Descriptive Statistics and Correlation Analysis

# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number]).dropna()

# Descriptive statistics
print(numeric_df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Influencer Metrics')
plt.show()

# 3. Visualizations for Deeper Analysis

# Distribution of Followers
plt.figure(figsize=(10,6))
sns.histplot(df['followers'], bins=20, kde=True)
plt.title('Distribution of Followers')
plt.xlabel('Number of Followers')
plt.show()

# Average Likes by Country (Top 10)
top_countries = df.groupby('country')['avg_likes'].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(12,6))
sns.barplot(x=top_countries.index, y=top_countries.values, palette='Blues_d')
plt.xticks(rotation=45)
plt.title('Top 10 Countries by Average Likes')
plt.xlabel('Country')
plt.ylabel('Average Likes')
plt.show()

# 4. Influencer Segmentation by Follower Count

def follower_segment(followers):
    if followers < 100000:
        return 'Micro'
    elif followers < 1000000:
        return 'Macro'
    else:
        return 'Mega'

df['follower_segment'] = df['followers'].apply(follower_segment)
print(df['follower_segment'].value_counts())

# Engagement Rate vs. Followers by Follower Segment
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='followers', y='60_day_eng_rate', hue='follower_segment')
plt.xscale('log')  # Use log scale for followers due to wide range
plt.xlabel('Followers (log scale)')
plt.ylabel('60-Day Engagement Rate (%)')
plt.title('Engagement Rate vs. Followers by Follower Segment')
plt.show()

# 5. Assign Personalized Marketing Offers

def assign_offer(row):
    influence_score = row['influence_score']
    followers = row['followers']
    engagement_rate = row['60_day_eng_rate']
    
    if influence_score >= 90 and engagement_rate >= 0.015:
        return 'Premium Collaboration'
    elif influence_score >= 80 and followers >= 1e6:
        return 'Standard Collaboration'
    else:
        return 'Micro-Influencer Campaign'

df['personalized_offer'] = df.apply(assign_offer, axis=1)
print(df['personalized_offer'].value_counts())

# Save the processed data for future use
df.to_csv('processed_influencer_data.csv', index=False)

# 6. Interactive Dashboard with Dash

# Save the processed data for future use
df.to_csv('processed_influencer_data.csv', index=False)

# Initialize the Flask app
server = Flask(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, server=server)

# 6. Interactive Dashboard with Dash
app.layout = html.Div([
    html.H1("Instagram Influencer Analysis Dashboard"),
    dcc.Tabs([
        dcc.Tab(label='Top Influencers', children=[
            dcc.Graph(
                id='top-influencers',
                figure=px.bar(df.head(10), x='channel_info', y='followers', color='60_day_eng_rate',
                              title="Top 10 Influencers by Followers and Engagement Rate")
            )
        ]),
        dcc.Tab(label='Segmentation', children=[
            dcc.Graph(
                id='segmentation',
                figure=px.scatter(df, x='followers', y='60_day_eng_rate', color='follower_segment',
                                  title="Engagement Rate vs Followers by Segment", log_x=True)
            )
        ]),
        dcc.Tab(label='Personalized Offers', children=[
            dcc.Graph(
                id='offers-pie',
                figure=px.pie(df, names='personalized_offer', title="Distribution of Personalized Offers")
            )
        ])
    ])
])

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))