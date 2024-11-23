import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_sentiment_chart(df):
    sentiment_counts = df['sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Sentiment Distribution',
        color_discrete_map={
            'positive': 'green',
            'neutral': 'gray',
            'negative': 'red'
        }
    )
    return fig

def create_word_cloud(texts):
    text = ' '.join(texts)
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig 