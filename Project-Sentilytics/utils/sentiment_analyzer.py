from textblob import TextBlob
import pandas as pd

class SentimentAnalyzer:
    def __init__(self, client):
        self.client = client
        self.system_prompt = """You are an expert sentiment analyzer specializing in 
        customer feedback analysis. Analyze the given text and provide sentiment 
        classification (positive, negative, or neutral) along with a confidence score."""

    def analyze_batch(self, texts):
        results = {
            'sentiment': [],
            'scores': []
        }
        
        for text in texts:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Analyze the sentiment of: {text}"}
                ]
            )
            
            # Process the response
            analysis = response.choices[0].message.content
            # Simple parsing - you might want to make this more robust
            if "positive" in analysis.lower():
                sentiment = "positive"
                score = 0.8
            elif "negative" in analysis.lower():
                sentiment = "negative"
                score = 0.2
            else:
                sentiment = "neutral"
                score = 0.5
                
            results['sentiment'].append(sentiment)
            results['scores'].append(score)
            
        return results

    def get_summary(self, df):
        sentiment_counts = df['sentiment'].value_counts()
        return {
            'total_feedback': len(df),
            'sentiment_distribution': sentiment_counts.to_dict(),
            'average_score': df['sentiment_score'].mean()
        } 