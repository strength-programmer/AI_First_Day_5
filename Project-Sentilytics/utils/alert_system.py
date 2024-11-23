class AlertSystem:
    def __init__(self, negative_threshold=0.3):
        self.negative_threshold = negative_threshold

    def check_alerts(self, df):
        alerts = []
        
        # Check for spike in negative sentiment
        negative_ratio = len(df[df['sentiment'] == 'negative']) / len(df)
        if negative_ratio > self.negative_threshold:
            alerts.append(f"Alert: High negative sentiment detected ({negative_ratio:.1%})")
            
        return alerts 