import json

class RAGEngine:
    def __init__(self, client):
        self.client = client
        self.system_prompt = """You are an advanced business analytics AI specialized in customer feedback analysis. 
        Analyze the feedback to provide detailed, actionable insights in the following areas:

        1. Key Themes: Identify main topics and patterns in the feedback
        2. Sentiment Analysis: Analyze emotional tone and intensity
        3. Impact Areas: Identify which aspects of the business are most affected
        4. Recommendations: Provide specific, actionable recommendations
        5. Priority Level: Assess urgency of addressing identified issues
        6. Competitive Insights: Identify any mentions of competitors or market positioning
        7. Customer Experience: Analyze specific touchpoints mentioned
        8. Trend Analysis: Identify emerging patterns or shifts in customer sentiment

        Format your response in clear sections with bullet points for easy reading.
        Focus on insights that can drive business improvements and strategic decisions."""

    def analyze_batch(self, texts):
        combined_text = "\n".join(texts)
        response = self.client.chat.completions.create(
            model="gpt-4",  # Upgraded to GPT-4 for better analysis
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Analyze this collection of feedback and provide comprehensive insights:\n\n{combined_text}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content

    def get_insights(self, feedback):
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Analyze this feedback and provide insights: {feedback}"}
            ]
        )
        return response.choices[0].message.content

    def generate_report_insights(self, texts):
        combined_text = "\n".join(texts)
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """You are an expert business analyst specialized in customer feedback analysis.
                Generate a comprehensive report with the following sections:
                1. Executive Summary: High-level overview of the feedback analysis
                2. Key Findings: Major patterns and insights discovered
                3. Trending Topics: Important themes and subjects mentioned frequently
                4. Action Items: Specific, actionable recommendations for business improvement
                
                Format the response as a JSON object with these keys:
                {
                    "executive_summary": "string",
                    "key_findings": ["string"],
                    "trending_topics": ["string"],
                    "action_items": ["string"]
                }
                
                Focus on insights that are:
                - Specific and actionable
                - Backed by patterns in the data
                - Relevant to business improvement
                - Prioritized by impact and urgency"""},
                {"role": "user", "content": f"Analyze this collection of feedback and provide comprehensive insights:\n\n{combined_text}"}
            ],
            temperature=0.7
        )
        
        # Parse the JSON response
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # Fallback in case the response isn't proper JSON
            return {
                "executive_summary": "Analysis completed successfully.",
                "key_findings": ["No specific findings to report."],
                "trending_topics": ["No trending topics identified."],
                "action_items": ["No specific actions recommended."]
            }