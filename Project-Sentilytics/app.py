import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path
from openai import OpenAI
import nltk
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils.sentiment_analyzer import SentimentAnalyzer
from utils.data_processor import DataProcessor
from utils.rag_engine import RAGEngine
from utils.visualization import create_sentiment_chart, create_word_cloud
from utils.alert_system import AlertSystem
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
from streamlit_option_menu import option_menu

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="Sentilytics Dashboard",
    page_icon="images/logo.png",
    layout="wide"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = None
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'alerts' not in st.session_state:
    st.session_state['alerts'] = []
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = False

def main():
    # Sidebar configuration
    with st.sidebar:
        st.image('images/logo.png')  # Add your logo image here
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        
        # Move file upload to sidebar
        uploaded_file = st.file_uploader("Upload your feedback data (CSV)", type=['csv'])
        
        # Updated navigation menu
        selected = option_menu(
            "Dashboard",
            ["Home", "Sentiment Analysis", "Reports"],  # Removed About Us
            icons=['house', 'graph-up', 'file-text'],  # Removed info-circle icon
            menu_icon="cast",
            default_index=0,
            styles={
                "icon": {"color": "#dec960", "font-size": "20px"},
                "nav-link": {"font-size": "17px", "text-align": "left", "margin": "5px", "--hover-color": "#262730"},
                "nav-link-selected": {"background-color": "#262730"}
            }
        )

    if not api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        return
        
    if not uploaded_file:
        st.warning("Please upload a CSV file with feedback data to proceed.")
        return
        
    # Process the file if it hasn't been processed yet
    if not st.session_state['processed_data']:
        try:
            data_processor = DataProcessor()
            df = data_processor.load_data(uploaded_file)
            
            if 'feedback' not in df.columns:
                st.error("Could not identify a suitable feedback column in your file. "
                        "Please ensure your file contains a column with customer feedback/comments.")
                return
                
            if len(df) == 0:
                st.warning("The uploaded file appears to be empty. Please upload a file with feedback data.")
                return
                
            st.session_state['data'] = df
            st.session_state['processed_data'] = True
            
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.info("Please ensure your file is properly formatted and contains feedback data.")
            return

    client = OpenAI(api_key=api_key)
    st.session_state['api_key'] = api_key

    # Page content based on selection
    if selected == "Home":
        display_home()
    elif selected == "Sentiment Analysis":
        display_sentiment_analysis(client)
    elif selected == "Reports":
        display_reports(client)  # New function

def display_home():
    # Hero section
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='font-size: 3.5rem; margin-bottom: 1rem;'>Welcome to Sentilytics</h1>
            <p style='font-size: 1.5rem; color: #666; margin-bottom: 2rem;'>
                Unlock the Power of Customer Feedback with AI-Driven Sentiment Analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add hero image
    # st.image('images/sentiment-hero.png', use_column_width=True)  # You'll need to add this image
    
    # Key Features section
    st.markdown("## 🚀 Key Features", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 200px;'>
                <h3 style='color: #1f77b4;'>📊 Real-Time Analysis</h3>
                <p>Instant sentiment analysis of customer feedback using advanced AI algorithms. Get immediate insights into customer sentiment patterns.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 200px;'>
                <h3 style='color: #1f77b4;'>🎯 Smart Insights</h3>
                <p>Advanced pattern recognition and trend analysis. Identify key themes and topics in customer feedback automatically.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 200px;'>
                <h3 style='color: #1f77b4;'>📈 Visual Reports</h3>
                <p>Generate comprehensive PDF reports with visualizations, trends, and actionable recommendations.</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    # How It Works section
    st.markdown("## 🔄 How It Works", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image('images/upload.png', width=100)  # Add this icon
        st.markdown("""
            <div style='text-align: center;'>
                <h4>1. Upload Data</h4>
                <p>Upload your CSV file containing customer feedback</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.image('images/analyze.png', width=100)  # Add this icon
        st.markdown("""
            <div style='text-align: center;'>
                <h4>2. Analyze</h4>
                <p>AI processes and analyzes the sentiment patterns</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.image('images/visualize.png', width=100)  # Add this icon
        st.markdown("""
            <div style='text-align: center;'>
                <h4>3. Visualize</h4>
                <p>View interactive charts and insights</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.image('images/report.png', width=100)  # Add this icon
        st.markdown("""
            <div style='text-align: center;'>
                <h4>4. Report</h4>
                <p>Generate detailed PDF reports</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Capabilities section
    st.markdown("## <br> Capabilities", unsafe_allow_html=True)
    
    capabilities = [
        "✨ Sentiment Classification (Positive, Negative, Neutral)",
        "📊 Sentiment Score Analysis",
        "🔍 Key Theme Extraction",
        "📈 Trend Analysis",
        "⚡ Real-time Processing",
        "📱 Interactive Visualizations",
        "🎯 Actionable Insights",
        "📄 PDF Report Generation"
    ]
    
    col1, col2 = st.columns(2)
    for i, capability in enumerate(capabilities):
        if i < len(capabilities) // 2:
            col1.markdown(f"### {capability}")
        else:
            col2.markdown(f"### {capability}")

def display_sentiment_analysis(client):
    st.title("Sentiment Analysis Dashboard")
    
    df = st.session_state['data']
    
    # Initialize components
    sentiment_analyzer = SentimentAnalyzer(client)
    rag_engine = RAGEngine(client)
    alert_system = AlertSystem()

    # Analysis tabs with improved structure
    tab1, tab2, tab3 = st.tabs([
        "Sentiment Overview", 
        "Detailed Analysis",
        "Insights & Recommendations"
    ])

    with tab1:
        st.header("Sentiment Overview")
        if len(df) > 0:
            # Perform sentiment analysis
            results = sentiment_analyzer.analyze_batch(df['feedback'].tolist())
            df['sentiment'] = results['sentiment']
            df['sentiment_score'] = results['scores']
            
            # Create metrics columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                positive_pct = (df['sentiment'] == 'positive').mean() * 100
                st.metric("Positive Feedback", f"{positive_pct:.1f}%")
                
            with col2:
                negative_pct = (df['sentiment'] == 'negative').mean() * 100
                st.metric("Negative Feedback", f"{negative_pct:.1f}%")
                
            with col3:
                avg_score = df['sentiment_score'].mean()
                st.metric("Average Sentiment Score", f"{avg_score:.2f}")
                
            with col4:
                total_feedback = len(df)
                st.metric("Total Feedback", total_feedback)
            
            # Display sentiment distribution
            st.subheader("Sentiment Distribution")
            fig = create_sentiment_chart(df)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Detailed Analysis")
        if len(df) > 0:
            # Time series analysis (if timestamp available)
            if 'timestamp' in df.columns:
                st.subheader("Sentiment Trends Over Time")
                # Add time series visualization here
            
            # Keyword analysis
            st.subheader("Key Terms Analysis")
            word_cloud_fig = create_word_cloud(df['feedback'].tolist())
            st.pyplot(word_cloud_fig)
            
            # Replace the dropdown with side-by-side columns
            st.subheader("Sample Feedback by Sentiment")
            
            # Get sample feedback for each sentiment
            positive_samples = df[df['sentiment'] == 'positive']['feedback'].head()
            negative_samples = df[df['sentiment'] == 'negative']['feedback'].head()
            neutral_samples = df[df['sentiment'] == 'neutral']['feedback'].head()
            
            # Create columns for each sentiment
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Positive Feedback**")
                for idx, feedback in enumerate(positive_samples, 1):
                    st.write(f"{idx}. {feedback}")
            
            with col2:
                st.markdown("**Negative Feedback**")
                for idx, feedback in enumerate(negative_samples, 1):
                    st.write(f"{idx}. {feedback}")
            
            with col3:
                st.markdown("**Neutral Feedback**")
                for idx, feedback in enumerate(neutral_samples, 1):
                    st.write(f"{idx}. {feedback}")

    with tab3:
        st.header("Insights & Recommendations")
        if len(df) > 0:
            # Overall insights
            st.subheader("Overall Analysis")
            overall_insights = rag_engine.analyze_batch(df['feedback'].tolist())
            st.write(overall_insights)
            
            # Display critical alerts
            alerts = alert_system.check_alerts(df)
            if alerts:
                st.subheader("Critical Alerts")
                for alert in alerts:
                    st.warning(alert)

def display_reports(client):
    st.title("Analytics Reports")
    
    if st.session_state['data'] is None:
        st.warning("Please upload data first to generate reports.")
        return
        
    df = st.session_state['data']
    
    # Initialize components
    sentiment_analyzer = SentimentAnalyzer(client)
    rag_engine = RAGEngine(client)
    
    st.write("### Generate Comprehensive Analysis Report")
    st.write("This report includes detailed sentiment analysis, key patterns, and actionable recommendations.")
    
    # Report customization options
    col1, col2 = st.columns(2)
    with col1:
        include_samples = st.checkbox("Include sample feedback", value=True)
        include_wordcloud = st.checkbox("Include word cloud visualization", value=True)
    with col2:
        sample_size = st.slider("Number of sample feedback to include", 3, 10, 5)
        confidence_threshold = st.slider("Minimum confidence threshold", 0.0, 1.0, 0.7)
    
    if st.button("Generate Report"):
        with st.spinner("Generating comprehensive report..."):
            try:
                # Get enhanced insights from RAG engine
                insights = rag_engine.generate_report_insights(df['feedback'].tolist())
                
                # Generate PDF report
                pdf_bytes = generate_enhanced_report(
                    df, 
                    sentiment_analyzer, 
                    rag_engine,
                    insights,
                    include_samples=include_samples,
                    include_wordcloud=include_wordcloud,
                    sample_size=sample_size,
                    confidence_threshold=confidence_threshold
                )
                
                # Provide download button
                st.download_button(
                    label="Download Report",
                    data=pdf_bytes,
                    file_name=f"sentiment_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                
                st.success("Report generated successfully!")
                
            except Exception as e:
                st.error(f"An error occurred while generating the report: {str(e)}")

def generate_enhanced_report(df, sentiment_analyzer, rag_engine, insights, include_samples=True, 
                           include_wordcloud=True, sample_size=5, confidence_threshold=0.7):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Enhanced title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    # Custom styles for sections
    section_style = ParagraphStyle(
        'SectionStyle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10
    )
    
    # Add title and date
    story.append(Paragraph("Sentiment Analysis Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", section_style))
    summary = sentiment_analyzer.get_summary(df)
    story.append(Paragraph(insights['executive_summary'], styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Key Metrics
    story.append(Paragraph("Key Metrics", section_style))
    story.append(Paragraph(f"Total Feedback Analyzed: {summary['total_feedback']}", styles['Normal']))
    story.append(Paragraph(f"Average Sentiment Score: {summary['average_score']:.2f}", styles['Normal']))
    story.append(Paragraph(f"Sentiment Distribution:", styles['Normal']))
    for sentiment, count in summary['sentiment_distribution'].items():
        percentage = (count / summary['total_feedback']) * 100
        story.append(Paragraph(f"• {sentiment.capitalize()}: {count} ({percentage:.1f}%)", styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Key Findings
    story.append(Paragraph("Key Findings", section_style))
    for finding in insights['key_findings']:
        story.append(Paragraph(f"• {finding}", styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Trending Topics
    story.append(Paragraph("Trending Topics", section_style))
    for topic in insights['trending_topics']:
        story.append(Paragraph(f"• {topic}", styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Action Items
    story.append(Paragraph("Recommended Actions", section_style))
    for action in insights['action_items']:
        story.append(Paragraph(f"• {action}", styles['Normal']))
    story.append(Spacer(1, 15))
    
    # Sample Feedback (if enabled)
    if include_samples:
        story.append(Paragraph("Representative Feedback Samples", section_style))
        samples = df.sample(min(sample_size, len(df)))
        for _, row in samples.iterrows():
            story.append(Paragraph(f"Feedback: {row['feedback']}", styles['Normal']))
            story.append(Paragraph(f"Sentiment: {row['sentiment']}", styles['Normal']))
            story.append(Spacer(1, 10))
    
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

if __name__ == "__main__":
    main() 