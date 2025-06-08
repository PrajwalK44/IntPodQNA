import streamlit as st
import pandas as pd
import json
import os
import time
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import tempfile
import zipfile
from main import PodcastSummarizer, PodcastSummary, QAPair  # Import your main classes

# Configure Streamlit page
st.set_page_config(
    page_title="🎙️ Podcast AI Summarizer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        color: #000;
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .qa-pair {
        color: #000;
        background: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .speaker-tag {
        background: #667eea;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .timestamp {
        background: #28a745;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.75rem;
    }
    
    .topic-tag {
        background: #17a2b8;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
    if 'summary' not in st.session_state:
        st.session_state.summary = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def create_api_key_section():
    """Create API key input section"""
    st.sidebar.header("🔑 API Configuration")
    
    # AssemblyAI API Key
    assemblyai_key = st.sidebar.text_input(
        "AssemblyAI API Key",
        type="password",
        help="Get your free API key from AssemblyAI"
    )
    
    # Groq API Key
    groq_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        help="Get your free API key from Groq"
    )
    
    if st.sidebar.button("🚀 Initialize Summarizer"):
        if assemblyai_key and groq_key:
            try:
                st.session_state.summarizer = PodcastSummarizer(assemblyai_key, groq_key)
                st.sidebar.success("✅ Summarizer initialized successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to initialize: {str(e)}")
        else:
            st.sidebar.error("Please provide both API keys")
    
    return assemblyai_key, groq_key

def create_file_upload_section():
    """Create file upload section"""
    st.header("📁 Upload Audio File")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
        help="Supported formats: MP3, WAV, M4A, FLAC, OGG"
    )
    
    # URL input as alternative
    st.subheader("Or provide audio URL")
    audio_url = st.text_input(
        "Audio URL",
        placeholder="https://example.com/podcast.mp3",
        help="Direct link to audio file"
    )
    
    # Podcast title
    podcast_title = st.text_input(
        "Podcast Title (Optional)",
        placeholder="Enter podcast title...",
        value="Podcast Analysis"
    )
    
    return uploaded_file, audio_url, podcast_title

def display_processing_status():
    """Display processing status with progress"""
    if st.session_state.processing:
        st.info("🔄 Processing your podcast...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate progress (you can integrate real progress from your main.py)
        for i in range(100):
            time.sleep(0.1)
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text("📝 Transcribing audio...")
            elif i < 60:
                status_text.text("🤖 Generating summary with AI...")
            elif i < 80:
                status_text.text("🔍 Extracting Q&A pairs...")
            else:
                status_text.text("📊 Analyzing insights...")

def display_summary_overview(summary: PodcastSummary):
    """Display summary overview with metrics"""
    st.markdown('<div class="main-header"><h1>🎙️ Podcast Summary Complete!</h1></div>', unsafe_allow_html=True)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📈 Duration",
            value=summary.duration,
            help="Total podcast duration"
        )
    
    with col2:
        st.metric(
            label="👥 Speakers",
            value=len(summary.speakers),
            help="Number of speakers identified"
        )
    
    with col3:
        st.metric(
            label="❓ Q&A Pairs",
            value=len(summary.qa_pairs),
            help="Question-Answer pairs extracted"
        )
    
    with col4:
        st.metric(
            label="🏷️ Topics",
            value=len(summary.key_topics),
            help="Key topics identified"
        )

def display_overall_summary(summary: PodcastSummary):
    """Display the overall summary"""
    st.header("📝 Overall Summary")
    st.markdown(f"""
    <div class="metric-card">
        <h4>{summary.title}</h4>
        <p>{summary.overall_summary}</p>
    </div>
    """, unsafe_allow_html=True)

def display_key_topics(summary: PodcastSummary):
    """Display key topics as tags"""
    st.header("🏷️ Key Topics")
    
    # Create topic tags
    topics_html = ""
    for topic in summary.key_topics:
        topics_html += f'<span class="topic-tag">{topic}</span>'
    
    st.markdown(topics_html, unsafe_allow_html=True)
    
    # Topic frequency chart
    if len(summary.key_topics) > 0:
        st.subheader("📊 Topic Distribution")
        topic_df = pd.DataFrame({
            'Topic': summary.key_topics[:10],  # Top 10 topics
            'Relevance': [100 - i*5 for i in range(len(summary.key_topics[:10]))]  # Mock relevance scores
        })
        
        fig = px.bar(
            topic_df, 
            x='Relevance', 
            y='Topic',
            orientation='h',
            title="Top Topics by Relevance",
            color='Relevance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def display_speakers_analysis(summary: PodcastSummary):
    """Display speakers analysis"""
    st.header("👥 Speakers Analysis")
    
    # Speakers info
    for speaker, role in summary.speakers.items():
        st.markdown(f"""
        <div class="metric-card">
            <span class="speaker-tag">{speaker}</span>
            <strong> - {role}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Speaker distribution chart if available
    if 'speaker_distribution' in summary.insights:
        st.subheader("📊 Speaking Time Distribution")
        speaker_data = summary.insights['speaker_distribution']
        
        fig = px.pie(
            values=list(speaker_data.values()),
            names=list(speaker_data.keys()),
            title="Word Count by Speaker"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_qa_pairs(summary: PodcastSummary):
    """Display Q&A pairs in an interactive format"""
    st.header("❓ Question & Answer Pairs")
    
    if not summary.qa_pairs:
        st.info("No Q&A pairs were extracted from this podcast.")
        return
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        speaker_filter = st.selectbox(
            "Filter by Question Speaker",
            options=["All"] + list(set([qa.speaker_q for qa in summary.qa_pairs]))
        )
    
    with col2:
        max_pairs = st.slider("Number of Q&A pairs to show", 1, len(summary.qa_pairs), min(10, len(summary.qa_pairs)))
    
    # Filter Q&A pairs
    filtered_pairs = summary.qa_pairs
    if speaker_filter != "All":
        filtered_pairs = [qa for qa in summary.qa_pairs if qa.speaker_q == speaker_filter]
    
    # Display Q&A pairs
    for i, qa in enumerate(filtered_pairs[:max_pairs], 1):
        st.markdown(f"""
        <div class="qa-pair">
            <div style="margin-bottom: 0.5rem;">
                <span class="timestamp">{qa.timestamp}</span>
                <span class="speaker-tag">Q: {qa.speaker_q}</span>
                <span class="speaker-tag">A: {qa.speaker_a}</span>
            </div>
            <div style="margin-bottom: 0.5rem;">
                <strong>Q:</strong> {qa.question}
            </div>
            <div>
                <strong>A:</strong> {qa.answer}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_insights(summary: PodcastSummary):
    """Display detailed insights"""
    st.header("📊 Detailed Insights")
    
    if not summary.insights:
        st.info("No insights available.")
        return
    
    # Create insights cards
    col1, col2 = st.columns(2)
    
    with col1:
        if 'total_words' in summary.insights:
            st.metric("📝 Total Words", summary.insights['total_words'])
        
        if 'estimated_reading_time' in summary.insights:
            st.metric("⏱️ Reading Time", summary.insights['estimated_reading_time'])
        
        if 'question_density' in summary.insights:
            st.metric("❓ Question Density", f"{summary.insights['question_density']:.2f} per 1000 words")
    
    with col2:
        if 'dominant_speaker' in summary.insights:
            st.metric("🎤 Dominant Speaker", summary.insights['dominant_speaker'])
        
        if 'overall_sentiment' in summary.insights:
            st.metric("😊 Overall Sentiment", summary.insights['overall_sentiment'].title())
    
    # Additional insights visualization
    if 'speaker_distribution' in summary.insights:
        st.subheader("📈 Speaking Statistics")
        speaker_stats = summary.insights['speaker_distribution']
        stats_df = pd.DataFrame(list(speaker_stats.items()), columns=['Speaker', 'Word Count'])
        
        fig = px.bar(stats_df, x='Speaker', y='Word Count', title="Word Count by Speaker")
        st.plotly_chart(fig, use_container_width=True)

def create_export_section(summary: PodcastSummary):
    """Create export functionality"""
    st.header("📥 Export Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Download JSON"):
            summary_dict = {
                'title': summary.title,
                'duration': summary.duration,
                'overall_summary': summary.overall_summary,
                'key_topics': summary.key_topics,
                'speakers': summary.speakers,
                'qa_pairs': [
                    {
                        'question': qa.question,
                        'answer': qa.answer,
                        'timestamp': qa.timestamp,
                        'speaker_q': qa.speaker_q,
                        'speaker_a': qa.speaker_a
                    }
                    for qa in summary.qa_pairs
                ],
                'insights': summary.insights
            }
            
            json_str = json.dumps(summary_dict, indent=2, ensure_ascii=False)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{summary.title.replace(' ', '_')}_summary.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Download CSV"):
            if summary.qa_pairs:
                qa_data = []
                for qa in summary.qa_pairs:
                    qa_data.append({
                        'Timestamp': qa.timestamp,
                        'Question_Speaker': qa.speaker_q,
                        'Question': qa.question,
                        'Answer_Speaker': qa.speaker_a,
                        'Answer': qa.answer
                    })
                
                df = pd.DataFrame(qa_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{summary.title.replace(' ', '_')}_qa_pairs.csv",
                    mime="text/csv"
                )
    
    with col3:
        if st.button("📝 Download Text"):
            text_content = f"""PODCAST SUMMARY: {summary.title}
{'='*60}

Duration: {summary.duration}
Speakers: {len(summary.speakers)}
Q&A Pairs: {len(summary.qa_pairs)}

OVERALL SUMMARY:
{summary.overall_summary}

KEY TOPICS:
{', '.join(summary.key_topics)}

SPEAKERS:
{chr(10).join([f'- {speaker}: {role}' for speaker, role in summary.speakers.items()])}

TOP Q&A PAIRS:
{chr(10).join([f'{i}. [{qa.timestamp}] Q ({qa.speaker_q}): {qa.question} A ({qa.speaker_a}): {qa.answer}' for i, qa in enumerate(summary.qa_pairs, 1)])}
"""
            
            st.download_button(
                label="Download Text",
                data=text_content,
                file_name=f"{summary.title.replace(' ', '_')}_summary.txt",
                mime="text/plain"
            )

def main():
    """Main Streamlit app"""
    initialize_session_state()
    
    # App header
    st.markdown("""
    # 🎙️ AI Podcast Summarizer
    ### Transform your podcasts into structured summaries with AI-powered analysis
    """)
    
    # API Configuration
    assemblyai_key, groq_key = create_api_key_section()
    
    # Main content
    if st.session_state.summarizer is None:
        st.info("👆 Please configure your API keys in the sidebar to get started.")
        st.markdown("""
        ### Features:
        - 🎯 **AI-Powered Transcription** using AssemblyAI
        - 🤖 **Enhanced Summaries** with Groq LLM
        - ❓ **Q&A Extraction** from conversations
        - 👥 **Speaker Analysis** and identification
        - 📊 **Interactive Visualizations**
        - 📥 **Multiple Export Formats**
        
        ### How to get API Keys:
        1. **AssemblyAI**: Visit [AssemblyAI](https://www.assemblyai.com/) and sign up for a free account
        2. **Groq**: Visit [Groq](https://console.groq.com/) and get your free API key
        """)
        return
    
    # File upload section
    uploaded_file, audio_url, podcast_title = create_file_upload_section()
    
    # Process button
    if st.button("🚀 Analyze Podcast", type="primary"):
        if not uploaded_file and not audio_url:
            st.error("Please upload a file or provide an audio URL.")
            return
        
        st.session_state.processing = True
        
        try:
            with st.spinner("Processing your podcast... This may take a few minutes."):
                # Handle file upload
                audio_source = None
                if uploaded_file:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        audio_source = tmp_file.name
                else:
                    audio_source = audio_url
                
                # Process the audio
                summary = st.session_state.summarizer.transcribe_and_analyze(
                    audio_file=audio_source,
                    audio_title=podcast_title
                )
                
                st.session_state.summary = summary
                st.session_state.processing = False
                
                # Clean up temporary file
                if uploaded_file and audio_source:
                    try:
                        os.unlink(audio_source)
                    except:
                        pass
                
                st.success("✅ Podcast analysis completed successfully!")
                st.rerun()
        
        except Exception as e:
            st.session_state.processing = False
            st.error(f"❌ Error processing podcast: {str(e)}")
    
    # Display results if available
    if st.session_state.summary:
        summary = st.session_state.summary
        
        # Summary overview
        display_summary_overview(summary)
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📝 Summary", "🏷️ Topics", "👥 Speakers", "❓ Q&A Pairs", "📊 Insights", "📥 Export"
        ])
        
        with tab1:
            display_overall_summary(summary)
        
        with tab2:
            display_key_topics(summary)
        
        with tab3:
            display_speakers_analysis(summary)
        
        with tab4:
            display_qa_pairs(summary)
        
        with tab5:
            display_insights(summary)
        
        with tab6:
            create_export_section(summary)

if __name__ == "__main__":
    main()