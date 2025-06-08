import assemblyai as aai
import pandas as pd
import numpy as np
import re
import json
import os
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# Groq API
try:
    from groq import Groq
    print("✅ Groq library loaded successfully")
except ImportError as e:
    print(f"❌ Groq library not available: {e}")
    print("Install with: pip install groq")

# NLP libraries for better text processing
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✅ NLP libraries loaded successfully")
except ImportError as e:
    print(f"⚠️ Some NLP libraries not available: {e}")

@dataclass
class QAPair:
    """Clean Q&A pair with metadata"""
    question: str
    answer: str
    timestamp: str
    speaker_q: str
    speaker_a: str
    confidence: float = 0.0
    topic: str = ""

@dataclass
class PodcastSummary:
    """Complete podcast summary"""
    title: str
    duration: str
    overall_summary: str
    key_topics: List[str]
    speakers: Dict[str, str]  # speaker_id: role
    qa_pairs: List[QAPair]
    insights: Dict[str, any]

class PodcastSummarizer:
    """Enhanced Podcast Summarizer using AssemblyAI and Groq LLM"""
    
    def __init__(self, assemblyai_key: str, groq_key: str):
        """Initialize with AssemblyAI and Groq API keys"""
        aai.settings.api_key = assemblyai_key
        self.transcript = None
        self.summary = None
        self.qa_pairs = []
        
        # Initialize Groq client
        try:
            self.groq_client = Groq(api_key=groq_key)
            print("✅ Groq client initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Groq client: {e}")
            self.groq_client = None
        
        # Configuration for chunking and rate limiting
        self.max_chunk_size = 8000  # Characters per chunk for Groq
        self.rate_limit_delay = 2   # Seconds between API calls
        self.max_retries = 3
    
    def _chunk_text(self, text: str, max_size: int = None) -> List[str]:
        """Split text into chunks to avoid hitting rate limits and token limits"""
        if max_size is None:
            max_size = self.max_chunk_size
            
        # Split by sentences first
        sentences = sent_tokenize(text) if text else []
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max_size, start a new chunk
            if len(current_chunk) + len(sentence) + 1 > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Handle case where single sentence is too long
                    chunks.append(sentence[:max_size])
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _call_groq_with_retry(self, messages: List[Dict], max_tokens: int = 1000) -> str:
        """Call Groq API with retry logic and rate limiting"""
        if not self.groq_client:
            return "Groq client not available"
        
        for attempt in range(self.max_retries):
            try:
                # Add delay to respect rate limits
                time.sleep(self.rate_limit_delay)
                
                response = self.groq_client.chat.completions.create(
                    model="llama3-70b-8192",  # Fast and good model
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    stream=False
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                print(f"⚠️ Groq API attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff
                    print(f"🔄 Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ All Groq API attempts failed")
                    return f"Failed to generate summary: {e}"
        
        return "Failed after all retries"
    
    def transcribe_and_analyze(self, audio_file: str, audio_title: str = "Podcast") -> PodcastSummary:
        """Complete pipeline: transcribe, analyze, and summarize"""
        print(f"🚀 Starting analysis of: {audio_title}")
        
        # Step 1: Configure AssemblyAI for comprehensive analysis
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            speaker_labels=True,  # Enable speaker diarization
            summarization=True,   # Enable built-in summarization
            summary_model=aai.SummarizationModel.informative,
            summary_type=aai.SummarizationType.bullets,
            entity_detection=True,  # Detect entities
            sentiment_analysis=True,  # Analyze sentiment
            language_detection=True,  # Auto-detect language
        )
        
        # Step 2: Transcribe
        print("🔄 Transcribing audio with AssemblyAI...")
        transcriber = aai.Transcriber(config=config)
        self.transcript = transcriber.transcribe(audio_file)
        
        if self.transcript.status == "error":
            raise RuntimeError(f"Transcription failed: {self.transcript.error}")
        
        print("✅ Transcription completed successfully!")
        
        # Step 3: Process and create summary
        return self._create_comprehensive_summary(audio_title)
    
    def _create_enhanced_summary_with_groq(self, text: str, builtin_summary: str = "") -> str:
        """Create an enhanced summary using Groq LLM with chunking"""
        if not self.groq_client:
            return builtin_summary if builtin_summary else self._create_basic_summary(text)
        
        print("🔄 Generating enhanced summary with Groq...")
        
        # Chunk the text to handle large transcripts
        chunks = self._chunk_text(text, max_size=6000)  # Smaller chunks for safety
        chunk_summaries = []
        
        print(f"📄 Processing {len(chunks)} chunks...")
        
        # Summarize each chunk
        for i, chunk in enumerate(chunks[:10]):  # Limit to first 10 chunks to avoid rate limits
            print(f"🔄 Processing chunk {i+1}/{min(len(chunks), 10)}...")
            
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert podcast summarizer. Create a concise, informative summary of the given text chunk. Focus on:
1. Main topics and key points discussed
2. Important insights or takeaways
3. Notable quotes or statements
4. Any actionable advice mentioned
Keep it concise but comprehensive."""
                },
                {
                    "role": "user",
                    "content": f"Summarize this podcast transcript chunk:\n\n{chunk}"
                }
            ]
            
            chunk_summary = self._call_groq_with_retry(messages, max_tokens=500)
            if chunk_summary and not chunk_summary.startswith("Failed"):
                chunk_summaries.append(chunk_summary)
        
        # Combine chunk summaries into final summary
        if chunk_summaries:
            combined_text = "\n\n".join(chunk_summaries)
            
            # Create final comprehensive summary
            final_messages = [
                {
                    "role": "system",
                    "content": """You are creating a final comprehensive podcast summary. Combine and synthesize the following chunk summaries into one cohesive, well-structured summary. Organize it with clear sections and highlight the most important points."""
                },
                {
                    "role": "user",
                    "content": f"Create a comprehensive final summary from these chunk summaries:\n\n{combined_text}"
                }
            ]
            
            final_summary = self._call_groq_with_retry(final_messages, max_tokens=800)
                        
            if final_summary and not final_summary.startswith("Failed"):
                return final_summary
        
        # Fallback to builtin summary
        return builtin_summary if builtin_summary else self._create_basic_summary(text)
    
    def _extract_key_topics_with_groq(self, text: str) -> List[str]:
        """Extract key topics using Groq LLM"""
        if not self.groq_client:
            return self._extract_topics_from_text(text)
        
        # Use a smaller sample of text for topic extraction
        sample_text = text[:4000] if len(text) > 4000 else text
        
        messages = [
            {
                "role": "system",
                "content": "Extract the main topics discussed in this podcast transcript. Return only a comma-separated list of topics (maximum 15 topics). Be concise and specific."
            },
            {
                "role": "user",
                "content": f"Extract key topics from this podcast transcript:\n\n{sample_text}"
            }
        ]
        
        response = self._call_groq_with_retry(messages, max_tokens=200)
        
        if response and not response.startswith("Failed"):
            # Parse the comma-separated topics
            topics = [topic.strip() for topic in response.split(',')]
            topics = [topic for topic in topics if topic and len(topic) > 2]
            return topics[:15]
        
        # Fallback to basic extraction
        return self._extract_topics_from_text(text)
    
    def _create_comprehensive_summary(self, title: str) -> PodcastSummary:
        """Create a comprehensive summary from transcript"""
        print("🔄 Creating comprehensive summary...")
        
        # Extract basic info
        duration = self._format_duration(self.transcript.audio_duration)
        
        # Get AssemblyAI's built-in summary
        builtin_summary = self.transcript.summary if hasattr(self.transcript, 'summary') else ""
        
        # Create enhanced summary using Groq
        overall_summary = self._create_enhanced_summary_with_groq(self.transcript.text, builtin_summary)
        
        # Extract key topics using Groq
        key_topics = self._extract_key_topics_with_groq(self.transcript.text)
        
        # Identify speakers and their roles
        speakers = self._identify_speakers()
        
        # Extract Q&A pairs
        qa_pairs = self._extract_qa_pairs()
        
        # Generate insights
        insights = self._generate_insights()
        
        summary = PodcastSummary(
            title=title,
            duration=duration,
            overall_summary=overall_summary,
            key_topics=key_topics,
            speakers=speakers,
            qa_pairs=qa_pairs,
            insights=insights
        )
        
        print("✅ Comprehensive summary created!")
        return summary
    
    def _create_basic_summary(self, text: str) -> str:
        """Create a basic summary using text processing"""
        sentences = sent_tokenize(text)
        
        # Take first few and last few sentences, plus some middle ones
        if len(sentences) > 10:
            summary_sentences = sentences[:3] + sentences[len(sentences)//2-1:len(sentences)//2+2] + sentences[-2:]
        else:
            summary_sentences = sentences[:5]
        
        return ' '.join(summary_sentences)
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics using basic NLP"""
        # Common terms for context - you can customize this
        domain_terms = ['business', 'technology', 'ai', 'machine learning', 'startup', 'innovation',
                       'strategy', 'market', 'product', 'customer', 'growth', 'development',
                       'data', 'analytics', 'digital', 'transformation', 'leadership']
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words and len(w) > 3]
        except:
            words = [w for w in words if len(w) > 3]
        
        # Count frequency
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # Prioritize domain terms and high-frequency words
        topics = []
        for term in domain_terms:
            if term in word_freq and word_freq[term] > 2:
                topics.append(term.title())
        
        # Add other high-frequency words
        other_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, freq in other_topics:
            if len(topics) >= 15:
                break
            if word.title() not in topics and freq > 3:
                topics.append(word.title())
        
        return topics
    
    def _identify_speakers(self) -> Dict[str, str]:
        """Identify speakers and their roles"""
        speakers = {}
        
        if hasattr(self.transcript, 'utterances') and self.transcript.utterances:
            speaker_stats = defaultdict(lambda: {'words': 0, 'utterances': 0})
            
            for utterance in self.transcript.utterances:
                speaker = utterance.speaker
                word_count = len(utterance.text.split())
                speaker_stats[speaker]['words'] += word_count
                speaker_stats[speaker]['utterances'] += 1
            
            # Assign roles based on speaking patterns
            sorted_speakers = sorted(speaker_stats.items(), key=lambda x: x[1]['words'], reverse=True)
            
            for i, (speaker, stats) in enumerate(sorted_speakers):
                if i == 0:
                    role = "Host/Interviewer"
                else:
                    role = "Guest/Interviewee"
                speakers[speaker] = role
        else:
            speakers = {"Speaker_A": "Host", "Speaker_B": "Guest"}
        
        return speakers
    
    def _extract_qa_pairs(self) -> List[QAPair]:
        """Extract question-answer pairs from utterances"""
        qa_pairs = []
        
        if not hasattr(self.transcript, 'utterances') or not self.transcript.utterances:
            return qa_pairs
        
        utterances = self.transcript.utterances
        question_patterns = [
            r'\?',  # Direct question mark
            r'^(what|how|why|when|where|who|which|can|could|would|should|do|does|did|is|are|was|were)',
            r'^(tell me|explain|describe|talk about)',
            r'(right|correct|agree|think)\?$',
        ]
        
        for i, utterance in enumerate(utterances):
            text = utterance.text.strip()
            
            # Check if this is a question
            is_question = any(re.search(pattern, text.lower()) for pattern in question_patterns)
            
            if is_question and len(text.split()) >= 3:
                # Look for answer in next few utterances
                for j in range(i + 1, min(i + 4, len(utterances))):
                    candidate = utterances[j]
                    
                    # Skip if same speaker
                    if candidate.speaker == utterance.speaker:
                        continue
                    
                    # Skip if too short
                    if len(candidate.text.split()) < 5:
                        continue
                    
                    # Skip if it's another question
                    candidate_is_question = any(re.search(pattern, candidate.text.lower()) for pattern in question_patterns)
                    if candidate_is_question:
                        continue
                    
                    # Create Q&A pair
                    qa_pair = QAPair(
                        question=text,
                        answer=candidate.text.strip(),
                        timestamp=self._format_timestamp(utterance.start),
                        speaker_q=utterance.speaker,
                        speaker_a=candidate.speaker,
                        confidence=0.8  # Base confidence
                    )
                    qa_pairs.append(qa_pair)
                    break
        
        return qa_pairs[:20]  # Return top 20 Q&A pairs
    
    def _generate_insights(self) -> Dict[str, any]:
        """Generate insights about the podcast"""
        insights = {}
        
        # Basic stats
        total_words = len(self.transcript.text.split())
        insights['total_words'] = total_words
        insights['estimated_reading_time'] = f"{total_words // 200} minutes"
        
        # Speaker analysis
        if hasattr(self.transcript, 'utterances') and self.transcript.utterances:
            speaker_word_count = defaultdict(int)
            for utterance in self.transcript.utterances:
                speaker_word_count[utterance.speaker] += len(utterance.text.split())
            
            insights['speaker_distribution'] = dict(speaker_word_count)
            insights['dominant_speaker'] = max(speaker_word_count, key=speaker_word_count.get)
        
        # Q&A analysis
        insights['total_qa_pairs'] = len(self.qa_pairs)
        insights['question_density'] = len(self.qa_pairs) / (total_words / 1000) if total_words > 0 else 0
        
        # Sentiment if available
        if hasattr(self.transcript, 'sentiment_analysis_results'):
            sentiments = [s.sentiment for s in self.transcript.sentiment_analysis_results]
            insights['overall_sentiment'] = max(set(sentiments), key=sentiments.count) if sentiments else "neutral"
        
        return insights
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration from seconds to readable format"""
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        remaining_seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"
        else:
            return f"{minutes}:{remaining_seconds:02d}"
    
    def _format_timestamp(self, milliseconds: int) -> str:
        """Format timestamp from milliseconds"""
        total_seconds = int(milliseconds / 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def display_summary(self, summary: PodcastSummary):
        """Display a beautiful summary"""
        print("\n" + "="*80)
        print(f"🎙️  PODCAST SUMMARY: {summary.title}")
        print("="*80)
        
        print(f"\n📊 BASIC INFO:")
        print(f"⏱️  Duration: {summary.duration}")
        print(f"👥 Speakers: {len(summary.speakers)}")
        print(f"❓ Q&A Pairs: {len(summary.qa_pairs)}")
        
        print(f"\n📝 OVERALL SUMMARY:")
        print(f"{summary.overall_summary}")
        
        print(f"\n🏷️  KEY TOPICS:")
        topics_str = " • ".join(summary.key_topics[:10])
        print(f"{topics_str}")
        
        print(f"\n👥 SPEAKERS:")
        for speaker, role in summary.speakers.items():
            print(f"   🎤 {speaker}: {role}")
        
        print(f"\n❓ TOP QUESTION-ANSWER PAIRS:")
        for i, qa in enumerate(summary.qa_pairs[:5], 1):
            print(f"\n   {i}. [{qa.timestamp}] {qa.speaker_q} ➤ {qa.speaker_a}")
            print(f"      Q: {qa.question}")
            print(f"      A: {qa.answer[:200]}{'...' if len(qa.answer) > 200 else ''}")
            print("      " + "-"*60)
        
        print(f"\n📈 INSIGHTS:")
        for key, value in summary.insights.items():
            if key == 'speaker_distribution':
                print(f"   📊 Speaker Word Distribution:")
                for speaker, words in value.items():
                    print(f"      {speaker}: {words} words")
            else:
                print(f"   📌 {key.replace('_', ' ').title()}: {value}")
    
    def export_summary(self, summary: PodcastSummary, output_dir: str = "./podcast_summary"):
        """Export summary to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export as JSON
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
        
        with open(f"{output_dir}/summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        
        # Export Q&A as CSV
        if summary.qa_pairs:
            qa_df = pd.DataFrame([
                {
                    'Timestamp': qa.timestamp,
                    'Question Speaker': qa.speaker_q,
                    'Question': qa.question,
                    'Answer Speaker': qa.speaker_a,
                    'Answer': qa.answer
                }
                for qa in summary.qa_pairs
            ])
            qa_df.to_csv(f"{output_dir}/qa_pairs.csv", index=False)
        
        # Export readable summary
        with open(f"{output_dir}/readable_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"PODCAST SUMMARY: {summary.title}\n")
            f.write("="*60 + "\n\n")
            f.write(f"Duration: {summary.duration}\n")
            f.write(f"Speakers: {len(summary.speakers)}\n")
            f.write(f"Q&A Pairs: {len(summary.qa_pairs)}\n\n")
            f.write("OVERALL SUMMARY:\n")
            f.write(summary.overall_summary + "\n\n")
            f.write("KEY TOPICS:\n")
            f.write(", ".join(summary.key_topics) + "\n\n")
            f.write("SPEAKERS:\n")
            for speaker, role in summary.speakers.items():
                f.write(f"- {speaker}: {role}\n")
            f.write("\nTOP Q&A PAIRS:\n")
            for i, qa in enumerate(summary.qa_pairs, 1):
                f.write(f"\n{i}. [{qa.timestamp}]\n")
                f.write(f"Q ({qa.speaker_q}): {qa.question}\n")
                f.write(f"A ({qa.speaker_a}): {qa.answer}\n")
                f.write("-" * 40 + "\n")
        
        print(f"📁 Summary exported to: {output_dir}")

# Example usage
def main():
    """Example usage of the PodcastSummarizer with Groq"""
    
    # Initialize with your API keys
    ASSEMBLYAI_KEY = ""  # Your AssemblyAI key
    GROQ_KEY = ""  # Replace with your Groq API key
    
    summarizer = PodcastSummarizer(ASSEMBLYAI_KEY, GROQ_KEY)
    
    # Process audio file (local file or URL)
    audio_file = "interview2.mp3"  # Replace with your file path
    
    try:
        # Generate comprehensive summary
        summary = summarizer.transcribe_and_analyze(
            audio_file=audio_file,
            audio_title="Podcast Analysis with Groq LLM"
        )
        
        # Display the summary
        summarizer.display_summary(summary)
        
        # Export to files
        summarizer.export_summary(summary)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()