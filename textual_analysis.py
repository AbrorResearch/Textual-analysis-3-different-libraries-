"""
Textual Analysis using Loughran McDonald, VADER, NLTK Opinion Lexicon, and Readability Metrics
This script analyzes text from an Excel file column using three different sentiment/lexicon dictionaries
plus readability indices (FOG and Flesch-Kincaid).

Dictionaries Used:
1. Loughran McDonald - Financial sentiment analysis (pyloughranmcdonald)
2. VADER - Valence Aware Dictionary and sEntiment Reasoner (from nltk)
3. NLTK Opinion Lexicon - General sentiment analysis (Harvard IV-4 based)

Readability Metrics:
1. FOG (Gunning FOG) Index - Years of education needed to understand the text
2. Flesch Kincaid Grade Level - US school grade level needed to understand the text
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon
from nltk.tokenize import sent_tokenize, word_tokenize
from pyloughranmcdonald import LMClassifier

# Download required NLTK data
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('corpora/opinion_lexicon')
except LookupError:
    nltk.download('opinion_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TextualAnalyzer:
    """
    Performs comprehensive textual analysis including:
    1. Loughran McDonald Dictionary (financial sentiment via pyloughranmcdonald)
    2. VADER (Valence Aware Dictionary and sEntiment Reasoner)
    3. NLTK Opinion Lexicon (Harvard IV-4 based)
    4. Readability Metrics (FOG Index and Flesch-Kincaid Grade Level)
    """
    
    def __init__(self):
        """Initialize the analyzer with dictionaries"""
        # Initialize Loughran McDonald classifier
        self.lm_classifier = LMClassifier()
        
        # Initialize VADER sentiment analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Load NLTK Opinion Lexicon (Harvard IV-4 based)
        self.opinion_positive_words = set(opinion_lexicon.positive())
        self.opinion_negative_words = set(opinion_lexicon.negative())
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: convert to lowercase, remove punctuation, tokenize
        """
        if pd.isna(text):
            return []
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces and letters
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Split into words
        words = text.split()
        
        return words
    
    def count_syllables(self, word: str) -> int:
        """
        Estimate syllable count in a word.
        This is a simplified approximation.
        Rules:
        - Count vowel groups
        - Subtract silent 'e'
        - Adjust for common patterns
        """
        word = word.lower()
        syllable_count = 0
        vowels = "aeiou"
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        # Adjust for 'le' at end
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1
        
        # Ensure at least 1 syllable
        return max(1, syllable_count)
    
    def is_complex_word(self, word: str) -> bool:
        """
        Determine if a word is complex (3+ syllables, excluding proper nouns and common words)
        Used for FOG index calculation
        """
        if len(word) <= 3:
            return False
        
        syllables = self.count_syllables(word)
        return syllables >= 3
    
    def calculate_fog_index(self, text: str) -> Dict:
        """
        Calculate Gunning FOG (Frequency of Gobbledygook) Index
        
        Formula: 0.4 * ((words/sentences) + 100 * (complex_words/words))
        
        Interpretation:
        - 6: Elementary school level
        - 9: High school level
        - 13: College level
        - 16+: Graduate school level
        
        Returns:
        - fog_index: The FOG score
        - complexity: Percentage of complex words
        """
        if pd.isna(text):
            return {'FOG_index': 0, 'FOG_complexity': 0}
        
        # Count sentences
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return {'FOG_index': 0, 'FOG_complexity': 0}
        
        # Count words
        words = self.preprocess_text(text)
        num_words = len(words)
        
        if num_words == 0:
            return {'FOG_index': 0, 'FOG_complexity': 0}
        
        # Count complex words (3+ syllables)
        complex_words = sum(1 for word in words if self.is_complex_word(word))
        
        # Calculate FOG index
        fog_index = 0.4 * ((num_words / num_sentences) + 100 * (complex_words / num_words))
        complexity_percent = (complex_words / num_words) * 100
        
        return {
            'FOG_index': round(fog_index, 2),
            'FOG_complexity': round(complexity_percent, 2)  # % of complex words
        }
    
    def calculate_flesch_kincaid(self, text: str) -> Dict:
        """
        Calculate Flesch-Kincaid Grade Level Index
        
        Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        
        Interpretation:
        - 6-8: Easy to read
        - 9-12: Medium difficulty
        - 13-15: Difficult
        - 16+: Very difficult
        
        Returns:
        - flesch_kincaid_grade: The grade level
        """
        if pd.isna(text):
            return {'Flesch_Kincaid_Grade': 0}
        
        # Count sentences
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return {'Flesch_Kincaid_Grade': 0}
        
        # Count words
        words = self.preprocess_text(text)
        num_words = len(words)
        
        if num_words == 0:
            return {'Flesch_Kincaid_Grade': 0}
        
        # Count syllables
        total_syllables = sum(self.count_syllables(word) for word in words)
        
        # Calculate Flesch-Kincaid grade level
        flesch_kincaid = (0.39 * (num_words / num_sentences) + 
                          11.8 * (total_syllables / num_words) - 15.59)
        
        return {
            'Flesch_Kincaid_Grade': round(max(0, flesch_kincaid), 2)  # Ensure non-negative
        }
    
    def calculate_readability_metrics(self, text: str) -> Dict:
        """
        Calculate all readability metrics
        """
        fog_results = self.calculate_fog_index(text)
        flesch_kincaid_results = self.calculate_flesch_kincaid(text)
        
        # Combine results
        readability_results = {}
        readability_results.update(fog_results)
        readability_results.update(flesch_kincaid_results)
        
        return readability_results
    
    def analyze_loughran_mcdonald(self, text: str) -> Dict:
        """
        Analyze text using Loughran McDonald dictionary (Financial Sentiment)
        
        Uses the official pyloughranmcdonald library for accurate financial sentiment analysis.
        
        Categories:
        - positive: Financial gains, profits, growth
        - negative: Losses, decline, risk
        - uncertainty: Uncertain, vague terms
        - litigious: Legal and regulatory terms
        - constraining: Risk and uncertainty modifiers
        - strong_modal: Will, must, definitely
        - weak_modal: May, might, could
        """
        score = self.lm_classifier.score(text)
        
        # Extract raw counts
        lm_results = {
            'LM_positive_count': score.get('positive', 0),
            'LM_negative_count': score.get('negative', 0),
            'LM_uncertainty_count': score.get('uncertainty', 0),
            'LM_litigious_count': score.get('litigious', 0),
            'LM_constraining_count': score.get('constraining', 0),
            'LM_strong_modal_count': score.get('strong_modal', 0),
            'LM_weak_modal_count': score.get('weak_modal', 0),
        }
        
        # Calculate total word count for ratios
        words = self.preprocess_text(text)
        total_words = len(words)
        
        # Add ratios
        for key, value in lm_results.items():
            ratio_key = key.replace('_count', '_ratio')
            lm_results[ratio_key] = value / total_words if total_words > 0 else 0
        
        # Calculate net sentiment (positive - negative)
        lm_results['LM_net_sentiment'] = (lm_results['LM_positive_count'] - 
                                          lm_results['LM_negative_count'])
        
        return lm_results
    
    def analyze_vader(self, text: str) -> Dict:
        """
        Analyze text using VADER (Valence Aware Dictionary and sEntiment Reasoner)
        
        VADER is specifically designed for social media and modern text.
        It's excellent for capturing nuanced sentiment including emoticons, slang, etc.
        
        Returns:
        - negative: Proportion of negative sentiment (0-1)
        - neutral: Proportion of neutral sentiment (0-1)
        - positive: Proportion of positive sentiment (0-1)
        - compound: Normalized composite sentiment score (-1 to +1)
          * -1.0 = most negative
          * +1.0 = most positive
          * -0.05 to +0.05 = neutral
        """
        if pd.isna(text):
            return {
                'VADER_negative': 0,
                'VADER_neutral': 0,
                'VADER_positive': 0,
                'VADER_compound': 0
            }
        
        scores = self.vader_analyzer.polarity_scores(text)
        
        return {
            'VADER_negative': round(scores['neg'], 4),
            'VADER_neutral': round(scores['neu'], 4),
            'VADER_positive': round(scores['pos'], 4),
            'VADER_compound': round(scores['compound'], 4)  # Key metric: -1 to +1
        }
    
    def analyze_opinion_lexicon(self, text: str) -> Dict:
        """
        Analyze text using NLTK Opinion Lexicon (Harvard IV-4 based)
        
        The Opinion Lexicon is derived from the Harvard IV-4 dictionary,
        which is a well-established general-purpose sentiment dictionary
        used in academic research for decades.
        
        Categories:
        - positive_count: Count of positive words
        - negative_count: Count of negative words
        - ratio: (positive - negative) / (positive + negative)
        - sentiment_score: -1 to +1 scale
        """
        words = self.preprocess_text(text)
        
        positive_count = sum(1 for word in words if word in self.opinion_positive_words)
        negative_count = sum(1 for word in words if word in self.opinion_negative_words)
        
        opinion_results = {
            'Opinion_Lexicon_positive_count': positive_count,
            'Opinion_Lexicon_negative_count': negative_count,
        }
        
        # Calculate ratios
        total_words = len(words)
        opinion_results['Opinion_Lexicon_positive_ratio'] = (positive_count / total_words 
                                                             if total_words > 0 else 0)
        opinion_results['Opinion_Lexicon_negative_ratio'] = (negative_count / total_words 
                                                             if total_words > 0 else 0)
        
        # Calculate sentiment score (-1 to +1)
        if (positive_count + negative_count) > 0:
            opinion_results['Opinion_Lexicon_sentiment_score'] = (
                (positive_count - negative_count) / (positive_count + negative_count)
            )
        else:
            opinion_results['Opinion_Lexicon_sentiment_score'] = 0
        
        return opinion_results
    
    def analyze_text(self, text: str) -> Dict:
        """
        Perform comprehensive analysis on a text using all three dictionaries plus readability metrics
        """
        lm_results = self.analyze_loughran_mcdonald(text)
        vader_results = self.analyze_vader(text)
        opinion_results = self.analyze_opinion_lexicon(text)
        readability_results = self.calculate_readability_metrics(text)
        
        # Combine all results
        combined_results = {}
        combined_results.update(lm_results)
        combined_results.update(vader_results)
        combined_results.update(opinion_results)
        combined_results.update(readability_results)
        
        return combined_results
    
    def analyze_excel_column(self, excel_file: str, sheet_name: str, column_name: str, 
                            output_file: str = None) -> pd.DataFrame:
        """
        Analyze a column in an Excel file using all three dictionaries plus readability metrics
        
        Parameters:
        -----------
        excel_file : str
            Path to the Excel file
        sheet_name : str
            Name of the sheet
        column_name : str
            Name of the column to analyze
        output_file : str, optional
            Path to save the results to a new Excel file
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with original text and analysis results
        """
        # Read Excel file
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the Excel file")
        
        # Perform analysis on each row
        results_list = []
        for idx, text in enumerate(df[column_name]):
            print(f"Processing row {idx + 1}/{len(df)}")
            analysis_results = self.analyze_text(text)
            results_list.append(analysis_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Combine with original data
        output_df = pd.concat([df, results_df], axis=1)
        
        # Save to Excel if output file is specified
        if output_file:
            output_df.to_excel(output_file, index=False, sheet_name=sheet_name)
            print(f"Results saved to {output_file}")
        
        return output_df


def main():
    """
    Main function demonstrating how to use the TextualAnalyzer
    """
    # Initialize analyzer
    analyzer = TextualAnalyzer()
    
    # Example usage:
    # Replace these with your actual file paths and column name
    excel_file = "your_data.xlsx"  # Change to your Excel file
    sheet_name = "Sheet1"           # Change to your sheet name
    column_name = "text_column"     # Change to your column name
    output_file = "analysis_results.xlsx"
    
    # Check if file exists
    if not Path(excel_file).exists():
        print(f"Error: File '{excel_file}' not found.")
        print("\nExample usage:")
        print("================")
        print("# Create a sample Excel file first:")
        print("import pandas as pd")
        print('df = pd.DataFrame({"text_column": ["This is good text", "This is bad text"]})')
        print('df.to_excel("your_data.xlsx", index=False)')
        print("\n# Then run:")
        print("analyzer = TextualAnalyzer()")
        print("results_df = analyzer.analyze_excel_column(")
        print('    "your_data.xlsx",')
        print('    "Sheet1",')
        print('    "text_column",')
        print('    "analysis_results.xlsx"')
        print(")")
        print("print(results_df.head())")
        return
    
    # Analyze Excel column
    results_df = analyzer.analyze_excel_column(
        excel_file=excel_file,
        sheet_name=sheet_name,
        column_name=column_name,
        output_file=output_file
    )
    
    # Display results
    print("\nAnalysis Results:")
    print(results_df.head())
    print(f"\nShape: {results_df.shape}")
    print(f"\nColumns: {list(results_df.columns)}")


if __name__ == "__main__":
    main()
