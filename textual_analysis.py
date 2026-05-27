"""
Textual Analysis using Loughran McDonald, VADER, and NLTK Opinion Lexicon
This script analyzes text from an Excel file column using three different sentiment/lexicon dictionaries.

Dictionaries Used:
1. Loughran McDonald - Financial sentiment analysis (pyloughranmcdonald)
2. VADER - Valence Aware Dictionary and sEntiment Reasoner (from nltk)
3. NLTK Opinion Lexicon - General sentiment analysis (Harvard IV-4 based)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon
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


class TextualAnalyzer:
    """
    Performs textual analysis using three professional dictionaries:
    1. Loughran McDonald Dictionary (financial sentiment via pyloughranmcdonald)
    2. VADER (Valence Aware Dictionary and sEntiment Reasoner)
    3. NLTK Opinion Lexicon (Harvard IV-4 based)
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
            'VADER_negative': scores['neg'],
            'VADER_neutral': scores['neu'],
            'VADER_positive': scores['pos'],
            'VADER_compound': scores['compound']  # Key metric: -1 to +1
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
        Perform comprehensive analysis on a text using all three dictionaries
        """
        lm_results = self.analyze_loughran_mcdonald(text)
        vader_results = self.analyze_vader(text)
        opinion_results = self.analyze_opinion_lexicon(text)
        
        # Combine all results
        combined_results = {}
        combined_results.update(lm_results)
        combined_results.update(vader_results)
        combined_results.update(opinion_results)
        
        return combined_results
    
    def analyze_excel_column(self, excel_file: str, sheet_name: str, column_name: str, 
                            output_file: str = None) -> pd.DataFrame:
        """
        Analyze a column in an Excel file using all three dictionaries
        
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
