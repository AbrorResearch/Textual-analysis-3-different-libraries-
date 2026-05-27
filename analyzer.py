"""
Textual Analysis Library
Core module for sentiment analysis and readability metrics
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon
from nltk.tokenize import sent_tokenize
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
    1. Loughran McDonald Dictionary (financial sentiment)
    2. VADER (Valence Aware Dictionary and sEntiment Reasoner)
    3. NLTK Opinion Lexicon (Harvard IV-4 based)
    4. Readability Metrics (FOG Index and Flesch-Kincaid Grade Level)
    """
    
    def __init__(self):
        """Initialize the analyzer with dictionaries"""
        self.lm_classifier = LMClassifier()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.opinion_positive_words = set(opinion_lexicon.positive())
        self.opinion_negative_words = set(opinion_lexicon.negative())
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text: convert to lowercase, remove punctuation, tokenize"""
        if pd.isna(text):
            return []
        
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = text.split()
        
        return words
    
    def count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word"""
        word = word.lower()
        syllable_count = 0
        vowels = "aeiou"
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        if word.endswith('e'):
            syllable_count -= 1
        
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1
        
        return max(1, syllable_count)
    
    def is_complex_word(self, word: str) -> bool:
        """Determine if a word is complex (3+ syllables)"""
        if len(word) <= 3:
            return False
        syllables = self.count_syllables(word)
        return syllables >= 3
    
    def calculate_fog_index(self, text: str) -> Dict:
        """Calculate Gunning FOG Index"""
        if pd.isna(text):
            return {'FOG_index': 0, 'FOG_complexity': 0}
        
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return {'FOG_index': 0, 'FOG_complexity': 0}
        
        words = self.preprocess_text(text)
        num_words = len(words)
        
        if num_words == 0:
            return {'FOG_index': 0, 'FOG_complexity': 0}
        
        complex_words = sum(1 for word in words if self.is_complex_word(word))
        fog_index = 0.4 * ((num_words / num_sentences) + 100 * (complex_words / num_words))
        complexity_percent = (complex_words / num_words) * 100
        
        return {
            'FOG_index': round(fog_index, 2),
            'FOG_complexity': round(complexity_percent, 2)
        }
    
    def calculate_flesch_kincaid(self, text: str) -> Dict:
        """Calculate Flesch-Kincaid Grade Level"""
        if pd.isna(text):
            return {'Flesch_Kincaid_Grade': 0}
        
        sentences = sent_tokenize(text)
        num_sentences = len(sentences)
        
        if num_sentences == 0:
            return {'Flesch_Kincaid_Grade': 0}
        
        words = self.preprocess_text(text)
        num_words = len(words)
        
        if num_words == 0:
            return {'Flesch_Kincaid_Grade': 0}
        
        total_syllables = sum(self.count_syllables(word) for word in words)
        flesch_kincaid = (0.39 * (num_words / num_sentences) + 
                          11.8 * (total_syllables / num_words) - 15.59)
        
        return {
            'Flesch_Kincaid_Grade': round(max(0, flesch_kincaid), 2)
        }
    
    def calculate_readability_metrics(self, text: str) -> Dict:
        """Calculate all readability metrics"""
        fog_results = self.calculate_fog_index(text)
        flesch_kincaid_results = self.calculate_flesch_kincaid(text)
        
        readability_results = {}
        readability_results.update(fog_results)
        readability_results.update(flesch_kincaid_results)
        
        return readability_results
    
    def analyze_loughran_mcdonald(self, text: str) -> Dict:
        """Analyze text using Loughran McDonald dictionary (Financial Sentiment)"""
        score = self.lm_classifier.score(text)
        
        lm_results = {
            'LM_positive_count': score.get('positive', 0),
            'LM_negative_count': score.get('negative', 0),
            'LM_uncertainty_count': score.get('uncertainty', 0),
            'LM_litigious_count': score.get('litigious', 0),
            'LM_constraining_count': score.get('constraining', 0),
            'LM_strong_modal_count': score.get('strong_modal', 0),
            'LM_weak_modal_count': score.get('weak_modal', 0),
        }
        
        words = self.preprocess_text(text)
        total_words = len(words)
        
        for key, value in lm_results.items():
            ratio_key = key.replace('_count', '_ratio')
            lm_results[ratio_key] = value / total_words if total_words > 0 else 0
        
        lm_results['LM_net_sentiment'] = (lm_results['LM_positive_count'] - 
                                          lm_results['LM_negative_count'])
        
        return lm_results
    
    def analyze_vader(self, text: str) -> Dict:
        """Analyze text using VADER sentiment analyzer"""
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
            'VADER_compound': round(scores['compound'], 4)
        }
    
    def analyze_opinion_lexicon(self, text: str) -> Dict:
        """Analyze text using NLTK Opinion Lexicon (Harvard IV-4 based)"""
        words = self.preprocess_text(text)
        
        positive_count = sum(1 for word in words if word in self.opinion_positive_words)
        negative_count = sum(1 for word in words if word in self.opinion_negative_words)
        
        opinion_results = {
            'Opinion_Lexicon_positive_count': positive_count,
            'Opinion_Lexicon_negative_count': negative_count,
        }
        
        total_words = len(words)
        opinion_results['Opinion_Lexicon_positive_ratio'] = (positive_count / total_words 
                                                             if total_words > 0 else 0)
        opinion_results['Opinion_Lexicon_negative_ratio'] = (negative_count / total_words 
                                                             if total_words > 0 else 0)
        
        if (positive_count + negative_count) > 0:
            opinion_results['Opinion_Lexicon_sentiment_score'] = (
                (positive_count - negative_count) / (positive_count + negative_count)
            )
        else:
            opinion_results['Opinion_Lexicon_sentiment_score'] = 0
        
        return opinion_results
    
    def analyze_text(self, text: str) -> Dict:
        """Perform comprehensive analysis on a text"""
        lm_results = self.analyze_loughran_mcdonald(text)
        vader_results = self.analyze_vader(text)
        opinion_results = self.analyze_opinion_lexicon(text)
        readability_results = self.calculate_readability_metrics(text)
        
        combined_results = {}
        combined_results.update(lm_results)
        combined_results.update(vader_results)
        combined_results.update(opinion_results)
        combined_results.update(readability_results)
        
        return combined_results
    
    def analyze_excel_column(self, excel_file: str, sheet_name: str, column_name: str, 
                            output_file: str = None) -> pd.DataFrame:
        """
        Analyze a column in an Excel file
        
        Parameters:
        -----------
        excel_file : str
            Path to the Excel file
        sheet_name : str
            Name of the sheet
        column_name : str
            Name of the column to analyze (e.g., 'B' or 'text')
        output_file : str, optional
            Path to save the results to a new Excel file
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with original data + analysis results
        """
        # Read Excel file
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        # Handle column selection (by name or letter)
        if column_name in df.columns:
            text_column = column_name
        elif column_name.upper() in [chr(65 + i) for i in range(26)]:
            # Convert letter to column index
            col_index = ord(column_name.upper()) - 65
            if col_index < len(df.columns):
                text_column = df.columns[col_index]
            else:
                raise ValueError(f"Column {column_name} does not exist")
        else:
            raise ValueError(f"Column '{column_name}' not found in the Excel file")
        
        print(f"Analyzing column: {text_column}")
        print(f"Total rows: {len(df)}")
        
        # Perform analysis on each row
        results_list = []
        for idx, text in enumerate(df[text_column]):
            if (idx + 1) % 10 == 0:
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
            print(f"✓ Results saved to {output_file}")
        
        return output_df
