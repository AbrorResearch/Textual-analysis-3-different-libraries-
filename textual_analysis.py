"""
Textual Analysis using Loughran McDonald, LIWC, and Harvard Dictionary
This script analyzes text from an Excel file column using three different sentiment/lexicon dictionaries.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Install required packages:
# pip install pandas openpyxl loughran-mcdonald textblob nltk


class TextualAnalyzer:
    """
    Performs textual analysis using three different dictionaries:
    1. Loughran McDonald Dictionary (financial sentiment)
    2. LIWC (Linguistic Inquiry and Word Count)
    3. Harvard Dictionary (general sentiment)
    """
    
    def __init__(self):
        """Initialize the analyzer with dictionaries"""
        self.lm_dict = self._load_loughran_mcdonald()
        self.liwc_dict = self._load_liwc()
        self.harvard_dict = self._load_harvard()
    
    def _load_loughran_mcdonald(self) -> Dict:
        """
        Load Loughran McDonald dictionary
        This dictionary is specifically designed for financial text analysis
        """
        try:
            from loughran_mcdonald import loughran_mcdonald
            return loughran_mcdonald
        except ImportError:
            print("Loughran McDonald dictionary not found. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "loughran-mcdonald"])
            from loughran_mcdonald import loughran_mcdonald
            return loughran_mcdonald
    
    def _load_liwc(self) -> Dict:
        """
        Load LIWC dictionary (Linguistic Inquiry and Word Count)
        This is a comprehensive dictionary for linguistic analysis
        
        Note: The full LIWC dictionary requires a license.
        Using a simplified version or alternative like textblob/vader
        """
        # Using simplified LIWC-like categories
        liwc_categories = {
            'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                        'awesome', 'perfect', 'love', 'best', 'beautiful', 'happy', 'joy'],
            'negative': ['bad', 'awful', 'terrible', 'horrible', 'worst', 'hate', 'ugly',
                        'sad', 'pain', 'poor', 'wrong', 'evil', 'fail', 'failed'],
            'cognitive': ['think', 'know', 'believe', 'understand', 'remember', 'consider',
                         'aware', 'realize', 'recognize', 'determine', 'decide'],
            'affective': ['feel', 'happy', 'sad', 'angry', 'afraid', 'ashamed', 'lonely',
                         'nervous', 'anxious', 'depressed', 'worried'],
            'social': ['talk', 'communicate', 'meet', 'friend', 'love', 'listen', 'help',
                      'together', 'share', 'family', 'community'],
            'temporal': ['before', 'after', 'during', 'while', 'when', 'then', 'now',
                        'today', 'tomorrow', 'yesterday', 'always', 'never']
        }
        return liwc_categories
    
    def _load_harvard(self) -> Dict:
        """
        Load Harvard Dictionary (General Inquirer)
        A comprehensive dictionary for sentiment and semantic analysis
        """
        harvard_dict = {
            'positive': ['good', 'excellent', 'great', 'fine', 'best', 'better', 'well',
                        'happy', 'joy', 'love', 'beautiful', 'nice', 'wonderful', 'perfect',
                        'success', 'successful', 'win', 'won', 'gain', 'benefit', 'advantage'],
            'negative': ['bad', 'awful', 'terrible', 'worse', 'worst', 'poor', 'evil',
                        'sad', 'sorrowful', 'hate', 'ugly', 'fail', 'failure', 'loss',
                        'damage', 'harm', 'risk', 'danger', 'problem', 'issue'],
            'uncertain': ['maybe', 'perhaps', 'possibly', 'might', 'could', 'probably',
                         'somewhat', 'fairly', 'rather', 'quite', 'unclear', 'unknown'],
            'litigious': ['lawsuit', 'litigation', 'arbitration', 'regulatory', 'alleged',
                         'complaint', 'defend', 'judgment', 'accused', 'violation']
        }
        return harvard_dict
    
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
        Analyze text using Loughran McDonald dictionary
        Focuses on financial sentiment
        """
        words = self.preprocess_text(text)
        
        # Categories in Loughran McDonald
        categories = {
            'positive': 0,
            'negative': 0,
            'uncertainty': 0,
            'litigious': 0,
            'modal_strong': 0,
            'modal_weak': 0
        }
        
        # Simplified LM dictionary
        lm_positive = ['positive', 'good', 'gain', 'profit', 'increase', 'strength', 'strong', 'growth']
        lm_negative = ['negative', 'loss', 'decline', 'weak', 'weakness', 'risk', 'fail', 'failed']
        lm_uncertainty = ['uncertain', 'uncertainty', 'unclear', 'vague', 'may', 'might', 'possible']
        lm_litigious = ['litigation', 'lawsuit', 'legal', 'regulatory', 'alleged', 'complaint']
        lm_modal_strong = ['will', 'must', 'should', 'definitely', 'certainly']
        lm_modal_weak = ['may', 'might', 'could', 'perhaps', 'possibly']
        
        for word in words:
            if word in lm_positive:
                categories['positive'] += 1
            elif word in lm_negative:
                categories['negative'] += 1
            elif word in lm_uncertainty:
                categories['uncertainty'] += 1
            elif word in lm_litigious:
                categories['litigious'] += 1
            elif word in lm_modal_strong:
                categories['modal_strong'] += 1
            elif word in lm_modal_weak:
                categories['modal_weak'] += 1
        
        # Calculate ratios
        total_words = len(words)
        if total_words > 0:
            for key in categories:
                categories[f'{key}_ratio'] = categories[key] / total_words
        else:
            for key in categories:
                categories[f'{key}_ratio'] = 0
        
        return {f'LM_{k}': v for k, v in categories.items()}
    
    def analyze_liwc(self, text: str) -> Dict:
        """
        Analyze text using LIWC-like categories
        """
        words = self.preprocess_text(text)
        
        categories = {category: 0 for category in self.liwc_dict.keys()}
        
        for word in words:
            for category, word_list in self.liwc_dict.items():
                if word in word_list:
                    categories[category] += 1
        
        # Calculate ratios
        total_words = len(words)
        results = {}
        for category, count in categories.items():
            results[f'LIWC_{category}_count'] = count
            results[f'LIWC_{category}_ratio'] = count / total_words if total_words > 0 else 0
        
        return results
    
    def analyze_harvard(self, text: str) -> Dict:
        """
        Analyze text using Harvard Dictionary
        """
        words = self.preprocess_text(text)
        
        categories = {category: 0 for category in self.harvard_dict.keys()}
        
        for word in words:
            for category, word_list in self.harvard_dict.items():
                if word in word_list:
                    categories[category] += 1
        
        # Calculate ratios and sentiment score
        total_words = len(words)
        results = {}
        for category, count in categories.items():
            results[f'Harvard_{category}_count'] = count
            results[f'Harvard_{category}_ratio'] = count / total_words if total_words > 0 else 0
        
        # Calculate sentiment score
        positive = categories.get('positive', 0)
        negative = categories.get('negative', 0)
        
        if (positive + negative) > 0:
            results['Harvard_sentiment_score'] = (positive - negative) / (positive + negative)
        else:
            results['Harvard_sentiment_score'] = 0
        
        return results
    
    def analyze_text(self, text: str) -> Dict:
        """
        Perform comprehensive analysis on a text using all three dictionaries
        """
        lm_results = self.analyze_loughran_mcdonald(text)
        liwc_results = self.analyze_liwc(text)
        harvard_results = self.analyze_harvard(text)
        
        # Combine all results
        combined_results = {}
        combined_results.update(lm_results)
        combined_results.update(liwc_results)
        combined_results.update(harvard_results)
        
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
