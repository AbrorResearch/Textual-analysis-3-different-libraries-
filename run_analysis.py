"""
Quick Start Script - Run this file to analyze your Excel data
"""

from analyzer import TextualAnalyzer

# ==================== CONFIGURATION ====================
# UPDATE THESE THREE LINES WITH YOUR FILE INFORMATION

INPUT_FILE = "your_file.xlsx"           # Your Excel file name/path
SHEET_NAME = "Sheet1"                   # Your sheet name (usually "Sheet1")
TEXT_COLUMN = "B"                       # Column with text (use 'B' for column B, or column name)
OUTPUT_FILE = "analysis_results.xlsx"   # Output file name

# ========================================================

def main():
    print("=" * 80)
    print("TEXTUAL ANALYSIS TOOL")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input file:  {INPUT_FILE}")
    print(f"  Sheet:       {SHEET_NAME}")
    print(f"  Text column: {TEXT_COLUMN}")
    print(f"  Output file: {OUTPUT_FILE}")
    print("\n" + "=" * 80)
    print("Starting analysis...")
    print("=" * 80 + "\n")
    
    try:
        # Initialize analyzer
        analyzer = TextualAnalyzer()
        
        # Run analysis
        results_df = analyzer.analyze_excel_column(
            excel_file=INPUT_FILE,
            sheet_name=SHEET_NAME,
            column_name=TEXT_COLUMN,
            output_file=OUTPUT_FILE
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\n✓ Total rows analyzed: {len(results_df)}")
        print(f"✓ Total columns: {len(results_df.columns)}")
        print(f"✓ Output file: {OUTPUT_FILE}")
        
        print("\n📊 Output Columns:")
        print("-" * 80)
        for i, col in enumerate(results_df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        print("\n" + "=" * 80)
        print("KEY METRICS IN OUTPUT:")
        print("=" * 80)
        print("""
✓ LM_net_sentiment        - Loughran McDonald financial sentiment
✓ VADER_compound          - VADER sentiment score (-1 to +1)
✓ Opinion_Lexicon_sentiment_score - Harvard-based sentiment score
✓ FOG_index               - Years of education needed to read
✓ Flesch_Kincaid_Grade    - US grade level needed to read

Interpretation Guide:
  - Sentiment scores:
    * +1.0 = Completely positive
    * 0.0  = Neutral
    * -1.0 = Completely negative
  
  - Readability:
    * 6-8   = Easy (elementary school)
    * 9-12  = Medium (high school)
    * 13+   = Difficult (college+)
        """)
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File '{INPUT_FILE}' not found!")
        print(f"   Make sure the file exists in your current directory")
        print(f"   Or provide the full path: C:/Users/YourName/Documents/file.xlsx")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"   Please check your configuration and try again")

if __name__ == "__main__":
    main()
