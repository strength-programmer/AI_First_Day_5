import pandas as pd
from typing import Optional

class DataProcessor:
    FEEDBACK_SYNONYMS = {
        'feedback', 'comment', 'review', 'response', 'sentiment', 'opinion', 
        'text', 'message', 'note', 'input', 'comments', 'reviews', 'feedback_text',
        'customer_feedback', 'user_feedback', 'description', 'remarks'
    }

    def detect_feedback_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect the most likely feedback column from the DataFrame.
        Returns the column name or None if no suitable column is found.
        """
        # Convert all column names to lowercase for comparison
        columns_lower = {col.lower(): col for col in df.columns}
        
        # First, look for exact matches
        for synonym in self.FEEDBACK_SYNONYMS:
            if synonym in columns_lower:
                return columns_lower[synonym]
        
        # Then, look for partial matches
        for col_lower, col_original in columns_lower.items():
            for synonym in self.FEEDBACK_SYNONYMS:
                if synonym in col_lower or col_lower in synonym:
                    return col_original
        
        # If no matches found, try to find text-heavy columns
        text_col = self._find_text_column(df)
        if text_col:
            return text_col
            
        return None

    def _find_text_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the column that most likely contains text feedback
        based on average string length and unique value ratio.
        """
        text_scores = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':  # Only consider string columns
                # Calculate average string length
                avg_len = df[col].astype(str).str.len().mean()
                # Calculate unique value ratio
                unique_ratio = len(df[col].unique()) / len(df)
                # Score based on length and uniqueness
                text_scores[col] = avg_len * unique_ratio
        
        if text_scores:
            # Return column with highest score
            return max(text_scores.items(), key=lambda x: x[1])[0]
        return None

    def load_data(self, file):
        try:
            if not file.name.endswith('.csv'):
                raise ValueError("Please upload a CSV file.")
            
            df = pd.read_csv(file)
            
            # Detect feedback column
            feedback_column = self.detect_feedback_column(df)
            
            if not feedback_column:
                raise ValueError("Could not identify a suitable feedback column in the uploaded file. "
                               "Please ensure your file contains a column with customer feedback/comments.")
            
            # Rename the detected column to 'feedback' for consistency
            df = df.rename(columns={feedback_column: 'feedback'})
            
            # Basic data cleaning
            df['feedback'] = df['feedback'].astype(str)
            df['feedback'] = df['feedback'].apply(self.preprocess_text)
            
            # Remove empty feedback
            df = df[df['feedback'].str.strip() != '']
            
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError("The uploaded file is empty")
        except pd.errors.ParserError:
            raise ValueError("Unable to parse the file. Please ensure it's a valid CSV file")
        except Exception as e:
            raise ValueError(f"Error processing file: {str(e)}")

    def preprocess_text(self, text):
        # Add text preprocessing logic here
        return text.lower().strip() 