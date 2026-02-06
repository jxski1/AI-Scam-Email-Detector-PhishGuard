import re
import html

def clean_text(text):
    """Simple text cleaning: remove HTML entities, URLs, punctuation, and extra whitespace."""
    if not isinstance(text, str):
        return ""
    
    text = html.unescape(text)
    
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    
    text = re.sub(r'\S+@\S+', ' ', text)
    
    text = re.sub(r'[^\w\s]', ' ', text)
    
    text = re.sub(r'\d+', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip().lower()
    
    return text
