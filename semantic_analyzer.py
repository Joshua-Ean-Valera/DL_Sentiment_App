"""
Sentiment Analysis System for Deep Learning Models
Compatible with Python 3.13.5
Classifies text into: Positive, Negative, or Neutral sentiment
"""

import joblib
import numpy as np
import pandas as pd
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import json
import re

warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """
    Sentiment Analysis System using Deep Learning models
    Classifies text sentiment as Positive, Negative, or Neutral
    """
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the Sentiment Analyzer
        
        Args:
            model_dir: Directory containing the model files
        """
        self.model_dir = model_dir or os.path.dirname(os.path.abspath(__file__))
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.metadata = None
        
        # Sentiment mappings
        self.sentiment_labels = ['negative', 'neutral', 'positive']
        
        # Load all models
        self.load_models()
        
    def load_models(self) -> bool:
        """Load all required model components"""
        try:
            print("🔄 Loading Sentiment Analysis Models...")
            
            # Define model file paths
            model_files = {
                'model': 'best_llm_model.obj',
                'tokenizer': 'tokenizer.obj',
                'label_encoder': 'label_encoder.obj',
                'metadata': 'model_metadata.obj'
            }
            
            # Load each component
            for component, filename in model_files.items():
                filepath = os.path.join(self.model_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        loaded_obj = joblib.load(filepath)
                        setattr(self, component, loaded_obj)
                        print(f"✅ Loaded {component}: {type(loaded_obj).__name__}")
                        
                        # Debug information
                        if component == 'model' and hasattr(loaded_obj, 'classes_'):
                            print(f"   Model classes: {loaded_obj.classes_}")
                        elif component == 'label_encoder' and hasattr(loaded_obj, 'classes_'):
                            print(f"   Label encoder classes: {loaded_obj.classes_}")
                        elif component == 'metadata' and isinstance(loaded_obj, dict):
                            print(f"   Metadata keys: {list(loaded_obj.keys())}")
                            
                    except Exception as e:
                        print(f"⚠️  Error loading {component}: {e}")
                        continue
                else:
                    print(f"❌ File not found: {filepath}")
            
            # Verify models are loaded
            if self.model is not None:
                print(f"✅ Sentiment model successfully loaded!")
                return True
            else:
                print("❌ Failed to load main sentiment model")
                return False
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Basic preprocessing for sentiment analysis
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\w\s.,!?-]', '', text)  # Remove special characters but keep punctuation
        
        return text
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment of text (Positive, Negative, or Neutral)
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing sentiment prediction results
        """
        try:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {
                    'error': 'Empty text after preprocessing',
                    'original_text': text
                }
            
            # Initialize result structure
            result = {
                'original_text': text,
                'processed_text': processed_text,
                'timestamp': datetime.now().isoformat(),
                'sentiment': 'neutral',
                'confidence': 0.0,
                'probabilities': {
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 0.0
                }
            }
            
            # Prepare text for model
            text_features = self._prepare_text_features(processed_text)
            
            # Perform prediction using the model
            if self.model is not None:
                try:
                    # Handle different model types
                    if hasattr(self.model, 'predict'):
                        # Make prediction
                        prediction = self.model.predict(text_features)
                        
                        # Get probabilities if available
                        if hasattr(self.model, 'predict_proba'):
                            probabilities = self.model.predict_proba(text_features)
                            
                            # Map to sentiment labels
                            sentiment_probs = self._map_probabilities_to_sentiment(probabilities[0])
                            result['probabilities'] = sentiment_probs
                            result['confidence'] = max(sentiment_probs.values())
                        
                        # Decode prediction to sentiment label
                        sentiment = self._decode_prediction_to_sentiment(prediction[0])
                        result['sentiment'] = sentiment
                        
                    else:
                        result['error'] = 'Model does not support prediction'
                        
                except Exception as e:
                    result['error'] = f"Prediction error: {e}"
                    
            else:
                result['error'] = 'No model loaded'
            
            return result
            
        except Exception as e:
            return {
                'error': f"Sentiment analysis failed: {e}",
                'original_text': text,
                'timestamp': datetime.now().isoformat()
            }
    
    def _prepare_text_features(self, text: str):
        """Prepare text features for the model"""
        try:
            # Use tokenizer if available
            if self.tokenizer is not None:
                if hasattr(self.tokenizer, 'transform'):
                    return self.tokenizer.transform([text])
                elif hasattr(self.tokenizer, 'encode'):
                    encoded = self.tokenizer.encode(text, return_tensors='np')
                    return encoded if hasattr(encoded, 'shape') else [text]
                elif callable(self.tokenizer):
                    return self.tokenizer([text])
                else:
                    return [text]
            else:
                return [text]
        except Exception as e:
            print(f"⚠️  Tokenizer error: {e}")
            return [text]
    
    def _map_probabilities_to_sentiment(self, probabilities) -> Dict[str, float]:
        """Map model probabilities to sentiment labels"""
        try:
            # Default mapping
            sentiment_probs = {
                'negative': 0.0,
                'neutral': 0.0,
                'positive': 0.0
            }
            
            # Check if we have model classes
            if hasattr(self.model, 'classes_'):
                classes = self.model.classes_
                for i, class_name in enumerate(classes):
                    if i < len(probabilities):
                        class_str = str(class_name).lower()
                        if 'positive' in class_str or 'pos' in class_str or class_str == '1':
                            sentiment_probs['positive'] = float(probabilities[i])
                        elif 'negative' in class_str or 'neg' in class_str or class_str == '0':
                            sentiment_probs['negative'] = float(probabilities[i])
                        elif 'neutral' in class_str or 'neu' in class_str or class_str == '2':
                            sentiment_probs['neutral'] = float(probabilities[i])
            else:
                # Fallback: assume order is negative, neutral, positive
                if len(probabilities) >= 3:
                    sentiment_probs['negative'] = float(probabilities[0])
                    sentiment_probs['neutral'] = float(probabilities[1])
                    sentiment_probs['positive'] = float(probabilities[2])
                elif len(probabilities) == 2:
                    # Binary classification: negative, positive
                    sentiment_probs['negative'] = float(probabilities[0])
                    sentiment_probs['positive'] = float(probabilities[1])
                    sentiment_probs['neutral'] = 0.0
            
            return sentiment_probs
            
        except Exception as e:
            print(f"⚠️  Error mapping probabilities: {e}")
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33}
    
    def _decode_prediction_to_sentiment(self, prediction) -> str:
        """Decode model prediction to sentiment label"""
        try:
            # Handle label encoder
            if self.label_encoder is not None and hasattr(self.label_encoder, 'inverse_transform'):
                try:
                    decoded = self.label_encoder.inverse_transform([prediction])[0]
                    decoded_str = str(decoded).lower()
                    
                    # Map to standard sentiment labels
                    if 'positive' in decoded_str or 'pos' in decoded_str:
                        return 'positive'
                    elif 'negative' in decoded_str or 'neg' in decoded_str:
                        return 'negative'
                    else:
                        return 'neutral'
                except:
                    pass
            
            # Handle direct model classes
            if hasattr(self.model, 'classes_'):
                classes = self.model.classes_
                if prediction < len(classes):
                    class_name = str(classes[prediction]).lower()
                    if 'positive' in class_name or 'pos' in class_name or class_name == '1':
                        return 'positive'
                    elif 'negative' in class_name or 'neg' in class_name or class_name == '0':
                        return 'negative'
                    else:
                        return 'neutral'
            
            # Fallback: numeric mapping
            prediction_val = int(prediction) if isinstance(prediction, (int, float, np.integer, np.floating)) else 1
            if prediction_val == 0:
                return 'negative'
            elif prediction_val == 1:
                return 'neutral'
            elif prediction_val == 2:
                return 'positive'
            else:
                return 'neutral'
                
        except Exception as e:
            print(f"⚠️  Error decoding prediction: {e}")
            return 'neutral'
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        print(f"🔄 Analyzing sentiment of {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            print(f"   Processing {i+1}/{len(texts)}")
            result = self.predict_sentiment(text)
            results.append(result)
        
        print("✅ Batch sentiment analysis complete!")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded sentiment models
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'timestamp': datetime.now().isoformat(),
            'model_directory': self.model_dir,
            'sentiment_labels': self.sentiment_labels,
            'components': {}
        }
        
        # Check each component
        components = ['model', 'tokenizer', 'label_encoder', 'metadata']
        
        for component in components:
            obj = getattr(self, component, None)
            if obj is not None:
                comp_info = {
                    'loaded': True,
                    'type': type(obj).__name__
                }
                
                # Add specific information based on object type
                if hasattr(obj, 'classes_'):
                    comp_info['classes'] = obj.classes_.tolist() if hasattr(obj.classes_, 'tolist') else list(obj.classes_)
                
                if hasattr(obj, 'shape'):
                    comp_info['shape'] = obj.shape
                
                if isinstance(obj, dict):
                    comp_info['keys'] = list(obj.keys())
                
                info['components'][component] = comp_info
            else:
                info['components'][component] = {'loaded': False}
        
        return info
    
    def export_results(self, results: List[Dict], filename: str = None) -> str:
        """
        Export sentiment analysis results to JSON file
        
        Args:
            results: List of sentiment analysis results
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sentiment_analysis_results_{timestamp}.json"
        
        filepath = os.path.join(self.model_dir, filename)
        
        # Prepare data for export
        export_data = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_texts_analyzed': len(results),
                'sentiment_labels': self.sentiment_labels,
                'analyzer_info': self.get_model_info()
            },
            'results': results
        }
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Sentiment analysis results exported to: {filepath}")
        return filepath


def main():
    """
    Main function to demonstrate the Sentiment Analyzer
    """
    print("🚀 Starting Sentiment Analysis System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Display model information
    print("\n📊 Model Information:")
    model_info = analyzer.get_model_info()
    for component, info in model_info['components'].items():
        status = "✅ Loaded" if info['loaded'] else "❌ Not loaded"
        print(f"   {component}: {status}")
        if info['loaded']:
            print(f"      Type: {info['type']}")
    
    # Sample texts for sentiment analysis
    sample_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is terrible. I hate it and want my money back.",
        "The weather is okay today, nothing special.",
        "I'm feeling quite disappointed with the service quality.",
        "Excellent work! You've exceeded my expectations completely.",
        "It's not bad, but it's not great either.",
        "I'm so happy with this purchase!",
        "This is the worst experience I've ever had.",
        "The product is fine, does what it's supposed to do."
    ]
    
    print(f"\n🔍 Analyzing sentiment of {len(sample_texts)} sample texts:")
    print("-" * 40)
    
    # Analyze each text
    results = []
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{i}. Text: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        
        result = analyzer.predict_sentiment(text)
        results.append(result)
        
        if 'error' not in result:
            sentiment = result['sentiment']
            confidence = result['confidence']
            
            # Emoji for sentiment
            emoji = "😊" if sentiment == 'positive' else "😞" if sentiment == 'negative' else "😐"
            
            print(f"   {emoji} Sentiment: {sentiment.upper()}")
            print(f"   🎯 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
            
            # Show probabilities
            probs = result['probabilities']
            print(f"   📊 Probabilities: Pos:{probs['positive']:.2f} | Neu:{probs['neutral']:.2f} | Neg:{probs['negative']:.2f}")
        else:
            print(f"   ❌ Error: {result['error']}")
    
    # Export results
    export_path = analyzer.export_results(results)
    
    print(f"\n✅ Sentiment analysis complete!")
    print(f"📄 Detailed results saved to: {export_path}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
