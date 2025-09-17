"""
Web Interface for Sentiment Analysis System
Flask application for sentiment classification (Positive, Negative, Neutral)
"""

from flask import Flask, render_template, request, jsonify
import os
from semantic_analyzer import SentimentAnalyzer
from datetime import datetime
import json

app = Flask(__name__)

# Initialize the sentiment analyzer
analyzer = None

def initialize_analyzer():
    """Initialize the sentiment analyzer"""
    global analyzer
    try:
        print("Initializing Sentiment Analyzer...")
        analyzer = SentimentAnalyzer()
        print("Sentiment Analyzer initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize analyzer: {e}")
        return False

@app.route('/')
def home():
    """Home page with sentiment analysis interface"""
    return render_template('sentiment_analyzer.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze sentiment of submitted text"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'error': 'Please provide text to analyze'
            }), 400
        
        if analyzer is None:
            return jsonify({
                'error': 'Sentiment analyzer not initialized'
            }), 500
        
        # Perform sentiment analysis
        result = analyzer.predict_sentiment(text)
        
        # Add status
        result['status'] = 'success'
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Sentiment analysis failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze sentiment of multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts or not isinstance(texts, list):
            return jsonify({
                'error': 'Please provide a list of texts to analyze'
            }), 400
        
        if analyzer is None:
            return jsonify({
                'error': 'Sentiment analyzer not initialized'
            }), 500
        
        # Perform batch sentiment analysis
        results = analyzer.analyze_batch(texts)
        
        response = {
            'results': results,
            'total_analyzed': len(results),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Batch sentiment analysis failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model_info')
def model_info():
    """Get information about the loaded models"""
    try:
        if analyzer is None:
            return jsonify({
                'error': 'Semantic analyzer not initialized'
            }), 500
        
        info = analyzer.get_model_info()
        info['status'] = 'success'
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({
            'error': f'Failed to get model info: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    analyzer_status = 'initialized' if analyzer is not None else 'not_initialized'
    
    health_info = {
        'status': 'healthy',
        'analyzer_status': analyzer_status,
        'timestamp': datetime.now().isoformat()
    }
    
    if analyzer is not None:
        model_info = analyzer.get_model_info()
        health_info['models_loaded'] = {
            component: info['loaded'] 
            for component, info in model_info['components'].items()
        }
    
    return jsonify(health_info)

@app.route('/export_results', methods=['POST'])
def export_results():
    """Export sentiment analysis results"""
    try:
        data = request.get_json()
        results = data.get('results', [])
        filename = data.get('filename', None)
        
        if not results:
            return jsonify({
                'error': 'No results to export'
            }), 400
        
        if analyzer is None:
            return jsonify({
                'error': 'Sentiment analyzer not initialized'
            }), 500
        
        # Export results
        export_path = analyzer.export_results(results, filename)
        
        return jsonify({
            'status': 'success',
            'message': 'Results exported successfully',
            'file_path': export_path,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Export failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("Starting Sentiment Analysis Web Application")
    print("=" * 70)
    
    # Initialize the analyzer
    if initialize_analyzer():
        print("Ready to analyze sentiment! (Positive, Negative, Neutral)")
        print("Starting Flask server...")
        print("Access the web interface at: http://localhost:5001")
        print("-" * 70)
        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("Failed to initialize sentiment analyzer.")
        print("Please ensure all model files are present in the directory.")
        print("Required files:")
        print("- best_llm_model.obj")
        print("- tokenizer.obj") 
        print("- label_encoder.obj")
        print("- model_metadata.obj")
