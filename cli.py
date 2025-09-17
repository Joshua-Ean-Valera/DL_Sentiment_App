"""
Command Line Interface for Sentiment Analysis System
Provides direct command-line access to sentiment classification (Positive, Negative, Neutral)
"""

import argparse
import sys
import json
from semantic_analyzer import SentimentAnalyzer
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis System - Classify text as Positive, Negative, or Neutral",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "I love this product!"
  python cli.py --file input.txt
  python cli.py --batch "Great service" "Terrible quality" "It's okay"
  python cli.py --interactive
        """
    )
    
    # Text input options
    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('--file', '-f', help='File containing text to analyze')
    parser.add_argument('--batch', '-b', nargs='+', help='Multiple texts to analyze')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    # Output options
    parser.add_argument('--output', '-o', help='Output file for results (JSON format)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    # Model options
    parser.add_argument('--model-info', action='store_true', help='Show model information')
    
    args = parser.parse_args()
    
    # Print header
    if not args.quiet:
        print("😊😐😞 Sentiment Analysis System")
        print("=" * 50)
    
    # Initialize analyzer
    try:
        if not args.quiet:
            print("🔄 Loading sentiment models...")
        analyzer = SentimentAnalyzer()
        if not args.quiet:
            print("✅ Sentiment models loaded successfully!")
            print()
    except Exception as e:
        print(f"❌ Error initializing sentiment analyzer: {e}")
        sys.exit(1)
    
    # Handle model info request
    if args.model_info:
        show_model_info(analyzer, args.verbose)
        return
    
    # Handle interactive mode
    if args.interactive:
        interactive_mode(analyzer)
        return
    
    # Determine input method
    texts_to_analyze = []
    
    if args.text:
        texts_to_analyze = [args.text]
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    texts_to_analyze = [content]
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            sys.exit(1)
    elif args.batch:
        texts_to_analyze = args.batch
    else:
        parser.print_help()
        sys.exit(1)
    
    if not texts_to_analyze:
        print("❌ No text provided for analysis")
        sys.exit(1)
    
    # Perform analysis
    results = []
    
    for i, text in enumerate(texts_to_analyze):
        if not args.quiet and len(texts_to_analyze) > 1:
            print(f"🔍 Analyzing text {i+1}/{len(texts_to_analyze)}...")
        
        result = analyzer.predict_sentiment(text)
        results.append(result)
        
        if not args.quiet:
            display_result(result, args.verbose)
            if i < len(texts_to_analyze) - 1:
                print("-" * 50)
    
    # Save results if requested
    if args.output:
        save_results(results, args.output, analyzer)
        if not args.quiet:
            print(f"💾 Results saved to: {args.output}")

def show_model_info(analyzer, verbose=False):
    """Display model information"""
    info = analyzer.get_model_info()
    
    print("📊 Model Information:")
    print(f"Directory: {info['model_directory']}")
    print(f"Last updated: {info['timestamp']}")
    print()
    
    print("Components:")
    for component, details in info['components'].items():
        status = "✅" if details['loaded'] else "❌"
        print(f"  {status} {component.title()}")
        
        if details['loaded'] and verbose:
            print(f"    Type: {details['type']}")
            if 'classes' in details:
                print(f"    Classes: {details['classes']}")
            if 'shape' in details:
                print(f"    Shape: {details['shape']}")
            if 'keys' in details:
                print(f"    Keys: {details['keys']}")

def display_result(result, verbose=False):
    """Display sentiment analysis result"""
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Basic information
    text = result['original_text']
    if len(text) > 80:
        display_text = text[:77] + "..."
    else:
        display_text = text
    
    print(f"📝 Text: \"{display_text}\"")
    
    # Sentiment results
    if 'sentiment' in result:
        sentiment = result['sentiment']
        emoji = get_sentiment_emoji(sentiment)
        print(f"{emoji} Sentiment: {sentiment.upper()}")
    
    if 'confidence' in result:
        confidence = result['confidence']
        print(f"🎯 Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
    
    if 'probabilities' in result and verbose:
        print("📊 Probability Distribution:")
        for sentiment, prob in result['probabilities'].items():
            emoji = get_sentiment_emoji(sentiment)
            bar_length = int(prob * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"   {emoji} {sentiment.title()}: {prob:.3f} [{bar}]")

def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment"""
    sentiment_lower = sentiment.lower()
    if sentiment_lower == 'positive':
        return "😊"
    elif sentiment_lower == 'negative':
        return "😞"
    else:
        return "😐"

def interactive_mode(analyzer):
    """Interactive sentiment analysis mode"""
    print("🔄 Interactive Sentiment Analysis Mode - Enter 'quit' to exit")
    print("-" * 50)
    
    while True:
        try:
            text = input("\n📝 Enter text to analyze sentiment: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                print("⚠️  Please enter some text")
                continue
            
            print("🔍 Analyzing sentiment...")
            result = analyzer.predict_sentiment(text)
            print()
            display_result(result, verbose=True)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def save_results(results, filename, analyzer):
    """Save results to JSON file"""
    export_data = {
        'metadata': {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_texts_analyzed': len(results),
            'analyzer_version': '1.0',
            'model_info': analyzer.get_model_info()
        },
        'results': results
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
