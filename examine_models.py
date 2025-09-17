import joblib
import os

def examine_model_files():
    """Examine the model files to understand their structure"""
    
    files_to_check = [
        'best_llm_model.obj',
        'label_encoder.obj', 
        'model_metadata.obj',
        'tokenizer.obj'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            print(f"\n{'='*50}")
            print(f"Examining: {file}")
            print(f"{'='*50}")
            
            try:
                obj = joblib.load(file)
                print(f"Type: {type(obj)}")
                print(f"Object: {obj}")
                
                # Try to get more info about the object
                if hasattr(obj, '__dict__'):
                    print(f"Attributes: {list(obj.__dict__.keys())}")
                
                if hasattr(obj, 'shape'):
                    print(f"Shape: {obj.shape}")
                    
                if hasattr(obj, 'classes_'):
                    print(f"Classes: {obj.classes_}")
                    
                if isinstance(obj, dict):
                    print(f"Dictionary keys: {list(obj.keys())}")
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
        else:
            print(f"File not found: {file}")

if __name__ == "__main__":
    examine_model_files()
