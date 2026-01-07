# Minimal import test
import sys
sys.path.insert(0, '.')

try:
    from src.world_model_training import WorldModelTrainingConfig
    print("OK!")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
