import warnings
warnings.simplefilter('always')

try:
    import numpy
    print(f"NumPy Version: {numpy.__version__}")
    
    import bottleneck
    print(f"Bottleneck Version: {bottleneck.__version__}")
    
    import scipy
    print(f"Scipy Version: {scipy.__version__}")
    
    from scipy.special import expit
    print("Scipy Special (expit) loaded successfully.")
    
    import pandas
    
    print("SUCCESS: Library checks completed without crash.")
except Exception as e:
    print(f"FAIL: Error: {e}")
