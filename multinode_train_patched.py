#!/usr/bin/env python
"""
Wrapper script that applies the torchvision patch before running the training
"""
# MUST BE FIRST - Apply the patch before any other imports
import startup_patch

# Now run the original training script
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Import and run the training
from multinode_train import *

# Call main if it exists
if __name__ == "__main__":
    main() 