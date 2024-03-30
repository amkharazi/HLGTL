# Check Test Plan for more details 
# Test VGG19 model on MNIST dataset
# No change to classifier -Basic Model
# Optimizer Adam - Default
# No Scheduler
# MNIST dataset -> (3, 192, 192) 
# Pretrained - Frozen Non Classifier Parameters
# Transfer Learning - Classifier parameters
########################################################

# Add all .py files to path
import sys
sys.path.append('..')

# Import Libraries
from Utils.Accuracy_measures import topk_accuracy



