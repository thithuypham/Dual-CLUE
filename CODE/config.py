"""
Configuration Parameters for Dual-CLUE: Underwater Image Enhancement

This file centralizes all configuration parameters for the Dual-CLUE project,
including paths, model architecture settings, and hyperparameters for
training and testing.

Configuration is organized into logical sections:
- System: Device and hardware settings
- Paths: Data directories and file paths
- Model: Architecture and model parameters
- Training: Training hyperparameters
- Testing: Testing parameters
- Preprocessing: Parameters for preprocessing
"""
import os
import torch

# System Configuration
class SystemConfig:
    """System and hardware settings."""
    # CUDA settings
    CUDA_DEVICE_ORDER = "PCI_BUS_ID"
    CUDA_VISIBLE_DEVICES = "0"  # comma-separated device IDs, e.g., "0,1,2"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Random seed for reproducibility
    SEED = 42
    
    # Multi-GPU settings
    USE_DATA_PARALLEL = True
    
    # Number of workers for data loading
    NUM_WORKERS = 4

# Path Configuration
class PathConfig:
    """Path configuration for datasets and model checkpoints."""
    # Base directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "save_path")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Dataset paths
    DATASET_DIR = DATA_DIR  # Main dataset directory
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    RAW_INPUT_DIR = os.path.join(DATA_DIR, "raw_input")
    
    # Model checkpoint
    DEFAULT_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "model.pth")
    
    # Make sure directories exist
    @staticmethod
    def ensure_directories():
        """Create necessary directories if they don't exist."""
        directories = [
            PathConfig.DATA_DIR,
            PathConfig.CHECKPOINT_DIR, 
            PathConfig.RESULTS_DIR,
            PathConfig.RAW_INPUT_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Model Configuration
class ModelConfig:
    """Model architecture and parameters."""
    # Dual-Net parameters
    NUM_UNFOLDING_LAYERS = 5
    
    # Network structure parameters
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    NUM_FEATURES = 40
    SCALE_UNET_FEATS = 20
    SCALE_ORSNET_FEATS = 16
    NUM_CAB = 8
    KERNEL_SIZE = 3
    REDUCTION = 4
    USE_BIAS = False
    DEPTH = 5
    
    # RDN parameters
    RDN_NUM_FEATURES = 64  # Different from NUM_FEATURES (40)
    RDN_GROWTH_RATE = 64
    RDN_NUM_BLOCKS = 3
    RDN_NUM_LAYERS = 4
    
    # BasicBlock parameters
    PATCH_SIZE = 35
    EPSILON = 1e-6
    
    # Fixed lambda parameters
    LAMBDA_1 = 1.0
    LAMBDA_2 = 0.7
    LAMBDA_3 = 0.3
    LAMBDA_4 = 1.0
    LAMBDA_5 = 1.0
    
    # Initial gamma values
    INIT_GAMMA = 3.001
    INIT_ETA = 1.001

# Training Configuration
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Basic training parameters
    BATCH_SIZE = 4
    EPOCHS = 200
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    RESIZE_SIZE = 256
    
    # Optimizer settings
    OPTIMIZER = "Adam"  # One of ["Adam", "AdamW", "SGD"]
    BETA1 = 0.9
    BETA2 = 0.999
    MOMENTUM = 0.9  # Only used for SGD
    
    # Learning rate scheduler
    USE_LR_SCHEDULER = True
    LR_SCHEDULER = "StepLR"  # One of ["StepLR", "CosineAnnealingLR", "ReduceLROnPlateau"]
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.5
    LR_T_MAX = 200
    LR_MIN = 1e-6
    LR_PATIENCE = 10  # For ReduceLROnPlateau
    
    # Loss weights
    LAMBDA_J = 1.0  # Weight for image reconstruction loss
    LAMBDA_T = 0.1  # Weight for transmission map loss
    LAMBDA_B = 0.1  # Weight for background light loss
    
    # Logging
    LOG_INTERVAL = 100  # Log metrics every N iterations
    
    # Checkpointing
    CHECKPOINT_FREQUENCY = 10  # Save checkpoint every N epochs

# Testing Configuration
class TestingConfig:
    """Testing parameters and settings."""
    # Testing parameters
    BATCH_SIZE = 1
    RESIZE_SIZE = 256
    
    # Output settings
    SAVE_INTERMEDIATE = False  # Whether to save intermediate outputs from each unfolding layer

# Preprocessing Configuration
class PreprocessingConfig:
    """Parameters for data preprocessing."""
    # Image preprocessing
    RESIZE_WIDTH = 400
    RESIZE_HEIGHT = 300
    NORMALIZE = True
    
    # Guided filter parameters
    GUIDED_FILTER_RADIUS = 50
    GUIDED_FILTER_EPSILON = 0.001
    
    # Transmission map parameters
    GAMMA_CORRECTION = 1.2
    BLOCK_SIZE = 15
    MIN_TRANSMISSION = 0.1
    MAX_TRANSMISSION = 0.9
    
    # Directory structure
    INPUT_DIR = "input"
    GT_DIR = "gt"
    T_PRIOR_DIR = "t_prior"
    B_PRIOR_DIR = "B_prior"
    RAW_INPUT_DIR = "raw_input"

# Export all configs as a single dictionary for easy access
CONFIG = {
    "system": SystemConfig,
    "paths": PathConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "testing": TestingConfig,
    "preprocessing": PreprocessingConfig
} 