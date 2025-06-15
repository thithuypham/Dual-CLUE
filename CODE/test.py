"""
Testing script for Dual-CLUE: Underwater Image Enhancement Model

This script loads a trained Dual-CLUE model and runs inference on test images,
saving the enhanced results along with estimated transmission maps and
background light.

Usage:
    python test.py --model_dir <path_to_model> --test_dir <test_dataset_dir> --output <output_dir>
"""
import sys
import os
import glob
import time
import random
import argparse
from datetime import datetime
from ntpath import basename

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

from data_utils import UIEBDataset
from model import Dual_Net
from CODE.config import CONFIG

# Set device for computation
DEVICE = CONFIG["system"].DEVICE

def set_seed(seed=CONFIG["system"].SEED, tf2_set_seed=True, torch_set_seed=True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        tf2_set_seed: Whether to set TensorFlow seeds
        torch_set_seed: Whether to set PyTorch seeds
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if torch_set_seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_output_dirs(output_base):
    """
    Create output directories for saving results.
    
    Args:
        output_base: Base output directory
        
    Returns:
        tuple: Paths to output directories for results, transmission maps, and background light
    """
    output_dirs = {
        'results': os.path.join(output_base, 'results'),
        'transmission': os.path.join(output_base, 'estimated_t'),
        'background': os.path.join(output_base, 'estimated_B')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return output_dirs

def load_model(model_path, layers=CONFIG["model"].NUM_UNFOLDING_LAYERS):
    """
    Load a trained Dual-CLUE model.
    
    Args:
        model_path: Path to the model checkpoint
        layers: Number of unfolding layers in the model
        
    Returns:
        nn.Module: Loaded and initialized model
    """
    try:
        # Initialize model
        model = Dual_Net(layers)
        
        # Use DataParallel if configured
        if CONFIG["system"].USE_DATA_PARALLEL and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Count parameters and move model to device
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of model parameters: {pytorch_total_params:,}")
        
        model.to(DEVICE)
        model.eval()
        print(f"Loaded model from {model_path}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

def load_and_preprocess_image(image_path, transform):
    """
    Load and preprocess a single image.
    
    Args:
        image_path: Path to the image file
        transform: Torchvision transform to apply
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    try:
        img = Image.open(image_path)
        return transform(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

def process_image(model, input_img, t_p, B_p, resize):
    """
    Process a single image through the model.
    
    Args:
        model: Dual-CLUE model
        input_img: Input image tensor
        t_p: Transmission prior tensor
        B_p: Background light prior tensor
        resize: Resize transform for outputs
        
    Returns:
        tuple: Enhanced image, transmission map, and background light
    """
    # Convert to proper format
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    # Add batch dimension
    input_img = Variable(input_img.type(Tensor)).unsqueeze(0)
    t_p = Variable(t_p.type(Tensor)).unsqueeze(0)
    B_p = Variable(B_p.type(Tensor)).unsqueeze(0)
    
    # Handle single-channel transmission prior
    if t_p.shape[1] == 1:
        t_p = t_p.repeat(1, 3, 1, 1)
    
    # Run inference
    with torch.no_grad():
        output_J, output_B, output_t, *_ = model(input_img, t_p, B_p)
    
    # Get final outputs (last element in each list)
    enhanced_img = resize(output_J[-1])
    transmission = resize(output_t[-1])
    background = resize(output_B[-1])
    
    return enhanced_img, transmission, background

def batch_process_images(model, test_files, test_files_t, test_files_B_p, transform, resize, output_dirs, batch_size=CONFIG["testing"].BATCH_SIZE):
    """
    Process images in batches for better performance.
    
    Args:
        model: Dual-CLUE model
        test_files: List of input image paths
        test_files_t: List of transmission prior paths
        test_files_B_p: List of background light prior paths
        transform: Image transform function
        resize: Resize transform for outputs
        output_dirs: Output directories for saving results
        batch_size: Batch size for processing
        
    Returns:
        list: Processing times for each batch
    """
    times = []
    
    # Process in batches
    for i in range(0, len(test_files), batch_size):
        batch_files = test_files[i:i+batch_size]
        batch_t_files = test_files_t[i:i+batch_size]
        batch_B_p_files = test_files_B_p[i:i+batch_size]
        
        # Load and preprocess batch
        input_batch = []
        t_p_batch = []
        B_p_batch = []
        valid_indices = []
        
        for j, (img_path, t_path, b_path) in enumerate(zip(batch_files, batch_t_files, batch_B_p_files)):
            input_img = load_and_preprocess_image(img_path, transform)
            t_p = load_and_preprocess_image(t_path, transform)
            B_p = load_and_preprocess_image(b_path, transform)
            
            if input_img is not None and t_p is not None and B_p is not None:
                input_batch.append(input_img)
                t_p_batch.append(t_p)
                B_p_batch.append(B_p)
                valid_indices.append(j)
        
        if not input_batch:  # Skip if all images failed to load
            continue
            
        # Convert to tensors
        input_batch = torch.stack(input_batch).to(DEVICE)
        t_p_batch = torch.stack(t_p_batch).to(DEVICE)
        B_p_batch = torch.stack(B_p_batch).to(DEVICE)
        
        # Handle single-channel transmission prior
        if t_p_batch.shape[1] == 1:
            t_p_batch = t_p_batch.repeat(1, 3, 1, 1)
        
        # Measure processing time
        start_time = time.time()
        
        # Process batch
        with torch.no_grad():
            output_J, output_B, output_t, *_ = model(input_batch, t_p_batch, B_p_batch)
        
        # Get final outputs
        enhanced_imgs = resize(output_J[-1])
        transmissions = resize(output_t[-1])
        backgrounds = resize(output_B[-1])
        
        process_time = time.time() - start_time
        times.append(process_time)
        
        # Save results
        for k, idx in enumerate(valid_indices):
            img_path = batch_files[idx]
            filename = basename(img_path).split('.')[0] + '.png'
            
            save_image(enhanced_imgs[k].data, os.path.join(output_dirs['results'], filename))
            save_image(transmissions[k].data, os.path.join(output_dirs['transmission'], filename))
            save_image(backgrounds[k].data, os.path.join(output_dirs['background'], filename))
    
    return times

def testing(args):
    """
    Main testing function.
    
    Args:
        args: Command-line arguments
    """
    # Create output directories
    output_dirs = create_output_dirs(args.output)
    
    # Load model
    model = load_model(args.model_dir)
    
    # Initialize transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((CONFIG["testing"].RESIZE_SIZE, CONFIG["testing"].RESIZE_SIZE))
    ])
    resize = transforms.Resize((CONFIG["testing"].RESIZE_SIZE, CONFIG["testing"].RESIZE_SIZE))
    
    # Get list of test files
    try:
        test_files = sorted(glob.glob(os.path.join(args.test_dir, CONFIG["preprocessing"].INPUT_DIR, "*.*")))
        test_files_t = sorted(glob.glob(os.path.join(args.test_dir, CONFIG["preprocessing"].T_PRIOR_DIR, "*.*")))
        test_files_B_p = sorted(glob.glob(os.path.join(args.test_dir, CONFIG["preprocessing"].B_PRIOR_DIR, "*.*")))
        
        if not test_files:
            print(f"No test images found in {os.path.join(args.test_dir, CONFIG['preprocessing'].INPUT_DIR)}")
            return
            
        print(f"Found {len(test_files)} test images")
    except Exception as e:
        print(f"Error reading test files: {str(e)}")
        return
    
    # Process images
    if CONFIG["testing"].BATCH_SIZE > 1 and len(test_files) > 1:
        print(f"Processing images in batches of {CONFIG['testing'].BATCH_SIZE}...")
        with tqdm(total=len(test_files)) as pbar:
            batch_times = batch_process_images(
                model, test_files, test_files_t, test_files_B_p, 
                transform, resize, output_dirs
            )
            pbar.update(len(test_files))
        
        # Calculate statistics
        if len(batch_times) > 1:
            avg_time = np.mean(batch_times[1:])  # Skip first batch (warmup)
            total_time = np.sum(batch_times[1:])
            avg_time_per_image = total_time / len(test_files)
            fps = 1.0 / avg_time_per_image
            
            print(f"\nTotal samples processed: {len(test_files)}")
            print(f"Average processing time: {avg_time_per_image:.4f} seconds per image ({fps:.2f} FPS)")
            print(f"Total processing time: {total_time:.4f} seconds")
            print(f"Results saved to {output_dirs['results']}\n")
    else:
        # Process images one by one
        times = []
        for i, img_path in enumerate(tqdm(test_files, desc="Processing images")):
            try:
                # Load input images and priors
                input_img = load_and_preprocess_image(test_files[i], transform)
                t_p = load_and_preprocess_image(test_files_t[i], transform)
                B_p = load_and_preprocess_image(test_files_B_p[i], transform)
                
                if input_img is None or t_p is None or B_p is None:
                    continue
                    
                # Measure processing time
                start_time = time.time()
                enhanced_img, transmission, background = process_image(model, input_img, t_p, B_p, resize)
                process_time = time.time() - start_time
                times.append(process_time)
                
                # Generate output filename
                filename = basename(test_files[i]).split('.')[0] + '.png'
                
                # Save results
                save_image(enhanced_img.data, os.path.join(output_dirs['results'], filename))
                save_image(transmission.data, os.path.join(output_dirs['transmission'], filename))
                save_image(background.data, os.path.join(output_dirs['background'], filename))
                    
            except Exception as e:
                print(f"Error processing image {test_files[i]}: {str(e)}")
                continue
        
        # Print timing statistics
        if len(times) > 1:
            # Skip the first image (warm-up)
            avg_time = np.mean(times[1:])
            total_time = np.sum(times[1:])
            fps = 1.0 / avg_time
            
            print(f"\nTotal samples processed: {len(test_files)}")
            print(f"Average processing time: {avg_time:.4f} seconds per image ({fps:.2f} FPS)")
            print(f"Total processing time: {total_time:.4f} seconds")
            print(f"Results saved to {output_dirs['results']}\n")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Dual-CLUE: Deep Unfolding Network for Underwater Image Enhancement')
    parser.add_argument('--model_dir', type=str, default=CONFIG["paths"].DEFAULT_MODEL_PATH,
                        help=f'Path to pretrained model (default: {CONFIG["paths"].DEFAULT_MODEL_PATH})')
    parser.add_argument('--test_dir', type=str, default=CONFIG["paths"].TEST_DIR,
                        help=f'Path to test dataset directory (default: {CONFIG["paths"].TEST_DIR})')
    parser.add_argument('--output', type=str, default=CONFIG["paths"].RESULTS_DIR,
                        help=f'Path to save outputs (default: {CONFIG["paths"].RESULTS_DIR})')
    parser.add_argument('--seed', type=int, default=CONFIG["system"].SEED,
                        help=f'Random seed for reproducibility (default: {CONFIG["system"].SEED})')
    parser.add_argument('--batch_size', type=int, default=CONFIG["testing"].BATCH_SIZE,
                        help=f'Batch size for testing (default: {CONFIG["testing"].BATCH_SIZE})')

    args = parser.parse_args()
    
    # Update CONFIG with command-line arguments
    CONFIG["testing"].BATCH_SIZE = args.batch_size

    # Create timestamped output directory
    run_version = datetime.now().strftime("%Y%m%d")
    print(f"Test run: {run_version}")
    args.output = os.path.join(args.output, run_version)
    os.makedirs(args.output, exist_ok=True)

    # Set random seeds
    set_seed(args.seed)

    # Ensure directories exist
    CONFIG["paths"].ensure_directories()

    # Run testing
    testing(args)