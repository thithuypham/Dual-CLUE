'''
Transmission Map Estimation for Underwater Image Enhancement

This script implements transmission map and background light estimation for underwater images,
following the method described in:

Reference: 
Liang, Zheng, et al. "Single underwater image enhancement by attenuation map guided 
color correction and detail preserved dehazing." Neurocomputing 425 (2021).

The script processes raw underwater images to generate:
1. Transmission maps (t_prior)
2. Background light estimates (B_prior)
3. Normalized input images

Usage:
    python preprocessing_data.py
'''
import numpy as np
import cv2
import os
import datetime
import natsort
import argparse
from tqdm import tqdm
from scipy import ndimage

from CODE.config import CONFIG

class GuidedFilter:
    """
    Implementation of the Guided Filter for edge-preserving smoothing.
    
    The guided filter is an edge-preserving smoothing filter like the bilateral filter,
    but with better behavior near edges. It's used here to refine the transmission map.
    
    Args:
        I: Guidance image (usually the source RGB image)
        radius: Filter radius
        epsilon: Regularization parameter
    """
    def __init__(self, I, radius=CONFIG["preprocessing"].GUIDED_FILTER_RADIUS, 
                 epsilon=CONFIG["preprocessing"].GUIDED_FILTER_EPSILON):
        self._radius = 2 * radius + 1
        self._epsilon = epsilon
        self._I = self._toFloatImg(I)
        self._initFilter()

    def _toFloatImg(self, img):
        """Convert image to float32 format with proper normalization."""
        if img.dtype == np.float32:
            return img
        return (1.0 / 255.0) * np.float32(img)

    def _initFilter(self):
        """Initialize filter parameters by computing covariance matrices."""
        I = self._I
        r = self._radius
        eps = self._epsilon

        # Extract RGB channels
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        # Compute mean values for each channel
        self._Ir_mean = cv2.blur(Ir, (r, r))
        self._Ig_mean = cv2.blur(Ig, (r, r))
        self._Ib_mean = cv2.blur(Ib, (r, r))

        # Compute variance and covariance values
        Irr_var = cv2.blur(Ir ** 2, (r, r)) - self._Ir_mean ** 2 + eps                                       
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - self._Ir_mean * self._Ig_mean                                  
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - self._Ir_mean * self._Ib_mean                                  
        Igg_var = cv2.blur(Ig * Ig, (r, r)) - self._Ig_mean * self._Ig_mean + eps                            
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - self._Ig_mean * self._Ib_mean                                  
        Ibb_var = cv2.blur(Ib * Ib, (r, r)) - self._Ib_mean * self._Ib_mean + eps                                                       

        # Compute inverse of covariance matrix
        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var                                                      
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var                                                      
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var                                                      
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var                                                      
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var                                                      
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var                                                      
        
        # Normalize by determinant
        I_cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var                                    
        Irr_inv /= I_cov                                                                                     
        Irg_inv /= I_cov                                                                                     
        Irb_inv /= I_cov                                                                                     
        Igg_inv /= I_cov                                                                                     
        Igb_inv /= I_cov                                                                                     
        Ibb_inv /= I_cov                                                                                     
        
        # Store inverse matrix
        self._Irr_inv = Irr_inv                                                                              
        self._Irg_inv = Irg_inv                                                                              
        self._Irb_inv = Irb_inv                                                                              
        self._Igg_inv = Igg_inv                                                                              
        self._Igb_inv = Igb_inv                                                                              
        self._Ibb_inv = Ibb_inv                  

    def _computeCoefficients(self, p):
        """
        Compute linear coefficients for the guided filter.
        
        Args:
            p: Input image to be filtered
            
        Returns:
            tuple: Linear coefficients (a, b) for each channel
        """
        r = self._radius                                                             
        I = self._I                                                                 
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]                                                          
        
        # Compute mean values
        p_mean = cv2.blur(p, (r, r))                             
        Ipr_mean = cv2.blur(Ir * p, (r, r))                                                         
        Ipg_mean = cv2.blur(Ig * p, (r, r))                                                    
        Ipb_mean = cv2.blur(Ib * p, (r, r))             

        # Compute covariance values
        Ipr_cov = Ipr_mean - self._Ir_mean * p_mean                                                 
        Ipg_cov = Ipg_mean - self._Ig_mean * p_mean                                                     
        Ipb_cov = Ipb_mean - self._Ib_mean * p_mean                                                       
                                                                                                                 
        # Compute a and b coefficients
        ar = self._Irr_inv * Ipr_cov + self._Irg_inv * Ipg_cov + self._Irb_inv * Ipb_cov                 
        ag = self._Irg_inv * Ipr_cov + self._Igg_inv * Ipg_cov + self._Igb_inv * Ipb_cov                
        ab = self._Irb_inv * Ipr_cov + self._Igb_inv * Ipg_cov + self._Ibb_inv * Ipb_cov    

        b = p_mean - ar * self._Ir_mean - ag * self._Ig_mean - ab * self._Ib_mean                                                                                                                                         

        # Mean of a and b for each pixel
        ar_mean = cv2.blur(ar, (r, r))          
        ag_mean = cv2.blur(ag, (r, r))                                                                   
        ab_mean = cv2.blur(ab, (r, r))                                                                      
        b_mean = cv2.blur(b, (r, r))                                                                                                                                              

        return ar_mean, ag_mean, ab_mean, b_mean            

    def _computeOutput(self, ab, I):
        """
        Compute the filtered output using the linear coefficients.
        
        Args:
            ab: Linear coefficients (a, b)
            I: Guidance image
            
        Returns:
            np.ndarray: Filtered output image
        """
        ar_mean, ag_mean, ab_mean, b_mean = ab
        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]
        q = ar_mean * Ir + ag_mean * Ig + ab_mean * Ib + b_mean
        return q

    def filter(self, p):
        """
        Apply guided filter to input image p.
        
        Args:
            p: Input image to be filtered
            
        Returns:
            np.ndarray: Filtered output image
        """
        p_32F = self._toFloatImg(p)
        ab = self._computeCoefficients(p_32F)
        return self._computeOutput(ab, self._I)

def get_attenuation(image, gamma=CONFIG["preprocessing"].GAMMA_CORRECTION):
    """
    Estimate wavelength-dependent attenuation indices for underwater image.
    
    Args:
        image: Input underwater image (normalized to [0,1])
        gamma: Gamma correction factor
        
    Returns:
        np.ndarray: Sorted indices of channels by attenuation rate
    """
    # Split image into R, G, B channel
    image_b = image[:,:,0]
    image_g = image[:,:,1]
    image_r = image[:,:,2]
    
    # Estimate attenuation map for each channel
    attenuation_b = 1 - image_b**(gamma)
    attenuation_g = 1 - image_g**(gamma)
    attenuation_r = 1 - image_r**(gamma)
    
    # Determine the color channel that attenuates at the highest rate
    mean_att_b = np.mean(attenuation_b)
    mean_att_g = np.mean(attenuation_g)
    mean_att_r = np.mean(attenuation_r)

    list_att = [mean_att_b, mean_att_g, mean_att_r]
    index = np.argsort(list_att)
    
    return index

def getMaxChannel(img):
    """
    Get maximum channel value for each pixel in the image.
    
    Args:
        img: Input image
        
    Returns:
        np.ndarray: Maximum channel values as a grayscale image
    """
    # Use NumPy's max function along the channel axis for vectorized operation
    return np.max(img[:, :, :2], axis=2).astype(np.float64)

def getMaxChannel_window(img, blockSize):
    """
    Get maximum value in a sliding window of blockSize for each pixel.
    
    Args:
        img: Input grayscale image
        blockSize: Size of the sliding window
        
    Returns:
        np.ndarray: Maximum values in local windows
    """
    # Vectorized version of the window maximum using max_filter from scipy
    # This replaces the nested loops for better performance
    imgDark = ndimage.maximum_filter(img, size=blockSize)
    
    return imgDark

def DepthMap(img, blockSize, index_att):
    """
    Estimate the depth map based on attenuation differences.
    
    Args:
        img: Input underwater image
        blockSize: Block size for local maximum computation
        index_att: Indices of channels sorted by attenuation rate
        
    Returns:
        np.ndarray: Estimated depth map
    """
    # Estimate depth map based on color channel differences
    img_c = np.zeros(img[:,:,0:2].shape)
    img_c_star = img[:,:,index_att[-1]]
    img_c[:,:,0] = img[:,:,index_att[0]]
    img_c[:,:,1] = img[:,:,index_att[1]]

    # Determine the color channel that attenuates at the highest rate
    max_c_star = getMaxChannel_window(img_c_star, blockSize)
    img_c = getMaxChannel(img_c)
    max_c = getMaxChannel_window(img_c, blockSize)
    largestDiff = max_c_star - max_c

    return largestDiff

def RefinedTransmission(transmission, img):
    """
    Refine the transmission map using guided filter.
    
    Args:
        transmission: Initial transmission map
        img: Guidance image (source RGB image)
        
    Returns:
        np.ndarray: Refined transmission map
    """
    guided_filter = GuidedFilter(img)
    transmission = guided_filter.filter(transmission)
    transmission = np.clip(transmission, 0, 255)

    return transmission

class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value
        
    def printInfo(self):
        print(self.x, self.y, self.value)

def getAtmosphericLight(depth, img):
    """
    Get atmospheric light based on the depth map.
    
    Args:
        depth: Depth map
        img: Input image
        
    Returns:
        np.ndarray: Estimated atmospheric light
    """
    height = len(depth)
    width = len(depth[0])
    nodes = []
    
    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, depth[i, j])
            nodes.append(oneNode)
            
    nodes = sorted(nodes, key=lambda node: node.value, reverse=False)
    atmosphericLight = img[nodes[0].x, nodes[0].y, :]
    
    return atmosphericLight

def estimateBackgroundLight(image):
    """
    Estimate background light for the image.
    
    Args:
        image: Input underwater image
        
    Returns:
        np.ndarray: Estimated background light
    """
    index = get_attenuation(image)
    largestDiff = DepthMap(image, CONFIG["preprocessing"].BLOCK_SIZE, index)
    atmosphericLight = getAtmosphericLight(largestDiff, image)
    
    return atmosphericLight

def process_image(filepath, resize_shape=(CONFIG["preprocessing"].RESIZE_WIDTH, CONFIG["preprocessing"].RESIZE_HEIGHT)):
    """
    Process a single underwater image to generate transmission and background light priors.
    
    Args:
        filepath: Path to the input image
        resize_shape: Target size for resizing (width, height)
        
    Returns:
        tuple: Processed image, transmission map, and background light
    """
    try:
        # Read and preprocess image
        img = cv2.imread(filepath)
        if img is None:
            return None, None, None
            
        # Resize for consistent processing
        img = cv2.resize(img, resize_shape)
        image = img/255.0
        
        # Estimate attenuation and depth
        index = get_attenuation(image)
        largestDiff = DepthMap(image, CONFIG["preprocessing"].BLOCK_SIZE, index)
        
        # Calculate and refine transmission map
        transmission = largestDiff + (1 - np.max(largestDiff))
        transmission = np.clip(transmission, 
                              CONFIG["preprocessing"].MIN_TRANSMISSION, 
                              CONFIG["preprocessing"].MAX_TRANSMISSION)
        transmission = RefinedTransmission(transmission*255, image*255)
        
        # Estimate background light
        BackgroundLight = estimateBackgroundLight(image)
        D = np.ones(image.shape)
        B = D*BackgroundLight
        
        return np.uint8(image*255), np.uint8(transmission), np.uint8(B*255)
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None, None, None

def main(args):
    """
    Main function to process underwater images and generate transmission and background light priors.
    
    Args:
        args: Command-line arguments
    """
    try:
        # Set dataset directory - adjust this to your dataset path
        path = args.input_dir
        
        # Check if input directory exists
        if not os.path.exists(path):
            print(f"Error: Input directory {path} does not exist!")
            return
            
        # Get all files in natural sort order
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        files = natsort.natsorted(files)

        # Set output directories
        output_base = args.output_dir
        t_prior_dir = os.path.join(output_base, CONFIG["preprocessing"].T_PRIOR_DIR)
        B_prior_dir = os.path.join(output_base, CONFIG["preprocessing"].B_PRIOR_DIR)
        input_dir = os.path.join(output_base, CONFIG["preprocessing"].INPUT_DIR)

        # Create output directories if they don't exist
        for directory in [t_prior_dir, B_prior_dir, input_dir]:
            if not os.path.exists(directory):   
                os.makedirs(directory)
                print(f"Directory '{directory}' created")

        # Process each file
        print(f"Found {len(files)} images to process")
        for i, file in enumerate(tqdm(files, desc="Processing images")):
            filepath = os.path.join(path, file)
            prefix = os.path.splitext(file)[0] + '.png'
            
            if not os.path.isfile(filepath):
                continue
                
            # Process image
            image, transmission, background = process_image(filepath)
            
            if image is None:
                continue
                
            # Save outputs
            cv2.imwrite(os.path.join(t_prior_dir, prefix), transmission)
            cv2.imwrite(os.path.join(B_prior_dir, prefix), background)
            cv2.imwrite(os.path.join(input_dir, prefix), image)
                
        print(f"Processing complete. Results saved to {output_base}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess underwater images for Dual-CLUE')
    
    parser.add_argument('--input_dir', type=str, default=CONFIG["paths"].RAW_INPUT_DIR,
                        help=f'Directory containing raw underwater images (default: {CONFIG["paths"].RAW_INPUT_DIR})')
    parser.add_argument('--output_dir', type=str, default=CONFIG["paths"].DATA_DIR,
                        help=f'Base directory to save outputs (default: {CONFIG["paths"].DATA_DIR})')
    parser.add_argument('--resize', type=str, default=f"{CONFIG['preprocessing'].RESIZE_WIDTH}x{CONFIG['preprocessing'].RESIZE_HEIGHT}",
                        help=f'Resize dimensions as WIDTHxHEIGHT (default: {CONFIG["preprocessing"].RESIZE_WIDTH}x{CONFIG["preprocessing"].RESIZE_HEIGHT})')
    
    args = parser.parse_args()
    
    # Parse resize parameter
    if args.resize:
        try:
            width, height = map(int, args.resize.split('x'))
            CONFIG["preprocessing"].RESIZE_WIDTH = width
            CONFIG["preprocessing"].RESIZE_HEIGHT = height
        except:
            print(f"Invalid resize format. Using default: {CONFIG['preprocessing'].RESIZE_WIDTH}x{CONFIG['preprocessing'].RESIZE_HEIGHT}")
    
    # Ensure directories exist
    CONFIG["paths"].ensure_directories()
    
    # Run preprocessing
    main(args) 