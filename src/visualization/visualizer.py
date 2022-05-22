import os

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def visualize_features(ground_truth: np.ndarray, synthetic: np.ndarray, 
                       input_file: str, output_path: str):

    font = {
        'size'   : 18
    }
    matplotlib.rc('font', **font)
    
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 1, 1).set_title('Ground truth signal - vocoder features')
    plt.imshow(ground_truth[:, :-2].T, aspect='auto', origin='lower', cmap='magma')
    plt.xlabel('Block number')
    plt.ylabel('Features')
    plt.subplot(2, 1, 2).set_title('Synthetic signal - vocoder features')
    plt.imshow(synthetic[:, :-2].T, aspect='auto', origin='lower', cmap='magma')
    plt.xlabel('Block number')
    plt.ylabel('Features')
    plt.tight_layout()
    
    fname = os.path.basename(input_file).replace('hdf5', 'png')
    output = os.path.join(output_path, fname)
    plt.savefig(output)
