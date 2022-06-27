#!/usr/bin/env python3
import os
import sys

import numpy as np
import skimage
import skimage.measure as measure


def mse(x, y):
    return np.mean(np.square(x - y))

def psnr(x, y):
    return 10 * np.log10(255 ** 2 / mse(x, y))

    
def render_loss(target_filepath, img_filepath):
    # https://stackoverflow.com/questions/52799031/skimage-measure-produce-strangely-high-mean-square-error
    target = skimage.io.imread(target_filepath)
    # If the image is [,,4] then take only [,,3]
    target = target[:,:,:3]

    img = skimage.io.imread(img_filepath)
    # If the image is [,,4] then take only [,,3]
    img = img[:,:,:3]


    mse_loss = mse(target.astype(float), img.astype(float))
    psnr_loss = psnr(target.astype(float), img.astype(float))


    print("mse_loss: ", mse_loss)
    print("psnr_loss: ", psnr_loss)





if __name__ == "__main__":
    # target_filepath = "./apps/imgs/ball.png"
    # img_filepath = "./apps/results/ball__num_paths_512__max_width2.0__use_lpips_lossFalse__num_iter1001__use_blobTrue/iter_1000.png"

    # target_filepath = "./apps/imgs/puke.png"
    # img_filepath = "/Users/henryleemr/Documents/workplace/lottie-files/raster-to-vector/diffvecg/apps/results/puke__num_paths_512__max_width2.0__use_lpips_lossFalse__num_iter602__use_blobTrue/iter_601.png"
    

    # target_filepath = "./apps/imgs/shapes.png"
    # img_filepath = "/Users/henryleemr/Documents/workplace/lottie-files/raster-to-vector/diffvecg/apps/results/shapes__num_paths_512__max_width2.0__use_lpips_lossFalse__num_iter602__use_blobTrue/iter_601.png"
    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # target_filepath = "./apps/imgs/ball.png"
    # img_filepath = "./apps/imgs/ball_untuned_raster.png"


    # target_filepath = "./apps/imgs/puke.png"
    # img_filepath = "./apps/imgs/puke_untuned_raster.png"


    # target_filepath = "./apps/imgs/shapes.png"
    # img_filepath = "./apps/imgs/shapes_untuned_raster.png"




    render_loss(target_filepath=target_filepath, img_filepath=img_filepath)       
