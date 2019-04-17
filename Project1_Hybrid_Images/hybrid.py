from __future__ import division
from __future__ import absolute_import

import sys
import cv2
import numpy as np



def cross_prod(img,kernel):

    padding = max(kernel.shape[0],kernel.shape[1])
    k_x = (kernel.shape[0] - 1)//2
    k_y = (kernel.shape[1] - 1)//2
    offset = (padding-1)//2

    #preserve original 2D shape of image
    shape = img.shape
    new_img = np.zeros(shape[:2])

    #pad image with zeros
    img = np.pad(img,(offset,offset),'constant')
    # print('padded img',pad)
    for i in range(offset,offset+shape[0]):
        for j in range(offset,offset+shape[1]):
            running_sum = 0
            for x in range(-k_x,k_x+1):
                for y in range(-k_y,k_y+1):
                    running_sum += kernel[k_x+x,k_y+y]*img[i+x,j+y] 

            new_img[i-offset,j-offset] = running_sum
    return new_img

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # preserve original 2D shape of image
    shape = img.shape

    img_set = []

    if len(shape) == 3:
        for dim in range(3):
            img_set.append(cross_prod(img[:,:,dim],kernel))
        return np.stack(img_set,axis=2)
       
    elif len(shape) == 2:
        return cross_prod(img,kernel)


def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    return cross_correlation_2d(img, np.flip(kernel,axis=[0,1]))

def gaussian_blur_kernel_2d(sigma, height, width):
    u'''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    x = np.arange(int(-width/2), int(width/2 +1))
    y = np.arange(int(-height/2), int(height/2 +1))

    x2d, y2d = np.meshgrid(x, y)

    G = np.exp(-(x2d**2+y2d**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    return G/ G.sum()

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernel = gaussian_blur_kernel_2d(sigma, size,size)
    return convolve_2d(img,kernel)


def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    low_img = low_pass(img,sigma,size)
    high_img = img - low_img
    return high_img


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

<<<<<<< HEAD

if __name__ == '__main__':
    k = np.array([[0,0,0],[0,1,0],[0,0,0]])
    i = np.array([[1,2,3,4,5]]*5)
    print(cross_correlation_2d(i,k))
    print(gaussian_blur_kernel_2d(5, 5, 5))
=======
>>>>>>> d692ce94ac66bc4ace0d921be9801d35fa0354a9
