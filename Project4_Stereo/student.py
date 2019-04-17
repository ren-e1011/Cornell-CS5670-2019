# Please place imports here.
# BEGIN IMPORTS
import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
# import util_sweep
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x 3 image with dimensions matching the
                  input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    start = time.time()
    height, width, channel = images[0].shape[0], images[0].shape[1],images[0].shape[2]
    num_images = len(images)

    images = np.array(images)
    I = images.reshape(num_images, height*width*channel)
    L = lights.T
    step1 = np.linalg.inv(np.dot(L, L.T))
    G = np.dot(step1, np.dot(L, I))
    p = np.linalg.norm(G, axis = 0)
    
    G3 = np.reshape(G.T, (height, width, channel, 3))
    Ggray = np.mean(G3, axis = 2)
    p_norm = np.linalg.norm(Ggray, axis = 2)
    mark = p_norm < 1e-7
    
    normal = Ggray/np.maximum(1e-7, p_norm[:,:,np.newaxis])
    normal[mark] = 0
    albedo = p.reshape(height, width, channel)
    albedo[mark] = 0
    
    end = time.time()
    print (end - start)

    return albedo, normal



def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """

    kRt = np.dot(K, Rt)
    height = points.shape[0]
    width = points.shape[1]
    
    #add a one vector to each of the x,y,z points
    homography = np.concatenate((points, np.ones((height, width, 1))), axis = 2)
    print (homography.shape)
    #dot product between the homography and the intrinsics
    projection = np.tensordot(homography, kRt.T, axes = 1)
    print (projection.shape)
    #divide by the last axis
    projection_return = projection/(projection[:,:,2])[:,:,np.newaxis]
    print (projection_return.shape)
    #take the first two axes
    return projection_return[:,:,0:2]



def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x112, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    #get channel size of image
    #for each channel in image: 
    for channel in range(image.shape[2]):
        #confirm this does what you think it should do
        channel_mean =  image[:,:,channel].mean()
        if channel_mean < 1e-6: return np.zeros(ncc_size**2*image.shape[2])
        image[:,:,channel] -= channel_mean
        
    ## compute mean
    # subtract mean
    # vectorize
    # consider padding
    for height in range(0,image.shape[0],ncc_size):
        for width in range(0,image.shape[1],ncc_size):
            v = np.reshape(a=image[height:height+ncc_size+1,width:width+ncc_size+1,:],newshape=(ncc_size**2*image.shape[2]),order='F')
    ## divide by std.dev
    ##INplace??
            image[height:height+ncc_size+1,width:width+ncc_size+1] =  v/np.sqrt(np.linalg.norm(v))
    return image
    
    # raise NotImplementedError()


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = np.zeros(image1.shape[0],image1.shape[1])
    for height in range(image1.shape[0]):
        for width in range(image1.shape[1]):
            ncc[height,width] = np.correlate(image1[height],image2[height])
    return ncc
    # raise NotImplementedE