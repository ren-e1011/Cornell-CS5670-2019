import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial

import transformations


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True

def atan2d(y, x):
    """compute atan2(y, x) with the result in degrees"""
    
    if abs(y) > abs(x):
        q = 2; x, y = y, x
    else:
        q = 0
    if x < 0:
        q += 1; x = -x
    
    ang = math.degrees(math.atan2(y, x))
    if q == 1:
        ang = (180 if y >= 0 else -180) - ang
    elif q == 2:
        ang =  90 - ang
    elif q == 3:
        ang = -90 + ang
    return ang


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
            Input:
            image -- uint8 BGR image with values between [0, 255]
            Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
            '''
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
        Compute silly example features. This doesn't do anything meaningful, but
        may be useful to use as an example.
        '''
    
    def detectKeypoints(self, image):
        '''
            Input:
            image -- uint8 BGR image with values between [0, 255]
            Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
            '''
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]
        
        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]
                
                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.
                    
                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10
                    
                    features.append(f)
        
        return features


class HarrisKeypointDetector(KeypointDetector):
    
    def saveHarrisImage(self, harrisImage, srcImage):
        '''
            Saves a visualization of the harrisImage, by overlaying the harris
            response image as red over the srcImage.
            
            Input:
            srcImage -- Grayscale input image in a numpy array with
            values in [0, 1]. The dimensions are (rows, cols).
            harrisImage -- Grayscale input image in a numpy array with
            values in [0, 1]. The dimensions are (rows, cols).
            '''
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)
        
        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)
    
    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
            Input:
            srcImage -- Grayscale input image in a numpy array with
            values in [0, 1]. The dimensions are (rows, cols).
            Output:
            harrisImage -- numpy array containing the Harris score at
            each pixel.
            orientationImage -- numpy array containing the orientation of the
            gradient at each pixel in degrees.
            '''
        height, width = srcImage.shape[:2]
        print ('made it here')
        
        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])
        
        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN

        window_size = 5
        offset = (window_size - 1)/2
        
        gradientX = scipy.ndimage.sobel(srcImage, axis=0, mode='reflect')
        gradientY = scipy.ndimage.sobel(srcImage, axis=1, mode='reflect')

        Ixx = cv2.GaussianBlur(gradientX*gradientX, (5,5), sigmaX = 0.5, sigmaY = 0, borderType = cv2.BORDER_REFLECT)
        Ixy = cv2.GaussianBlur(gradientX*gradientY, (5,5) ,sigmaX = 0.5, sigmaY = 0, borderType = cv2.BORDER_REFLECT)
        Iyy = cv2.GaussianBlur(gradientY*gradientY, (5,5) ,sigmaX = 0.5, sigmaY = 0, borderType = cv2.BORDER_REFLECT)
        
        Wdet = Ixx*Iyy - Ixy**2
        Wtr = Ixx + Iyy
        
        harrisImage = Wdet - 0.1*(Wtr**2)

        
        for i in range(height):
            for j in range(width):
                orientationImage[i,j] = atan2d(gradientX[i,j], gradientY[i,j])
        
        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)
        
        return harrisImage, orientationImage
    
    def computeLocalMaxima(self, harrisImage):
        '''
            Input:
            harrisImage -- numpy array containing the Harris score at
            each pixel.
            Output:
            destImage -- numpy array containing True/False at
            each pixel, depending on whether
            the pixel value is the local maxima in
            its 7x7 neighborhood.
            '''
        #destImage = np.zeros_like(harrisImage, np.bool)
        height, width = harrisImage.shape[:2]
        result = ndimage.maximum_filter(harrisImage, size = (7,7))
        
        destImage = (harrisImage == result)
        
        window_size = 7
        #height, width = harrisImage.shape[:2]
        for i in range(height):
            for j in range(width):
                
                if not inbounds([height, width], [i-window_size, j-window_size]) and inbounds([height, width], [i+window_size + 1, j+window_size + 1]):
                    
                    destImage[i][j] = True
        
        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
                        #raise Exception("TODO 2: in features.py not implemented")
        # TODO-BLOCK-END
        
        return destImage
    
    def detectKeypoints(self, image):
        '''
            Input:
            image -- BGR image with values between [0, 255]
            Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
            '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []
        
        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)
        
        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)
        
        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        #raise Exception("TODO 3: in features.py not implemented")
                                            
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y,x]:
                    continue
                
                f = cv2.KeyPoint()
                f.pt = (x,y)
                f.response = harrisImage[y,x]
                f.angle = orientationImage[y,x]
                f.size = 10.0
                features.append(f)
                
  
        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
            Input:
            image -- uint8 BGR image with values between [0, 255]
            Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
            '''
        detector = cv2.ORB()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
            Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
            Output:
            Descriptor numpy array, dimensions:
            keypoint number x feature descriptor dimension
            '''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
            Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
            Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
            '''

        # image = image.astype(np.float32)
        # image /= 255.
        # grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # desc_window = 5
        # desc = np.zeros((len(keypoints), desc_window * desc_window))
        # for i, f in enumerate(keypoints):
        #     x, y = f.pt
        #     x, y = int(x), int(y)

        #     # TODO 4: The simple descriptor is a 5x5 window of intensities
        #     # sampled centered on the feature point. Store the descriptor
        #     # as a row-major vector. Treat pixels outside the image as zero.

        #     #consider points' rotation??

        #     desc_pos = 0
        #     for row_coord in range(y-desc_window//2,y+desc_window//2+1):
        #         for col_coord in range(x-desc_window//2,x+desc_window//2+1): 
        #             #if the limit exists
        #             if col_coord > 0 and col_coord < len(image) and row_coord > 0 and row_coord < len(image[0]):
        #                 desc[i][desc_pos] = grayImage[row_coord,col_coord]
                        
        #             desc_pos +=1 
        # return desc

        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc_window = 5
        offset = (desc_window - 1)/2
        desc = np.zeros((len(keypoints), desc_window * desc_window))
        for i, f in enumerate(keypoints):
                x, y = f.pt
                x, y = int(x), int(y)
                                    
                desc_pos = 0
                for x_coord in range(y-offset,y+offset+1):
                        for y_coord in range(x-offset, x+ offset + 1):
                                #if the limit exists
                            if inbounds([grayImage.shape[0], grayImage.shape[1]], [x_coord, y_coord]):
                                desc[i][desc_pos] = grayImage[x_coord,y_coord]

                            desc_pos +=1

        return desc

#WORK IN PROGRESS
class MOPSFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''

        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
            
            x, y = f.pt
            # x, y = int(x), int(y)
            # print(x,y)
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.

            #in rad
            # theta = f.angle* np.pi / 180
            theta = math.radians(-f.angle)
            
            # move from keypoint center to be centered at origin: left x + windowSize//2, up y + windowSize//2
            T1 = np.array([
                [1,0,-x],
                [0,1,-y],
                [0,0,1]])

            # rotate horizontally [inv rotate by theta]

            R = np.array([
                [np.cos(theta),-np.sin(theta),0],
                [np.sin(theta),np.cos(theta),0],
                [0,0,1]
            ])

            # scale (gauss blurred img) to 1/5
            S = np.array([
                [.2,0,0],
                [0,.2,0],
                [0,0,1]])

            #push to be upper left aligned with origin
            T2 = np.array([
                [1,0,+(windowSize//2)],
                [0,1,(windowSize//2)],
                [0,0,1]
                ])
            #Left-multiplied transformations are combined right-to-left so the transformation matrix is the matrix product T2 S R T1. The figures below illustrate the sequence.
            # F = T2*S*R*T1
            F = np.dot(np.dot(np.dot(T2,S),R),T1)

            # warp affine does not require affine row
            transMx = F[:-1]

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

            # Intensity normalize the window: subtract the mean, divide by the SD
            windowMean = np.mean(destImage)
            windowSD = np.std(destImage)
            # If the standard deviation is very close to zero (less than 10**-5 in magnitude) then you should just return an all-zeros vector to avoid a divide by zero error
           
            if windowSD < 10**-5: 
                destImage = np.zeros(destImage.shape)
            
            else: 
                destImage = (destImage - windowMean)/windowSD

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
            # raise Exception("TODO 6: in features.py not implemented")
            # TODO-BLOCK-END
            # print(destImage.shape) 
            desc_pos = 0
            for row_coord in range(len(destImage)):
                for col_coord in range(len(destImage[0])): 
                    #if the limit exists
                    # if col_coord > 0 and col_coord < len(image) and row_coord > 0 and row_coord < len(image[0]):
                    desc[i][desc_pos] = destImage[row_coord,col_coord]
                    desc_pos += 1
        return desc

class MOPSFeatureDescriptor_legacy(FeatureDescriptor):
    
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
            Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
            Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
            and W is the window size
            '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        
        for i, f in enumerate(keypoints):
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.

            x, y = f.pt
            x, y = int(x), int(y)
            desc_pos = 0

            # copy 40x40 ORIENTED window from grayscaled gauss prefiltered image
            # 40 x 40 window around POI
            grayImage_windowSize = 40
            grayImage_window = np.zeros([40,40])


            col_slice = np.arange(start=x-grayImage_windowSize//2,stop=x+grayImage_windowSize//2+1,step=1)
            row_slice = np.arange(start=y-grayImage_windowSize//2,stop=y+grayImage_windowSize//2+1,step=1)

            col_offset = 0
            row_offset = 0

            col_stop = len(col_slice) 
            row_stop = len(row_slice) 

            if col_slice[0] < 0:
                col_offset = abs(col_slice[0])
                col_slice = col_slice[col_offset:]
            #len -1 for 0-based indexing
            if col_slice[-1] > len(grayImage[0])-1:
                #over-bloat, negative index, will peel off last negative ix elements 
                col_stop = len(grayImage[0])-col_slice[-1]

            if row_slice[0] < 0:
                row_offset = abs(row_slice[0])
                row_slice = row_slice[row_offset:]
            #len -1 for 0-based indexing
            if row_slice[-1] > len(grayImage)-1:
                row_stop = len(grayImage)-row_slice[-1]

            grayImage_window[row_offset:row_stop,col_offset:col_stop] = grayImage[row_slice[row_offset:row_stop],col_slice[col_offset:col_stop]]



            # for row_coord in range(y-desc_window//2,y+desc_window//2+1):
            #     for col_coord in range(x-desc_window//2,x+desc_window//2+1): 
            #         #if the limit exists
            #         if col_coord > 0 and col_coord < len(image) and row_coord > 0 and row_coord < len(image[0]):
            #             desc[i][desc_pos] = grayImage[row_coord,col_coord]

            #USE WARP AFFINE

            # shrink to 1/5, 8x8
            desc_ix = 0
            for row in grayImage_window:
                for col in grayImage_window[row]:
                    if row % 5 == 0:
                        desc[row,desc_ix] = grayImage_window[row,col]
                        desc_ix += 1

            
            #Rotate pixels to horizontal

            # Intensity normalize the window: subtract the mean, divide by the SD
            windowMean = np.mean(grayImage_window)
            windowSD = np.std(grayImage_window)
            # If the standard deviation is very close to zero (less than 10**-5 in magnitude) then you should just return an all-zeros vector to avoid a divide by zero error
            if windowSD < 10**-5: return np.zeros(grayImage_window.shape)
            
            grayImage_window = (grayImage_window - windowMean)/windowSD

            #WAT
            transMx = np.zeros((2, 3))
            
            
            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                                       (windowSize, windowSize), flags=cv2.INTER_LINEAR)
                                       
                                       # TODO 6: Normalize the descriptor to have zero mean and unit
                                       # variance. If the variance is zero then set the descriptor
                                       # vector to zero. Lastly, write the vector to desc.
                                       # TODO-BLOCK-BEGIN
            raise Exception("TODO 6: in features.py not implemented")
        # TODO-BLOCK-END
        
        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
            Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
            Output:
            Descriptor numpy array, dimensions:
            keypoint number x feature descriptor dimension
            '''
        descriptor = cv2.ORB()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))
        
        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
            Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
            Output:
            Descriptor numpy array, dimensions:
            keypoint number x feature descriptor dimension
            '''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
            Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
            Output:
            features matches: a list of cv2.DMatch objects
            How to set attributes:
            queryIdx: The index of the feature in the first image
            trainIdx: The index of the feature in the second image
            distance: The distance between the two features
            '''
        raise NotImplementedError
    
    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0
        
        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)
            
            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1
        
        return d / n if n != 0 else 0
    
    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]
        
        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
                         (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
            Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
            Output:
            features matches: a list of cv2.DMatch objects
            How to set attributes:
            queryIdx: The index of the feature in the first image
            trainIdx: The index of the feature in the second image
            distance: The distance between the two features
            '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]
        
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []
        
        # TODO 7: Perform simple feature matching.  This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        raise Exception("TODO 7: in features.py not implemented")
        # TODO-BLOCK-END
        
        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
            Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
            Output:
            features matches: a list of cv2.DMatch objects
            How to set attributes:
            queryIdx: The index of the feature in the first image
            trainIdx: The index of the feature in the second image
            distance: The ratio test score
            '''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]
        
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []
        
        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN
        raise Exception("TODO 8: in features.py not implemented")
        # TODO-BLOCK-END
        
        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()
    
    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))
