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

<<<<<<< HEAD
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

=======
>>>>>>> origin/master

## Keypoint detectors ##########################################################

class KeypointDetector(object):
    def detectKeypoints(self, image):
        '''
<<<<<<< HEAD
            Input:
            image -- uint8 BGR image with values between [0, 255]
            Output:
=======
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
>>>>>>> origin/master
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
<<<<<<< HEAD
            '''
=======
        '''
>>>>>>> origin/master
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
<<<<<<< HEAD
        Compute silly example features. This doesn't do anything meaningful, but
        may be useful to use as an example.
        '''
    
    def detectKeypoints(self, image):
        '''
            Input:
            image -- uint8 BGR image with values between [0, 255]
            Output:
=======
    Compute silly example features. This doesn't do anything meaningful, but
    may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        '''
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
>>>>>>> origin/master
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
<<<<<<< HEAD
            '''
=======
        '''
>>>>>>> origin/master
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]
<<<<<<< HEAD
        
=======

>>>>>>> origin/master
        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]
<<<<<<< HEAD
                
                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.
                    
=======

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

>>>>>>> origin/master
                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10
<<<<<<< HEAD
                    
                    features.append(f)
        
=======

                    features.append(f)

>>>>>>> origin/master
        return features


class HarrisKeypointDetector(KeypointDetector):
<<<<<<< HEAD
    
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
=======

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
>>>>>>> origin/master
        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)
<<<<<<< HEAD
        
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
        
=======

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

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

>>>>>>> origin/master
        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
<<<<<<< HEAD

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
=======
        raise Exception("TODO 1: in features.py not implemented")
        # TODO-BLOCK-END

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
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        raise Exception("TODO 2: in features.py not implemented")
        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
>>>>>>> origin/master
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
<<<<<<< HEAD
            '''
=======
        '''
>>>>>>> origin/master
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []
<<<<<<< HEAD
        
        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
=======

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

>>>>>>> origin/master
        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)
<<<<<<< HEAD
        
=======

>>>>>>> origin/master
        # Compute local maxima in the Harris image.  You will need to
        # implement this function. Create image to store local maximum harris
        # values as True, other pixels False
        harrisMaxImage = self.computeLocalMaxima(harrisImage)
<<<<<<< HEAD
        
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
                
  
=======

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN
                raise Exception("TODO 3: in features.py not implemented")
                # TODO-BLOCK-END

                features.append(f)

>>>>>>> origin/master
        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
<<<<<<< HEAD
            Input:
            image -- uint8 BGR image with values between [0, 255]
            Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
            '''
=======
        Input:
            image -- uint8 BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees) and set the size to 10.
        '''
>>>>>>> origin/master
        detector = cv2.ORB()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
<<<<<<< HEAD
            Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
            Output:
            Descriptor numpy array, dimensions:
            keypoint number x feature descriptor dimension
            '''
=======
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
>>>>>>> origin/master
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
<<<<<<< HEAD
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

=======
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):
            x, y = f.pt
            x, y = int(x), int(y)

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # TODO-BLOCK-BEGIN
            raise Exception("TODO 4: in features.py not implemented")
            # TODO-BLOCK-END

        return desc


>>>>>>> origin/master
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
<<<<<<< HEAD

=======
>>>>>>> origin/master
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        windowSize = 8
        desc = np.zeros((len(keypoints), windowSize * windowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        for i, f in enumerate(keypoints):
<<<<<<< HEAD
            
            x, y = f.pt
=======
>>>>>>> origin/master
            # TODO 5: Compute the transform as described by the feature
            # location/orientation. You will need to compute the transform
            # from each pixel in the 40x40 rotated window surrounding
            # the feature to the appropriate pixels in the 8x8 feature
            # descriptor image.
<<<<<<< HEAD

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
=======
            transMx = np.zeros((2, 3))

            # TODO-BLOCK-BEGIN
            raise Exception("TODO 5: in features.py not implemented")
            # TODO-BLOCK-END
>>>>>>> origin/master

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx,
                (windowSize, windowSize), flags=cv2.INTER_LINEAR)

<<<<<<< HEAD
            # Intensity normalize the window: subtract the mean, divide by the SD
            windowMean = np.mean(destImage)
            windowSD = np.std(destImage)
            # If the standard deviation is very close to zero (less than 10**-5 in magnitude) then you should just return an all-zeros vector to avoid a divide by zero error
           
            if windowSD < 10**-5: 
                destImage = np.zeros(destImage.shape)
            
            else: 
                destImage = (destImage - windowMean)/windowSD

=======
>>>>>>> origin/master
            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is zero then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN
<<<<<<< HEAD
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
=======
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
>>>>>>> origin/master
        descriptor = cv2.ORB()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))
<<<<<<< HEAD
        
=======

>>>>>>> origin/master
        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
<<<<<<< HEAD
            Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
            Output:
            Descriptor numpy array, dimensions:
            keypoint number x feature descriptor dimension
            '''
=======
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            Descriptor numpy array, dimensions:
                keypoint number x feature descriptor dimension
        '''
>>>>>>> origin/master
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
<<<<<<< HEAD
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
    
=======
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

>>>>>>> origin/master
    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0
<<<<<<< HEAD
        
=======

>>>>>>> origin/master
        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)
<<<<<<< HEAD
            
            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1
        
        return d / n if n != 0 else 0
    
=======

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

>>>>>>> origin/master
    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]
<<<<<<< HEAD
        
        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
                         (h[3]*x + h[4]*y + h[5]) / d])
=======

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])
>>>>>>> origin/master


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
<<<<<<< HEAD
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

=======
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
>>>>>>> origin/master
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]
<<<<<<< HEAD
        
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []
        
        dist = scipy.spatial.distance.cdist(desc1, desc2, metric = 'euclidean')
        index = np.argmin(dist, axis = 1)
        
        for i in range(desc1.shape[0]):
            m = cv2.DMatch()
            m.queryIdx = i
            m.trainIdx = index[i]
            m.distance = dist[i][index[i]]
            
            matches.append(m)
        
=======

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

>>>>>>> origin/master
        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
<<<<<<< HEAD
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
=======
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
>>>>>>> origin/master
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]
<<<<<<< HEAD
        
        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []
        
        dist = scipy.spatial.distance.cdist(desc1, desc2, metric = 'euclidean')
        index = np.argmin(dist, axis = 1)
        second_index = np.argsort(dist, axis=1)[:,1]
        for i in range(desc1.shape[0]):
            m = cv2.DMatch()
            m.queryIdx = i
            m.trainIdx = index[i]
            m.distance = (dist[i][index[i]])/(dist[i][second_index[i]])
            
            matches.append(m)
        
=======

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

>>>>>>> origin/master
        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()
<<<<<<< HEAD
    
    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))
=======

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))

>>>>>>> origin/master
