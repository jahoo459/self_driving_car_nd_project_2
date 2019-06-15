## Advanced Lane Finding Project

# Global variables for camera calibration
ret = 0
mtx = 0
dist = 0
rvecs = 0
tvecs = 0

# The goals / steps of this project are the following:
#
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## First, I'll calibrate the camera using chessboard images

from functions import *

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


def pipeline(raw_img):
    # Create Line objects that will track the state of the both lines
    lines = [Line(), Line()]

    # apply undistortion to the image
    undistorted_img = cv2.undistort(raw_img, mtx, dist, None, mtx)

    # * Use color transforms, gradients, etc., to create a thresholded binary image.
    combined_binary = apply_threshold_operations(undistorted_img)

    # * Apply a perspective transform to rectify binary image ("birds-eye view").
    warped_img = apply_perspective_transform(undistorted_img, combined_binary)

    # Initially fit polynomial using sliding windows approach (only in first step)
    if(lines[0].detected == False or lines[1].detected == False):
    # out_img, left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial_init(warped_img, lines)
        fit_polynomial_init(warped_img, lines)

    else:
        # In the next steps try to search for a line in a defined area close
        # to previously detected line
        result = search_around_poly(warped_img, lines)

    # * Determine the curvature of the lane and vehicle position with respect to center.
    measure_curvature_pixels(lines)

    # * Warp the detected lane boundaries back onto the original image.

    # * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.








if __name__ == '__main__':
    test_img = cv2.imread('test_images/test4.jpg')
    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrateCamera()
    pipeline(test_img)




