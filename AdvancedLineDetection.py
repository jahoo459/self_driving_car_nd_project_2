## Advanced Lane Finding Project

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
def bgr2rgb(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)


import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt

def calibrateCamera():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    # cv2.destroyAllWindows()

    # undistort the camera

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # plt.figure()
    # test_img = cv2.imread('test_images/straight_lines1.jpg')
    # plt.subplot(2, 1, 1)
    # plt.imshow(bgr2rgb(img))
    # plt.subplot(2, 1, 2)
    #
    # dst = cv2.undistort(img, mtx, dist, None, mtx)
    # plt.imshow(bgr2rgb(dst))
    # cv2.waitKey(1000)

    return ret, mtx, dist, rvecs, tvecs


def pipeline(raw_img):
    # calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrateCamera()

    # apply undistortion to the image
    undistorted_img = cv2.undistort(raw_img, mtx, dist, None, mtx)

    gray_undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

    # Apply threshold on gradient (xsobel)
    sobelx = cv2.Sobel(gray_undistorted_img, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # * Use color transforms, gradients, etc., to create a thresholded binary image.
    # transform from RGB to HLS
    hls_image = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2HLS)
    S = hls_image[:, :, 2]

    # apply threshold on S channel
    s_thres = (90, 255)
    s_binary = np.zeros_like(S)
    s_binary[(S > s_thres[0]) & (S <= s_thres[1])] = 1
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    # Plotting thresholded images
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Stacked thresholds')
    ax1.imshow(color_binary)

    ax2.set_title('Combined S channel and gradient thresholds')
    ax2.imshow(combined_binary, cmap='gray')

    # * Apply a perspective transform to rectify binary image ("birds-eye view").

    # use gray_undistorted_img

    # # For source points I'm grabbing the outer four detected corners
    pts = np.float32([[130, 720], [1210, 720], [700, 450], [585, 450]])
    offset = 0

    copy = np.copy(gray_undistorted_img)

    # lines = cv2.polylines(copy, pts, isClosed=False, color=(255, 0, 0), thickness=10)
    cv2.line(copy, tuple(pts[0]), tuple(pts[1]), (255, 0, 0), 10)
    cv2.line(copy, tuple(pts[1]), tuple(pts[2]), (255, 0, 0), 10)
    cv2.line(copy, tuple(pts[2]), tuple(pts[3]), (255, 0, 0), 10)
    cv2.line(copy, tuple(pts[3]), tuple(pts[0]), (255, 0, 0), 10)

    plt.figure()
    plt.imshow(copy)

    img_size = gray_undistorted_img.shape[::-1]
    # # For destination points, I'm arbitrarily choosing some points to be
    # # a nice fit for displaying our warped result
    # # again, not exact, but close enough for our purposes
    dst = np.float32([[offset, img_size[1] - offset],
                      [img_size[0] - offset, img_size[1] - offset],
                      [img_size[0] - offset, offset],
                      [offset, offset]])
    # dst = np.float32([[offset, offset], [img_size[0] - offset, offset],
    #                   [img_size[0] - offset, img_size[1] - offset],
    #                   [offset, img_size[1] - offset]])
    # # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst)
    # # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(gray_undistorted_img, M, gray_undistorted_img.shape[::-1])
    # warped = cv2.warpPerspective(copy, M, copy.shape[::-1])
    plt.figure()
    plt.imshow(warped)


    # * Detect lane pixels and fit to find the lane boundary.
    # Calculate the histogram
    # * Determine the curvature of the lane and vehicle position with respect to center.
    # * Warp the detected lane boundaries back onto the original image.
    # * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


if __name__ == '__main__':
    test_img = cv2.imread('test_images/straight_lines1.jpg')
    pipeline(test_img)




