import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from AdvancedLineDetection import Line

# Global variables
# Real world dist calculation
ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 700

def bgr2rgb(rgb_img):
    return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)


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

    # undistort the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs


def apply_threshold_operations(img):

    # input to sobel shall be grayscale undistored image
    # First convert to hls and go for yellow and white color
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold on gradient (xsobel)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # transform from RGB to HLS
    hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls_image[:, :, 2]

    # apply threshold on S channel
    s_thres = (90, 255)
    s_binary = np.zeros_like(S)
    s_binary[(S > s_thres[0]) & (S <= s_thres[1])] = 1
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    ### VISUALIZATION ###
    # Plotting thresholded images
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.set_title('Stacked thresholds')
    # ax1.imshow(color_binary)
    #
    # ax2.set_title('Combined S channel and gradient thresholds')
    # ax2.imshow(combined_binary, cmap='gray')
    ### VISUALIZATION ###

    return combined_binary


def apply_perspective_transform(undistored_img, combined_binary):

    # use gray undistored image for visualization
    gray_undistored_img = cv2.cvtColor(undistored_img, cv2.COLOR_BGR2GRAY)

    # Select the soruce points on original image
    pts = np.float32([[231, 690], [1075, 690], [713, 465], [570, 465]])

    # define offset
    offset = 200
    img_size = combined_binary.shape[::-1]

    # Choose destination points
    dst = np.float32([[offset, img_size[1]],
                      [img_size[0] - offset, img_size[1]],
                      [img_size[0] - offset, 0],
                      [offset, 0]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(pts, dst)
    Minv = cv2.getPerspectiveTransform(dst, pts)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(combined_binary, M, combined_binary.shape[::-1])

    ### VISUALIZATION ###
    # copy = np.copy(gray_undistored_img)
    #
    # cv2.line(copy, tuple(pts[0]), tuple(pts[1]), (255, 0, 0), 2)
    # cv2.line(copy, tuple(pts[1]), tuple(pts[2]), (255, 0, 0), 2)
    # cv2.line(copy, tuple(pts[2]), tuple(pts[3]), (255, 0, 0), 2)
    # cv2.line(copy, tuple(pts[3]), tuple(pts[0]), (255, 0, 0), 2)
    #
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    # ax1.set_title('selected points in original image')
    # ax1.imshow(copy, cmap='gray')
    #
    # warped_visu = cv2.warpPerspective(copy, M, copy.shape[::-1])
    # ax2.set_title('image after perspective transform')
    # ax2.imshow(warped_visu, cmap='gray')
    ### VISUALIZATION ###

    return warped, M, Minv


def fit_polynomial_init(binary_warped, lines: Line):
    # Find our lane pixels first using sliding windows approach
    leftx, lefty, rightx, righty, out_img = find_lane_pixels_sliding_windows(binary_warped)

    # Fit a second order polynomial for both lines
    left_fit, right_fit, left_fitx, right_fitx, ploty, left_fit_cr, right_fit_cr, offset_m = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    # Update the lines
    update_lines(lines, left_fit, right_fit, leftx, rightx,
                 lefty, righty, left_fitx, right_fitx, left_fit_cr, right_fit_cr, ploty, offset_m)

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    ### Visualization ###
    # Plots the left and right polynomials on the lane lines
    # plt.figure()
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    ### Visualization ###


def fit_poly(img_shape, leftx, lefty, rightx, righty):

    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty *ym_per_pix, rightx *xm_per_pix, 2)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])

    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Calculate distance of the car from the center of the line
    # Assumption: center of the car is pixel 1280/2 = 640
    center = 640
    # The value is calucalte for y = 719 (closest to the bottom of the image)
    offset_left = (center - left_fitx[-1])
    offset_right = (right_fitx[-1] - center)
    offset_m = (offset_left - offset_right) * xm_per_pix



    return left_fit, right_fit, left_fitx, right_fitx, ploty, left_fit_cr, right_fit_cr, offset_m


def find_lane_pixels_sliding_windows(warped):
    # calculate histogram
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)

    ### Visualize ###
    # Visualize histogram
    # plt.figure()
    # plt.plot(histogram)
    ### Visualize ###

    # Prepare the output (3-channel) image
    out_img = np.dstack((warped, warped, warped)) * 255

    # Search for lines to the left and right from the midpoint
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # number of sliding windows
    nwindows = 9
    # width of the windows +/- margin
    margin = 80
    # minimum number of pixels found to recenter window
    minpix = 150
    # height of windows - based on nwindows above and image shape
    window_height = np.int(warped.shape[0] // nwindows)

    # x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        ### VISUALIZATION ###
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
        ### VISUALIZATION ###

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def update_lines(lines: Line, left_fit, right_fit, leftx, rightx,
                 lefty, righty, left_fitx, right_fitx, left_fit_cr, right_fit_cr, ploty, offset_m):
    if (left_fit != []): # Line was detected
        lines[0].detected = True
        lines[0].detetction_counter = 0
        lines[0].previous_fit = lines[0].current_fit
        lines[0].recent_xfitted = left_fitx
        lines[0].current_fit = left_fit
        lines[0].allx = leftx
        lines[0].ally = lefty
        lines[0].current_fit_cr = left_fit_cr
        lines[0].ploty = ploty
        lines[0].dist_from_center_m = offset_m
    else: #Line wasn't detected
        lines[0].detected = False
        lines[0].detection_counter =+ 1
        # lines[0].current_fit = lines[0].previous_fit

    if (right_fit != []):
        lines[1].detected = True
        lines[1].detetction_counter = 0
        lines[1].previous_fit = lines[0].current_fit
        lines[1].recent_xfitted = right_fitx
        lines[1].current_fit = right_fit
        lines[0].allx = rightx
        lines[0].ally = righty
        lines[1].current_fit_cr = right_fit_cr
        lines[1].ploty = ploty
        lines[1].dist_from_center_m = offset_m
    else:
        lines[1].detected = False
        lines[1].detection_counter =+ 1
        # lines[0].current_fit = lines[0].previous_fit

def search_around_poly(binary_warped, lines: Line):
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_fit = lines[0].current_fit
    right_fit = lines[1].current_fit

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fit, right_fit, left_fitx, right_fitx, ploty, left_fit_cr, right_fit_cr, offset_m = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    # Update the lines status
    update_lines(lines, left_fit, right_fit, leftx, rightx,
                 lefty, righty, left_fitx, right_fitx, left_fit_cr, right_fit_cr, ploty, offset_m)

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    ### Visualization ###
    # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    ### Visualization ###


def measure_curvature_pixels(lines):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''

    ploty = lines[0].ally
    left_fit = lines[0].current_fit
    right_fit = lines[1].current_fit

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    lines[0].radius_of_curvature = left_curverad
    lines[1].radius_of_curvature = right_curverad


def measure_curvature_real(lines):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    ploty = lines[0].ploty
    left_fit_cr = lines[0].current_fit_cr
    right_fit_cr = lines[1].current_fit_cr

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    lines[0].radius_of_curvature_cr = left_curverad
    lines[1].radius_of_curvature_cr = right_curverad

def reproject_lines(lines, warped, Minv, image):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    left_fitx = lines[0].recent_xfitted
    right_fitx = lines[1].recent_xfitted
    ploty = lines[0].ploty

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result

def write_measured_info(img, lines):
    curvature_left = lines[0].radius_of_curvature_cr
    curvature_right = lines[1].radius_of_curvature_cr
    dist_from_center = lines[0].dist_from_center_m

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerY = 100
    bottomLeftCornerX = 10
    delta = 50
    fontScale = 1
    fontColor = (255,255,255)
    lineType = 2

    text = []
    if(curvature_left > 5000):
        text.append("Radius Left: inf")
    else:
        text.append(str("Radius Left: " + str(int(curvature_left)) + "m."))
    if(curvature_right > 5000):
        text.append("Radius right: inf")
    else:
        text.append(str("Radius right: " + str(int(curvature_right)) + "m."))
    text.append(str("Distance from center: " + str("{:.3f}".format(dist_from_center)) + "m."))

    # bottom left corner changes
    blc = [bottomLeftCornerX, 0]
    # Draw text on the image
    for i in range(len(text)):
        blc[1] = bottomLeftCornerY + i * delta
        cv2.putText(img, text[i], tuple(blc), font, fontScale, fontColor, lineType)

    return img