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
        # not detected counter
        self.detection_counter = 0
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #polynomial coefficients for the previous fit
        self.previous_fit = [np.array([False])]
        # polynomial coefficients for the most recent fit in world distance
        self.current_fit_cr = [np.array([False])]
        #radius of curvature of the line in pixel
        self.radius_of_curvature = None
        # radius of curvature of the line in meter
        self.radius_of_curvature_cr = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        self.ploty = None
        self.dist_from_center_m = 0


def pipeline(raw_img):

    # apply undistortion to the image
    undistorted_img = cv2.undistort(raw_img, mtx, dist, None, mtx)

    # * Use color transforms, gradients, etc., to create a thresholded binary image.
    combined_binary = apply_threshold_operations(undistorted_img)

    # * Apply a perspective transform to rectify binary image ("birds-eye view").
    warped_img, M, Minv = apply_perspective_transform(undistorted_img, combined_binary)

    # Initially fit polynomial using sliding windows approach (only in first step)
    if(lines[0].detected == False or lines[1].detected == False):
        fit_polynomial_init(warped_img, lines)

    else:
        # In the next steps try to search for a line in a defined area close
        # to previously detected line
        search_around_poly(warped_img, lines)

    # * Determine the curvature of the lane and vehicle position with respect to center.
    # measure_curvature_pixels(lines)
    measure_curvature_real(lines)

    # * Warp the detected lane boundaries back onto the original image.
    final_img = reproject_lines(lines, warped_img, Minv, raw_img)
    # * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    final_img = write_measured_info(final_img, lines)

    return final_img



def run_on_images():
    # Run for all test images and seve the output in output_images
    images = glob.glob('test_images/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        result = pipeline(img)

        name = fname.split("\\")
        name = name[-1]

        cv2.imwrite('output_images/' + 'result_' + name, result)

def run_on_video():
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    from IPython.display import HTML

    project_video_path = "project_video.mp4"
    project_video_path_output = "project_video_output.mp4"
    challenge_video_path = "challenge_video.mp4"
    challenge_video_path_output = "challenge_video_output.mp4"
    harder_challenge_video_path = "harder_challenge_video.mp4"
    harder_challenge_video_path_output = "harder_challenge_video_output.mp4"

    clip1 = VideoFileClip(project_video_path)
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(project_video_path_output, audio=False)

if __name__ == '__main__':

    # Create Line objects that will track the state of the both lines
    lines = [Line(), Line()]

    # calibrate camera only once
    ret, mtx, dist, rvecs, tvecs = calibrateCamera()

    # run_on_video()
    run_on_images()




