import time

import cv2
import numpy as np
import math

cam = cv2.VideoCapture('.\\data\\Lane_Detection_Test_Video_01.mp4')

<<<<<<< Updated upstream
NEW_WIDTH = 450
NEW_HEIGHT = 235

screen_corners = np.array([
    (NEW_WIDTH, 0),   # Top right
    (0, 0),           # Top left
=======
NEW_WIDTH = 640
NEW_HEIGHT = 270

screen_corners = np.array([
    (NEW_WIDTH, 0),  # Top right
    (0, 0),  # Top left
>>>>>>> Stashed changes
    (0, NEW_HEIGHT),  # Bottom left
    (NEW_WIDTH, NEW_HEIGHT)  # Bottom right
], dtype=np.float32)
# Define the corners of the trapezoid from the previous exercise
<<<<<<< Updated upstream
upper_left = (NEW_WIDTH*0.55 - 45, NEW_HEIGHT*0.55 + 50)
bottom_right = (NEW_WIDTH, NEW_HEIGHT)
upper_right = (NEW_WIDTH*0.55 - 10, NEW_HEIGHT*0.55 + 50)
=======
upper_left = (NEW_WIDTH * 0.45, NEW_HEIGHT * 0.75)
bottom_right = (NEW_WIDTH, NEW_HEIGHT)
upper_right = (NEW_WIDTH * 0.55, NEW_HEIGHT * 0.75)
>>>>>>> Stashed changes
bottom_left = (0, NEW_HEIGHT)

# Round the coordinates and convert them to integers
trapezoid_bounds = np.array([upper_left, upper_right, bottom_right, bottom_left], dtype=np.int32)
trapezoid_bounds_float = np.float32(trapezoid_bounds)

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
# Get the perspective transformation matrix
perspective_matrix = cv2.getPerspectiveTransform(trapezoid_bounds_float, screen_corners)

empty_frame = np.zeros((NEW_HEIGHT, NEW_WIDTH), dtype=np.uint8)

# Fill the trapezoid area with white color (1)
cv2.fillConvexPoly(empty_frame, trapezoid_bounds, 1)

<<<<<<< Updated upstream
cv2.imshow("Trapezoid", empty_frame*255)

threshold_value = int(255/2)
while True:
    ret, frame = cam.read()

    if ret is False:
        break
    width, height, _ = frame.shape

    frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))
    cv2.imshow("Original", frame)

=======
cv2.imshow("Trapezoid", empty_frame * 255)

threshold_value = int(255 / 2)

left_top = (0, 0)
left_bottom = (0, NEW_HEIGHT)
right_top = (int(NEW_WIDTH / 2), 0)
right_bottom = (int(NEW_WIDTH / 2), NEW_HEIGHT)


def get_line_top_and_bottom(left_line, right_line):
    left_top_y = 0
    left_bottom_y = NEW_HEIGHT
    right_top_y = 0
    right_bottom_y = NEW_HEIGHT

    left_top_x = int((left_top_y - left_line[0]) / left_line[1])
    left_bottom_x = int((left_bottom_y - left_line[0]) / left_line[1])
    right_top_x = int((right_top_y - right_line[0]) / right_line[1])
    right_bottom_x = int((right_bottom_y - right_line[0]) / right_line[1])

    if -10 ** 8 <= left_top_x <= 10 ** 8:
        left_top = (left_top_x, left_top_y)
    if -10 ** 8 <= left_bottom_x <= 10 ** 8:
        left_bottom = (left_bottom_x, left_bottom_y)
    if -10 ** 8 <= right_top_x <= 10 ** 8:
        right_top = (right_top_x, right_top_y)
    if -10 ** 8 <= right_bottom_x <= 10 ** 8:
        right_bottom = (right_bottom_x, right_bottom_y)

    return left_top, left_bottom, right_top, right_bottom


def draw_final_lines(frame, left_line, right_line, original_frame):
    blank_frame_left = np.zeros_like(frame)
    blank_frame_right = np.zeros_like(frame)

    left_top, left_bottom, right_top, right_bottom = get_line_top_and_bottom(left_line, right_line)

    cv2.line(blank_frame_left, left_top, left_bottom, (255, 255, 255), 3)
    cv2.line(blank_frame_right, right_top, right_bottom, (255, 255, 255), 3)

    current_frame = np.float32([[0, 0], [NEW_WIDTH, 0], [NEW_WIDTH, NEW_HEIGHT], [0, NEW_HEIGHT]])
    target_frame = np.float32(trapezoid_bounds)

    magic_matrix = cv2.getPerspectiveTransform(current_frame, target_frame)
    base_frame_left = cv2.warpPerspective(blank_frame_left, magic_matrix, (NEW_WIDTH, NEW_HEIGHT))
    base_frame_right = cv2.warpPerspective(blank_frame_right, magic_matrix, (NEW_WIDTH, NEW_HEIGHT))

    coordinates_left = np.argwhere(base_frame_left > 0)
    coordinates_right = np.argwhere(base_frame_right > 0)

    left_x = coordinates_left[:, 1]
    left_y = coordinates_left[:, 0]

    right_x = coordinates_right[:, 1]
    right_y = coordinates_right[:, 0]

    original_frame_copy = original_frame.copy()

    original_frame_copy[left_y, left_x] = [50, 50, 250]
    original_frame_copy[right_y, right_x] = [50, 250, 50]

    return original_frame_copy


def get_coordinates_of_street_markings(frame):
    noiseless_frame = frame.copy()
    noiseless_frame[:, :int(NEW_WIDTH * 0.05)] = 0
    noiseless_frame[:, int(NEW_WIDTH * 0.95):] = 0

    # cv2.imshow('Noiseless', noiseless_frame)

    coordinates_left = np.argwhere(noiseless_frame[:, :int(NEW_WIDTH * 0.5)] > 0)
    coordinates_right = np.argwhere(noiseless_frame[:, int(NEW_WIDTH * 0.5):] > 0)

    left_xs = coordinates_left[:, 1]
    left_ys = coordinates_left[:, 0]

    right_xs = coordinates_right[:, 1] + int(NEW_WIDTH * 0.5)
    right_ys = coordinates_right[:, 0]

    return left_xs, left_ys, right_xs, right_ys


def draw_lines(frame, left_line, right_line):
    left_top, left_bottom, right_top, right_bottom = get_line_top_and_bottom(left_line, right_line)

    cv2.line(frame, left_top, left_bottom, 200, 5)
    cv2.line(frame, right_top, right_bottom, 100, 5)

    middle_of_screen_x = NEW_WIDTH // 2

    cv2.line(frame, (middle_of_screen_x, 0), (middle_of_screen_x, NEW_WIDTH), 255, 1)

    return frame


def process(frame):
    # ex 2
    frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))
    cv2.imshow("Original", frame)

>>>>>>> Stashed changes
    # ex 3
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    empty_frame_color = cv2.cvtColor(empty_frame, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Gray", frame_gray)

    # ex 4
<<<<<<< Updated upstream
    road_frame = cv2.multiply(frame, empty_frame_color)
=======
    road_frame = cv2.multiply(frame_gray, empty_frame)
>>>>>>> Stashed changes
    cv2.imshow("Road", road_frame)

    # ex 5
    stretched = np.float32([[[0, 0], [NEW_WIDTH, 0], [NEW_WIDTH, NEW_HEIGHT], [0, NEW_HEIGHT]]])
    matrix = cv2.getPerspectiveTransform(trapezoid_bounds_float, stretched)
    top_down = cv2.warpPerspective(frame_gray, matrix, (NEW_WIDTH, NEW_HEIGHT))
    cv2.imshow("Top-Down", top_down)

    # ex 6
    blurred = cv2.blur(top_down, (5, 5))
    cv2.imshow("Blurred", blurred)

    # ex 7
    sobel_vertical = np.float32([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)

    float_frame_1 = np.float32(blurred)
    float_frame_2 = np.float32(blurred)

    sobel_1 = cv2.filter2D(float_frame_1, -1, sobel_vertical)
    sobel_2 = cv2.filter2D(float_frame_2, -1, sobel_horizontal)

<<<<<<< Updated upstream
    sobel_final = np.sqrt(sobel_1**2 + sobel_2**2)
=======
    sobel_final = np.sqrt(sobel_1 ** 2 + sobel_2 ** 2)
>>>>>>> Stashed changes

    sobel_final = cv2.convertScaleAbs(sobel_final)
    cv2.imshow("Sobel", sobel_final)

    # ex 8
    _, thresholded = cv2.threshold(sobel_final, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresholded", thresholded)

<<<<<<< Updated upstream
    # ex 9
    frame_copy = thresholded.copy()
    thresholded[:, :int(NEW_WIDTH*0.05)] = 0
    thresholded[:, int(NEW_WIDTH*0.95):] = 0

    left_half = thresholded[:, :int(NEW_WIDTH / 2)]
    right_half = thresholded[:, int(NEW_WIDTH / 2):]

    # Get the coordinates of white points on each side of the road
    left_points = np.argwhere(left_half == 255)
    right_points = np.argwhere(right_half == 255)

    # Separate X and Y coordinates
    left_x = left_points[:, 1]
    left_y = left_points[:, 0]
    right_x = right_points[:, 1] + int(NEW_WIDTH / 2)  # Add half of the width to the X coordinates
    right_y = right_points[:, 0]

    cv2.imshow("Binarized", thresholded)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
=======
    left_xs, left_ys, right_xs, right_ys = get_coordinates_of_street_markings(thresholded)

    # Ex9: Find the lines that detect the edges of the lane and draw them
    left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, 1)
    right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, 1)
    lines_frame = draw_lines(thresholded, left_line, right_line)
    cv2.imshow('Lines', lines_frame)

    # Ex10: Draw the final lines
    final_frame = draw_final_lines(frame, left_line, right_line, frame)
    cv2.imshow('Final Frame', final_frame)


def main():
    while True:
        start_time = time.time()
        ret, frame = cam.read()

        if ret is False:
            print("End of video")
            break

        process(frame)

        elapsed_time = time.time() - start_time

        wait_time = max(0, elapsed_time)
        time.sleep(wait_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


main()
>>>>>>> Stashed changes
