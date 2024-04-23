import cv2
import numpy as np

cam = cv2.VideoCapture("Lane_Detection_Test_Video_01.mp4")

NEW_WIDTH = 450
NEW_HEIGHT = 235

screen_corners = np.array([
    (NEW_WIDTH, 0),   # Top right
    (0, 0),           # Top left
    (0, NEW_HEIGHT),  # Bottom left
    (NEW_WIDTH, NEW_HEIGHT)  # Bottom right
], dtype=np.float32)
# Define the corners of the trapezoid from the previous exercise
upper_left = (NEW_WIDTH*0.55 - 45, NEW_HEIGHT*0.55 + 50)
bottom_right = (NEW_WIDTH, NEW_HEIGHT)
upper_right = (NEW_WIDTH*0.55 - 10, NEW_HEIGHT*0.55 + 50)
bottom_left = (0, NEW_HEIGHT)

# Round the coordinates and convert them to integers
trapezoid_bounds = np.array([upper_left, upper_right, bottom_right, bottom_left], dtype=np.int32)
trapezoid_bounds_float = np.float32(trapezoid_bounds)


# Get the perspective transformation matrix
perspective_matrix = cv2.getPerspectiveTransform(trapezoid_bounds_float, screen_corners)

empty_frame = np.zeros((NEW_HEIGHT, NEW_WIDTH), dtype=np.uint8)

# Fill the trapezoid area with white color (1)
cv2.fillConvexPoly(empty_frame, trapezoid_bounds, 1)

cv2.imshow("Trapezoid", empty_frame*255)

threshold_value = int(255/2)
while True:
    ret, frame = cam.read()

    if ret is False:
        break
    width, height, _ = frame.shape

    frame = cv2.resize(frame, (NEW_WIDTH, NEW_HEIGHT))
    cv2.imshow("Original", frame)

    # ex 3
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    empty_frame_color = cv2.cvtColor(empty_frame, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Gray", frame_gray)

    # ex 4
    road_frame = cv2.multiply(frame, empty_frame_color)
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

    sobel_final = np.sqrt(sobel_1**2 + sobel_2**2)

    sobel_final = cv2.convertScaleAbs(sobel_final)
    cv2.imshow("Sobel", sobel_final)

    # ex 8
    _, thresholded = cv2.threshold(sobel_final, threshold_value, 255, cv2.THRESH_BINARY)
    cv2.imshow("Thresholded", thresholded)

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
