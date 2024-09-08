import cv2
import numpy as np

#Estimate the background color by averaging the color of the corners of the image.
def estimate_background_color(image, corner_crop_percent=0.05):

    # Get the dimensions of the image
    h, w, _ = image.shape

    # Calculate the size of the cropped areas from the corners based on the percentage of the image size
    crop_size = (int(h * corner_crop_percent), int(w * corner_crop_percent))

    # Extract the four corner regions of the image
    corners = [image[:crop_size[0], :crop_size[1]],               # Top-left
               image[:crop_size[0], w - crop_size[1]:w],          # Top-right
               image[h - crop_size[0]:h, :crop_size[1]],          # Bottom-left
               image[h - crop_size[0]:h, w - crop_size[1]:w]]     # Bottom-right

    # Flatten the corner arrays into a single list of pixel values
    corners = np.concatenate([c.reshape(-1, 3) for c in corners])

    # Calculate the median color value across all the corner pixels
    median_color = np.median(corners, axis=0)

    return median_color

#Process each frame of the video
def process_frame(image, lower_bound, upper_bound):
    #Create a slightly blurred image to create better contours
    blurred2 = cv2.GaussianBlur(image, (31, 31), 15, 15)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(blurred2, cv2.COLOR_BGR2HSV)        

    #Create a mask and reverse it
    mask_inv = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_inv = cv2.bitwise_not(mask_inv)
    
    # Find contours based on the masked image
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()

    # Define a minimum area threshold for contours
    min_area = 10000

    # Loop over all the contours
    for contour in contours:
        # Check if the contour is large enough to be considered
        if cv2.contourArea(contour) > min_area:

            # Draw the exact contour on the output image
            cv2.drawContours(output, [contour], -1, (0, 0, 0), 2)
            
            # Calculate the center using the moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # Draw a dot at the center of the contour and label it
            cv2.circle(output, (cX, cY), 5, (0, 0, 0), -1)
            cv2.putText(output, "Center" + "[" + str(cX) + ", " + str(cY) + "]", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return output

#Find the bounds that should be used to make the masks
def find_bounds(hsv_background_color):

    #Sensitivity of the "hue" value
    sensitivityh = 5

    #Sensitivity of the "saturation" value
    sensitivitys = 40

    #Sensitivity of the "value" value
    sensitivityv = 40
    
    h_val = int(hsv_background_color[0])
    s_val = int(hsv_background_color[1])
    v_val = int(hsv_background_color[2])

    #Set the bounds for masking
    lower_bound = np.array([h_val - sensitivityh, max(s_val-sensitivitys,0), max(v_val-sensitivityv, 0)])
    upper_bound = np.array([h_val + sensitivityh, min(s_val+sensitivitys,255), min(v_val+sensitivityv, 255)])
    return lower_bound, upper_bound

# Define the input video file or camera
input_video_path = 'Input/PennAir 2024 App Dynamic.mp4'
output_video_path = 'Output/Output Part 2.mp4'

# Open the video capture
cap = cv2.VideoCapture(input_video_path)

# Check if the video capture opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the width, height, and frame rate of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

ret, frame = cap.read()

#Create an aggressively blurred image to get a more accurate estimate of background color
blurred = cv2.GaussianBlur(frame, (151, 151), 100, 100)

# Estimate the background color
background_color = estimate_background_color(frame)
hsv_background_color = cv2.cvtColor(np.uint8([[background_color]]), cv2.COLOR_BGR2HSV)[0][0]

# Define color range used for masking everything but the background
lower_bound, upper_bound = find_bounds(hsv_background_color)

# Process the video frame by frame
while True:
    ret, frame = cap.read()

    # If no more frames
    if not ret:
        break

    output = process_frame(frame, lower_bound, upper_bound)

    out.write(output)

    cv2.imshow('Processed Video', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

cv2.destroyAllWindows()
