import cv2
import numpy as np

def estimate_background_color(image, corner_crop_percent=0.05):
    #Estimate the background color by averaging the color of the corners of the image.
    h, w, _ = image.shape
    crop_size = (int(h * corner_crop_percent), int(w * corner_crop_percent))
    corners = [image[:crop_size[0], :crop_size[1]],               # Top-left
               image[:crop_size[0], w - crop_size[1]:w],          # Top-right
               image[h - crop_size[0]:h, :crop_size[1]],          # Bottom-left
               image[h - crop_size[0]:h, w - crop_size[1]:w]]     # Bottom-right
    corners = np.concatenate([c.reshape(-1, 3) for c in corners])
    median_color = np.median(corners, axis=0)
    return median_color


def process_frame(image, lower_bound, upper_bound, needs_invert):
    #Create a slightly blurred image to create better contours
    
    blurred2 = cv2.GaussianBlur(image, (31, 31), 15, 15)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(blurred2, cv2.COLOR_BGR2HSV)        
        
    mask_inv = cv2.inRange(hsv, lower_bound, upper_bound)
    if needs_invert == 2:
        mask_inv = cv2.bitwise_not(mask_inv)
        
    elif needs_invert == 1:
        mask1 = cv2.inRange(hsv, lower_bound, np.array([179,int(upper_bound[1]),int(upper_bound[2])]))
        mask2 = cv2.inRange(hsv, np.array([0,int(lower_bound[1]),int(lower_bound[2])]), upper_bound)
        mask_inv = cv2.bitwise_or(mask1, mask2)
    
    # Find contours based on the masked image
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare the output image
    output = image.copy()

    # Define a minimum area threshold for contours
    min_area = 5000  # You can adjust this value based on your specific needs

    # Loop over the contours
    for contour in contours:
        if cv2.contourArea(contour) > min_area:  # Check if the contour is large enough
            # Draw the exact contour on the output image
            cv2.drawContours(output, [contour], -1, (0, 0, 0), 2)
            
            # Calculate the moments of the contour
            M = cv2.moments(contour)
            
            # Calculate the centroid using the moments
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0
            
            # Draw a circle at the centroid
            cv2.circle(output, (cX, cY), 5, (0, 0, 0), -1)

            cv2.putText(output, "Center" + "[" + str(cX) + ", " + str(cY) + "]", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return output

def find_bounds(hsv_background_color):
    sensitivityh = 5  # Adjust this value based on your specific image and background
    sensitivitys = 40
    sensitivityv = 40
    sensitivityValue1 = 15
    sensitivityValue2 = 10
    sensitivity2 = 20

    h_val = int(hsv_background_color[0])
    s_val = int(hsv_background_color[1])
    v_val = int(hsv_background_color[2])
    
    if (v_val < 70 and s_val < sensitivityValue1):
        lower_bound = np.array([0, 0, hsv_background_color[2]+sensitivity2])
        upper_bound = np.array([179, 255, 255])
        return lower_bound, upper_bound, 0
    
    if (v_val > 204 and s_val < sensitivityValue2):
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([179, 255, hsv_background_color[2]-sensitivity2])
        return lower_bound, upper_bound, 0
    
    if (h_val - sensitivityh < 0):
        lower_bound = np.array([179-(h_val - sensitivityh), max(s_val-sensitivitys,0), max(v_val-sensitivityv, 0)])
        upper_bound = np.array([h_val + sensitivityh, min(s_val+sensitivitys,255), min(v_val+sensitivityv, 255)])
        return lower_bound, upper_bound, 1
    
    if (h_val + sensitivityh > 179):
        lower_bound = np.array([h_val - sensitivityh, max(s_val-sensitivitys,0), max(v_val-sensitivityv, 0)])
        upper_bound = np.array([(h_val + sensitivityh)%179, min(s_val+sensitivitys,255), min(v_val+sensitivityv, 255)])
        return lower_bound, upper_bound, 1
    
    lower_bound = np.array([h_val - sensitivityh, max(s_val-sensitivitys,0), max(v_val-sensitivityv, 0)])
    upper_bound = np.array([h_val + sensitivityh, min(s_val+sensitivitys,255), min(v_val+sensitivityv, 255)])
    return lower_bound, upper_bound, 2

def getBounds(frame):
    #Create an aggressively blurred image to get a more accurate estimate of background color
    blurred = cv2.GaussianBlur(frame, (151, 151), 100, 100)

    # Estimate the background color
    background_color = estimate_background_color(frame)
    hsv_background_color = cv2.cvtColor(np.uint8([[background_color]]), cv2.COLOR_BGR2HSV)[0][0]

    # Define color range for masking everything but the background
    return find_bounds(hsv_background_color)



# Define the input video file
input_video_path = 'PennAir 2024 App Dynamic Hard.mp4'
output_video_path = 'final2.mp4'

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
fourcc = cv2.VideoWriter_fourcc(*'mp4v') #MPEG-4 codec
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

ret, frame = cap.read()

if not ret:
    print("Error: Could not read the first frame")
    cap.release()
    out.release()
    exit()

lower_bound, upper_bound, needs_invert = getBounds(frame)

# Process the video frame by frame
while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # Exit loop if no more frames are available

    output = process_frame(frame, lower_bound, upper_bound, needs_invert)

    out.write(output)
    # Optionally, display the processed frame
    cv2.imshow('Processed Video', output)

    # Press 'q' to exit the video processing loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()




