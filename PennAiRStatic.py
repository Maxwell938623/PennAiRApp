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

def process_image(image, lower_bound, upper_bound):
    #Create a slightly blurred image to create better contours
    blurred2 = cv2.GaussianBlur(image, (21, 21), 4, 0)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(blurred2, cv2.COLOR_BGR2HSV)        
        
    mask_inv = cv2.inRange(hsv, lower_bound, upper_bound)
    mask_inv = cv2.bitwise_not(mask_inv)
    
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
    
    h_val = int(hsv_background_color[0])
    s_val = int(hsv_background_color[1])
    v_val = int(hsv_background_color[2])
    
    lower_bound = np.array([h_val - sensitivityh, max(s_val-sensitivitys,0), max(v_val-sensitivityv, 0)])
    upper_bound = np.array([h_val + sensitivityh, min(s_val+sensitivitys,255), min(v_val+sensitivityv, 255)])
    return lower_bound, upper_bound


image = cv2.imread('PennAir 2024 App Static.png')

if image is None:
    quit()

blurred = cv2.GaussianBlur(image, (151, 151), 100, 100)

# Estimate the background color
background_color = estimate_background_color(image)
hsv_background_color = cv2.cvtColor(np.uint8([[background_color]]), cv2.COLOR_BGR2HSV)[0][0]

# Define color range for masking everything but the background

lower_bound, upper_bound = find_bounds(hsv_background_color)

output = process_image(image, lower_bound, upper_bound)

cv2.imshow('Processed Image', output)
cv2.waitKey(0)

cv2.destroyAllWindows()
