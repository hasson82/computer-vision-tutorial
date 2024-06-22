import numpy as np
import cv2
import matplotlib.pyplot as plt

def gen_cover2_images(x_offset, y_offset):
    print("Generating cover2 image")
    blank_image = np.ones((512, 512, 3))
    circle_color = (0,0,0)
    cv2.circle(blank_image, (130+x_offset, 135+y_offset), 50, circle_color, -1)
    cv2.circle(blank_image, (370+x_offset, 135+y_offset), 50, circle_color, -1)
    cv2.circle(blank_image, (55+x_offset, 375+y_offset), 20, circle_color, -1)
    cv2.circle(blank_image, (255+x_offset, 375+y_offset), 20, circle_color, -1)
    cv2.circle(blank_image, (460+x_offset, 375+y_offset), 20, circle_color, -1)
    
    result = cv2.normalize(blank_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image_name = f"cover2_{x_offset}_{y_offset}.jpg"
    cv2.imwrite(f"cover2\\{image_name}", result)

def gen_cover1_images(x_offset, y_offset):
    print("Generating cover1 image")
    blank_image = np.ones((512, 512, 3))
    circle_color = (0,0,0)
    cv2.circle(blank_image, (255+x_offset, 135+y_offset), 50, circle_color, -1)
    cv2.circle(blank_image, (60+x_offset, 375+y_offset), 20, circle_color, -1)
    cv2.circle(blank_image, (185+x_offset, 375+y_offset), 20, circle_color, -1)
    cv2.circle(blank_image, (350+x_offset, 375+y_offset), 20, circle_color, -1)
    cv2.circle(blank_image, (460+x_offset, 375+y_offset), 20, circle_color, -1)
    
    result = cv2.normalize(blank_image, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image_name = f"cover1_{x_offset}_{y_offset}.jpg"
    cv2.imwrite(f"cover1\\{image_name}", result)

def main():
    print("Clustering main")
    number_of_images = 50
    for i in range(number_of_images):
        x_offset = np.random.randint(-15, 15)
        y_offset = np.random.randint(-15, 15)
        gen_cover1_images(x_offset, y_offset)
        
    for i in range(number_of_images):
        x_offset = np.random.randint(-15, 15)
        y_offset = np.random.randint(-15, 15)
        gen_cover2_images(x_offset, y_offset)


if __name__ == "__main__":
    main()