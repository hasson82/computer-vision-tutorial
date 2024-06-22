import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('data\\jj_and_omer.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

x = np.random.randint(0, 400)
y = np.random.randint(0, 800)
radius = min(np.random.randint(1, 400-x), np.random.randint(1, 800-y))

print(x, y, radius)
cv2.circle(img, center=(x,y), radius=radius, color=(0, 255, 0), thickness=10)
gray = cv2.GaussianBlur(img, (15, 15), 0)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 200, param1=100, param2=50, minRadius=10, maxRadius=140)
circles = np.uint16(np.around(circles))

# Create a blank white image
output = np.ones_like(img) * 255

# Draw the circles on the white image
for i in circles[0,:]:
    cv2.circle(output, (i[0], i[1]), i[2], (0, 0, 0), -1)

# Save the result
cv2.imwrite('output.jpg', output)

# First plot
plt.figure()
plt.imshow(img)
plt.title('Original Image')

# Second plot
plt.figure()
plt.imshow(output, cmap='gray')
plt.title('Output Image')

plt.show()