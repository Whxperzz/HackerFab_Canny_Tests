
import cv2
import numpy as np

# make test image
img = np.zeros((500, 500, 3), dtype=np.uint8)

# wafer outline
cv2.circle(img, (250, 250), 200, (180, 180, 180), 2)

# add patterns
for x in range(120, 380, 80):
    for y in range(120, 380, 80):
        cv2.rectangle(img, (x, y), (x + 40, y + 40),
                      (255, 255, 255), 2)

# grayscale and blur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# canny edge detection
edges = cv2.Canny(blur, 50, 150)

# save images
cv2.imwrite("original.png", img)
cv2.imwrite("edges.png", edges)

# compare images
cv2.imshow("Original", img)
cv2.imshow("Canny", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()


# test with noise and uneven lighting
noise = np.random.normal(0, 25, gray.shape)
noisy = np.clip(gray.astype(float) + noise, 0, 255).astype(np.uint8)

# add uneven lighting
light = np.linspace(0.5, 1.0, 500)
light = np.tile(light, (500, 1))
noisy = (noisy * light).astype(np.uint8)

# run canny again
blur2 = cv2.GaussianBlur(noisy, (5, 5), 0)
edges2 = cv2.Canny(blur2, 50, 150)

# save results
cv2.imwrite("noisy.png", noisy)
cv2.imwrite("noisy_edges.png", edges2)

# show results
cv2.imshow("Noisy Image", noisy)
cv2.imshow("Canny with Noise", edges2)

cv2.waitKey(0)
cv2.destroyAllWindows()
