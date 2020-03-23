import cv2

# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

img = cv2.imread("C:\\Users\\tunis\\OneDrive\\Documents\\UofM\\ECE588\\HW1\\Original.jpeg", cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()

keypoints_sift, descriptors = sift.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, keypoints_sift, None)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
