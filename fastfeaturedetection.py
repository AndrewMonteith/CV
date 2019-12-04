import scipy.io

mat = scipy.io.loadmat("C:\\Users\\Hp\\Downloads\\n04037443.sbow")

print(mat["image_sbow"][0])

# # Alternate to YOLO for detecting images and stuff
# import cv2
#
#
# p = "C:\\Users\\Hp\\Downloads\\TTBB - Bad\\left-images\\1506943010.480501_L.png"
# img = cv2.imread(p)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# # Initiate FAST object with default values
# fast = cv2.FastFeatureDetector_create(threshold=10)
#
# # find and draw the keypoints
# kp = fast.detect(gray, None)
#
# #Print all default params
# print("Threshold: ", fast.getThreshold())
# print("nonmaxSuppression: ", fast.getNonmaxSuppression())
# print("neighborhood: ", fast.getType())
# print("Total Keypoints with nonmaxSuppression: ", len(kp))
#
# # Output the points on the mats
# img3 = cv2.drawKeypoints(img, kp, None, color=(0, 0, 255))
#
# # Show the mats
# cv2.imshow("gray", gray)
# cv2.imshow("img1", img) # Original mat
# cv2.imshow("img3", img3) # Without nonmaxSuppression
# cv2.waitKey()
