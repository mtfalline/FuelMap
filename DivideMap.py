import cv2

image = cv2.imread("m_3912045_ne_10_060_20180916.tif")
#cv2.imshow("original", image)
print(image.shape)
ylim = int(image.shape[0] / 100)
xlim = int(image.shape[1] / 100)
for i in range(ylim):
    x = i * 100
    print(x)
    for j in range(xlim):
        y = j * 100
        cropped = image[x:x+100, y:y+100]
        #cv2.imshow("cropped", cropped)
        #cv2.waitKey(0)
        cv2.imwrite(r"C:\images\uthumbnail" + str(i) + "X" + str(j) + ".jpg", cropped)
