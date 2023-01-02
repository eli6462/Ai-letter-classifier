import cv2
import matplotlib.pyplot as plt

img = cv2.imread('b.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.bitwise_not(img)
img = cv2.GaussianBlur(img, (7,7), 0)

# !!! resize the img so it won't be too big (if it's big)
if len(img) > 1500 or len(img[0]) > 1500:
    img = cv2.resize(img, (int(len(img[0])/2.5),int(len(img)/2.5)), interpolation = cv2.INTER_CUBIC)

plt.imshow(img)
plt.show()

#threshold filter to image(make the dark areas absolutely black)
sum = 0
after_sum = 0
for i in range(len(img)):
    for j in range(len(img[0])):
        #sum += img[i, j]
        if img [i,j] < 155:
            img[i,j] = 0
        #after_sum += img[i,j]
#avg = sum / (len(img) * len(img[0]))
#after_avg = after_sum / (len(img) * len(img[0]))
#print("avg = " + str(avg))
#print("after_avg = " + str(after_avg))

plt.imshow(img)
plt.show()

#crop to square
points = cv2.findNonZero(img)
x, y, w, h = cv2.boundingRect(points)
if (w > 0 and h > 0):
    if w > h:
      y = y - (w-h)//2
      img = img[y:y+w, x:x+w]
    else:
      x = x - (h-w)//2
      img = img[y:y+h, x:x+h]


img = cv2.resize(img, (28,28), interpolation = cv2.INTER_CUBIC)


plt.imshow(img)
plt.show()


