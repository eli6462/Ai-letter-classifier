from emnist import extract_training_samples
import cv2
import numpy
import pickle
import matplotlib.pyplot as plt

X, y = extract_training_samples('letters')

# # rescale pixels value to be between 1 and 0
X = X / 255.

# set first 60,000 instances as train, and last 10,000 as test
X_train, X_test = X[:60000], X[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

# reshape into a 1d array
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)

# import the pretrained model
with open('mlp2.pickle', 'rb') as f:
    mlp2 = pickle.load(f)

# test the model
print("Test set score: %f" % mlp2.score(X_test, y_test))


'''  classify the images of the letters'''

images = []


images.append(cv2.imread('z.jpg',cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('b.jpg',cv2.IMREAD_GRAYSCALE))
images.append(cv2.imread('c.jpg',cv2.IMREAD_GRAYSCALE))

for img in images:
    img = cv2.bitwise_not(img)
    #img = cv2.GaussianBlur(img, (7,7), 0)

    # apply filter to image(make the dark areas absolutely black)
    counter = 0
    sum = 0
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img [i,j] < 155:
                #img[i,j] = 0
                counter +=1
                sum += img[i,j]
    avg = int(sum/counter)
    #print("avg = " + str(avg))

    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i,j] < avg + 50:
                img[i,j] = 0

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

    #show the image
    plt.imshow(img)
    plt.show()

    #predict
    img = (numpy.array(img)).reshape(1,784)
    prediction = mlp2.predict(img)
    print(prediction)
    prediction = str(chr(prediction[0]+64))

    print("the letter is: " + prediction)