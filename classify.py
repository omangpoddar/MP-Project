import numpy as np
import matplotlib.pyplot as pl
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import cv2
import glob
import _pickle as cPickle
STANDARD_SIZE = (300, 167)
data=[]
Y=[]
for filename in glob.glob('D:\MP\project\Mp_project\crowd\*.jpg'):
        img=cv2.imread(filename)
        res = cv2.resize(img,STANDARD_SIZE, interpolation = cv2.INTER_CUBIC)
        flatten=res.flatten()
        data.append(flatten)
        Y.append([0])       
        #cv2.imshow('e',res)
#        print(filename)
#for filename in glob.glob('D:\MP\project\Mp_project\pitch\*.jpg'):
#        img=cv2.imread(filename)
#        res = cv2.resize(img,STANDARD_SIZE, interpolation = cv2.INTER_CUBIC)
#        flatten=res.flatten()
#        data.append(flatten)
#        Y.append([1])


for filename in glob.glob('D:\MP\project\Mp_project\Boundary\*.jpg'):
        img=cv2.imread(filename)
        res = cv2.resize(img,STANDARD_SIZE, interpolation = cv2.INTER_CUBIC)
        flatten=res.flatten()
        data.append(flatten)
        Y.append([2])   


for filename in glob.glob('D:\MP\project\Mp_project/Player/*.jpg'):
        img=cv2.imread(filename)
        res = cv2.resize(img,STANDARD_SIZE, interpolation = cv2.INTER_CUBIC)
        flatten=res.flatten()
        data.append(flatten)
        Y.append([3])   




for filename in glob.glob('D:\MP\project\Mp_project\Ground\*.jpg'):
        img=cv2.imread(filename)
        res = cv2.resize(img,STANDARD_SIZE, interpolation = cv2.INTER_CUBIC)
        flatten=res.flatten()
        data.append(flatten)
        Y.append([4])

for filename in glob.glob('D:\MP\project\Mp_project\batsman\*.jpg'):
        img=cv2.imread(filename)
        res = cv2.resize(img,STANDARD_SIZE, interpolation = cv2.INTER_CUBIC)
        flatten=res.flatten()
        data.append(flatten)
        Y.append([5])

for filename in glob.glob('D:\MP\project\Mp_project\Sky\*.jpg'):
        img=cv2.imread(filename)
        res = cv2.resize(img,STANDARD_SIZE, interpolation = cv2.INTER_CUBIC)
        flatten=res.flatten()
        data.append(flatten)
        Y.append([6])   




Y=np.array(Y)
Y=Y.ravel()
        
data=np.array(data)
pca = RandomizedPCA(n_components=5)
X = pca.fit_transform(data)


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)

#############################...... now running KNN ..........
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
with open('trained_classifier.pkl','wb') as fid:
        cPickle.dump(knn,fid)

##################### EVALUATING THE MODEL
knn1 = pickle.load(open('trained_classifier.pkl', 'rb'))
y_pred = knn1.predict(X_test)

print("Test set predictions:\n {}".format(y_pred))

#print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

#We can also use the score method of the knn object....
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))




