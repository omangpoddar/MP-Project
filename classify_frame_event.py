 #
#
# Testing Classifier
#
#
#  Author: Harshit Joshi
#  Date: 16/4/2017
#
#  Content-based image classification :: VBoW + Randomized PCA + KNeighborsClassifier  
#
#  

from pymining import seqmining
import pickle as cp
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
counter=0
filename_list=[]

def isSubSequence(string1, string2, m, n):
    # Base Cases
    if m == 0:    return True
    if n == 0:    return False
 
    # If last characters of two strings are matching
    if string1[m-1] == string2[n-1]:
        return isSubSequence(string1, string2, m-1, n-1)
 
    # If last characters are not matching
    return isSubSequence(string1, string2, m, n-1)

def EPR(img):
	edges = cv2.Canny(img,100,200)

	numberOfTruePixels = cv2.countNonZero(edges)
	# r=float(numberOfTruePixels)/float(img.size)
	epr=numberOfTruePixels
	return epr

def PPR(img):
	h_value=0
	rows,cols,t = img.shape
	for i in range(rows):
		for j in range(cols):
			if img[i][j][0]>150 and img[i][j][0]<175 and img[i][j][1]>160 and img[i][j][1]<190 and img[i][j][2]>130 and img[i][j][2]<150   :
				h_value=h_value+1

	# ppr=(float)(h_value)/(float)(img.size)
	ppr=h_value
	return ppr

def GPR(img):
	hsv_image=cv2.cvtColor(img,cv2.COLOR_BGR2HSV);
	h, s, v = cv2.split(hsv_image)
	h_value=0
	rows,cols = h.shape
	for i in range(rows):
		for j in range(cols):
			k = h[i,j]
			if k>48 and k<68:
				h_value=h_value+1
	# gpr=(float)(h_value)/(float)(h.size)
	gpr= h_value 
	return gpr

def VisualFeatures(X):
	X_Visual=[]
	for img in X:
		gpr=GPR(img)
		ppr=PPR(img)
		epr=EPR(img)
		X_Visual.append([gpr,ppr,epr])
	X_Visual=np.array(X_Visual)
	
	## Scaling the words for a uniform scale
	scale = StandardScaler().fit(X_Visual)
	image_features = scale.transform(X_Visual)  		## for better prediction

	return X_Visual
	
def VBoW(X,dict_size):
	X_VBoW=[]
	# _init_  sift 
	sift = cv2.xfeatures2d.SIFT_create()
	point_list=[]    
	desc_list=[]
	no_kp_list=[]
	
	try:
		#1. Add desc to desc_list and points to point_list	 
		for img in X:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #as sift is 2d_feature
			kp, des = sift.detectAndCompute(gray, None)
			no_kp_list.append(len(kp))
			
			
			if len(kp) ==0 :      
				desc_list.append(np.zeros((128,)))  #appending empty list of descriptors-- with zero entry
				# point_list.append(np.zeroes((128))
				# should we pass this to kmeans ???  
				# kmean may assign it to richer cluster_point than the point itself
			else : #Our img was rich ... {no._of_kp >=10}
				desc_list.append(des)
				for i in des:  #des -> our descriptor
					point_list.append(i)
					

		#2 KMeans cluster :
		kmeans = KMeans(n_clusters=dict_size, n_jobs=-1)  #init kmeans
		point_list=np.array(point_list)
		print("point_list shape=",point_list.shape)
		kmeans.fit(point_list)				 	
		
		#validation check
		if len(desc_list)==len(X) and len(no_kp_list)==len(X) :
			print("All images are processed.. :)")
		else :
			print("ERROR : Missed/Dropped some images ")
		
		
		# 3. Building Visual Daictionary
		# init X_VBoW 	
		X_VBoW = np.zeros((len(X),dict_size), "float32")  
		
		for i in range(len(X)):
		    if no_kp_list[i]<10 :
		    	continue
		    else :
			    for j in desc_list[i]:
			    	j=np.array(j).reshape(1,-1)
			    	word=kmeans.predict(j)
			    	X_VBoW[i][word] += 1
			
		## Scaling the words for a uniform scale
		scale = StandardScaler().fit(X_VBoW)
		image_features = scale.transform(X_VBoW)  		## for better prediction


	except Exception as i:	
		print("Error while creating VBoW")
		print(i)

	return X_VBoW

def clean_img(img) :
	#prepreocess img : normalize and resize
	try:
		STANDARD_SIZE = (int(img.shape[1]/2), int(img.shape[0]/2))
		resized_img = cv2.resize(img,STANDARD_SIZE, interpolation = cv2.INTER_AREA) # inter_area for shrinking
		#cv2.normalize(resized_img,resized_img, 0, 255, cv2.NORM_MINMAX)   #normType= MINMAX #can try other norms
		## ! Is normalizing any good ? 
	except Exception as i:	
		print("Error while cleaning")
		print(i)	
	return resized_img 

def pca_features(X,n_comp) :  # n_comp= no. of components
	
	n_img=len(X)

	# for i in range(n_img):
	#  	X_flat.append(X[i].flatten())
	# #!!! Try writing above lines using numpy index in one line(w/o looping) !!! X.reshape(len(X),-1) ?
	# X_flat=np.array(X_flat)
	
	X_flat=X.reshape(len(X),-1)
	print("Shape of X in pca_func",X.shape)
	print("Shape of X_flat in pca_func",X_flat.shape)
	print("\n")

	pca = PCA(n_components=n_comp, svd_solver='randomized') #initialize
	X_new=pca.fit_transform(X_flat)
	return X_new


def load_data(path_list) :
	# Function than return numpy arrays : X_train and y  (ready for ML model)
	X_train=[]
	y_train=[]
	global counter
	global filename_list
	no_of_folder=len(path_list)
	try:
		for y_label in range(no_of_folder):
			for filename in glob.glob(path_list[y_label]):
				img=cv2.imread(filename)
				filename_list.append(filename)
				img_cleaned=clean_img(img)
				
				X_train.append(img_cleaned)
				y_train.append(y_label)
				#print(path_list[i])
				counter+=1
	except Exception as i:	
		print("Error in loading data")
		print(i)
	
	X_train = np.array(X_train)
	y=np.array(y_train)
	
	
	return X_train,y


def FeatureUnion(X):
	X_pca = pca_features(X,n_comp=5)
	X_VBoW = VBoW(X,dict_size= 14)
	X_Visual = VisualFeatures(X)    # gpr+ppr+epr
	
	X = np.concatenate([X_pca,X_VBoW,X_Visual],axis=1)

	return X

if __name__ == "__main__":
	# Make Pathlist
	# global filename_list
	#path_list=[]
    
	#path_list.append("Images/Folder0/*.jpg")
	#path_list.append("Images/Folder1/*.jpg")
	path_list = glob.glob('Images/*/')
	
	#print(path_list[1])
	s=[]
	for k in range(len(path_list)):
		print(k)
		p=[]
		path_list[k]+='*.jpg'
		p.append(path_list[k])
		X,y = load_data(p)
		print (X.shape , y.shape)
		print("\n")
		#Prepare Data : Get Features 
		X_pca = pca_features(X,n_comp=5)
		print("X_shape after pca=", X_pca.shape)
		X_VBoW = VBoW(X,dict_size= 14)
		X_Visual = VisualFeatures(X)    # gpr+ppr+epr
		X = np.concatenate([X_pca,X_VBoW,X_Visual],axis=1)
		clf=cp.load(open("AdaBoost.pkl",'rb'))
		output=clf.predict(X)
		print (output)
		no_of_frames=len(output)
		
		str1=""
		for i in range(no_of_frames) :
			if output[i]==0:
				str1=str1+'P'    #pitch
			if output[i]==1:
				str1=str1+'B'   #batsmen
			if output[i]==2:
				str1=str1+'G'   #Ground
			if output[i]==3:
				str1=str1+'p'    #player
			if output[i]==4:
				str1=str1+'b'   #boundary
			if output[i]==5:
				str1=str1+'C'   #crowd
			if output[i]==6:
				str1=str1+'S'   #sky
		s.append(str1)
		#print(str1)
	#print(len(s))	
	
	print (s)
	freq_seqs = seqmining.freq_seq_enum(s, 2)
	print(sorted(freq_seqs))
	a=list(freq_seqs)
	a1=[]
	print(a)
	for i in range(len(freq_seqs)):
		s1=""
		for j in range(len(a[i][0])):
			s1 = s1+(a[i][0][j])
		a1.append(s1)
	print(a1)
	file = open("t.txt","w")
	for i in range(len(a1)):
		file.write(a1[i]+"\n")
	file.close() 
	'''ans={0:"pitch",1:"batsmen",2:"ground",3:"player",
		 4:"boundary",5:"crowd",6:"sky"}
	for i in range(len(output)) :
		img=cv2.imread(filename_list[i])
		plt.imshow(img)
		plt.title(ans[output[i]])
		plt.show()
	''' 
