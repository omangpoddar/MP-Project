import numpy as np
import cv2
import os
import glob



def frames(path_list):
	no_of_folder=len(path_list)
	print no_of_folder
	k=0
	for i in range(no_of_folder):
		for filename in glob.glob(path_list[i]):
			print filename
			cap = cv2.VideoCapture(filename)
			t=0
			while True:
				(grabbed,frame) = cap.read()
				if not grabbed:
					break
				t+=1
			fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
			vid_duration =  t/fps
			time_length = vid_duration
			frame_seq = time_length*fps
			print (frame_seq)
			i = 1.0
			
			while (i < frame_seq):
				print (i)
				
				cap.set(1,i)
				ret, frame = cap.read()
				print (frame)
				crop_img = frame[200:520,300:980]
				if i==1:
					dirname='Folder'+str(k)
					os.mkdir(dirname)
				cv2.imwrite(os.path.join(dirname,str(i)+'.jpg'),crop_img)
				i=i+20
				
					
			k=k+1
				


if __name__ == "__main__":
	# Make Pathlist
	path_list=[]
	path_list.append("Videos/four/*.mp4")
	print path_list
	path_list.append("Videos/run/*.mp4")
	path_list.append("Videos/six/*.mp4")
	path_list.append("Videos/wicket/*.mp4")
	frames(path_list)

cv2.destroyAllWindows()
