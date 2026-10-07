import cv2,glob,sys,numpy as np
fs=sorted(glob.glob(sys.argv[1]))
ims=[cv2.resize(cv2.imread(f),(640,360)) for f in fs]
for im,f in zip(ims,fs): cv2.putText(im,f.split('/')[-1][-8:-4],(10,30),0,1,(0,255,255),2)
while len(ims)%4: ims.append(np.zeros_like(ims[0]))
cv2.imwrite(sys.argv[2],np.vstack([np.hstack(ims[i:i+4]) for i in range(0,len(ims),4)]))
