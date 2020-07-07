import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv(r'C:\Users\7590 inspiron\Desktop\AIML\data\data_linear.csv')
dientich = data['Diện tích'].values
gia = data['Giá'].values
plt.plot(dientich,gia)
plt.show()

"""
*****************
"""

import cv2 

img = cv2.imread(r'C:\Users\7590 inspiron\Desktop\AIML\data\image.png')
w = img.shape[0]
h = img.shape[1]
h_w = int(img.shape[0]/2)
h_h = int(img.shape[1]/2)

# Crop 
crp_img = img[0:h_h, 0:h_w, : ]
cv2.imwrite(r'C:\Users\7590 inspiron\Desktop\AIML\data\cropped.png', crp_img)

# Resize
rs_img = cv2.resize(src = img, dsize = (h_h, h_w))
cv2.imwrite(r'C:\Users\7590 inspiron\Desktop\AIML\data\resized.png', rs_img)

# GaussianBlur
gb = cv2.GaussianBlur(img,(5,5),0)
cv2.imwrite(r'C:\Users\7590 inspiron\Desktop\AIML\data\GB-ed.png', gb)

# Using Canny Edge Detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert color to Gray
gr_gb = cv2.GaussianBlur(gray,(5,5),0)
img_canny = cv2.Canny(gr_gb,100,50) 
cv2.imwrite(r'C:\Users\7590 inspiron\Desktop\AIML\data\edge_dt-ed.png',img_canny)
