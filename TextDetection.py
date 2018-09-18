import cv2
import numpy as np


###################################################

img = cv2.imread('Sample Data Receipts-page-004.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('img',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
###################################################

kernel = np.ones((5,5),np.uint8)
img_grad = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)

#cv2.imshow('img',img_grad)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
###################################################

_,img_bin = cv2.threshold(img_grad,0,255,cv2.THRESH_OTSU)
#img_bin = cv2.medianBlur(img_bin,5)
#cv2.imshow('img',img_bin)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
###################################################
kernel2 = np.ones((1,20),np.uint8)
#kernel3 = np.ones((10,1),np.uint8)
img_closing = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, kernel2)
#img_closing = cv2.morphologyEx(img_closing, cv2.MORPH_CLOSE, kernel3)

#cv2.imshow('img',img_closing)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

###################################################

_,contours,hierarchy = cv2.findContours(img_closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

#cv2.drawContours(img, contours, -1, (0,255,0), 3)

#cv2.imshow('Contours', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

imgs = []

###################################################

for i in range(hierarchy.shape[1]):
    x,y,w,h = cv2.boundingRect(contours[i])
    if(not (h < 15 or w < 15) and (h < img.shape[0]/20)):
        count = cv2.countNonZero(img_bin[y:y+h,x:x+w])
        r = count/(w*h)
        #if (r>0.25):
        
        cropped = img_gray[y:y+h,x:x+w]
        imgs.append(cropped)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0 ,255), 3);
#        cv2.imshow('Contours', cropped)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    
        
cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
        




