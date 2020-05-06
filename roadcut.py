from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data,io,data_dir,filters, feature
from skimage.color import label2rgb
import skimage
import matplotlib.pyplot as plt
from PIL import Image

import cv2
import numpy as np
import random
import glob as gb
import copy

def Variance_compare(LBP_1,LBP_2,cluster_num):
    lbp_cluster_piece = (np.power(2,n_points)-1) / cluster_num
    LBP_Hist_1 = np.zeros((cluster_num),dtype=float)
    LBP_Hist_2 = np.zeros((cluster_num),dtype=float)

    for i in range(cluster_num):
        LBP_Hist_1[i] = len(LBP_1[(LBP_1 >= i * lbp_cluster_piece) & (LBP_1 < (i+1) * lbp_cluster_piece)])
        LBP_Hist_2[i] = len(LBP_2[(LBP_2 >= i * lbp_cluster_piece) & (LBP_2 < (i+1) * lbp_cluster_piece)])
    
    LBP_Hist_1 /= len(LBP_1)
    LBP_Hist_2 /= len(LBP_2)

    D0 = LBP_Hist_1 - LBP_Hist_2
    D0 = np.power(D0,2)
    D0 = np.sqrt(np.sum(D0))
    return D0
# fourcc = 0x00000021 #存取影片
# cv2.VideoWriter_fourcc('H', '2', '6', '4')
# videoWriter = cv2.VideoWriter('./cut3.mp4', fourcc , 24, (1392,512)) # 建立 VideoWriter 物件，輸出影片至 output.avi ,falus

height=375
width=1242
radius = 1 # LBP算法中范围半径的取值
n_points = 8* radius  # 领域像素点
vediocount=5

marker_0 = np.zeros((height,width),np.uint8)
marker_1 = np.zeros((height,width),np.uint8)
marker_2 = np.zeros((height,width),np.uint8)
roadbase = np.zeros((height,width),np.uint8)

# marker = np.zeros((height,width),np.uint8)

imgs = gb.glob("D:/F222/project_Road/data4/0000000???.png")
for S in range (0,1,1):
    for path in range(500,801,1):
        if path/10 <1 :
            image = cv2.imread(".\data4\\000000000%d.png" % path)
            print("000000000%d.png" % path)
        elif path/10 <10 :
            image = cv2.imread(".\data4\\00000000%d.png" % path)
            print("00000000%d.png" % path)
        elif path/10 <100 :
            image = cv2.imread(".\data4\\0000000%d.png" % path)
            print("0000000%d.png" % path)
        gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("gray",0)
        cv2.imshow('gray',gray)
        lbp = local_binary_pattern(gray, n_points, radius)
        # cv2.namedWindow("lbp",0)
        # cv2.imshow('lbp',lbp)

        kernel_size = 15+(S*2) ####
        kernel_range = kernel_size // 2
        marker = np.zeros((height,width),np.uint8)

        if vediocount%4 == 0 :
            marker = marker_1
            marker_0 = np.zeros((height,width),np.uint8)
        elif vediocount%4 == 1 :
            marker = marker_0
            marker_1 = np.zeros((height,width),np.uint8)

        for w in range((kernel_size*2),(width-kernel_size),kernel_range):
            for h in range((kernel_size*2),(height-kernel_size),kernel_range):
                
                Diff1=Variance_compare(lbp[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range],lbp[h-kernel_range:h+kernel_range,w:w+(kernel_range*2)],32)
                #print("Diff1=",Diff1)
                Diff2=Variance_compare(lbp[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range],lbp[h:h+(kernel_range*2),w-kernel_range:w+kernel_range],32)
                #print("Diff2=",Diff2)
                Diff3=Variance_compare(lbp[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range],lbp[h-kernel_range:h+kernel_range,w-(kernel_range*2):w],32)
                #print("Diff3=",Diff3)
                Diff4=Variance_compare(lbp[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range],lbp[h-(kernel_range*2):h,w-kernel_range:w+kernel_range],32)
                #print("Diff4=",Diff4)
                
                # print("Diff_total = ",(Diff1+Diff2+Diff3+Diff4))
                if ((Diff1+Diff2+Diff3+Diff4)/4)<0.9:
                    marker[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range] = 255
                    marker_0[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range] = 255
                    marker_1[h-kernel_range:h+kernel_range,w-kernel_range:w+kernel_range] = 255
        
        marker[0:10,0:10] = 255
        marker[0:10,(width-10):width] = 255
        marker[(int(height/2)-5):(int(height/2)+5),0:10] = 255
        marker[(int(height/2)-5):(int(height/2)+5),(width-10):width] = 255
        marker[0:10,(int(width/3)):(int(width/3*2))] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        marker = cv2.morphologyEx(marker, cv2.MORPH_CLOSE, kernel)
        # marker = cv2.dilate(marker,kernel)

        # cv2.namedWindow("marker",0)
        # cv2.imshow('marker',marker)
        # cv2.imwrite('Results/marker.jpg',marker)
        # water shed---------------------------------------------------------------------------------
        sure_fg = np.uint8(marker)
        ret, markers = cv2.connectedComponents(sure_fg)

        markers = cv2.watershed(image,markers) 
        cut = np.zeros((height,width),np.uint8)
        cut[markers == -1] = [255]

        # cv2.namedWindow("out",0)
        # cv2.imshow("out", cut)
        # cv2.imwrite('Results/watershed.jpg',cut)
        ###---------------------------------------------------------------------------------------

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        dilate = np.zeros((height,width),np.uint8)
        # dilate = cv2.erode(cut,kernel)
        dilate = cv2.dilate(cut,kernel)

        dilate = 255- dilate

        # cv2.namedWindow("dilate",0)
        # cv2.imshow("dilate", dilate)
        # # cv2.imwrite('Results/watershed_dilate.jpg',dilate)

        ###connectedComponents-----------------------------------------------------------------------
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate)
        # print ("nlabels",nlabels)
        # print ("centroids",centroids.shape)

        max_size = 0 
        road_number = -1
        roadbase = np.zeros((height,width),np.uint8)

        for i in range(0,nlabels-1,1):
            x,y = centroids[i]
            # print("i",i)
            # print("x : ",x," y : ",y)
            # cv2.putText(roadbase, str(centroids[i]), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if  (y>(height/3*2)) and (x<((width/3)*2)) and (x>width/3) and np.all(dilate[labels==i] == 255 ):
                if(dilate[labels==i].size > max_size):
                    max_size =  dilate[labels==i].size 
                    road_number = i
        if road_number == -1 :
            for i in range(0,nlabels-1,1):
                x,y = centroids[i]
                # print("i",i)
                # print("x : ",x," y : ",y)
                # cv2.putText(roadbase, str(centroids[i]), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if  (y>(height/2)) and (x<((width/3)*2)) and (x>width/3) and np.all(dilate[labels==i] == 255 ):
                    if(dilate[labels==i].size > max_size):
                        max_size =  dilate[labels==i].size 
                        road_number = i
        if road_number == -1 :
            for i in range(0,nlabels-1,1):
                x,y = centroids[i]
                # print("i",i)
                # print("x : ",x," y : ",y)
                # cv2.putText(roadbase, str(centroids[i]), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if  (y>(height/3)) and (x<((width/4)*3)) and (x>width/4) and np.all(dilate[labels==i] == 255 ):
                    if(dilate[labels==i].size > max_size):
                        max_size =  dilate[labels==i].size 
                        road_number = i
        if road_number == -1 :
            for i in range(0,nlabels-1,1):
                x,y = centroids[i]
                # print("i",i)
                # print("x : ",x," y : ",y)
                # cv2.putText(roadbase, str(centroids[i]), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255, 255), 1, cv2.LINE_AA)

                if  (y>(height/2)) and (x<((width/4)*3)) and (x>width/4) and np.all(dilate[labels==i] == 255 ):
                    if(dilate[labels==i].size > max_size):
                        max_size =  dilate[labels==i].size 
                        road_number = i
        print ("road_number : ",road_number)

        roadbase[labels==road_number] = (255)

        # print (labels)
        # cv2.namedWindow("labels",0)
        # cv2.imshow("labels", roadbase)
        # cv2.imwrite('Results/roadbase.jpg',roadbase)

        for i in range(0,nlabels,1):
            x,y = centroids[i]
            Diff=Variance_compare(lbp[ labels==road_number],lbp[ (labels==i) & (dilate==255) ],32)      
            # print("x : ",x,"y : ",y)
            # print(Diff)

            # if Diff < 0.04 :
            #     roadbase[ (labels==i)] = (255)
            cv2.putText(cut, str(Diff), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (255, 255,255), 1, cv2.LINE_AA)

            if (((y >  height/2) and (x<((width/3)*2)) and (x>width/3)) or ((y >  height/3*2) and (x<((width/4)*3)) and (x>width/4))) and np.all(dilate[labels==i] == 255 ):     
                if  Diff < 0.09 :
                    print("x : ",x,"y : ",y)
                    print(Diff)
                    roadbase[ (labels==i)] = (255)
                    cv2.putText(roadbase, str(Diff), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 0, 0), 1, cv2.LINE_AA)
            elif (y > height/4*3) and np.all(dilate[labels==i] == 255 ):
                if  Diff < 0.06 :
                    print("x : ",x,"y : ",y)
                    print(Diff)
                    roadbase[ (labels==i)] = (255)
                    cv2.putText(roadbase, str(Diff), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 0, 0), 1, cv2.LINE_AA)
            elif (y > height/5*4) and np.all(dilate[labels==i] == 255 ):
                if  Diff < 0.12 :
                    print("x : ",x,"y : ",y)
                    print(Diff)
                    roadbase[ (labels==i)] = (255)
                    cv2.putText(roadbase, str(Diff), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
        # print (labels)
        # cv2.namedWindow("labels2",0)
        # cv2.imshow("labels2", roadbase)
        # # filename = ("data4_R3_%d_labels"  % kernel_size)
        # cv2.imwrite('data4_R3_%d' % kernel_size + '_labels/%d.jpg' % path ,roadbase)

        # cv2.namedWindow("out",0)
        # cv2.imshow("out", cut)
        # cv2.imwrite('data4_R3_%d' % kernel_size + '_cut/%d.jpg' % path ,cut)

        ret,roadbase = cv2.threshold(roadbase,127,255,cv2.THRESH_BINARY)

        kernel = np.ones((9,9),np.uint8)
        roadbase = cv2.morphologyEx(roadbase, cv2.MORPH_CLOSE, kernel)
        roadbase = cv2.morphologyEx(roadbase, cv2.MORPH_OPEN, kernel)

        # cv2.namedWindow("labels3",0)
        # cv2.imshow("labels3", roadbase)
        # videoWriter.write(roadbase)

        # water shed2---------------------------------------------------------------------------------
        roadbase = cv2.erode(roadbase,kernel,iterations = 5)
        marker[roadbase== 255] = 255 
        cv2.imwrite('data4_R3_%d'  % kernel_size + '_maker/%d.jpg' % path ,marker)
        sure_fg = np.uint8(marker)
        ret, markers = cv2.connectedComponents(sure_fg)

        markers = cv2.watershed(image,markers) 
        cut2 = np.zeros((height,width),np.uint8)
        cut2[markers == -1] = [255]
        ###---------------------------------------------------------------------------------------
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        cut2 = cv2.dilate(cut2,kernel)
        cut2 = 255 - cut2
        cv2.namedWindow("out2",0)
        cv2.imshow("out2", cut2)
        ###connectedComponents-----------------------------------------------------------------------

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(cut2)
        max_size = 0 
        road_number = -1
        roadbase = np.zeros((height,width),np.uint8)

        for i in range(0,nlabels-1,1):
            x,y = centroids[i]
            if  (y>(height/3*2)) and (x<((width/3)*2)) and (x>width/3) and np.all(cut2[labels==i] == 255 ):
                if(cut2[labels==i].size > max_size):
                    max_size =  cut2[labels==i].size 
                    road_number = i
        if road_number == -1 :
            for i in range(0,nlabels-1,1):
                x,y = centroids[i]
                if  (y>(height/2)) and (x<((width/3)*2)) and (x>width/3) and np.all(cut2[labels==i] == 255 ):
                    if(cut2[labels==i].size > max_size):
                        max_size =  cut2[labels==i].size 
                        road_number = i
        # print ("road_number : ",road_number)
        out = np.zeros((height,width),np.uint8)
        out[labels==road_number] = (255)

        cv2.namedWindow("Results",0)
        cv2.imshow("Results", out)   
        cv2.imwrite('data4_R3_15_mak/%d.jpg' % path ,out)
        
        vediocount +=1
        if cv2.waitKey(10) == 27:
            break
        cv2.waitKey(1)

# videoWriter.release()
cv2.destroyAllWindows() 

