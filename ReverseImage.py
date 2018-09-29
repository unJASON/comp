import XmlProcess
import numpy as np
import cv2
import Models
tagPath = 'Annotations/'
imgPath = 'BMPImages/'
reverseAnnoPath = 'reverseAnno'
reverseBMPPath = 'reverseBMP'
imgObjMap = XmlProcess.extractAllXML(tagPath)
resImags, resAnswers = Models.mode(imgObjMap, imgPath)
#python 图像翻转,使用openCV flip()方法翻转
# xImg = cv2.flip(resImags[0],1,dst=None) #水平镜像




for imgObjK in imgObjMap.keys():
    imgObjK_noSufix = imgObjK[:-7]  #名字
    img = cv2.imread(imgPath + imgObjK_noSufix + '.bmp')  # img (h,w,channel)
    img = cv2.flip(img,1,dst=None)
    for obj in imgObjMap[imgObjK]:
        obj['x'] = img.shape[1] - int(obj['x']) - int(obj['width'])
        # cv2.rectangle(img, (int(obj['x']), int(obj['y'])),
        #               (int(obj['x'])+int(obj['width']),int(obj['y'])+int(obj['height']))
        #               , color=(0, 0, 0), thickness=2)
    savePath = reverseAnnoPath + '/reverse_' + imgObjK
    XmlProcess.saveXML(savePath, imgObjMap[imgObjK])
    cv2.imwrite(reverseBMPPath+'/reverse_'+ imgObjK_noSufix + '.bmp',img)

