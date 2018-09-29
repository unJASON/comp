import xml.etree.cElementTree as et
from xml.dom.minidom import Document
import os


def extractAllXML(path):
    files = os.listdir(path)
    dict_xml = {}
    for file in files:
        objList = []
        root = et.parse(path + file).getroot()
        for obj in root.getchildren():
            objList.append(obj.find('Rect').attrib)
        dict_xml[file] = objList
    return dict_xml


# 保存单个XML
def saveXML(fileName, objects):
    doc = Document()
    objectList = doc.createElement('ObjectList')
    doc.appendChild(objectList)
    for obj in objects:
        object_dat = doc.createElement('Object')
        rec = doc.createElement('Rect')
        rec.setAttribute('x', str(obj['x']))
        rec.setAttribute('y', str(obj['y']))
        rec.setAttribute('width', str(obj['width']))
        rec.setAttribute('height', str(obj['height']))
        object_dat.appendChild(rec)
        objectList.appendChild(object_dat)
    # 将dom对象写入本地xml文件
    with open(fileName, 'wb+') as f:
        f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    pass
