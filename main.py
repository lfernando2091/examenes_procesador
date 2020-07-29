import numpy as np
import sys
import cv2

img = cv2.imread('./img05.jpeg')
cRgba = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(cRgba, 7)
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
closeKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, closeKernel, iterations=3)

def findByArea(item): 
  min_area = 100
  max_area = 1000
  area = cv2.contourArea(item)
  if area > min_area and area < max_area:
    return True
  else: 
    return False

def sort_contours(cnts, method="left-to-right"):
  reverse = False
  i = 0

  if method == "right-to-left" or method == "bottom-to-top":
      reverse = True

  if method == "top-to-bottom" or method == "bottom-to-top":
      i = 1

  boundingBoxes = [cv2.boundingRect(c) for c in cnts]
  (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
      key=lambda b:b[1][i], reverse=reverse))
  return (cnts, boundingBoxes)


cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# No contours, stop the execution
if len(cnts) <= 1:
  sys.exit()

cnts = list(filter(findByArea, cnts)) 
(cntsFilt, boundingBoxes) = sort_contours(cnts, "left-to-right")

markRefLetf = (
    cntsFilt[0], 
    cntsFilt[1], 
    cntsFilt[2])

markRefRight = (
    cntsFilt[len(cntsFilt) - 3], 
    cntsFilt[len(cntsFilt) - 2],
    cntsFilt[len(cntsFilt) - 1])

(markRefLetf, boundingBoxes1) = sort_contours(markRefLetf, "top-to-bottom")
(markRefRight, boundingBoxes2) = sort_contours(markRefRight, "bottom-to-top")

x1,y1,w1,h1 = cv2.boundingRect(markRefLetf[0])
x2,y2,w2,h2 = cv2.boundingRect(markRefRight[0])
ROI = close[y1+h1:y2, x1+h1:x2]
OriginalROI = img[y1+h1:y2, x1+h1:x2]
cv2.rectangle(img, (x1, y1), (x2 + w2, y2 + h2), (36,255,12), 2)

ROI = cv2.medianBlur(ROI, 7)
ROI = cv2.filter2D(ROI, -1, sharpen_kernel)

circles = cv2.HoughCircles(ROI, cv2.HOUGH_GRADIENT, 2, 25, param1=7, param2=20, minRadius=7, maxRadius=15)
if circles is not None:
  circles = np.uint16(np.around(circles))
  # for i in circles[0, :]:
  #   cv2.circle(OriginalROI, (i[0], i[1]), i[2], (0, 255, 0), 2)
  #   cv2.circle(OriginalROI, (i[0], i[1]), 2, (0, 0, 255), 3)
else:
  print("No circles")

def drwSection(indice, img, height, width, rows, colums, posX, posY, counterIndex):
  cellWidth = width / colums
  cellHeight = height / rows
  cellWidth = np.uint16(np.around(cellWidth))
  cellHeight = np.uint16(np.around(cellHeight))
  thickness = 2
  cellColor = (0, 102, 255)
  # cv2.putText(img, "Section {}".format(indice + 1), (posX, posY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
  for r in range(rows):
    counterIndex = counterIndex + 1
    cellsMatrix = []
    for c in range(colums):
      colIndex = c + 1
      rowIndex = r + 1
      cellStartPoint = (posX + (cellWidth * c), posY + (cellHeight * r)) 
      cellEndPoint = (posX + (cellWidth * colIndex), posY + (cellHeight * rowIndex)) 
      cellLocation = (0, cellStartPoint, cellEndPoint)
      cellsMatrix.append(cellLocation)
      # cv2.rectangle(img, cellStartPoint, cellEndPoint, cellColor, thickness)
    rowsMatrix.append((counterIndex, cellsMatrix))

if circles is not None:
  height=1020
  width=182
  posX=26
  posY=19
  rows=25
  colums=4
  sections=4
  spaceBetSections = 33
  
  rowsMatrix = []
  start_point = (posX, posY) 
  end_point = (posX + width, posY + height) 
  color = (255, 0, 0) 
  thickness = 2
  counterIndex = 0
  for s in range(sections):
    secStartPoint = posX + (s * (width + spaceBetSections))
    counterIndex = (s * rows)
    drwSection(s, OriginalROI, height, width, rows, colums, secStartPoint, posY, counterIndex)
else:
  print("No circles")

if circles is not None:
  circles = sorted(circles[0], key=lambda x: x[1], reverse=False)
  rowsMatrix = sorted(rowsMatrix, key=lambda x: x[1][0][1][1], reverse=False)
  colClassifierCounter = 0 
  result = []
  lengthRowsMatrix = len(rowsMatrix) 
  for i in range(lengthRowsMatrix): 
    sizeSectionColum = (rowsMatrix[i][1][0][1][0], rowsMatrix[i][1][colums - 1][2][0])
    sizeRow = (rowsMatrix[i][1][0][1][1], rowsMatrix[i][1][0][2][1])
    locatedInCurrentSection = list(
        filter(
            lambda x: (x[0] >= sizeSectionColum[0] and x[0] < sizeSectionColum[1]) and
                      (x[1] >= sizeRow[0] and x[1] < sizeRow[1]), 
            circles)
        ) 
    cellCounter = 0
    for cell in rowsMatrix[i][1]:
      cellCounter += 1
      widthColum = (cell[1][0], cell[2][0])
      if(len(locatedInCurrentSection) > 0):
        if locatedInCurrentSection[0][0] >= widthColum[0] and locatedInCurrentSection[0][0] < widthColum[1]:
          letterTranlation = ''
          if cellCounter == 1: 
            letterTranlation = 'A'
          if cellCounter == 2: 
            letterTranlation = 'B'
          if cellCounter == 3: 
            letterTranlation = 'C'
          if cellCounter == 4: 
            letterTranlation = 'D'
          cellLocation = (rowsMatrix[i][0], letterTranlation)
          result.append(cellLocation)
  result = sorted(result, key=lambda x: x[0], reverse=False)
  for elem in result:
    print(elem)
  # cv2_imshow(OriginalROI)
else:
  print("No circles")