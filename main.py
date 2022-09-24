import cv2 as cv
import numpy as np
from deskew import determine_skew 
import os
import argparse

def determineAngle(img):
    try:
        return determine_skew(img)
    except:
        raise Exception("can't determine the image angle")

def fixRotation(img):
    angle = determineAngle(img)
    rows = img.shape[0]
    cols = img.shape[1]

    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angle,1)

    return cv.warpAffine(img,M,(cols,rows))

def getAutoEdges(img):
    v = np.median(img)
    sigma = 0.33

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv.Canny(img, lower, upper, apertureSize=3)

def getEdges(img):
    lower = 10
    upper = 300
    return cv.Canny(img, lower, upper, apertureSize=3)

def getContours(edges):
    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    return contours

def getPossibleContour(contours):
    sortedContours = sorted(contours, key=cv.contourArea, reverse=True)
    
    for contour in sortedContours: 
        approx = cv.approxPolyDP(contour, 0.018*cv.arcLength(contour, True), True)
        
        if len(approx) == 4:
            approxContour = approx
            return approxContour

def run(files): 
    output = "./output"
    for file in files:
        img = cv.imread(file.path)
        
        rotated_img = fixRotation(img)
        
        gray = cv.cvtColor(rotated_img.copy(), cv.COLOR_RGB2GRAY)

        filtered_img = cv.bilateralFilter(gray, 11, 17, 17)

        edges = getAutoEdges(filtered_img)
        cnts = getContours(edges)
        cnts_img = cv.drawContours(rotated_img.copy(), cnts, -1, (0,255,0), 1)
        cv.imshow('contours', cnts_img)

        cnt = getPossibleContour(cnts)

        if len(cnt) > 0:
            plate_img = cv.drawContours(rotated_img.copy(), [cnt], 0, (0,0,255), 2)
            cv.imshow('ANPR', plate_img)
            x,y,w,h = cv.boundingRect(cnt) 
            new_img = rotated_img[y:y+h,x:x+w]

            if not os.path.exists(output):
                os.mkdir(output)
            cv.imwrite('./output/'+file.name,new_img)

        cv.waitKey(100)

def main():
    parser = argparse.ArgumentParser("Vechicle number plate recognition program")

    parser.add_argument("-i", "--input_dir", required=True)

    args = parser.parse_args()
    files = []
    try:
        if args.input_dir:
            with os.scandir(args.input_dir) as entries:
                for entry in entries:
                    if entry.is_file():
                        files.append(entry)

        
        run(files)
    except:
        return

main()