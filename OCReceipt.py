import requests
import os
from os import listdir
import json
import pillow_heif 
from pillow_heif import register_heif_opener

import gspread
from oauth2client.service_account import ServiceAccountCredentials

from PIL import Image
import pytesseract
import argparse
import cv2
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet 

scope = ['https://www.googleapis.com/auth/spreadsheets', "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json',scope)

client = gspread.authorize(creds)

sheet = client.open('SheetConnectionTestSheet').sheet1


receiptOcrEndpoint = 'https://ocr.asprise.com/api/v1/receipt' # Receipt OCR API endpoint
 

folder_dir = "./Receipt"


for images in os.listdir(folder_dir):
    # Method 1
    # r = requests.post(receiptOcrEndpoint, data = { \
    # 'client_id': 'TEST',        # Use 'TEST' for testing purpose \
    # 'recognizer': 'auto',       # can be 'US', 'CA', 'JP', 'SG' or 'auto' \
    # 'ref_no': 'ocr_python_123', # optional caller provided ref code \
    # }, \
    # files = {"file": open(f"./Receipt/{images}", "rb")})

    # json_data = json.loads(r.text)
    # print(json_data) 

    # print(json_data['receipts']['date']) 
    # print(json_data['merchant_name']) 
    # print(json_data['items'][0]['amount']) 
    # print(json_data['total']) 

    # Method 2
    image=cv2.imread(f"./Receipt/{images}")
    #Convert image to black and white
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = cv2.medianBlur(gray,5)
    thresh1 = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
 
    # Applying dilation on the threshold image
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    
    # Finding contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_NONE)
    # text=(pytesseract.image_to_string(thresh1))
    # print(text)
    im2 = image.copy()
 

    # Looping through the identified contours
    # Then rectangular part is cropped and passed on
    # to pytesseract for extracting text from it
    # Extracted text is then written into the text file
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Cropping the text block for giving input to OCR
        cropped = im2[y:y + h, x:x + w]
        
        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)
        
        # Appending the text into file
        print(text)

        

    #identify the date

    match=re.findall(r'\d\d[/.-]\d\d[/.-]\d+', text)

    date=" "
    date=date.join(match)
    print(date)
    date = date

    #identify Total

    tokens = text.split()
    for n,i in enumerate(tokens):  
        if 'TOTAL' in i:
            print(tokens[n+1])
            total = tokens[n+1]
            break
    
    #identify Title
    title = tokens[0]
    print(tokens[0])


    #Locate last row in googlesheet
    last_row = len(sheet.get_values())

    #Write into googlesheet
    sheet.insert_row([date,title, "","", "","",total ],last_row +1 )
   