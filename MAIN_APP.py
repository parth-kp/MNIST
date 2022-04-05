import numpy as np
import cv2
from numpy import testing
from numpy.lib.type_check import imag
import pygame
from pygame import image
from pygame.locals import *
from keras.models import load_model
from tensorflow.python.keras.backend import constant

WINDOWSIZEX = 250
WINDOWSIZEY = 250
BOUNDRYINC = 5
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255, 0, 0 )
IMAGESAVE = False
MODEL = load_model ("bestmodel.h5")

LABELS = {  0: "Zero", 1: "One",
            2: "Two", 3:"Three",
            4: "Four", 5:"Five",
            6: "Six", 7: "Seven",
            8: "Eight", 9: "Nine"}

pygame.init()
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("WhiteBoard")


iswriting = True
image_cnt = 1
PREDICT = False

number_xcord=[]
number_ycord=[]
font = pygame.font.Font('freesansbold.ttf', 32)
 
while True:
    
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()

        xcord,ycord = pygame.mouse.get_pos()
        if pygame.mouse.get_pressed() == (1,0,0):
            pygame.draw.circle(DISPLAYSURF, (255,255,255),  (xcord, ycord),10,10)
            number_xcord.append(xcord)    
            number_ycord.append(ycord)

        if event.type == pygame.MOUSEBUTTONDOWN:
            iswriting = True
            PREDICT=False
            IMAGESAVE = False

        if event.type == pygame.MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)
            try:
                rect_min_x , rect_max_x = max(number_xcord[0]-BOUNDRYINC, 0 ), min(WINDOWSIZEX, number_xcord[-1]+BOUNDRYINC)
                rect_min_Y , rect_max_Y = max(number_ycord[0]-BOUNDRYINC, 0 ), min(number_ycord[-1]+BOUNDRYINC, WINDOWSIZEX)
                number_xcord = []
                number_ycord = []
                IMAGESAVE = True
                PREDICT=True
                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x:1, rect_min_Y:rect_max_Y:1].T.astype(np.float32)
            except:
                continue
            
        if IMAGESAVE:
            pygame.image.save(DISPLAYSURF, "image.jpg")
            im = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

        if PREDICT:
            PREDICT=False
            DISPLAYSURF.fill (WHITE)
            image = cv2.resize(np.asarray(im),(28,28))
            image = np.pad(image,(10,10),'constant',constant_values =0)
            image = cv2.resize(np.asarray(image), (28,28)) /255
            label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1,28,28,1)))])
            text = font.render(label, True, (0,0,0), (255,255,255))
            textRect = text.get_rect()
            textRect.center=(125,125)
            DISPLAYSURF.blit(text, textRect)
            
        if event. type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill (BLACK)

        pygame.display.update()
