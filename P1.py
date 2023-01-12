import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import math

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]) :
    m = inImage.shape[0]
    n = inImage.shape[1]
    outImage = np.zeros((m,n))
    if inRange == [] :
        minI = np.amin(inImage)
        maxI = np.amax(inImage)
        inRange = [minI,maxI]
    for i in range(m):
        for j in range(n):
            outImage[i,j] = outRange[0] + ((outRange[1]-outRange[0])*(inImage[i,j]-inRange[0]))/(inRange[1]-inRange[0])
            if outImage[i,j] < outRange[0]:
                outImage[i,j] = outRange[0]
            elif outImage[i,j] > outRange[1]:
                outImage[i,j] = outRange[1]
    return outImage

def equalizeIntensity(inImage, nBins=256):
    image_hist = np.zeros(nBins)
    # frequency count of each pixel
    for pixel in inImage:
        image_hist[pixel] += 1
    cumsum = np.cumsum(image_hist)
    norm = (cumsum - cumsum.min()) * 255
    # normalization of the pixel values
    n_pixel = cumsum.max() - cumsum.min()
    norm_uniforme = norm / n_pixel
    norm_uniforme = norm_uniforme.astype('int')
    outImage = norm_uniforme[inImage.flatten()]
    outImage = np.reshape(outImage, inImage.shape)
    return outImage

def filterImage(inImage,kernel):
    m = inImage.shape[0]
    n = inImage.shape[1]
    outImage = np.zeros((m,n))
    row = kernel.shape[0]
    col = kernel.shape[1]
    centerX = math.floor(row/2)
    centerY = math.floor(col/2)
    image = np.pad(inImage,((centerX,centerX),(centerY,centerY)),mode = "constant")
    for i in range(m):
        for j in range(n):
            outImage[i,j] = np.sum(image[i:i+row,j:j+col]*kernel)
    return outImage

def gaussKernel1D(sigma):
    size = round(2*(3*sigma)+1)
    kernel = np.zeros(size)
    mid = math.floor(size/2)
    kernel=[(1/(sigma*np.sqrt(2*np.pi)))*(1/(np.exp((i**2)/(2*sigma**2)))) for i in range(-mid,mid+1)]
    return kernel

def  gaussianFilter(inImage, sigma):
    kernel = gaussKernel1D(sigma)
    kernelT = np.transpose(kernel)
    kernel2 = np.outer(kernel, kernelT)
    outImage = filterImage(inImage, kernel2)
    return outImage

def  medianFilter (inImage, filterSize):
    m = inImage.shape[0]
    n = inImage.shape[1]
    outImage = np.zeros((m,n))
    image = np.zeros((filterSize,filterSize))
    row = image.shape[0]
    center = math.floor(row/2)
    image = np.pad(inImage,[center,center],mode = "constant")
    for i in range(m):
        for j in range(n):
            outImage[i,j] = np.median(image[i:i+row,j:j+row])
    return outImage

def highBoost(inImage, A, method, param):
    m = inImage.shape[0]
    n = inImage.shape[1]
    outImage = np.zeros((m,n))
    if (method == "median"):
        image = medianFilter(inImage,param)
        outImage = inImage*A-image
    elif (method == "gaussian"):
        image = gaussianFilter(inImage,param)
        outImage = inImage*A-image
    else: print("Este filtrado no existe")
    return outImage

def erode(inImage, SE, center =[]):
    m = inImage.shape[0]
    n = inImage.shape[1]
    kx,ky = SE.shape
    px,py = kx-1,ky-1
    minI = math.floor(kx/2)
    maxI = math.floor(ky/2)
    medio = [minI,maxI]
    if center == []:
        center = medio
    outImage = np.zeros((m,n),dtype = np.int32)
    image = np.pad(inImage,((px,px),(py,py)),mode = "constant")
    for i in range (m):
        for j in range (n):
            if(image[i+px-center[0]:i+kx+px-center[0], j+py-center[1]:j+ky+py-center[1]] >= SE).all():
                outImage[i,j] = 1
            else:
                outImage[i,j] = 0
    return outImage

def dilate(inImage, SE, center =[]):
    m = inImage.shape[0]
    n = inImage.shape[1]
    kx,ky = SE.shape
    px,py = kx-1,ky-1
    minI = math.floor(kx/2)
    maxI = math.floor(ky/2)
    mid = [minI,maxI]
    if not np.array(center).size :
        center = mid
    outImage = np.zeros((m,n))
    image = np.pad(inImage,((px,px),(py,py)),mode = "constant")
    outImageAux = np.pad(outImage,((px,px),(py,py)),mode = "constant")
    for i in range (m):
        for j in range (n):
            if(inImage[i,j] == 1):
                outImageAux[i+center[0]:i+kx+center[0], j+center[1]:j+ky+center[1]] = SE
    for i in range (m):
        for j in range (n):
            outImage[i,j] = outImageAux[i+px,j+py]
    return outImage

def opening (inImage, SE, center=[]):
    image = erode(inImage, SE,center)
    outImage = dilate(image, SE,center)
    return outImage

def closing (inImage, SE, center=[]):
    image = dilate(inImage, SE,center)
    outImage = erode(image, SE,center)
    return outImage


def hit_or_miss (inImage, objSEj, bgSE, center=[]):
    m = inImage.shape[0]
    n = inImage.shape[1]
    compImage = 1-inImage
    image = erode(inImage,objSEj,center)
    imageb = erode(compImage,bgSE,center)
    outImage = np.zeros((m,n), dtype = np.int32)
    for i in range(m):
        for j in range(n):
            if (image[i,j] == 1) and (imageb[i,j] == 1):
                outImage[i,j] = 1
    return outImage

def gradientImage(inImage, operator):
    if operator == "Roberts":
        fx = np.array([[-1, 0],[0, 1]])
        fy = np.array([[0, -1],[1, 0]])
    elif operator == "CentralDiff":
        fx = np.array([[-1, 0, 1]])
        fy = np.array([[-1], [0], [1]])
    elif operator == "Prewitt":
        fx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        fy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])   
    elif operator == "Sobel":
        fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gx = filterImage(inImage,fx)
    gy = filterImage(inImage,fy)
    return gx, gy

def edgeCanny(inImage, sigma, tlow, thigh):
    outImage = np.zeros(np.shape(inImage))
    gauss = gaussianFilter(inImage,sigma)
    gx,gy = gradientImage(gauss,"Sobel")
    eo = np.arctan2(gy,gx)
    orientacion = eo * 180./np.pi
    m = orientacion.shape[0]
    n = orientacion.shape[1]
    sup = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            if (orientacion[i,j] < 0):
                orientacion[i,j] +=180
    #np.hypot para hacer la raiz cuadrada de la suma de los cuadrados
    em = np.hypot(gx,gy)
    #Comprobamos la orientación
    for i in range(1,m-1):
        for j in range(1,n-1):
            if (0 <= orientacion[i,j] < 22.5) or (157.5 <= orientacion[i,j] <= 180):
                x = em[i,j+1]
                y = em[i,j-1]
            elif (22.5 <= orientacion[i,j] < 67.5):
                x = em[i+1,j-1]
                y = em[i-1,j+1]
            elif (67.5 <= orientacion[i,j] < 112.5):
                x = em[i+1,j]
                y = em[i-1,j]
            elif (112.5 <= orientacion[i,j] < 157.5):
                x = em[i-1,j-1]
                y = em[i+1,j+1]

            if (em[i,j] >= x) and (em[i,j] >= y):
                sup[i,j] = em[i,j]
            else:
                sup[i,j] = 0

    high = 180          
    low = 40
    high_i,high_j = np.where(sup >= thigh)
    outImage[high_i,high_j] = high 
    low_i,low_j = np.where((sup <= thigh) & (sup >= tlow))   
    outImage[low_i,low_j] = low
    #Comprobamos si hay cambio brusco entre pixeles vecinos
    for i in range(1, m-1):
        for j in range(1, n-1):
            if (outImage[i,j] == low):
                    if ((outImage[i+1, j] == high) or (outImage[i,j+1] == high) or (outImage[i+1,j-1] == high) or (outImage[i+1, j+1] == high)
                    or (outImage[i, j-1] == high) or (outImage[i-1, j] == high) or (outImage[i-1, j+1] == high) or (outImage[i-1, j-1] == high)):
                        outImage[i,j] = high
                    else:
                        outImage[i,j] = 0
    return outImage