from tkinter.filedialog import askopenfilenames
import cv2 as cv
import numpy as np
import matplotlib.pylab as plt


def showoutput(inputimage, outputimage, title="Output Image", cmap=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(cv.cvtColor(inputimage, cv.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(outputimage, cmap=cmap)
    plt.show()


def draw_rectangle(image, lines):
    img_copy = image.copy()
    if lines is not None:
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for line in lines:
            for x1, y1, x2, y2 in line:
                min_x = min(min_x, x1, x2)
                min_y = min(min_y, y1, y2)
                max_x = max(max_x, x1, x2)
                max_y = max(max_y, y1, y2)
        cv.rectangle(img_copy, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    return img_copy

def apply_edge_detection(gray_image, method):
    if method.lower() == 'canny':
        edges = cv.Canny(gray_image, 55, 150)
        return edges, "Canny Edge Detection"
    elif method.lower() == 'sobel':
        sobelx = cv.Sobel(gray_image, cv.CV_64F, 1, 0, ksize=3)
        sobely = cv.Sobel(gray_image, cv.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv.magnitude(sobelx, sobely)
        sobel_combined = cv.convertScaleAbs(sobel_combined)
        return sobel_combined, "Sobel Edge Detection"
    elif method.lower() == 'laplacian':
        laplacian = cv.Laplacian(gray_image, cv.CV_64F)
        laplacian = cv.convertScaleAbs(laplacian)
        return laplacian, "Laplacian Edge Detection"
    elif method.lower() == 'scharr':
        scharrx = cv.Scharr(gray_image, cv.CV_64F, 1, 0)
        scharry = cv.Scharr(gray_image, cv.CV_64F, 0, 1)
        scharr_combined = cv.magnitude(scharrx, scharry)
        scharr_combined = cv.convertScaleAbs(scharr_combined)
        return scharr_combined, "Scharr Edge Detection"
    else:
        raise ValueError("Invalid method. Choose 'canny', 'sobel', 'laplacian', or 'scharr'.")


imagepath = askopenfilenames()


method = input("Choose edge detection method (canny/sobel/laplacian/scharr): ").strip().lower()

for i in imagepath:
    img = cv.imread(i)
    
    grayimage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    showoutput(img, grayimage, title="Grayscale Image", cmap='gray')
    
    nonenoise = cv.medianBlur(grayimage, 3)
    showoutput(img, nonenoise, title="Median Blur Image", cmap='gray')
    
    edges, title = apply_edge_detection(nonenoise, method)
    showoutput(img, edges, title=title, cmap='gray')
    
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    outputimage = draw_rectangle(img, lines)
    showoutput(img, outputimage, title="Detected Fracture")
