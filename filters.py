import numpy as np
import cv2
import cvzone

def increase_brightness(frame, brightness_factor = 1.9):
    # Increase brightness by scaling pixel values
    brightened_frame = cv2.multiply(frame, np.array([brightness_factor]))
    brightened_frame = np.clip(brightened_frame, 0, 255).astype(np.uint8)
    return brightened_frame

def decrease_brightness(frame, darkness_factor = 1.9):
    darkened_frame = cv2.divide(frame, np.array([darkness_factor]))
    darkened_frame = np.clip(darkened_frame, 0, 255).astype(np.uint8)
    return darkened_frame

def sharpened(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame_edge=cv2.Sobel(gray,cv2.CV_8U,1,1,ksize=5)#edge detection
    sharp=np.zeros_like(frame) 
    sharp[:,:,0]=frame [:,:,0]+frame_edge
    sharp[:,:,1]=frame [:,:,1]+frame_edge
    sharp[:,:,2]=frame [:,:,2]+frame_edge
    return sharp

def fft(frame):
    
    image1 = cv2.imread('img.jpg')
    image2=frame

    width = frame.shape[0]  # Change this to your desired width
    height = frame.shape[1] # Change this to your desired height

    image1 = cv2.resize(image1, (width, height))
    b1, g1, r1 = cv2.split(image1)
    b2, g2, r2 = cv2.split(image2)

    fft_b1 = np.fft.fft2(b1)
    fft_g1 = np.fft.fft2(g1)
    fft_r1 = np.fft.fft2(r1)

    fft_b2 = np.fft.fft2(b2)
    fft_g2 = np.fft.fft2(g2)
    fft_r2 = np.fft.fft2(r2)

    new_fft_b1 = np.abs(fft_b1) * np.exp(1j * np.angle(fft_b2))
    new_fft_g1 = np.abs(fft_g1) * np.exp(1j * np.angle(fft_g2))
    new_fft_r1 = np.abs(fft_r1) * np.exp(1j * np.angle(fft_r2))

    new_b1 = np.fft.ifft2(new_fft_b1).real
    new_g1 = np.fft.ifft2(new_fft_g1).real
    new_r1 = np.fft.ifft2(new_fft_r1).real
    
    result_image = cv2.merge((new_b1.astype(np.uint8), new_g1.astype(np.uint8), new_r1.astype(np.uint8)))
    return result_image


def frame_mixing(frame):
    # Load the color image
    color_image = cv2.imread('img.jpg')

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    target_width = frame.shape[0]
    target_height = frame.shape[1]

    color_image = cv2.resize(color_image, (target_width, target_height))

    edges = cv2.Canny(gray_frame, threshold1=30, threshold2=100)

    mask = np.zeros_like(color_image)
    mask[edges > 0] = color_image[edges > 0]
    
    return mask

def frame_mixing2(frame):
    # Load the color image
    color_image = cv2.imread('img.jpg')

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    target_width = frame.shape[0]
    target_height = frame.shape[1]

    color_image = cv2.resize(color_image, (target_width, target_height))

    edges = cv2.Sobel(gray_frame,cv2.CV_64F,1,1,ksize=5)

    mask = np.zeros_like(color_image)
    mask[edges > 0] = color_image[edges > 0]
    
    return (frame + mask).astype(np.uint8)

def frame_edges(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, threshold1=30, threshold2=100)
    return edges.reshape(edges.shape[0],edges.shape[1],1)

def fft2(frame):
    
    image1 = cv2.imread('img2.png')
    image2=frame

    width = frame.shape[0]  # Change this to your desired width
    height = frame.shape[1] # Change this to your desired height

    image1 = cv2.resize(image1, (width, height))
    b1, g1, r1 = cv2.split(image1)
    b2, g2, r2 = cv2.split(image2)

    fft_b1 = np.fft.fft2(b1)
    fft_g1 = np.fft.fft2(g1)
    fft_r1 = np.fft.fft2(r1)

    fft_b2 = np.fft.fft2(b2)
    fft_g2 = np.fft.fft2(g2)
    fft_r2 = np.fft.fft2(r2)

    new_fft_b1 = np.abs(fft_b1) * np.exp(1j * np.angle(fft_b2))
    new_fft_g1 = np.abs(fft_g1) * np.exp(1j * np.angle(fft_g2))
    new_fft_r1 = np.abs(fft_r1) * np.exp(1j * np.angle(fft_r2))

    new_b1 = np.fft.ifft2(new_fft_b1).real
    new_g1 = np.fft.ifft2(new_fft_g1).real
    new_r1 = np.fft.ifft2(new_fft_r1).real
    
    result_image = cv2.merge((new_b1.astype(np.uint8), new_g1.astype(np.uint8), new_r1.astype(np.uint8)))
    return result_image


def frame_mixing21(frame):
    # Load the color image
    color_image = cv2.imread('img2.png')

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    target_width = frame.shape[0]
    target_height = frame.shape[1]

    color_image = cv2.resize(color_image, (target_width, target_height))

    edges = cv2.Canny(gray_frame, threshold1=30, threshold2=100)

    mask = np.zeros_like(color_image)
    mask[edges > 0] = color_image[edges > 0]
    
    return mask

def frame_mixing22(frame):
    # Load the color image
    color_image = cv2.imread('img2.png')

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    target_width = frame.shape[0]
    target_height = frame.shape[1]

    color_image = cv2.resize(color_image, (target_width, target_height))

    edges = cv2.Sobel(gray_frame,cv2.CV_64F,1,1,ksize=5)

    mask = np.zeros_like(color_image)
    mask[edges > 0] = color_image[edges > 0]
    
    return (frame + mask).astype(np.uint8)

def fft3(frame):
    
    image1 = cv2.imread('img3.jpg')
    image2=frame

    width = frame.shape[0]  # Change this to your desired width
    height = frame.shape[1] # Change this to your desired height

    image1 = cv2.resize(image1, (width, height))
    b1, g1, r1 = cv2.split(image1)
    b2, g2, r2 = cv2.split(image2)

    fft_b1 = np.fft.fft2(b1)
    fft_g1 = np.fft.fft2(g1)
    fft_r1 = np.fft.fft2(r1)

    fft_b2 = np.fft.fft2(b2)
    fft_g2 = np.fft.fft2(g2)
    fft_r2 = np.fft.fft2(r2)

    new_fft_b1 = np.abs(fft_b1) * np.exp(1j * np.angle(fft_b2))
    new_fft_g1 = np.abs(fft_g1) * np.exp(1j * np.angle(fft_g2))
    new_fft_r1 = np.abs(fft_r1) * np.exp(1j * np.angle(fft_r2))

    new_b1 = np.fft.ifft2(new_fft_b1).real
    new_g1 = np.fft.ifft2(new_fft_g1).real
    new_r1 = np.fft.ifft2(new_fft_r1).real
    
    result_image = cv2.merge((new_b1.astype(np.uint8), new_g1.astype(np.uint8), new_r1.astype(np.uint8)))
    return result_image


def frame_mixing3(frame):
    # Load the color image
    color_image = cv2.imread('img3.jpg')

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    target_width = frame.shape[0]
    target_height = frame.shape[1]

    color_image = cv2.resize(color_image, (target_width, target_height))

    edges = cv2.Canny(gray_frame, threshold1=30, threshold2=100)

    mask = np.zeros_like(color_image)
    mask[edges > 0] = color_image[edges > 0]
    
    return mask

def frame_mixing32(frame):
    # Load the color image
    color_image = cv2.imread('img3.jpg')

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    target_width = frame.shape[0]
    target_height = frame.shape[1]

    color_image = cv2.resize(color_image, (target_width, target_height))

    edges = cv2.Sobel(gray_frame,cv2.CV_64F,1,1,ksize=5)

    mask = np.zeros_like(color_image)
    mask[edges > 0] = color_image[edges > 0]
    
    return (frame + mask).astype(np.uint8)



def process_images(frame,x,y,w,h):
    overlay_images = 'sunglass.png'
    overlay = cv2.imread(overlay_images, cv2.IMREAD_UNCHANGED)
    overlay_resize = cv2.resize(overlay, (w, h))
    frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])
    return frame

def process_images2(frame,x,y,w,h):
    overlay_images = 'star.png'
    overlay = cv2.imread(overlay_images, cv2.IMREAD_UNCHANGED)
    overlay_resize = cv2.resize(overlay, (w, h))
    frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])
    return frame






# if __name__ == '__main__':
#     img = cv2.imread('output\out.jpg')
    
#     cv2.imshow('output',frame_edges(img))
#     cv2.waitKey(0)
