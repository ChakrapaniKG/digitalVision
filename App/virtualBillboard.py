import cv2
import numpy as np

# Create a named window for the display.
win_name = 'Destination Image'

def mouse_handler(event, x, y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:
        #Render points as yellow circles in destination image.
        cv2.circle(data['img'], (x, y), radius=5, color=(0,255,255),thickness=-1, lineType=cv2.LINE_AA)
        cv2.imshow(win_name, data['img'])
        if len(data['points']) < 4:
            data['points'].append([x, y])

def get_roi_points(img):
    data = {'img':img.copy(),'points':[]}
    cv2.imshow(win_name, img)
    cv2.setMouseCallback(win_name, mouse_handler, data)
    cv2.waitKey(0)

    roi_points = np.vstack(data['points']).astype(float)

    return roi_points


img_src = cv2.imread('../Applications/Apollo-8-Launch.png')

img_dst = cv2.imread('../Applications/times_square.jpg')

size = img_src.shape
src_pts = np.array([[0,0], [size[1] - 1,0],[size[1] - 1, size[0] - 1],[0,size[0] - 1]], dtype=float)

print("Click on four corners of a billboard and then press Enter")

roi_dst = get_roi_points(img_dst)

print(roi_dst)

h,status = cv2.findHomography(src_pts, roi_dst)

warped_img = cv2.warpPerspective(img_src, h, (img_dst.shape[1],img_dst.shape[0]))


cv2.fillConvexPoly(img_dst, roi_dst.astype(int), 0, 16)

img_dst = img_dst + warped_img

cv2.imshow(win_name, img_dst)

cv2.waitKey(0)

cv2.destroyAllWindows()

