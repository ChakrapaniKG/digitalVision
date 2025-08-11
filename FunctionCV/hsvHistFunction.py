def draw_image_histogram_hsv(image,title='',yscale=''):
    '''Utility to get the histogram of the image for HSV'''
    hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_image],[0],None,[180],[0,179])
    hist_s = cv2.calcHist([hsv_image],[1],None,[256],[0,255])
    hist_v = cv2.calcHist([hsv_image],[2],None,[256],[0,255])

    fig = plt.figure(figsize=(20,10))
    fig.suptitle(title)
    ax = fig.add_subplot(1,3,1)
    ax.set_yscale(yscale)
    plt.plot(hist_h,color='b',label='Hue')
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(1,3,2)
    ax.set_yscale(yscale)
    plt.plot(hist_s,color='g',label='Saturation')
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(1,3,3)
    ax.set_yscale(yscale)
    plt.plot(hist_v,color='r',label='Value')
    ax.grid()
    ax.legend()

    plt.show()