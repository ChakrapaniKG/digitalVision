#Plotting Linear and Logarthimic histograms


def draw_image_histogram_bgr(image,title = '',yscale = 'linear'):
    "Utility to plot bgr histograms for all color channels independently"
    hist_b = cv2.calcHist([image],[0],None,[256],[0,255])
    hist_g = cv2.calcHist([image],[1],None,[256],[0,255])
    hist_r = cv2.calcHist([image],[2],None,[256],[0,255])

    fig = plt.figure(figsize=(17,5))
    fig.suptitle(title)
    ax = fig.add_subplot(1,3,1)
    ax.set_yscale(yscale)
    plt.plot(hist_b,color='b',label='Blue')
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(1,3,2)
    ax.set_yscale(yscale)
    plt.plot(hist_g,color='g',label='Green')
    ax.grid()
    ax.legend()

    ax = fig.add_subplot(1,3,3)
    ax.set_yscale(yscale)
    plt.plot(hist_r,color='r',label='Red')
    ax.grid()
    ax.legend()

    plt.show()