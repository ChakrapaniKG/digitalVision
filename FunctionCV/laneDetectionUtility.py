def region_of_interest(img,vertices):
    ''' Select Region of interest from the specified Vertices '''
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices,ignore_mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image

def draw_lines(img, lines, color = [255,0,0], thickness = 2):
    ''' For Drawing Lines '''
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img,(x1,y1),(x2,y2),color,thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """Utility for defining Line Segments."""
    lines = cv2.HoughLinesP(
        img, rho, theta, threshold, np.array([]),
        minLineLength = min_line_len, maxLineGap = max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines

def separate_left_right_lines(lines):
    ''' Separate left and right lines depending on the slope '''
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if y1 > y2:
                    left_lines.append([x1,y1,x2,y2])
                elif y1 < y2 :
                    right_lines.append([x1,y1,x2,y2])
    return left_lines, right_lines

def cal_avg(values):
    if not (type(values) == 'NoneType'):
        if len(values) > 0:
            n = len(values)
        else:
            n = 0
        return sum(values) / n

def extrapolate_lines(lines, upper_border, lower_border):
    ''' Extrapolate lines keeping in the mind the lower and upper border intersections '''
    slopes = []
    consts = []

    if (lines is not None) and (len(lines) != 0):
        for x1,y1,x2,y2 in lines:
            slope = (y2 - y1) / (x2 - x1)
            slopes.append(slope)
            c = y1 - slope * x1
            consts.append(c)
        avg_slope = cal_avg(slopes)
        avg_consts = cal_avg(consts)

        x_lane_lower_point = int((lower_border - avg_consts) / avg_slope)
        x_lane_upper_point = int((upper_border - avg_consts) / avg_slope)

        return [x_lane_lower_point, lower_border, x_lane_upper_point, upper_border]


def extrapolated_lane_image(img,lines,roi_upper_border,roi_lower_border):
    ''' Main function called to get the final lane lines '''
    lanes_img = np.zeros((img.shape[0],img.shape[1],3),dtype = np.uint8)
    lines_left, lines_right = separate_left_right_lines(lines)
    lane_left = extrapolate_lines(lines_left, roi_upper_border, roi_lower_border)
    lane_right = extrapolate_lines(lines_right, roi_upper_border, roi_lower_border)
    if lane_left is not None and lane_right is not None:
        draw_con(lanes_img, [[lane_left], [lane_right]])
    return lanes_img


def draw_con(img, lines):
    """Fill in lane area."""
    points = []
    for x1,y1,x2,y2 in lines[0]:
        points.append([x1,y1])
        points.append([x2,y2])
    for x1,y1,x2,y2 in lines[1]:
        points.append([x2,y2])
        points.append([x1,y1])

    points = np.array([points], dtype = 'int32')        
    cv2.fillPoly(img, points, (0,255,0))