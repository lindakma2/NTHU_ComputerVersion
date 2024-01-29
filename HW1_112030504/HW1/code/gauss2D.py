import numpy as np

#二維高斯，去除圖像一中的高頻
def gauss2D(shape=(3, 3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    #上面的shape跟sigma是default
    #前面把shapesize定為(29,29)，sigma定為7
    
    m,n = [(ss-1.)/2. for ss in shape]
    #m=14,n=14
    #shape的資料型態為tuple，有點類似list，是不可變的型態
    #利用for loop計算每個tuple內的value
    
    y,x = np.ogrid[-m:m+1,-n:n+1]
    #第一組縱向產生第二維大小始終為1，-14~0~14
    #第二組橫向產生第一維大小始終為1，-14~0~14
    #ogrid產生numpy二維組數 資料格式(起始:結束:步長(defult=1))
    
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    #高斯常態分布h(x,y)=e^-(x^2+y^2/2*std^2)
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
