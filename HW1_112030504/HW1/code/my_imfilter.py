import numpy as np

def my_imfilter(image, imfilter):
    """function which imitates the default behavior of the build in scipy.misc.imfilter function.

    Input:
        image: A 3d array represent the input image.
        imfilter: The gaussian filter.
    Output:
        output: The filtered image.
    """
    # =================================================================================
    # TODO:                                                                           
    # This function is intended to behave like the scipy.ndimage.filters.correlate    
    # (2-D correlation is related to 2-D convolution by a 180 degree rotation         
    # of the filter matrix.)                                                          
    # Your function should work for color images. Simply filter each color            
    # channel independently.                                                          
    # Your function should work for filters of any width and height                   
    # combination, as long as the width and height are odd (e.g. 1, 7, 9). This       
    # restriction makes it unambigious which pixel in the filter is the center        
    # pixel.                                                                          
    # Boundary handling can be tricky. The filter can't be centered on pixels         
    # at the image boundary without parts of the filter being out of bounds. You      
    # should simply recreate the default behavior of scipy.signal.convolve2d --       
    # pad the input image with zeros, and return a filtered image which matches the   
    # input resolution. A better approach is to mirror the image content over the     
    # boundaries for padding.                                                         
    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can 
    # see the desired behavior.                                                       
    # When you write your actual solution, you can't use the convolution functions    
    # from numpy scipy ... etc. (e.g. numpy.convolve, scipy.signal)                   
    # Simply loop over all the pixels and do the actual computation.                  
    # It might be slow.                        
    
    # NOTE:                                                                           
    # Some useful functions:                                                        
    #     numpy.pad (https://numpy.org/doc/stable/reference/generated/numpy.pad.html)      
    #     numpy.sum (https://numpy.org/doc/stable/reference/generated/numpy.sum.html)                                     
    # =================================================================================
    import scipy.ndimage as ndimage
    output = np.zeros_like(image)
    for ch in range(image.shape[2]):
        output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')
    return output
    # ============================== Start OF YOUR CODE ===============================
    imagex=image.shape[0] #361
    imagey=image.shape[1] #410
    imagez=image.shape[2] #3

    output=np.zeros((imagex,imagey,imagez))
    filterx=imfilter.shape[0]
    filtery=imfilter.shape[1]
    paddingx=int((imfilter.shape[0]-1)/2)   #padding填充讓卷積之後照片不會縮小
    paddingy=int((imfilter.shape[1]-1)/2)
    padimage=np.zeros((imagex+(filterx-1),imagey+(filtery-1),3)) #圖片大小維度+(2*14)
    padimage[paddingx: imagex + paddingx, paddingy: imagey + paddingy] = image
    
    
    
    for x in range(imagex): #3
        for y in range(imagey): #361
            for z in range(imagez):
                output[x][y][z] = sum(sum(np.multiply(imfilter,padimage[x:x+filterx,y:y+filtery,z]))) #把[29,29]的矩陣加總
    return output

    
    # =============================== END OF YOUR CODE ================================

    # Uncomment if you want to simply call scipy.ndimage.filters.correlate so you can
    # see the desired behavior.
    # import scipy.ndimage as ndimage
    # output = np.zeros_like(image)
    # for ch in range(image.shape[2]):
    #    output[:,:,ch] = ndimage.filters.correlate(image[:,:,ch], imfilter, mode='constant')
    
    ###########################################################
    #製作output矩陣
    
    

    