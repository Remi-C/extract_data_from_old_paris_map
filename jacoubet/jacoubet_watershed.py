# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:50:13 2016

@author: remi
"""

#test on jacoubet to segment street/buildings
#the idea is to use watershed, based on part of vector lines 

    
def plot_image(im, cmap='Greys'):
    import matplotlib.pyplot as plt
    fig = plt.figure()  # a new figure window
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)   
    ax.imshow(im, cmap) 


def load_images():
    """ load images, returns the images loaded as numpy matrix
    """
    from osgeo import gdal 
    import numpy as np 
    
    #load jacoubet image
    image_path = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/Feuille31_extract.tif'
    image_path_axis = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/Feuille31_extract_route.tif'
    image_path_axis_broken = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/Feuille31_extract_route_broken.tif'
    image_path_original_scan = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/BHVP_FM11_0031_C_extract.png'
    # open dataset
    ds = gdal.Open(image_path)
    ds2 = gdal.Open(image_path_axis)
    ds3 = gdal.Open(image_path_axis_broken)
    ds4 = gdal.Open(image_path_original_scan)
    
    
    jac = np.array(ds.GetRasterBand(1).ReadAsArray())
    axis = np.array(ds2.GetRasterBand(1).ReadAsArray())
    axis = np.array(ds3.GetRasterBand(1).ReadAsArray())
    axis = np.logical_not(axis)
    
    ori_scan_r = np.array(ds4.GetRasterBand(1).ReadAsArray())
    ori_scan_g = np.array(ds4.GetRasterBand(2).ReadAsArray())
    ori_scan_b = np.array(ds4.GetRasterBand(3).ReadAsArray())
    
    ori_scan = np.dstack((ori_scan_r,ori_scan_g,ori_scan_b)) 
    
    jac_inverted = np.logical_not(jac) 
     
    #imgplot = plt.imshow(axis, cmap='Greys')
    
    return jac, jac_inverted, axis, ori_scan


#######################
# watershed
########################
def watershed(base_image, seed_image=None, threshold_distance=80):
    """ execute watershed with chosen seeds
    """
    from scipy import ndimage as ndi
    
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from skimage.morphology import label
    import matplotlib.pyplot as plt
    
    
    distance = ndi.distance_transform_edt(base_image)
    #    imgplot = plt.imshow(distance)
    fig = plt.figure()  # a new figure window
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    #    ax.imshow(distance>threshold_distance, cmap='Greys')
    
    thresh = distance > threshold_distance
    ax.imshow(thresh, cmap='Greys') 
    
    
    
    #local_maxi = peak_local_max(distance, labels=jac, footprint=np.ones((100, 100)), indices=False)
    if seed_image is None: 
        markers = label(thresh)
    else:
        markers = label(seed_image)

    #    imgplot = plt.imshow(markers)
     
      
    watersh = watershed(-distance, markers, mask=base_image) 
    #    plt.imshow(watersh, cmap=plt.cm.viridis, interpolation='nearest')
    
    return watersh
     

    #conclusion : it works but we have a problem with holes in building boundaries



###############################
# superpixel
############################### 
def superpixels(image):
    """ given an input image, create super pixels on it
    """
    # we could try to limit the problem of holes in boundary by first over segmenting the image
    import matplotlib.pyplot as plt
    from skimage.segmentation import felzenszwalb, slic, quickshift
    from skimage.segmentation import mark_boundaries
    from skimage.util import img_as_float
    
    jac_float = img_as_float(image)
    plt.imshow(jac_float)
    #segments_fz = felzenszwalb(jac_float, scale=100, sigma=0.5, min_size=50)
    segments_slic = slic(jac_float, n_segments=600, compactness=0.01, sigma=0.001
        #, multichannel = False
        , max_iter=50) 
      
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    fig.set_size_inches(8, 3, forward=True)
    fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
    
    #ax[0].imshow(mark_boundaries(jac, segments_fz))
    #ax[0].set_title("Felzenszwalbs's method")
    ax.imshow(mark_boundaries(jac_float, segments_slic))
    ax.set_title("SLIC")           
    return segments_slic                     

 
###############################
# SNAKE : skimage
###############################
def snake_skimage():
    # to mitigate the problem of holes in building footprints, we could use 
    # snake. We initialise the snale wioth the contour of axis, then 
    # make the snake balloon
    # the idea is to work on small street segment one by one
    # we couls also use a more sophisticated method to compute 
    # all snake at the same time, for instance 
     
         
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.color import rgb2gray
    from skimage import data
    from skimage.filters import gaussian
    from skimage.segmentation import active_contour
    from skimage import measure
    from skimage.transform import rotate
      
    rotated_image = rotate(axis, 180, resize=True)
    plt.imshow(rotated_image)
    np.max(axis); np.min(axis)
    contours = measure.find_contours(axis*1.0, 0)
    
    fig, ax = plt.subplots()
    ax.imshow(jac, interpolation='nearest', cmap=plt.cm.gray)
    
    for n, contour in enumerate(contours):
        contour[:,[0, 1]] = contour[:,[1, 0]]
        ax.plot(contour[:, 0], contour[:, 1], linewidth=2)
        print(n,contour[0,:] )
        
        
    for n, contour in enumerate(contours):
        print(n, len(contour))
    test_contour = contours[88]  
     
    
    snake = active_contour(jac,
                                test_contour
                                , alpha= -1
                                , beta=10
                                , gamma=0.001
                                , max_iterations=50 
                                , bc='periodic'
                                , w_line = -1)
    
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(jac)
    ax.plot(test_contour[:, 0], test_contour[:, 1], '--r', lw=3)
    
    #for n, contour in enumerate(contours):
    #    ax.plot(contours[n][:, 1], contours[n][:, 0], '--r', lw=3)
        
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, jac.shape[0], jac.shape[1], 0])


    # conclusion : snake from skimage seems to be unable to balloon


def resize_image(im, percent):
    from skimage.transform import resize  
    return resize(im, [int(im.shape[0]*percent), int(im.shape[1]*percent)])  
    

####################################
# SNAKE : other lib
####################################

def snake_research(base_image, seed_image, per_size=0.25,smoothing=1, lambda1=1, lambda2=1):
    # the skimage snake implementation does not seems to be able to balloon
    # we try another implementation, maybe better, but pure research
     
    
    import sys
    sys.path.append('/home/remi/soft/morphsnakes')
      
    import morphsnakes
    from matplotlib import pyplot as ppl
    import time   
    from skimage.transform import resize  
    from PIL import Image 
    from skimage.morphology import label
    #    # Morphological ACWE. Initialization of the level-set.
    #    macwe = morphsnakes.MorphACWE(base_image, smoothing=smoothing, lambda1=lambda1, lambda2=lambda2)
    #    macwe.levelset = seed_image
    #    plt.imshow(macwe.levelset)
    #    
    #    # Visual evolution.
    #    ppl.figure()
    #    last_levelset = morphsnakes.evolve_visual(macwe, num_iters=10, background=jac)
    #    return last_levelset 
    down_sampled_jac = resize_image(base_image, per_size) 
    gI = morphsnakes.gborders(down_sampled_jac, alpha=1000, sigma=2)
    
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=3, threshold=0.8, balloon=1.5)
    markers = label(seed_image)
    mgac.levelset = resize_image(markers>0, per_size) 
    
    # Visual evolution.
    ppl.figure()
    
      
    start_time = time.time()
    last_levelset = morphsnakes.evolve_visual(mgac, num_iters=50, background=down_sampled_jac )
    interval = time.time() - start_time  
    print('Total time in seconds:', interval)
    return last_levelset


#########################################
# removing small connected components
#########################################
def remove_small_ccomponents(base_label_image, size_closing=2, hist_thres=1000):
    # the number and lettering are a pain for a number of methods
    # we tried to get rid of it
    # to this end, we extract all markings, compute connected components,
    # compute the number of pixels in each components, and remove the components 
    # with few pixels
    from scipy import ndimage as ndi 
    from skimage.morphology import binary_closing, binary_opening, binary_dilation
    from skimage.morphology import disk 
    from skimage.restoration import denoise_bilateral
    from skimage import measure
    from skimage.morphology import label
    import matplotlib.pyplot as plt
    import numpy as np
    
    #dilate then retract using morph math to consolidate noisy results
    #binclos = binary_dilation(jac_inverted)
    binclos = binary_closing(base_label_image,disk(size_closing)).astype(float)
    
    #binclos = binary_opening(binclos,disk(1))
    #bil = denoise_bilateral(binclos, sigma_color=0.05, sigma_spatial=5, multichannel=False)
    
    
    #find the connected components 
     
    markers, n_label = label(binclos, connectivity=1, background=0, return_num=True)
    #plt.imshow(markers,vmin=-1, vmax=1)
     
    
    hist, bins = np.histogram(markers, bins=n_label)
    r_bins = bins[0:n_label]  
    
    print('hom much hist bins are we going to keep? : ',np.sum(hist>hist_thres) )
    
    bins_to_remove = r_bins[hist<hist_thres]
    
    #np.append(bins_to_remove)
    
    to_remove_mask = np.in1d(markers, bins_to_remove.astype(int))
    np.sum(to_remove_mask==True) 
    to_remove_mask_resh = np.reshape(to_remove_mask, markers.shape) 
    
    
    filtered_jac_inverted = np.copy(base_label_image)
    filtered_jac_inverted[to_remove_mask_resh == True] = 0
      
    return filtered_jac_inverted, to_remove_mask_resh



def round_angle(iangle, step):
    return round(iangle/(step*1.0))*step

def dist_point_line(point,line):
    """ line is [x1,y1,x2,y2], point is [x0,y0]"""
    import  numpy as np
    return np.abs((line[3]-[1])*point[0]-(line[2]-[0])*point[1] +line[2]*line[1] - line[3]*line[0]  )/np.sqrt(np.pow(line[3]-[1],2)+np.pow(line[2]-[0],2))        


def find_lines(base_image, percent_resize=0.25, min_support = 100 , _minLineLength = 1000, _maxLineGap = 0, hough_s_dist_thre = 1, hough_s_angle_thre = 3.14/180):
    """ using rransac ot find lines in the image. 
    First resize image so the process is faster
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    from skimage.morphology import skeletonize
    from skimage.morphology import binary_closing,disk
    
    bi = skeletonize(base_image)
    #    binclos = binary_closing(bi,disk(4) )
    
    #    plt.imshow(binclos)
    gray_cv= np.array(resize_image(bi, percent_resize) * 255, dtype = np.uint8) 
    #    edges = cv2.Canny(gray_cv,50,150,apertureSize = 3)
    #plt.imshow(edges) 
    lines = cv2.HoughLinesP(gray_cv,hough_s_dist_thre,hough_s_angle_thre,min_support,minLineLength =_minLineLength,maxLineGap=_maxLineGap)
    print('number of found lines: ',len(lines))
     
    fig = plt.figure()  # a new figure window
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    
    blank = np.zeros(gray_cv.shape)
    ax.imshow(blank, cmap='Greys')
    ax.axis([0.0,gray_cv.shape[1], 0,gray_cv.shape[0]])
    
    """note : for a proper removal of grid line, we could expres sline i n hough
     space, then filter based on angle (close to 0 or 90),
     then perform kmeans with k = 4 to remove only the relevant lines
    """
    i = 0
    if(len(lines)) < 10000:
        for x in range(0, len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                #filtering
                if (np.absolute(x1-x2) < 2 or np.absolute(y1-y2) < 2) and np.sqrt(np.square(x1-x2) + np.square(y1-y2) ) > 20 :
                    #do nothing ()
                    i+=1
                    
                else:
                    
        #        print(x1,y1,x2,y2)
        #        print('length : ',x1-x2,y1-y2)
                    ax.plot((x1,x2),(y1,y2),linewidth=2.50, color='black')
    print(i,'line filtered, looking like major vertical/horizontal line')
    return lines
 
def distance_map(base_image, threshold_distance):
    from scipy import ndimage as ndi
    import matplotlib.pyplot as plt
    
    distance = ndi.distance_transform_edt(base_image)
    #    imgplot = plt.imshow(distance)
    fig = plt.figure()  # a new figure window
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    #    ax.imshow(distance>threshold_distance, cmap='Greys')
    
    thresh = distance > threshold_distance
    ax.imshow(thresh, cmap='Greys') 

def adaptative_thresholding(img, block_size):
    """ based on the orginal color scan, filter it for smoothing, then 
    tries an adaptative threslholding for a better binarization
    as a bonus, the result can be used for other applications"""
    from skimage.color import rgb2grey
    from skimage.filters import threshold_otsu, threshold_adaptive 
    from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
    import cv2
    
    #ori_scan_smooth = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=4, multichannel=True)
    ori_scan_smooth = cv2.bilateralFilter(img, d=50, sigmaColor=40, sigmaSpace=50 )
    plot_image(ori_scan_smooth)
    ori_scan_smooth_g = rgb2grey(ori_scan_smooth)
    ori_scan_smooth_g = 1- ori_scan_smooth_g  
    #thresholding original scan 
    ori_scan_smooth_g_int =  (ori_scan_smooth_g*255 ).astype(np.uint8)
    plot_image(ori_scan_smooth_g_int)
      
    #binary_adaptive = threshold_adaptive(ori_scan_smooth_g, block_size, method='median') 
    ori_scan_bin  = adaptiveThreshold(ori_scan_smooth_g_int,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY,block_size,-3)
   
    plot_image(ori_scan_bin) 
    return ori_scan_bin

def single_out_annotation(base_image, small_cc_image):
    """ extracting individual annotations :
    starting from potential annotation + noise, we remove the noise and 
     consolidate annotation area, then return the coordinates of center of 
     potential annotations""" 
     
    #  remove small stuff
    filtered_small_cc, removed_small_cc_small = remove_small_ccomponents(small_cc_image, size_closing=5, hist_thres=120)
    #plot_image(removed_small_cc_small)
    
    # dilate 
    from skimage.morphology import binary_dilation, disk
    dilation_radius = 10
    small_cc_cleaned_mask = binary_dilation(filtered_small_cc, disk(dilation_radius))
    #plot_image(small_cc_cleaned_mask) 
    
    #label connected compoenents
    from skimage.morphology import label
    from skimage.measure import regionprops
    from skimage.io import imsave
    markers, n_label = label(small_cc_cleaned_mask, connectivity=1, background=0, return_num=True)
    
    #for each cc, defines a region    
    region_prop = regionprops(markers, (base_image*255).astype(np.uint8))
    
    #for each region, do something
    
    base_path = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/results/annotations/'
    for region in region_prop: 
        #print(region.bbox, region.area)
        imsave(base_path+str(region.bbox)+'.png', region.intensity_image) 
        
    return region_prop

def main():
    """ main script, call all  the functions
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    ######## load images ########
    jac, jac_inverted, axis, ori_scan = load_images() 
    
    ######## adaptative thresholding #########
    #adaptative thresholding on original scan:
    #adaptative_thresholding(ori_scan, 101)
     
     
    ######## watershed ########
    #watersh = watershed(jac, seed_image=axis)
    #anotehr option
    #watersh = watershed(jac, threshold_distance = 80)
    
     
    
    ######## snake ######## 
    
    #snake with another lib, able to ballon
    #last_levelset = snake_research(jac, axis)
    
    ######## removing small ccomponents ########  
    filtered_jac_inverted, to_remove_mask_resh = remove_small_ccomponents(jac_inverted, size_closing=2, hist_thres=1000)
    plot_image(to_remove_mask_resh) 
    
    ######## isolating each annotation ##########
    regions = single_out_annotation(jac_inverted, to_remove_mask_resh)
    reg = regions[0]
    ######## OCR on isolated annotations ########
    """    
    import pyocr
    from PIL import Image
    tools = pyocr.get_available_tools()
    print(tools)
    tool = tools[1]
    langs = tool.get_available_languages()
    print("Available languages: %s" % ", ".join(langs))
    lang = 'fra'
    im= Image.fromarray(  reg.intensity_image)
    plot_image(im)
    word_boxes = tool.image_to_string(im,
                                      lang=lang,
        builder=pyocr.builders.WordBoxBuilder()
    )   
    
    digits = tool.image_to_string(
        im,
        lang=lang,
        builder=pyocr.tesseract.detect_orientation(im)
    )
    """
    #filtered_jac = np.logical_not(filtered_jac_inverted)    
    
    #distance_map(filtered_jac, threshold_distance=80)
    
    #superpixels
#    segments_slic = superpixels(filtered_jac)
#    from skimage.segmentation import mark_boundaries
#    bound = mark_boundaries(filtered_jac, segments_slic, color=(0,0,0), mode='subpixel')
#    

    
    #snake_research(filtered_jac_inverted, axis)
    
    #finding segement in image
#    res = find_lines(filtered_jac_inverted, percent_resize=1 
#               , min_support = 40 
#               , _minLineLength = 35 #70 # for main
#               , _maxLineGap = 10
#               , hough_s_dist_thre = 1
#               , hough_s_angle_thre = np.pi/180)
#    print(len(res))
    
    
main()
 
