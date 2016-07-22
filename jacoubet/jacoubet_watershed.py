# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 11:50:13 2016

@author: remi
"""

#test on jacoubet to segment street/buildings
#the idea is to use watershed, based on part of vector lines 

def load_images():
    """ load images, returns the images loaded as numpy matrix
    """
    from osgeo import gdal 
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    #load jacoubet image
    image_path = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/Feuille31_extract.tif'
    image_path_axis = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/Feuille31_extract_route.tif'
    image_path_axis_broken = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/Feuille31_extract_route_broken.tif'
    
    # open dataset
    ds = gdal.Open(image_path)
    ds2 = gdal.Open(image_path_axis)
    ds3 = gdal.Open(image_path_axis_broken)
    
    
    jac = np.array(ds.GetRasterBand(1).ReadAsArray())
    axis = np.array(ds2.GetRasterBand(1).ReadAsArray())
    axis = np.array(ds3.GetRasterBand(1).ReadAsArray())
    axis = np.logical_not(axis)
    
    jac_inverted = np.logical_not(jac) 
     
    imgplot = plt.imshow(axis, cmap='Greys')
    
    return jac, jac_inverted, axis


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
    imgplot = plt.imshow(distance)
    imgplot = plt.imshow(distance>40)
    
    thresh = distance > threshold_distance
    imgplot = plt.imshow(thresh) 
    
    
    
    #local_maxi = peak_local_max(distance, labels=jac, footprint=np.ones((100, 100)), indices=False)
    if seed_image is None: 
        markers = label(thresh)
    else:
        markers = label(seed_image)

    imgplot = plt.imshow(markers)
     
      
    watersh = watershed(-distance, markers, mask=base_image) 
    plt.imshow(watersh, cmap=plt.cm.viridis, interpolation='nearest')
    
    return watersh
     

    #conclusion : it works but we have a problem with holes in building boundaries



###############################
# superpixel
############################### 
def superpixels(image):
    """ given an input image, create super pixels on it
    """
    # we could try to limit the problem of holes in boundary by first over segmenting the image
        
    from skimage.segmentation import felzenszwalb, slic, quickshift
    from skimage.segmentation import mark_boundaries
    from skimage.util import img_as_float
    
    jac_float = img_as_float(image/255.0)
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


####################################
# SNAKE : other lib
####################################
def snake_research(base_image, seed_image, per_size=0.25):
    # the skimage snake implementation does not seems to be able to balloon
    # we try another implementation, maybe better, but pure research
     
    
    import sys
    sys.path.append('/home/remi/soft/morphsnakes',smoothing=1, lambda1=1, lambda2=1)
      
    import morphsnakes
    from matplotlib import pyplot as ppl
    import time   
    from skimage.transform import resize  
    from PIL import Image 
#    # Morphological ACWE. Initialization of the level-set.
#    macwe = morphsnakes.MorphACWE(base_image, smoothing=smoothing, lambda1=lambda1, lambda2=lambda2)
#    macwe.levelset = seed_image
#    plt.imshow(macwe.levelset)
#    
#    # Visual evolution.
#    ppl.figure()
#    last_levelset = morphsnakes.evolve_visual(macwe, num_iters=10, background=jac)
#    return last_levelset

    
    
    def resize_image(im, percent):
        return resize(im, [int(im.shape[0]*per_size), int(im.shape[1]*per_size)])  
        
    down_sampled_jac = resize_image(jac, per_size) 
    gI = morphsnakes.gborders(resize_image(jac, per_size) , alpha=1000, sigma=2)
    
    # Morphological GAC. Initialization of the level-set.
    mgac = morphsnakes.MorphGAC(gI, smoothing=3, threshold=0.8, balloon=1.5)
    mgac.levelset = resize_image(markers>0, per_size) 
    
    # Visual evolution.
    ppl.figure()
    
      
    start_time = time.time()
    last_levelset = morphsnakes.evolve_visual(mgac, num_iters=50, background=resize_image(jac, per_size) )
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
    
    #dilate then retract using morph math to consolidate noisy results
    #binclos = binary_dilation(jac_inverted)
    binclos = binary_closing(base_label_image,disk(size_closing)).astype(float)
    
    #binclos = binary_opening(binclos,disk(1))
    #bil = denoise_bilateral(binclos, sigma_color=0.05, sigma_spatial=5, multichannel=False)
    
    
    #find the connected components
    plt.imshow(binclos, cmap='Greys')
     
    markers, n_label = label(binclos, connectivity=1, background=0, return_num=True)
    plt.imshow(markers,vmin=-1, vmax=1)
     
    
    hist, bins = np.histogram(markers, bins=n_label)
    r_bins = bins[0:n_label]  
    
    print('hom much hist bins are we going to keep? : ',np.sum(hist>hist_thres) )
    
    bins_to_remove = r_bins[hist<hist_thres]
    
    #np.append(bins_to_remove)
    
    to_remove_mask = np.in1d(markers, bins_to_remove.astype(int))
    np.sum(to_remove_mask==True) 
    to_remove_mask_resh = np.reshape(to_remove_mask, markers.shape)
    plt.imshow(to_remove_mask_resh, cmap='Greys')
    
    
    filtered_jac_inverted = np.copy(base_label_image)
    filtered_jac_inverted[to_remove_mask_resh == True] = 0
     
    plt.imshow(filtered_jac_inverted, cmap='Greys')
    return filtered_jac_inverted
 
 
def main():
    """ main script, call all  the functions
    """
    #load images
    jac, jac_inverted, axis = load_images()

    #watershed
    watersh = watershed(jac, seed_image=axis)
    #anotehr option
    watersh = watershed(jac, threshold_distance = 80)
#    
#    #superpixels
#    segments_slic = superpixels(image)
#    
#    #snake
#    
#    #snake with another lib, able to ballon
#    last_levelset = snake_research(jac, axis)
#    
#    #removing small ccomponents
#    filtered_jac_inverted = remove_small_ccomponents(jac_inverted, size_closing=2, hist_thres=1000)
    
main()
