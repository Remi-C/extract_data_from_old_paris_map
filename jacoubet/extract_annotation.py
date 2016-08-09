# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:12:47 2016

@author: remi
"""

def load_image(image_path):
    from osgeo import gdal 
    import numpy as np
    ds = gdal.Open(image_path)
    geoTransf = ds.GetGeoTransform()
    jac = np.array(ds.GetRasterBand(1).ReadAsArray())
    jac_inverted = np.logical_not(jac) 
    return jac, jac_inverted, ds
    
def remove_small_ccomponents(base_label_image, size_closing=2, hist_thres=1000):
    # the number and lettering are a pain for a number of methods
    # we tried to extract it 
    from skimage.morphology import binary_closing 
    from skimage.morphology import disk   
    from skimage.morphology import label 
    import numpy as np
    import cv2
    
    base_label_image_f = cv2.bilateralFilter((base_label_image).astype(np.uint8), d=5, sigmaColor=0.05, sigmaSpace=5 )

    #dilate then retract using morph math to consolidate noisy results
    #binclos = binary_dilation(jac_inverted)
    binclos = binary_closing(base_label_image_f,disk(size_closing)).astype(float)
    
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


def pixelOffset2coord(ds,xOffset,yOffset):
    import gdal 
    geotransform = ds.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    coordX = originX+pixelWidth*xOffset
    coordY = originY+pixelHeight*yOffset
    return coordX, coordY 


def newraster_from_region(old_raster,new_raster_path,region):
    import gdal, ogr, osr
    array = region.intensity_image.astype(np.uint8) 
    
    coordX, coordY  = pixelOffset2coord(old_raster,region.bbox[1],region.bbox[0])
    
    
    geotransform = old_raster.GetGeoTransform()
    originX = coordX
    originY = coordY
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]

    cols = array.shape[1]
    rows = array.shape[0]
    #print('array shape : ',cols,rows)

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(new_raster_path, cols, rows, 1, gdal.GDT_Float32, ['TFW','YES'])
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(old_raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
 

def deal_with_found_region(regions, output_dir_for_annotation, ds):
    """ gieven a list of annotation, save each annotation into a geotif """
    from os import path 
    from skimage.io import imsave
    import numpy as np
    
    
    for region in regions:  
        #print(region.bbox, region.area)
        name_of_output = path.join(output_dir_for_annotation,str(region.bbox)+'.tif')
        img =  region.intensity_image.astype(np.uint8) 
        
        newraster_from_region(ds,name_of_output,region)
        
        
        #imsave(name_of_output, img) 
        
    #print(name_of_output)
        
    
    
    
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
    
    markers, n_label = label(small_cc_cleaned_mask, connectivity=1, background=0, return_num=True)
    
    #for each cc, defines a region    
    image_for_region =  (base_image*255).astype(np.uint8) 
    region_prop = regionprops(markers,image_for_region)
    
    #for each region, do something 
        
    return region_prop
    
def main():
    """ main script, call all  the functions
    """
    im_path  = '/media/sf_RemiCura/DATA/EHESS/GIS_maurizio/jacoubet_l1/tiles/Feuille21_4000_8000.tif'
    output_dir_for_annotation = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/results/annotations/'
    jac, jac_inverted, ds = load_image(im_path) 
    
    ######## removing small ccomponents ########  
    filtered_jac_inverted, to_remove_mask_resh = remove_small_ccomponents(jac_inverted, size_closing=2, hist_thres=1000)
    #plot_image(to_remove_mask_resh) 
    
    ######## isolating each annotation ##########
    regions = single_out_annotation(jac_inverted, to_remove_mask_resh)
    deal_with_found_region(regions, output_dir_for_annotation, ds)


main()