# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 10:40:59 2016
Create a vector file describing the center of detected annotation images
@author: remi
"""

# create a
 
#for each tiff in a repository :
# open it, get coordinates of center
# write coordinate + image path into the shapefile

# close shapefile

def load_image(image_path):
    from osgeo import gdal
    import numpy as np
    ds = gdal.Open(image_path) 
    return ds


def pixelOffset2coord(ds, xOffset, yOffset):
    geotransform = ds.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    coordX = originX+pixelWidth*xOffset
    coordY = originY+pixelHeight*yOffset
    return coordX, coordY


def get_image_bbox(ds): 
    geotransform = ds.GetGeoTransform() 
    
    x_min = geotransform[0]
    y_min = geotransform[3]
    x_max, y_max = pixelOffset2coord(ds, ds.RasterXSize, ds.RasterYSize)
    return x_min, y_min, x_max, y_max


def bbox_to_wkt(x_min, y_min, x_max, y_max):
    from osgeo import ogr

    # Create ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(x_min, y_min)
    ring.AddPoint(x_min, y_max)
    ring.AddPoint(x_max, y_max)
    ring.AddPoint(x_max, y_min)
    ring.AddPoint(x_min, y_min)
    
    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    return poly.ExportToWkt()


def walklevel(some_dir, level=1):
    import os
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def find_all_tiff_in_directory(input_folder,file_basename=''):
    from os import path
    import fnmatch
    function_arg = []
    for root, dirnames, filenames in walklevel(input_folder, 0):
        # print(root, dirnames, filenames)
        for filename in fnmatch.filter(filenames, file_basename+'*.tif'):
            filename_ne, file_extension = path.splitext(filename)
            # print(path.join(root, filename), path.join(output_folder, filename_ne), tilesize)
            function_arg.append(path.join(root, filename) ) 
    return function_arg

def write_csv_result(file_path, to_write):
    import csv
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar="|", quoting=csv.QUOTE_MINIMAL )
        writer.writerow(to_write)
    return  


def deal_with_one_image(im_path, output_vector_file_path):
    from os import path
    from os import getpid
    from random import randint 
    if randint(0,100) == 0:
        print(getpid(),' : dealing with image ',path.basename(im_path))
    ds = load_image(im_path)
    x_min, y_min, x_max, y_max = get_image_bbox(ds)
    poly_wkt = bbox_to_wkt(x_min, y_min, x_max, y_max)
    ds = None
    with lock:
        write_csv_result(output_vector_file_path, (im_path, poly_wkt))


def deal_with_one_image_g(im_path):   
    deal_with_one_image(im_path, output_vector_file)

def get_file_base_name(ds):
    from os import path 
    file_list = ds.GetFileList()
    filename_ne, file_extension = path.splitext(file_list[0])
    base_file_name = path.basename(filename_ne) 
    return base_file_name


def test_on_one_image():
    
    one_image_path = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/results/annotations/Feuille24_4000_6000_(408, 376, 464, 429).tif'
    ds = load_image(one_image_path)
    x_min, y_min, x_max, y_max = get_image_bbox(ds)
    poly_wkt = bbox_to_wkt(x_min, y_min, x_max, y_max)


def init(l):
    global lock
    lock = l
    
def parallel_version(function_arg, num_process):
    import multiprocessing
    l = multiprocessing.Lock() 
     
    pool = multiprocessing.Pool(num_process, initializer=init, initargs=(l,))
    
    pool.map(deal_with_one_image_g, function_arg)
    pool.close()
    pool.join()

     
def main():
    from os import path
    from datetime import datetime 
    
    beginning = datetime.now()
    
    global output_vector_file 
    output_vector_file = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/results/annotations_ocr/annotations_index.csv'
    base_im_path = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/results/annotations/' 
    base_im_path_2  = '/home/remi/Desktop/annotations/'
    
    num_process = 3
    list_of_im = find_all_tiff_in_directory(base_im_path_2)
    parallel_version(list_of_im, num_process)
    print('duration ', datetime.now()-beginning)
    
    # 100: 30
    """
    list_of_list_of_im = []
    for i in range(1,52+1):
        basename = 'Feuille'+str(i)
        list_of_im = find_all_tiff_in_directory(base_im_path, basename)
        list_of_list_of_im.append([list_of_im, basename])"""
    
    
    
    """
    for im in list_of_im[1:100]:
        print('dealing with image ',path.basename(im))
        deal_with_one_image(im, output_vector_base_path)
    """
        
main()