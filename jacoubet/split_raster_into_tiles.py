# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 14:48:25 2016

@author: remi
"""

import os, sys
from osgeo import gdal


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
            
            
def cut_one_raster_into_tiles(i_list): 
    input_file_path, output_file_ne, tilesize  = i_list
    from os import path 
    dset = gdal.Open(input_file_path)
    
    width = dset.RasterXSize
    height = dset.RasterYSize
    
    print(width, 'x', height)
 
    
    for i in range(0, width, tilesize):
        for j in range(0, height, tilesize):
            w = min(i+tilesize, width) - i
            h = min(j+tilesize, height) - j
            gdaltranString = "gdal_translate -of GTIFF -co TFW=YES -srcwin "+str(i)+", "+str(j)+", "+str(w)+", " \
                +str(h)+" " + input_file_path + " " + output_file_ne + "_"+str(i)+"_"+str(j)+".tif"
#            print(gdaltranString)
            os.system(gdaltranString)
            
def cut_a_folder_into_tiles(input_folder, output_folder,tilesize, mutliprocess_version=False):
    import glob
    from os import path
    import fnmatch 
    function_arg = []
    for root, dirnames, filenames in walklevel(input_folder,0):
    #    print(root, dirnames, filenames)
        for filename in fnmatch.filter(filenames, '*.tif'):
            filename_ne, file_extension = os.path.splitext(filename)
            print('dealing with raster ',filename)
#            print(path.join(root, filename), path.join(output_folder, filename_ne), tilesize)
            arg = (path.join(root, filename), path.join(output_folder, filename_ne), tilesize)
            if mutliprocess_version == False:
               cut_one_raster_into_tiles(arg) 
            function_arg.append(arg)
    return function_arg 
            
            
def multi_process_version(num_processes, function_arg):
    import multiprocessing as mp 
    pool = mp.Pool(num_processes) 
    results = pool.map(cut_one_raster_into_tiles, function_arg)

def main():
    input_folder = '/media/sf_RemiCura/DATA/EHESS/GIS_maurizio/jacoubet_l1/'
    output_folder = '/media/sf_RemiCura/DATA/EHESS/GIS_maurizio/jacoubet_l1/tiles/'
    tilesize = 2000
    mutliprocess_version = True
    num_processes = 6
    
    
    function_arg = cut_a_folder_into_tiles(input_folder, output_folder,tilesize, mutliprocess_version)
    if mutliprocess_version == True:
        multi_process_version(num_processes, function_arg)
        

    
    
    
main()