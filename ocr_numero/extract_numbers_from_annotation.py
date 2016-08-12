# -*- coding: utf-8 -*-
"""
Extract numbers from annotation segementation from a Paris historical map
Jacoubet 
Created on Wed Aug 10 12:46:12 2016

@author: remi
"""

# read image : use gdal to keep georeferencing

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
            
            
def plot_image(im, cmap='Greys'):
    import matplotlib.pyplot as plt
    fig = plt.figure()  # a new figure window
    ax = fig.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)   
    ax.imshow(im, cmap) 

def load_image(image_path):
    from osgeo import gdal
    import numpy as np
    ds = gdal.Open(image_path)
    jac = np.array(ds.GetRasterBand(1).ReadAsArray())
    jac_inverted = np.logical_not(jac)
    return jac, jac_inverted, ds


def postgres_friendly_list_printing(s):
    with_curly = list(s).__str__().replace('[','{').replace(']','}')
    with_double_quote = with_curly.__str__().replace("'",'"')
    return with_double_quote

def write_csv_result(file_path, to_write):
    import csv
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                                quotechar="|", quoting=csv.QUOTE_MINIMAL )
        writer.writerow(to_write)
    return 

#try several orientation

def pixelOffset2coord(ds, xOffset, yOffset):
    geotransform = ds.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    coordX = originX+pixelWidth*xOffset
    coordY = originY+pixelHeight*yOffset
    return coordX, coordY


def find_image_center_coordinates(im_array, ds):
    xOffset = int( im_array.shape[0]/2)
    yOffset = int( im_array.shape[1]/2)
    
    coordX, coordY = pixelOffset2coord(ds, xOffset, yOffset)
    return coordX, coordY

    

# extract number from one orientatiobn
def extract_number_from_one_orientation(im, orientation,lang='fra'):
    import pyocr
    import pyocr.builders
    from PIL import Image
    import numpy as np
    
    tools = pyocr.get_available_tools()
    tool = tools[1]
     
    im_r = im.rotate(orientation, expand=True)
    
    digits = tool.image_to_string(im_r,
                                      lang=lang,
        # builder=pyocr.builders.WordBoxBuilder()
        builder = pyocr.builders.TextBuilder()
    )
    if not digits:
        return
    else:
        return digits 
        


def get_file_base_name(ds):
    from os import path 
    file_list = ds.GetFileList()
    filename_ne, file_extension = path.splitext(file_list[0])
    base_file_name = path.basename(filename_ne) 
    return base_file_name
    
def compute_image_area(ds): 
    geotransform = ds.GetGeoTransform() 
    pixelWidth = geotransform[1] 
    pixelHeight = geotransform[5]
    
    return  abs(pixelWidth * ds.RasterXSize * pixelHeight * ds.RasterYSize )
    
def detect_digits_all_orientations(im_path):
    # load image
    from PIL import Image
    import numpy as np
    
    jac, jac_inverted, ds = load_image(im_path)
    coordX, coordY = find_image_center_coordinates(jac, ds)
    
    from os import path 
    file_list = ds.GetFileList()
    filename_ne, file_extension = path.splitext(file_list[0])
    base_file_name = path.basename(filename_ne)
    base_file_name = get_file_base_name(ds)
    
    
    im_area = compute_image_area(ds) 
    
    if im_area < 19 or im_area >70:
        print('     image area : ',im_area)
        print('     image too small (<19) or too big (>70)')
        return base_file_name, coordX, coordY, None, None
     
    im = Image.fromarray((jac*255).astype(np.uint8))
    contents = []
    orientations = []
    for orientation in range(0,360,10): 
        content = extract_number_from_one_orientation(im, orientation,lang='fra')
        if content:
            contents.append(content)
            orientations.append(orientation)
    
    return base_file_name, coordX, coordY, orientations, contents

def process_one_file_arr(arr_i):
    im_path, result_file = arr_i
    from os import path
    from os import getpid
    print(getpid(),' : dealing with image ',path.basename(im_path))
    process_one_file(im_path, result_file)

def process_one_file(im_path, result_file):
    base_file_name, coordX, coordY, orientation, contents = detect_digits_all_orientations(im_path)
     
    
    #add result to file
    if contents:
        with lock:
            write_csv_result(result_file,
                         [base_file_name, coordX, coordY,
                          postgres_friendly_list_printing(orientation),
                          postgres_friendly_list_printing(contents)]) 

def init(l):
    global lock
    lock = l
    
def parallel_version(function_arg, num_process):
    import multiprocessing
    l = multiprocessing.Lock() 
     
    pool = multiprocessing.Pool(num_process, initializer=init, initargs=(l,))
    
    pool.map(process_one_file_arr, function_arg)
    pool.close()
    pool.join()

def find_all_tiff_in_directory(input_folder, output_file ):
    from os import path
    import fnmatch
    function_arg = []
    for root, dirnames, filenames in walklevel(input_folder, 0):
        # print(root, dirnames, filenames)
        for filename in fnmatch.filter(filenames, '*.tif'):
            filename_ne, file_extension = path.splitext(filename)
            # print(path.join(root, filename), path.join(output_folder, filename_ne), tilesize)
            function_arg.append((path.join(root, filename), output_file) ) 
    return function_arg
    
def main():
    base_im_path = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/results/annotations/' 
    base_im_path_2  = '/home/remi/Desktop/annotations/'
    
    im_path = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/results/annotations/Feuille24_4000_6000_(408, 376, 464, 429).tif'
    
    result_file = '/media/sf_RemiCura/PROJETS/belleepoque/extract_data_from_old_paris_map/jacoubet/results/annotations_ocr/annotation_ocr.csv'
      
    num_process = 3
    
    #find all tiff files of a base repo
    function_arg = find_all_tiff_in_directory(base_im_path_2[0:100], result_file )
    # function_arg = function_arg[0:3]  
    #detect all possible digits in one image
    parallel_version(function_arg, num_process)
    
    
    
main()