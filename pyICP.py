from PIL import Image
from PIL import ImageFilter
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageChops
from PIL import ImageTk
import numpy
import numpy.linalg
import math
import re
import sys, os, os.path, glob
import functools
import tkinter as Tkinter
from tkinter.constants import *
import tkinter.font as tkFont

#---------------------------------------- Support Functions ---------------------------------------
def _euclidean(p, q):
    p,q = numpy.array(p), numpy.array(q)
    return numpy.sqrt( numpy.dot(p-q,p-q) )

def _indexAndLeastDistance(distList):
    minVal = min(distList)
    return distList.index(minVal), minVal    

def _difference(p,q):
    return p[0]-q[0],p[1]-q[1]

def _mean_coordinates(coords_list):
    mean = functools.reduce(lambda x,y: x+y, [x[0] for x in coords_list]), \
           functools.reduce(lambda x,y: x+y, [x[1] for x in coords_list])

    mean = mean[0]/float(len(coords_list)), mean[1]/float(len(coords_list))
    return mean

def _display_data_matrix(data_matrix):
    print("\ndata matrix:")
    for col in data_matrix:
        print(str(col))

def _print_points(msg, points_arr):
    print(msg,)
    print(str( ['(' + ("%.1f, %.1f")%(x,y) + ')' for (x,y) in points_arr] ) )

def _print_points_in_dict(msg, points_dict):
    print(msg,)
    print( [item[0]+" : "+'('+("%.1f, %.1f")%(points_dict[item[0]][0],points_dict[item[0]][1])+')' 
                    for item in sorted(points_dict.items(), 
                             lambda x,y: cmp(int(x[0].lstrip('md')),int(y[0].lstrip('md'))))] )

def _print_float_values_dict(msg, dict_with_float_vals):
    print(msg,)
    print( [item[0]+" : " + ("%.2f")%(dict_with_float_vals[item[0]]) 
             for item in sorted(dict_with_float_vals.items(), 
                    lambda x,y: cmp(int(x[0].lstrip('md')),int(y[0].lstrip('md'))))
                            if dict_with_float_vals[item[0]] is not None] )

def _least_dist_mapping(data_dict, model_dict, dist_threshold):
    mapping = {d : None for d in data_dict}
    error_dict = {d : None for d in data_dict}
    for d in sorted(data_dict.keys(), key = lambda x: x.lstrip('md')):
        dist_values = {m : _euclidean(data_dict[d],model_dict[m]) for m in model_dict}
        for mlabel in sorted(dist_values.keys(), key = lambda x: dist_values[x]):
            if dist_values[mlabel] < dist_threshold:
                mapping[d] = mlabel
                error_dict[d] = dist_values[mlabel]
                break;
    return mapping, error_dict

#-------------------------------------- ICP Class Definition --------------------------------------

class ICP(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''ICP constructor can only be called with keyword arguments for 
                      the following keywords: model_image, data_image, binary_or_color,
                      calculation_image_size, iterations, corners_or_edges, 
                      corner_detection_threshold, pixel_correspondence_dist_threshold, 
                      edge_detection_threshold, max_num_of_pixels_used_for_icp,
                      image_polarity, auto_select_model_and_data, 
                      smoothing_low_medium_or_high, font_file, debug1, and debug2''')       
        model_image=data_image=calculation_image_size=iterations=corner_detection_threshold=None
        corners_or_edges=edge_detection_threshold=pixel_correspondence_dist_threshold=None
        max_num_of_pixels_used_for_icp=smoothing_low_medium_or_high=subimage_index=None
        image_polarity=auto_select_model_and_data=font_file=debug1=debug2=None

        if 'model_image' in kwargs                 :              model_image=kwargs.pop('model_image')
        if 'data_image' in kwargs                  :               data_image=kwargs.pop('data_image')
        if 'binary_or_color' in kwargs             :      binary_or_color=kwargs.pop('binary_or_color')
        if 'binary_or_color' in kwargs             :      binary_or_color=kwargs.pop('binary_or_color')
        if 'iterations' in kwargs                  :                iterations=kwargs.pop('iterations')
        if 'corners_or_edges' in kwargs            :    corners_or_edges=kwargs.pop('corners_or_edges')
        if 'image_polarity' in kwargs              :        image_polarity=kwargs.pop('image_polarity')
        if 'subimage_index' in kwargs              :        subimage_index=kwargs.pop('subimage_index')
        if 'font_file' in kwargs                   :                  font_file=kwargs.pop('font_file')
        if 'debug1' in kwargs                      :                        debug1=kwargs.pop('debug1')
        if 'debug2' in kwargs                      :                        debug2=kwargs.pop('debug2')
        if 'calculation_image_size' in kwargs      :  \
                                            calculation_image_size=kwargs.pop('calculation_image_size')
        if 'corner_detection_threshold' in kwargs  :  \
                                    corner_detection_threshold=kwargs.pop('corner_detection_threshold')
        if 'edge_detection_threshold' in kwargs    :  \
                                        edge_detection_threshold=kwargs.pop('edge_detection_threshold')
        if 'pixel_correspondence_dist_threshold' in kwargs:  \
                  pixel_correspondence_dist_threshold=kwargs.pop('pixel_correspondence_dist_threshold')
        if 'auto_select_model_and_data' in kwargs: \
                                    auto_select_model_and_data=kwargs.pop('auto_select_model_and_data')
        if 'max_num_of_pixels_used_for_icp' in kwargs: \
                            max_num_of_pixels_used_for_icp=kwargs.pop('max_num_of_pixels_used_for_icp')
        if 'smoothing_low_medium_or_high' in kwargs: \
                                smoothing_low_medium_or_high=kwargs.pop('smoothing_low_medium_or_high')
        if len(kwargs) != 0:
                                  raise ValueError('''You have provided unrecognizable keyword args''')
        if model_image: 
            self.model_im = Image.open(model_image)
            self.original_model_im = Image.open(model_image)
        else:
            self.model_im = None
        if data_image: 
            self.data_im =  Image.open(data_image)
            self.original_data_im = Image.open(data_image)
        else:
            self.data_im = None
        if binary_or_color:
            self.binary_or_color = binary_or_color
        else:
            raise ValueError('''You must specify either "binary" or "color" ''')
        if font_file:
            self.font_file = font_file
        elif os.path.isfile("FreeSerif.ttf"):
            self.font_file = "FreeSerif.ttf"
        elif os.path.isfile("/usr/share/fonts/truetype/freefont/FreeSerif.ttf"):
            self.font_file = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
        else:
            print("Unable to find the font file 'FreeSerif.ttf' needed for displaying the results")
            print("Use the 'font_file' option in the constructor to specify your own font file")
            sys.exit(1)
        if corners_or_edges:
            self.corners_or_edges = corners_or_edges
        else:
            self.corners_or_edges = "edges"
        if smoothing_low_medium_or_high:
            self.smoothing_low_medium_or_high = smoothing_low_medium_or_high
        else:
            self.smoothing_low_medium_or_high = "medium"  
        if image_polarity:
            self.image_polarity = image_polarity
        elif corners_or_edges == "corners":
            raise ValueError('''\n\nYou must specify image_polarity as 1 or -1 when using '''
                             '''corner-based ICP. The polarity is 1 when the object pixels are '''
                             '''generally brighter than the background pixels. Otherwise it is -1.''')
        if subimage_index is None: 
            self.subimage_index = None
        else:
            self.subimage_index = subimage_index
        if calculation_image_size:
            self.calculation_image_size = calculation_image_size
        elif self.model_im.size[0] <= 100:
                self.calculation_image_size = self.model_im.size[0]
        else:
            self.calculation_image_size = 200
        if iterations:
            self.iterations = iterations
        else:
            self.iterations = 24
        if corner_detection_threshold:
            self.corner_detection_threshold = corner_detection_threshold
        else:
            self.corner_detection_threshold = 0.2
        if edge_detection_threshold:
            self.edge_detection_threshold = edge_detection_threshold
        else:
            self.edge_detection_threshold = 50
        if pixel_correspondence_dist_threshold:
            self.pixel_correspondence_dist_threshold = pixel_correspondence_dist_threshold
        else:
            self.pixel_correspondence_dist_threshold = 100
        if max_num_of_pixels_used_for_icp:
            self.max_num_of_pixels_used_for_icp = max_num_of_pixels_used_for_icp
        else:
            self.max_num_of_pixels_used_for_icp = 100
        if debug1:
            self.debug1 = debug1
        else:
            self.debug1 = 0
        if debug2:
            self.debug2 = debug2
        else:
            self.debug2 = 0
        if auto_select_model_and_data:
            self.auto_select_model_and_data = auto_select_model_and_data
        else:
            self.auto_select_model_and_data = 0
        self.model_rval =  None
        self.data_rval  =  None
        self.model_segmentation = None
        self.model_all_corners = None
        self.model_list = []
        self.data_segmentation = None
        self.data_all_corners = None
        self.data_list = []
        self.model_edge_map = None
        self.data_edge_map = None

    def extract_pixels_from_color_image(self, model_or_data):
        if model_or_data == "model":
            im = self.model_im
        else:
            im = self.data_im
        im = im.convert('L')        ## convert to gray level
        im.thumbnail( (self.calculation_image_size, self.calculation_image_size), Image.ANTIALIAS )
        if self.debug1: im.show()
        width,height = im.size
        if self.debug1: print("width: %d    height: %d" % (width, height))
        if self.corners_or_edges == "edges":
            dx = numpy.zeros((height, width), dtype="float")
            dy = numpy.zeros((height, width), dtype="float")
            rval = numpy.zeros((height, width), dtype="float")
            result_im = Image.new("1", (width,height), 0)
            edge_im = Image.new("L", (width,height), 0)
            edge_pixel_list = []    
            corner_pixels = []
            for i in range(3,width-3):
                for j in range(3,height-3):
                    ip1,im1,jp1,jm1 = i+1,i-1,j+1,j-1
                    dx[(j,i)] = (im.getpixel((ip1,jm1)) + 2*im.getpixel((ip1,j)) + \
                                 im.getpixel((ip1,jp1))) -  (im.getpixel((im1,jm1)) + \
                                 2*im.getpixel((im1,j)) + im.getpixel((im1,jp1))) 
                    dy[(j,i)] = (im.getpixel((im1,jp1)) + 2*im.getpixel((i,jp1)) + \
                                 im.getpixel((ip1,jp1))) -  (im.getpixel((im1,jm1)) + \
                                 2*im.getpixel((i,jm1)) + im.getpixel((ip1,jm1))) 
            edge_pixel_dict = {}
            for i in range(3,width-3):             
                for j in range(3,height-3):         
                    edge_strength = math.sqrt(dx[(j,i)]**2 + dy[(j,i)]**2)
                    edge_im.putpixel((i,j), int(edge_strength))   
                    if edge_strength > self.edge_detection_threshold:            
                        edge_pixel_dict["e_" + str(i) + "_" + str(j)] = edge_strength
            sorted_edge_pixels = sorted(edge_pixel_dict.keys(), \
                                          key=lambda x: edge_pixel_dict[x], reverse=True)
            if len(sorted_edge_pixels) > self.max_num_of_pixels_used_for_icp:
                sorted_edge_pixels = sorted_edge_pixels[0:self.max_num_of_pixels_used_for_icp]
            for pixel_label in sorted_edge_pixels: 
                parts = re.split(r'_', pixel_label)
                i_index,j_index = int(parts[-2]), int(parts[-1])
                edge_pixel_list.append((i_index,j_index))
                result_im.putpixel((i_index,j_index), 255)
            if self.debug1: result_im.show()
            if model_or_data == "model":
                if self.debug1: result_im.save("model_image_pixels_retained.jpg")
                self.model_im = result_im
                self.model_list = edge_pixel_list
                self.model_edge_map = edge_im
                if self.debug1: edge_im.save("model_edge_image.jpg")
            else:
                if self.debug1: result_im.save("data_image_pixels_retained.jpg")
                self.data_im = result_im
                self.data_list = edge_pixel_list
                self.data_edge_map = edge_im
                if self.debug1: edge_im.save("data_edge_image.jpg")
        else:
            im2 = im.copy()
            if self.smoothing_low_medium_or_high == "low":
                how_much_smoothing = 1
            elif self.smoothing_low_medium_or_high == "medium":
                how_much_smoothing = 5
            elif self.smoothing_low_medium_or_high == "high":
                how_much_smoothing = 10
            else:
                sys.exit('''\n\nYour value for smoothing_low_medium_or_high parameter must be '''
                         '''either "low", or "medium", or "high" ''')
            for i in range(how_much_smoothing):
                im2 = im2.filter(ImageFilter.BLUR) 
            if self.debug1: im2.show()
            hist = im2.histogram()
            total_count = functools.reduce(lambda x,y: x+y, hist)
            coarseness = 8              # make it a divisor of 256
            probs = [functools.reduce(lambda x,y: x+y, hist[coarseness*i:coarseness*i+coarseness])/float(total_count)
                                                     for i in range(int(len(hist)/coarseness))]
            prob_times_graylevel = [coarseness * i * probs[i] for i in range(len(probs))]
            mu_T = functools.reduce(lambda x,y: x+y, prob_times_graylevel)       # mean for the image
            prob_times_graysquared = [(coarseness * i - mu_T)**2 * probs[i] for i in range(len(probs))]
            sigma_squared_T = functools.reduce(lambda x,y: x+y, prob_times_graysquared)
            m0 = [functools.reduce(lambda x,y: x+y, probs[:k]) for k in range(1,len(probs)+1)]
            m1 = [functools.reduce(lambda x,y: x+y, prob_times_graylevel[:k]) for k in range(1,len(probs)+1)]
            sigmaB_squared = [None] * len(m0)          # for between-class variance as a func of threshold
            sigmaW_squared = [None] * len(m0)          # for within-class variance as a func of threshold
            variance_ratio = [None] * len(m0)          # for the ratio of the two variances
            for k in range(len(m0)):
                if 0 < m0[k] < 1.0:
                    sigmaB_squared[k] = (mu_T * m0[k] - m1[k])**2 / (m0[k] * (1.0 - m0[k]))
                    sigmaW_squared[k] = sigma_squared_T - sigmaB_squared[k]
                    variance_ratio[k] = sigmaB_squared[k] / sigmaW_squared[k]
            variance_ratio_without_none = [x for x in variance_ratio if x is not None ]
            otsu_threshold = variance_ratio.index(max(variance_ratio_without_none)) * coarseness
            if self.debug1: print( "\nbest threshold: %s" % str(otsu_threshold))
            segmented_im2 = Image.new("1", (width,height), 0)            
            for i in range(width):
                for j in range(height):
                    if self.image_polarity == 1:
                        if im2.getpixel((i,j)) > otsu_threshold: segmented_im2.putpixel((i,j), 255)
                    elif self.image_polarity == -1:
                        if im2.getpixel((i,j)) < otsu_threshold: segmented_im2.putpixel((i,j), 255)
                    else:
                        sys.exit("You did not specify image polarity")
            if self.debug1: segmented_im2.show()
            dx = numpy.zeros((height, width), dtype="float")
            dy = numpy.zeros((height, width), dtype="float")
            rval = numpy.zeros((height, width), dtype="float")
            corner_pixels = []
            for i in range(3,width-3):
                for j in range(3,height-3):
                    if segmented_im2.getpixel((i,j)) == 255:
                        ip1,im1,jp1,jm1 = i+1,i-1,j+1,j-1
                        dx[(j,i)] = (im.getpixel((ip1,jm1)) + 2*im.getpixel((ip1,j)) + \
                                     im.getpixel((ip1,jp1))) -  (im.getpixel((im1,jm1)) + \
                                     2*im.getpixel((im1,j)) + im.getpixel((im1,jp1))) 
                        dy[(j,i)] = (im.getpixel((im1,jp1)) + 2*im.getpixel((i,jp1)) + \
                                     im.getpixel((ip1,jp1))) -  (im.getpixel((im1,jm1)) + \
                                     2*im.getpixel((i,jm1)) + im.getpixel((ip1,jm1))) 
            if self.debug1:
                self.display_array_as_image(dx)
                self.display_array_as_image(dy)
            corners_im = Image.new("1", (width,height), 0)
            for i in range(3,width-3):
                for j in range(3,height-3):
                    if segmented_im2.getpixel((i,j)) == 255:
                        Cmatrix = numpy.zeros((2, 2), dtype="float")
                        c11=c12=c22=0.0
                        for k in range(i-2,i+3):        
                            for l in range(j-2,j+3):    
                                c11 += dx[(l,k)] * dx[(l,k)] 
                                c12 += dx[(l,k)] * dy[(l,k)] 
                                c22 += dy[(l,k)] * dy[(l,k)] 
                        Cmatrix[(0,0)] = c11
                        Cmatrix[(0,1)] = c12
                        Cmatrix[(1,0)] = c12
                        Cmatrix[(1,1)] = c22
                        determinant = numpy.linalg.det(Cmatrix)
                        trace       = numpy.trace(Cmatrix)                
                        ratio = 0.0
                        if trace != 0.0:
                            ratio = determinant / (trace * trace)
                        rval[(j,i)] = ratio
                        if abs(ratio) > self.corner_detection_threshold:
                            corner_pixels.append((i,j))
                            corners_im.putpixel((i,j), 255)
            if self.debug1: corners_im.show()
            singular_corners_im = Image.new("1", (width,height), 0)      
            singular_corners = []
            for candidate in corner_pixels:
                i,j = candidate
                non_singular_corner_found = False
                for corner in corner_pixels:
                    if corner == candidate: continue
                    k,l = corner
                    if abs(i-k) <=1 and abs(j-l) <= 1:
                        if rval[(j,i)] <= rval[(l,k)]:
                            non_singular_corner_found = True
                            break
                if non_singular_corner_found: continue
                singular_corners.append(candidate)
                singular_corners_im.putpixel((i,j), 255)
            singular_corners.sort(key = lambda x: rval[(x[1],x[0])], reverse=True)
            sorted_singular_corners_im = Image.new("1", (width,height), 0)      
            if len(singular_corners) > self.max_num_of_pixels_used_for_icp:
                singular_corners = singular_corners[0:self.max_num_of_pixels_used_for_icp]
            for corner in singular_corners:
                sorted_singular_corners_im.putpixel(corner, 255)
            if self.debug1: sorted_singular_corners_im.show()
            if model_or_data == "model":
                if self.debug1: sorted_singular_corners_im.save("model_corners.jpg")
                self.model_im = sorted_singular_corners_im
                self.model_segmentation = segmented_im2
                self.model_all_corners = corners_im
                self.model_list = singular_corners
                self.model_rval = rval
            else:
                if self.debug1: sorted_singular_corners_im.save("data_corners.jpg")
                self.data_im = sorted_singular_corners_im
                self.data_segmentation = segmented_im2
                self.data_all_corners = corners_im
                self.data_list = singular_corners
                self.data_rval = rval

    def condition_data(self):
        cutoff = 1.0
        num_of_data_pixels = len(self.data_list)
        num_of_model_pixels = len(self.model_list)
        diff = abs(num_of_data_pixels - num_of_model_pixels)
        min_count = min(num_of_data_pixels, num_of_model_pixels)
        if diff < cutoff * min_count: return
        if num_of_data_pixels > num_of_model_pixels:
            how_many = int( num_of_model_pixels + cutoff * min_count )
            self.data_list = self.data_list[0:how_many]
        else:
            how_many = int( num_of_data_pixels + cutoff * min_count )
            self.model_list = self.model_list[0:how_many]
        if self.debug1:
            _print_points("\nmodel list (in pixel coords) in `condition_data()': ", self.model_list)
            _print_points("\ndata list (in pixel coords) in `condition_data()': ", self.data_list)
        self.display_pixel_list_as_image(self.model_list)
        self.display_pixel_list_as_image(self.data_list)      
        self.save_pixel_list_as_image(self.model_list, "final_pixels_retained_model.jpg")
        self.save_pixel_list_as_image(self.data_list, "final_pixels_retained_data.jpg")

    def display_pixel_list_as_image(self, pixel_list):
        width,height = self.model_im.size
        display_im = Image.new("1", (width,height), 0)
        for pixel in pixel_list:
            display_im.putpixel(pixel, 255)
        display_im.show()

    def save_pixel_list_as_image(self, pixel_list, filename):
        width,height = self.model_im.size
        save_im = Image.new("1", (width,height), 0)
        for pixel in pixel_list:
            save_im.putpixel(pixel, 255)
        save_im.save(filename)

    def display_array_as_image(self, numpy_arr):
        height,width = numpy_arr.shape
        display_im = Image.new("L", (width,height), 0)
        for i in range(3,width-3):
            for j in range(3,height-3):
                display_im.putpixel((i,j), int(abs(numpy_arr[(j,i)])))
        display_im.show()

    def save_array_as_image(self, numpy_arr, label):
        height,width = numpy_arr.shape
        save_im = Image.new("L", (width,height), 0)
        for i in range(3,width-3):
            for j in range(3,height-3):
                save_im.putpixel((i,j), int(abs(numpy_arr[(j,i)])))
        save_im.save(label + ".jpg")

    def extract_pixels_from_binary_image(self, model_or_data):
        if model_or_data == "model":
            self.model_list = []
            if self.model_im.size[0] > 100:
                self.model_im.thumbnail( (self.calculation_image_size,\
                                 self.calculation_image_size), Image.ANTIALIAS )
            self.model_im = self.model_im.convert("L").convert("1")
            width, height =  self.model_im.size
            for i in range(3,width-3): 
                for j in range(3,height-3):
                    if ( self.model_im.getpixel((i,j)) != 0 ):  
                        self.model_list.append( (i,j) )       
        elif model_or_data == "data":
            self.data_list = []
            if self.data_im.size[0] > 100:
                self.data_im.thumbnail( (self.calculation_image_size,\
                                     self.calculation_image_size), Image.ANTIALIAS )
            self.data_im = self.data_im.convert("L").convert("1")
            width, height =  self.data_im.size
            for i in range(3,width-3): 
                for j in range(3,height-3):
                    if ( self.data_im.getpixel((i,j)) != 0 ):  
                        self.data_list.append( (i,j) )       
        else: sys.exit("Wrong arg used for extract_pixels_from_binary_image()")
        if self.debug1:
            if model_or_data == "model":    
                print("model pixel list %s" % str(self.model_list))
                print("\nnumber of pixels in model_list: %s" % str(len(self.model_list)))
            else:
                print("\ndata pixel list %s" % str(self.data_list))
                print("\nnumber of pixels in data_list: %s" % str(len(self.data_list)))

    def move_to_model_origin(self):
        self.model_mean = (numpy.matrix(list(_mean_coordinates(self.model_list)))).T

        self.zero_mean_model_list = [(p[0] - self.model_mean[0,0], \
                                p[1] - self.model_mean[1,0]) for p in self.model_list]
        self.zero_mean_data_list = [(p[0] - self.model_mean[0,0], \
                                p[1] - self.model_mean[1,0]) for p in self.data_list]
        if self.debug1:
            print("\nmodel mean in pixel coords: %s\n" % str(self.model_mean))
            _print_points("\nzero mean model list (pixel coords): ", self.zero_mean_model_list)
            _print_points("\nzero mean data list (pixel coords): ", self.zero_mean_data_list)

    def icp(self):
        if self.auto_select_model_and_data:
            if len(self.data_list) > len(self.model_list):
                print("\n>>>> SWAPPING THE MODEL AND THE DATA IMAGES <<<<\n\n")
                self.data_im, self.model_im = self.model_im, self.data_im
                self.data_list, self.model_list = self.model_list, self.data_list
                self.data_rval, self.model_rval = self.model_rval, self.data_rval
                self.data_segmentation, self.model_segmentation = \
                                  self.model_segmentation, self.data_segmentation
                self.data_all_corners, self.model_all_corners = \
                                    self.model_all_corners, self.data_all_corners
                self.data_edge_map, self.model_edge_map = self.model_edge_map, self.data_edge_map
        self.move_to_model_origin()
        old_error = float('inf')
        R = numpy.matrix( [[1.0, 0.0],[0.0, 1.0]] )
        T = (numpy.matrix([[0.0, 0.0]])).T
        self.R = R
        self.T = T
        self.R_for_iterations = {i : None for i in range(self.iterations)}
        self.T_for_iterations = {i : None for i in range(self.iterations)}
        model = self.zero_mean_model_list
        data = self.zero_mean_data_list
        model_dict = {"m" + str(i) : model[i] for i in range(len(model))}
        data_dict  = {"d" + str(i) : data[i] for i in range(len(data))}
        if self.debug2:
            _print_points_in_dict("\nmodel dict: ", model_dict)
            _print_points_in_dict("\ndata_dict: ", data_dict)
        error_for_iterations = []
        iteration = 0
        self.dir_name_for_results = "__result_" + str(self.subimage_index)
        if os.path.exists(self.dir_name_for_results):
            files = glob.glob(self.dir_name_for_results + "/*")
            map(lambda x: os.remove(x), files)
        else:
            os.mkdir(self.dir_name_for_results)
        while 1:
            if iteration == self.iterations: 
                print("\n\n\n***** FINAL RESULTS ****** FINAL RESULTS ****** FINAL RESULTS ****** FINAL RESULTS ****")
                print("\n\nImage size used in calculations:  width: %d  height: %d" % self.data_im.size)
                print("\nModel mean used for calculations: ") 
                print(str(self.model_mean))
                print("\nFinal rotation and translation of the data image with respect to the model mean: ")
                print("\nRotation:")
                print(str(self.R)) 
                print("\nTranslation:")
                print(str(self.T))
                print("\nData to Model Image Registration Error as a function of iterations: %s" % str(error_for_iterations))
                break
            if self.subimage_index is None:
                print("\n\n             STARTING ITERATION %s OUT OF %s\n" % (str(iteration+1), str(self.iterations)))
            else:
                print("\n\n  For subimages indexed %s   ==>  STARTING ITERATION %s OUT OF %s\n" % (str(self.subimage_index), str(iteration+1), str(self.iterations)))
            if self.debug2: _print_points_in_dict("\ndata_dict in loop: ", data_dict)
            xformed_data_dict = {d : R * (numpy.matrix( list(data_dict[d]) )).T + T for d in data_dict}
            xformed_data_dict = {d : (xformed_data_dict[d][0,0], xformed_data_dict[d][1,0]) \
                                                                        for d in xformed_data_dict}
            if self.debug2: _print_points_in_dict("\ntransformed data_dict in loop: ", xformed_data_dict)
            leastDistMapping, error_dict = \
              _least_dist_mapping(xformed_data_dict,model_dict,self.pixel_correspondence_dist_threshold)
            number_of_points_matched = len([x for x in leastDistMapping \
                                               if leastDistMapping[x] is not None])
            if self.debug2:
                print("\nleastDistMapping: ", [item[0]+"=>"+item[1] for item in sorted(leastDistMapping.items(),
                    lambda x,y: cmp(int(x[0].lstrip('md')),int(y[0].lstrip('md')))) if item[1] is not None])
            if self.debug2: _print_float_values_dict("\nerror values at data points: ", error_dict)
            if self.debug2: print("\nnumber of points matched: %s" % str(number_of_points_matched))
            error = functools.reduce(lambda x, y: x + y, [error_dict[x] for x in error_dict if error_dict[x] is not None])
            error = error / number_of_points_matched
            error_for_iterations.append(error)
            if self.debug2: print("\nold_error: %s    error: %s" % (str(old_error), str(error)))
            old_error = error
            data_labels_used = [d for d in leastDistMapping.keys() if leastDistMapping[d] is not None]
            data_labels_used.sort(key = lambda x: int(x.lstrip('md')))
            model_labels_used = [leastDistMapping[d] for d in data_labels_used]
            if self.debug2: print("\ndata labels used: %s" % str(data_labels_used))
            if self.debug2: print("\nmodel labels used: %s" % str(model_labels_used)) 
            A = numpy.matrix([[data_dict[d][0] for d in data_labels_used],[data_dict[d][1] for d in data_labels_used]])
            if self.debug2: print("\nA matrix: %s" % str(A))
            AATI = A.T * numpy.linalg.inv( A * A.T )
            if self.debug2: print("\nAATI matrix: %s" % str(AATI))
            B =  numpy.matrix([ [model_dict[m][0] - T[0,0] for m in model_labels_used], 
                                [model_dict[m][1] - T[1,0] for m in model_labels_used] ])
            if self.debug2: print("\nB matrix: %s" % str(B))
            matched_model_mean = numpy.matrix(list(_mean_coordinates([(model_dict[m][0], 
                                          model_dict[m][1]) for m in model_labels_used]))).T
            if self.debug2: print("\nmatched model mean: %s" % str(matched_model_mean))
            R_update = B * AATI * R.T
            [U,S,VT] = numpy.linalg.svd(R_update)
            U,VT = numpy.matrix(U), numpy.matrix(VT) 
            deter = numpy.linalg.det(U * VT)
            U[0,1] = U[0,1] * deter
            U[1,1] = U[1,1] * deter
            R_update = U * VT
            R = R_update * R
            print("\nRotation:")
            print (R)
            data_matrix2 = R * A
            data_transformed_mean = numpy.matrix(list(_mean_coordinates([(data_matrix2[0,j], 
                                   data_matrix2[1,j]) for j in range(data_matrix2.shape[1])]))).T
            if self.debug2: print("\ndata transformed mean: %s" % str(data_transformed_mean))
            T = matched_model_mean - data_transformed_mean  
            print("\nTranslation:")
            print(T)
            data_matrix_new = [ R * (numpy.matrix( list(data[p]))).T + T for p in range(len(data)) ] 
            data_transformed_new = \
                [ ( p[0,0] + self.model_mean[0,0], p[1,0] + self.model_mean[1,0] ) for p in data_matrix_new]
            displayWidth,displayHeight = self.data_im.size
            result_im = Image.new("1", (displayWidth,displayHeight), 0)
            for p in data_transformed_new:
                x,y = int(p[0]), int(p[1])
                if ( (0 <= x < displayWidth) and (0 <= y < displayHeight ) ):
                    result_im.putpixel( (x,y), 255 )
            result_im.save( self.dir_name_for_results + "/__result" + str(iteration) + ".jpg")
            iteration = iteration + 1
            self.R,self.T = R,T

    def display_results_as_movie(self):
        mw = Tkinter.Tk()                       
        tkFont.nametofont('TkDefaultFont').configure(size=20)    
        helv36 = tkFont.Font(family="Helvetica", size=28, weight='bold')    
        width, height = mw.winfo_screenwidth()-500, mw.winfo_screenheight()-100
        mw.title('''SHOW ITERATIONS AS A MOVIE   (red pixels are from the model and green '''
                 '''pixels from the data)''')
        mw.geometry( "%dx%d+100+100" % (width,height) )
        mw.focus_set()
        mw.bind("<Escape>", lambda e: e.widget.destroy())
        tkim = [None] * self.iterations
        model_image = self.model_im
        w_model, h_model =  model_image.size
        model_image_for_movie = Image.new("RGB", (w_model,h_model), (0,0,0))
        (model_mingray,model_maxgray) = model_image.getextrema()
        for i in range(w_model):
            for j in range(h_model):
                if model_image.getpixel((i,j)) > 0:
                    color_val = model_image.getpixel((i,j)) * int(255/model_maxgray)
                    model_image_for_movie.putpixel((i,j),(color_val,0,0))
        model_image = model_image_for_movie
        if w_model > h_model:
            w_display = int(0.5 * width)
            h_display = int(w_display * (h_model/float(w_model)))
        else:
            h_display = int(0.5 * height)
            w_display = int(h_display * (w_model/float(h_model)))
        imageframe = Tkinter.Frame(mw, width=w_display, height=h_display+50).pack()
        iterationIndexFrame = Tkinter.Frame(mw, width=w_display, height=10).pack()
        iterationLabelText = Tkinter.StringVar()
        Tkinter.Label(iterationIndexFrame,
                      textvariable = iterationLabelText,
                      anchor = 'c',
                      relief = 'groove',
                     ).pack(side='top', padx=10, pady=10)
        separator = Tkinter.Frame(mw, height=2, bd=1, relief=Tkinter.SUNKEN)
        separator.pack(fill=Tkinter.X, padx=5, pady=5)
        buttonframe = Tkinter.Frame(mw, width=w_display, height=10).pack()
        Tkinter.Button(buttonframe, 
                       text = 'Play movie again',                
                       anchor = 'c',
                       relief = 'raised',
                       font = helv36,
                       command = lambda: self.callbak(mw)
                      ).pack(side='top', padx=10, pady=5)

        separator = Tkinter.Frame(mw, height=2, bd=1, relief=Tkinter.SUNKEN)
        separator.pack(fill=Tkinter.X, padx=5, pady=5)    
        messageFrame = Tkinter.Frame(mw, width=w_display, height=10).pack()
        messageLabelText = Tkinter.StringVar()
        Tkinter.Label(messageFrame,
                      textvariable = messageLabelText,
                      anchor = 'c',
                      relief = 'groove',
                     ).pack(side='top', padx=10, pady=10)
        messageLabelText.set('''NOTE: It is best to NOT close this window until all the '''
                             '''iterations are completed''')
        self.iteration_control_flag = 1
        xpos = int( (width - w_display)/2 )
        ypos = 20
        for i in range(0,self.iterations):
            result_im = Image.open(self.dir_name_for_results + "/__result" + str(i) + ".jpg")
            (mingray,maxgray) = result_im.getextrema()
            rwidth,rheight = result_im.size
            result_color_im = Image.new("RGB", (rwidth,rheight), (0,0,0))
            for m in range(rwidth):
                for n in range(rheight):
                    if result_im.getpixel((m,n)) > 0:
                        color_val = result_im.getpixel((m,n)) * int(255/maxgray)
                        result_color_im.putpixel((m,n),(0,color_val,0))
            result_color_im.save( self.dir_name_for_results + "/__result_color" + str(i) + ".jpg")
        while self.iteration_control_flag:
            for i in range(0,self.iterations):
                try:
                    tkim[i] = Image.open(self.dir_name_for_results + "/__result_color" + str(i) + ".jpg")
                    out_image = ImageChops.add( model_image, tkim[i] )
                    out_out_image = out_image.resize((w_display,h_display), 
                                                     Image.ANTIALIAS)
                    out_photo_image = ImageTk.PhotoImage( out_out_image )
                    label_image = Tkinter.Label(imageframe,image=out_photo_image )
                    label_image.place(x=xpos,y=ypos,width=w_display,height=h_display)
                    iterationLabelText.set( "Iteration Number: " + str(i+1) )
                    self.iteration_control_flag = 0
                    if i < self.iterations - 1: mw.after(1000, mw.quit)       
                    mw.mainloop(0)
                except IOError: pass       

    def callbak(self,arg):
        arg.quit()
        self.iteration_control_flag = 1

    def displayImage6(self, argimage, title=""):
        width,height = argimage.size
        mw = Tkinter.Tk()
        winsize_x,winsize_y = None,None
        screen_width,screen_height = mw.winfo_screenwidth(),mw.winfo_screenheight()
        if screen_width <= screen_height:
            winsize_x = int(0.5 * screen_width)
            winsize_y = int(winsize_x * (height * 1.0 / width))            
        else:
            winsize_y = int(0.5 * screen_height)
            winsize_x = int(winsize_y * (width * 1.0 / height))
        display_image = argimage.resize((winsize_x,winsize_y), Image.ANTIALIAS)
        mw.title(title)   
        canvas = Tkinter.Canvas( mw,                         
                             height = winsize_y,
                             width = winsize_x,
                             cursor = "crosshair" )   
        canvas.pack( side = 'top' )                               
        frame = Tkinter.Frame(mw)                            
        frame.pack( side = 'bottom' )                             
        Tkinter.Button( frame,         
                text = 'Save',                                    
                command = lambda: canvas.postscript(file = title.partition(' ')[0] + ".jpg") 
              ).pack( side = 'left' )                             
        Tkinter.Button( frame,                        
                text = 'Exit',                                    
                command = lambda: mw.destroy(),                    
              ).pack( side = 'right' )                            
        photo = ImageTk.PhotoImage(argimage.resize((winsize_x,winsize_y), Image.ANTIALIAS))
        canvas.create_image(winsize_x/2,winsize_y/2,image=photo)
        mw.mainloop()

    def display_images_used_for_edge_based_icp(self):
        tk_images = []
        image_labels = []
        rootWindow = Tkinter.Tk()
        screen_width,screen_height =rootWindow.winfo_screenwidth(),rootWindow.winfo_screenheight()
        rootWindow.geometry( str(int(0.8 * screen_width)) + "x" + str(int(0.9 * screen_height)) + "+50+50") 
        canvas = Tkinter.Canvas(rootWindow)
        canvas.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=Tkinter.TRUE)
        scrollbar = Tkinter.Scrollbar(rootWindow,orient=Tkinter.HORIZONTAL,command=canvas.xview)
        scrollbar.pack(side=Tkinter.BOTTOM, fill=Tkinter.X)
        canvas.configure(xscrollcommand=scrollbar.set)
        def set_scrollregion(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
        frame = Tkinter.Frame(canvas)
        canvas.create_window((0,0), window=frame, anchor=Tkinter.NW)
        frame.bind('<Configure>', set_scrollregion)
        cellwidth = 2* self.data_im.size[0]
        padding = 10
        if cellwidth > 80:
            fontsize = 25
        else:
            fontsize = 15
        font = ImageFont.truetype(self.font_file, fontsize)
        data_image_width, data_image_height = self.data_im.size
        orig_image_width,orig_image_height = self.original_model_im.size
        original_model_im = self.original_model_im.copy()
        displayWidth,displayHeight = None,None
        if data_image_width > data_image_height:
            displayWidth = int(0.9 * cellwidth)
            displayHeight = int(displayWidth * data_image_height * 1.0 / data_image_width)
        else:
            displayHeight = int(0.9 * cellwidth)
            displayWidth = int(displayHeight * data_image_width * 1.0 / data_image_height )
        original_model_im.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        model_edge_map = self.model_edge_map.copy()
        model_edge_map = model_edge_map.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        model_edge_map.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        model_im = self.model_im.copy()
        model_im = model_im.resize((orig_image_width,orig_image_height), Image.ANTIALIAS)
        model_im.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        original_data_im = self.original_data_im.copy()
        original_data_im = original_data_im.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        original_data_im.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        data_edge_map = self.data_edge_map.copy()
        data_edge_map = data_edge_map.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        data_edge_map.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        data_im = self.data_im.copy()
        data_im = data_im.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        data_im.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        textImage7 = Image.new( "F", (displayWidth,displayHeight), 200 )
        draw = ImageDraw.Draw(textImage7)
        draw.text((10,10), "Close this window", font=font) 
        draw.text((10,30), "to see a movie of", font=font) 
        draw.text((10,50), "image registration", font=font) 
        image_labels.append("Model")
        tk_images.append(ImageTk.PhotoImage( original_model_im ))
        image_labels.append("Model Edge Map")
        tk_images.append(ImageTk.PhotoImage( model_edge_map ))
        image_labels.append("Model Edge Pixels\nRetained for ICP")
        tk_images.append(ImageTk.PhotoImage( model_im ))
        image_labels.append("Data")
        tk_images.append(ImageTk.PhotoImage( original_data_im ))
        image_labels.append("Data Edge Map")
        tk_images.append(ImageTk.PhotoImage( data_edge_map ))
        image_labels.append("Data Edge Pixels\nRetained for ICP")
        tk_images.append(ImageTk.PhotoImage( data_im ))
        tk_images.append(ImageTk.PhotoImage( textImage7 ))
        for i in range(3):
            Tkinter.Label(frame,image=tk_images[i], text=image_labels[i], font=fontsize, compound=Tkinter.BOTTOM, width=cellwidth).grid(row=0,column=i,padx=10,pady=30)
        for i in range(3,6):
            Tkinter.Label(frame,image=tk_images[i], text=image_labels[i], font=fontsize, compound=Tkinter.BOTTOM, width=cellwidth).grid(row=1,column=i-3,padx=10,pady=30)
        messageFrame = Tkinter.Frame(frame, width=displayWidth, height=displayHeight).grid(row=2,padx=10,pady=50)
        messageLabelText = Tkinter.StringVar()
        Tkinter.Label(messageFrame,
                      textvariable = messageLabelText,
                      font = 2 * fontsize,
                      anchor = 'c',
                      relief = 'groove',
                     ).pack(side='top', padx=10, pady=10)
        messageLabelText.set('''Close this window to see a movie of image registration''')
        Tkinter.mainloop()

    def display_images_used_for_corner_based_icp(self):
        tk_images = []
        image_labels = []
        rootWindow = Tkinter.Tk()
        screen_width,screen_height =rootWindow.winfo_screenwidth(),rootWindow.winfo_screenheight()
        rootWindow.geometry( str(int(0.8 * screen_width)) + "x" + str(int(0.9 * screen_height)) + "+50+50") 
        canvas = Tkinter.Canvas(rootWindow)
        canvas.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=Tkinter.TRUE)
        scrollbar = Tkinter.Scrollbar(rootWindow,orient=Tkinter.HORIZONTAL,command=canvas.xview)
        scrollbar.pack(side=Tkinter.BOTTOM, fill=Tkinter.X)
        canvas.configure(xscrollcommand=scrollbar.set)
        def set_scrollregion(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
        frame = Tkinter.Frame(canvas)
        canvas.create_window((0,0), window=frame, anchor=Tkinter.NW)
        frame.bind('<Configure>', set_scrollregion)
        cellwidth = 3 * self.data_im.size[0]
        padding = 10
        if cellwidth > 80:
            fontsize = 25
        else:
            fontsize = 15
        font = ImageFont.truetype(self.font_file, fontsize)
        data_image_width, data_image_height = self.data_im.size
        orig_image_width,orig_image_height = self.original_model_im.size
        original_model_im = self.original_model_im.copy()
        displayWidth,displayHeight = None,None
        if data_image_width > data_image_height:
            displayWidth = int(0.9 * cellwidth)
            displayHeight = int(displayWidth * data_image_height * 1.0 / data_image_width)
        else:
            displayHeight = int(0.9 * cellwidth)
            displayWidth = int(displayHeight * data_image_width * 1.0 / data_image_height )
        image_labels.append("Model")
        original_model_im.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)                       # 1
        image_labels.append("Model\nSegmentation")
        model_segmentation = self.model_segmentation.copy()                                              # 2
        model_segmentation = model_segmentation.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        model_segmentation.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        image_labels.append("Model\nCorner Pixels")
        model_all_corners = self.model_all_corners.copy()                                                # 3
        model_all_corners = model_all_corners.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        model_all_corners.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        image_labels.append("Model Corners\nRetained for ICP")
        model_corners_retained = self.model_im.copy()                                                    # 4
        model_corners_retained = model_corners_retained.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        model_corners_retained.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        image_labels.append("Data")
        original_data_im = self.original_data_im.copy()                                                  # 5
        original_data_im = original_data_im.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        original_data_im.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        image_labels.append("Data\nSegmentation")
        data_segmentation = self.data_segmentation.copy()                                                # 6
        data_segmentation = data_segmentation.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        data_segmentation.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)        
        image_labels.append("Data\nCorner Pixels")
        data_all_corners = self.data_all_corners.copy()                                                  # 7
        data_all_corners = data_all_corners.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        data_all_corners.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        image_labels.append("Data Corners\nRetained for ICP")
        data_corners_retained = self.data_im.copy()                                                      # 8
        data_corners_retained = data_corners_retained.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        data_corners_retained.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        tk_images.append(ImageTk.PhotoImage( original_model_im ))
        tk_images.append(ImageTk.PhotoImage( model_segmentation ))
        tk_images.append(ImageTk.PhotoImage( model_all_corners ))
        tk_images.append(ImageTk.PhotoImage( model_corners_retained ))
        tk_images.append(ImageTk.PhotoImage( original_data_im ))
        tk_images.append(ImageTk.PhotoImage( data_segmentation ))
        tk_images.append(ImageTk.PhotoImage( data_all_corners ))
        tk_images.append(ImageTk.PhotoImage( data_corners_retained ))
        for i in range(4):
            Tkinter.Label(frame,image=tk_images[i], text=image_labels[i], font=fontsize, compound=Tkinter.BOTTOM, width=cellwidth).grid(row=0,column=i,padx=10,pady=30)
        for i in range(4,8):
            Tkinter.Label(frame,image=tk_images[i], text=image_labels[i], font=fontsize, compound=Tkinter.BOTTOM, width=cellwidth).grid(row=1,column=i-4,padx=10,pady=30)
        messageFrame = Tkinter.Frame(frame, width=displayWidth, height=displayHeight).grid(row=2,padx=10,pady=50)
        messageLabelText = Tkinter.StringVar()
        Tkinter.Label(messageFrame,
                      textvariable = messageLabelText,
                      font = 2 * fontsize,
                      anchor = 'c',
                      relief = 'groove',
                     ).pack(side='top', padx=10, pady=10)
        messageLabelText.set('''Close this window to see a movie of image registration''')
        Tkinter.mainloop()

    def display_images_used_for_binary_image_icp(self):
        tk_images = []
        image_labels = []
        rootWindow = Tkinter.Tk()
        screen_width,screen_height =rootWindow.winfo_screenwidth(),rootWindow.winfo_screenheight()
        rootWindow.geometry( str(int(0.8 * screen_width)) + "x" + str(int(0.9 * screen_height)) + "+50+50") 
        canvas = Tkinter.Canvas(rootWindow)
        canvas.pack(side=Tkinter.TOP, fill=Tkinter.BOTH, expand=Tkinter.TRUE)
        scrollbar = Tkinter.Scrollbar(rootWindow,orient=Tkinter.HORIZONTAL,command=canvas.xview)
        scrollbar.pack(side=Tkinter.BOTTOM, fill=Tkinter.X)
        canvas.configure(xscrollcommand=scrollbar.set)
        def set_scrollregion(event):
            canvas.configure(scrollregion=canvas.bbox('all'))
        frame = Tkinter.Frame(canvas)
        canvas.create_window((0,0), window=frame, anchor=Tkinter.NW)
        frame.bind('<Configure>', set_scrollregion)
        cellwidth = 5 * self.data_im.size[0]
        padding = 10
        if cellwidth > 80:
            fontsize = 25
        else:
            fontsize = 15
        font = ImageFont.truetype(self.font_file, fontsize)
        data_image_width, data_image_height = self.data_im.size
        orig_image_width,orig_image_height = self.original_model_im.size
        orig_image_width,orig_image_height = 5 * orig_image_width, 5 * orig_image_height
        original_model_im = self.original_model_im.copy()
        displayWidth,displayHeight = None,None
        if data_image_width > data_image_height:
            displayWidth = int(0.9 * cellwidth)
            displayHeight = int(displayWidth * data_image_height * 1.0 / data_image_width)
        else:
            displayHeight = int(0.9 * cellwidth)
            displayWidth = int(displayHeight * data_image_width * 1.0 / data_image_height )
        image_labels.append("Model")
        original_model_im = original_model_im.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        original_model_im.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)                  
        image_labels.append("Data")
        original_data_im = self.original_data_im.copy()
        original_data_im = original_data_im.resize((orig_image_width, orig_image_height), Image.ANTIALIAS)
        original_data_im.thumbnail((displayWidth,displayHeight), Image.ANTIALIAS)
        tk_images.append(ImageTk.PhotoImage( original_model_im ))
        tk_images.append(ImageTk.PhotoImage( original_data_im ))
        for i in range(2):
            Tkinter.Label(frame,image=tk_images[i], text=image_labels[i], font=fontsize, compound=Tkinter.BOTTOM, width=cellwidth).grid(row=0,column=i,padx=10,pady=30)
        messageFrame = Tkinter.Frame(frame, width=displayWidth, height=displayHeight).grid(row=2,padx=10,pady=50)
        messageLabelText = Tkinter.StringVar()
        Tkinter.Label(messageFrame,
                      textvariable = messageLabelText,
                      font = 2 * fontsize,
                      anchor = 'c',
                      relief = 'groove',
                     ).pack(side='top', padx=10, pady=10)
        messageLabelText.set('''Close this window to see a movie of image registration''')
        Tkinter.mainloop()

    def cleanup_directory(self):
        for filename in glob.glob( self.dir_name_for_results + '/__result*.jpg' ): os.unlink(filename)

    @staticmethod
    def gendata( feature, imagesize, position, orientation, output_image_name ):
        width,height = imagesize
        x,y = position
        theta = orientation
        tan_theta = scipy.tan( theta * scipy.pi / 180 )
        cos_theta =  scipy.cos( theta * scipy.pi / 180 )
        sin_theta =  scipy.sin( theta * scipy.pi / 180 )

        im = Image.new( "L", imagesize, 0 )
        draw = ImageDraw.Draw(im)

        if feature == 'line':
            delta =  y / tan_theta
            if delta <= x:
                x1 = x - y / tan_theta
                y1 = 0
            else:
                x1 = 0
                y1 = y - x * tan_theta
            x2 = x1 + height / tan_theta
            y2 = height
            if x2 > width:
                excess = x2 - width
                x2 = width
                y2 = height - excess * tan_theta
            draw.line( (x1,y1,x2,y2), fill=255)
            del draw
            im.save( output_image_name )

        elif feature == "triangle":
            x1 = int(width/2.0)
            y1 = int(0.7*height)
            x2 = x1 -  int(width/4.0)
            y2 = int(height/4.0)
            x3 = x1 +  int(width/4.0)
            y3 = y2
            draw.line( (x1,y1,x2,y2), fill=255 )
            draw.line( (x1,y1,x3,y3), fill=255 )
            draw.line( (x2,y2,x3,y3), fill=255 )
            del draw
            h2 = int(height/2)
            w2 = int(width/2)
            im = im.transform(imagesize, Image.AFFINE, \
              (cos_theta,sin_theta,-x,-sin_theta,cos_theta,-y), Image.BICUBIC )
            im.save( output_image_name )
        else:
            print("unknown feature requested")
            sys.exit(0)