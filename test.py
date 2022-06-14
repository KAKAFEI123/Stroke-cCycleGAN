""" 
Inference code for Stroke-cCycleGAN
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im
import cv2 as cv
import numpy

def getStrokeWidth(img0):
    ret, img = cv.threshold(img0, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE) #ignore the borders
    contourLen = 0
    for idx, c in enumerate(contours):
        if c.shape[0] <= 10:
            continue
        contourLen += c.shape[0]
    strokeWidth = numpy.sum(img!=0) * 1.0 / contourLen * 2
    strokeWidth = int(100*strokeWidth)/100.0
    return strokeWidth


if __name__ == '__main__':
    opt = TestOptions().parse() 
    opt.num_threads = 0   
    opt.batch_size = 1    
    opt.serial_batches = True  
    opt.no_flip = True    
    opt.display_id = -1  
    dataset = create_dataset(opt)  
    model = create_model(opt)     
    model.setup(opt)    

    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()      
        visuals = model.get_current_visuals() 
        img_path = model.get_image_paths()  
        
        path_suffix = img_path[0]
        save_path = os.path.join("results", opt.name, path_suffix)
        img_name = save_path.split("/")[-1]
        pth = save_path.split("/"+img_name)[0]
        if not os.path.exists(pth):
            os.makedirs(pth)

        fake_B = tensor2im(visuals["fake_B"])

        cv.imwrite(save_path,fake_B)

        image = cv.cvtColor(fake_B,cv.COLOR_RGB2GRAY)
        stroke_width = getStrokeWidth(image)
        print('processing (%04d)-th image... %s' % (i, img_path))
        print("generated_stroke_width:", stroke_width)
