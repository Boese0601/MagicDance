from PIL import Image, ImageChops
import numpy as np


class RemoveWhite(object):
    def trim(self, im):
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)
        return im

    def __call__(self, pil_image):
        out_image = self.trim(pil_image)        
        if out_image.size[0] < 0.5*out_image.size[0] or out_image.size[1] < 0.5*out_image.size[1]:
            return pil_image
        return out_image



class CenterCrop(object):
    def __call__(self, pil_image):     
        # input is a pil image 

        width, height = pil_image.size
        
        if width>height and width/height < 1.3:
            return pil_image
        elif height>=width and height/width < 1.3:
            return pil_image

        new_width = new_height = min(width, height)

        left = int(np.ceil((width - new_width) / 2))
        right = width - int(np.floor((width - new_width) / 2))

        top = int(np.ceil((height - new_height) / 2))
        bottom = height - int(np.floor((height - new_height) / 2))

        center_cropped_img = pil_image.crop((left, top, right, bottom))

        return center_cropped_img