import modules.scripts as scripts
import gradio as gr
import os
import cv2
from PIL import Image
import numpy as np
from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "Transparency & Outline"


    def show(self, is_img2img):
        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        transparency = gr.Checkbox(True, label="transparency")
        outline_size = gr.Slider(minimum=0, maximum=8, step=1, Outline="Pixel size", value=1, elem_id="outline_size")
        return [transparency, outline_size]

  

    def run(self, p, transparency, outline_size):
        # function which takes an image from the Processed object, 
        # and the angle and two booleans indicating horizontal and
        # vertical flips from the UI, then returns the 
        # image rotated and flipped accordingly
        def outline_run(img, transparency, outline_size):
            raf=[]

            raf.append(img)

            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
            #image_morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            # Create a kernel. The size of the kernel affects the operation; you may need to adjust this.
            kernel = np.ones((5,5), np.uint8)

            # Perform morphological closing
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find contours and remove small noise
            cnts ,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #cnts = cnts[0] if len(cnts) == 2 else cnts[1]

            # Find contours, obtain bounding box, extract and save ROI
            ROI_number = 0

            #mask = np.full_like(image, (0,0,0))
            #cv2.drawContours(mask,cnts, -1, (255,255,255), cv2.FILLED)

            sprites=[]
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x-4, y-4), (x + w+4, y + h+4), (255, 255, 255), 2)
                # Extract the ROI from the original image
                ROI = image[y-4:y+h+4, x-4:x+w+4]
                if ROI.shape[0] == 0 or ROI.shape[1] == 0:
                    continue
                hh, ww = ROI.shape[:2]

                
                
                roi_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
                roi_thresh = cv2.threshold(roi_gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                # Morph open to remove noise
                #roi_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                #roi_morph = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, roi_kernel, iterations=1)
                kernel = np.ones((3,3), np.uint8)
                # Perform morphological closing
                thresh = cv2.morphologyEx(roi_thresh, cv2.MORPH_CLOSE, kernel)

              
                # Find contours and remove small noise
                roi_cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                big_contour = max(roi_cnts, key=cv2.contourArea)


                roi_mask = np.zeros_like(thresh)
                #roi_mask = np.zeros((hh,ww), dtype=np.uint8)   
                cv2.drawContours(roi_mask, [big_contour], -1, (255, 255, 255), thickness=cv2.FILLED) 
                #cv2.drawContours(roi_mask, roi_cnts, -1, (255,255,255), cv2.FILLED)
                
              
                result1 = cv2.bitwise_and(ROI, ROI, mask=roi_mask)


                roi_mask2 = np.zeros_like(thresh)
                #roi_mask = np.zeros((hh,ww), dtype=np.uint8)   
                cv2.drawContours(roi_mask2, [big_contour], -1, (255, 255, 255), thickness=cv2.FILLED) 
                #cv2.drawContours(roi_mask, roi_cnts, -1, (255,255,255), cv2.FILLED)
                # Creating kernel
                kernel2 = np.ones((4, 4), np.uint8)
                roi_mask2 = cv2.erode(roi_mask2, kernel2)
                roi_mask2 = cv2.GaussianBlur(roi_mask2, (5,5), 0)

                result2 = cv2.bitwise_and(ROI, ROI, mask=roi_mask2)

              
                background = np.full_like(ROI, (0,0,0))
                masked_background = cv2.bitwise_and(background, background, mask=roi_mask)

                  
                #cv2.drawContours(result1, [big_contour], -1, (0,0,0), 3)
                result1 = cv2.add(masked_background,ROI )
                
                outline =result1.copy()
                cv2.drawContours(outline, [big_contour], -1, (0, 0, 0, 5), thickness=outline_size)
                # Merge the canvas with the original image

                result1 = cv2.addWeighted(result2,0.5,outline,0.5,0)

                result1 = cv2.bitwise_and(result1, result1, mask=roi_mask2)
                # cv2.drawContours(result1, roi_cnts, -1, (0,0,0,50), 2)

                # Create a 4-channel image (3 for RGB and 1 for alpha)
                result_with_alpha = cv2.cvtColor(result1, cv2.COLOR_BGR2BGRA)

                if transparency :
                    result_with_alpha[..., 3] = roi_mask

                sprites.append(result_with_alpha)
                #imshow('',result_with_alpha )
                

                #sprite_path = f'{output_folder}/{ROI_number}.png'
                #cv2.imwrite(sprite_path, result_with_alpha)

                ROI_number += 1

                b, g, r, a = cv2.split(result_with_alpha)

                
                #imshow("",result_with_alpha)
                #images.save_image(Image.fromarray(cv2.merge((r, g, b, a))), p.outpath_samples, basename + "_" + str(ROI_number), proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)

                #images.save_image(Image.fromarray(cv2.merge((r, g, b, a))), p.outpath_samples, basename + "_" + str(ROI_number), proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p) 
                raf.append(Image.fromarray(cv2.merge((r, g, b, a))))
            #raf = img

            hh, ww = image.shape[:2]

            # Define the tile size
            tile_width = tile_height = hh

            # Create an output image with a transparent background, of the size of the atlas
            output = np.zeros((tile_height, tile_width * len(sprites), 4), dtype=np.uint8)

            # Position of the image in the output image
            x_offset = 0

            # Iterate over the images and add them to the output image
            for image in sprites:
                # The size of this image
                height, width = image.shape[:2]

                # Calculate the y-coordinate to place the image at the bottom of the tile
                y_offset = tile_height - height

                # Calculate the x-coordinate to place the image at the center of the tile
                x_center_offset = x_offset + (tile_width - width) // 2

                # Put the image on the output
                output[y_offset:y_offset+height, x_center_offset:x_center_offset+width] = image

                # Shift the x offset
                x_offset += tile_width

            output = cv2.cvtColor(output.astype('uint8'), cv2.COLOR_RGBA2BGRA)
            raf.append(Image.fromarray(output.astype('uint8'), 'RGBA'))
            return raf

  

        # If overwrite is false, append the rotation information to the filename
        # using the "basename" parameter and save it in the same directory.
        # If overwrite is true, stop the model from saving its outputs and
        # save the rotated and flipped images instead.
        basename = ""
        #p.do_not_save_samples = True

        
        proc = process_images(p)

        # rotate and flip each image in the processed images
        # use the save_images method from images.py to save
        # them.

        outlined_images=[]
        for i in range(len(proc.images)):
            outlined_images.extend(outline_run(proc.images[i], transparency, outline_size))

        for i, ex_image in enumerate(outlined_images):
            images.save_image(ex_image, p.outpath_samples, str(proc.seed) + "_" + str(i),
            proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)

        proc.images=outlined_images
        return proc
