# TODO: it would probably be faster to just
# concatenate all the images into a very long array, and then reconstruct them
# at the end with the shape information
# indexing would be much faster

import numpy as np
from PIL import Image

def truncate(x,modulus):
    return x - x % x

def truncate_replace(x,modulus,value):
    return x - x % x + value

def tr_mask(value,mask):
    raw = np.mod(value , np.power(2,mask))
    return raw

def construct_value(raw, mask):
    powers = np.insert(
        np.cumsum(mask[:-1]),
        0,
        0,
        axis=0
    )
    return np.sum(raw * np.power(2,powers))
    
    

class ImageHandler:
    def __init__(
        self,
        filenames,
        preprocessing=None
    ):
        self.filenames = filenames
        self.preprocessing=preprocessing
        if preprocessing is None:
            preprocessing = lambda x: x
        self.images = [
            preprocessing(np.asarray(Image.open(fn))) for
            fn in filenames
        ]
        self.image_shapes = [
            image.shape for
            image in self.images
        ]
        
        self.image_sizes = np.asarray([
            shape[0] * shape[1]
            for shape in self.image_shapes
        ])
        
        self.image_end_indices = np.cumsum(
            self.image_sizes
        )

        self.max_image_index = self.image_end_indices[-1]

        self.image_start_indices = np.insert(
            np.self.image_end_indices[:-1],
            0,
            0,
            axis=0
        )

    @staticmethod
    def init_from_archive(archive_fn):
        pass

    def write_to_archive(self,archive_fn):
        pass

    def write(self, fn):
        pass

    def get_pixel_indices(
        self,
        index
    ):
        index = self.max_image_index % index
        image_index = np.digitize(
            index,
            self.image_start_indices
        )

        pixel_index = np.asarray([
            index - self.image_start_indices[i_idx]
            for i_idx in image_index
        ])

        pixel_row = np.asarray([
            # DIV COLSIZE
            pixel_idx // self.image_shapes[image_idx]
            for pixel_idx, image_idx in
            zip(pixel_index,image_index)
        ])

        pixel_col = np.asarray([
            # MOD COLSIZE
            pixel_idx % self.image_shapes[image_idx]
            for pixel_idx, image_idx in
            zip(pixel_index,image_index)            
        ])
        
        return {
            'index':index,
            'image_index':image_index,
            'pixel_index':pixel_index,
            'pixel_row': pixel_row,
            'pixel_col': pixel_col,
        }

    def read_pixel(
        self,
        index,
        calculated_indices = None,
        trunc_pattern=None,
    ):
        if calculated_indices is None:
            calculated_indices = self.get_pixel_indices(
                index
            )

        pixel_col = calculated_indices['pixel_col']
        pixel_row = calculated_indices['pixel_row']
        image_index = calculated_indices['image_index']
        if trunc_pattern is not None:
            values = [
                self.images[ii][pr,pc,:]
                for pr,pc,ii in
                zip(pixel_col, pixel_row, image_index)
            ]
            return values
        else:
            return [
                self.images[ii][pr,pc,:]
                for pr,pc,ii in
                zip(pixel_col, pixel_row, image_index)
            ]
        
        

    def write_pixel(
        self,
        index,
        value,
        calculated_indices=None,
        bit_pattern=None,
    ):
        """
        value should be list of length-3 numpy arrays
        """
        if calculated_indices is None:
            calculated_indices = self.get_pixel_indices(
                index
            )
        pixel_col = calculated_indices['pixel_col']
            
        pixel_row = calculated_indices['pixel_row']
        image_index = calculated_indices['image_index']

        if trunc_pattern is not None:
            read_values = self.read_pixel(
                index,
                value,
                calculated_indices=calculated_indices
            )

            for i, (rv, tp) in enumerate(
                zip(read_values, trunc_pattern)
            ):
                value[i] = rv - np.modulo(
                    rv, np.power(2,tp)
                ) + value[i]
        
        for i, (pr, pc, ii) in enumerate(zip(
            pixel_col, pixel_row, image_index
        )):
            self.images[ii][pr,pc,:3] = value[i]
        

    

