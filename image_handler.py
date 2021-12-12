"""
Contains methods to open images and write data to 
and read data from specific bits of the color 
channels.

Can read and write different archive formats for efficiency.

Designed to be used for cryptographic chaffing

"""
from collections import Counter
from functools import partial
import gzip
import io
import index_generators
import lzma
import numpy as np
import os
from PIL import Image
import re
import types
import tarfile
import zipfile

def png_fn(x, tracker=None):
    x = re.sub(r'\..*','.png',x)
    if tracker:
        x_copy = x
        i = 0
        while x_copy in tracker:
            x_copy = re.sub(r'\.png',f'_{i:04d}.png', x)
            i+=1
        x = x_copy
    return x

def to_image(x, channels='RGB'):
    return Image.fromarray(np.uint8(x))

def elemwise_equals(x,y):
    try:
        if isinstance(x, list):
            return all([np.all(ex == ey) for ex, ey in zip(x,y)])
        else:
            return np.all(x==y)
    except:
        return False

class ImageHandler:
    """
    Loads and saves images while allowing generalized reading and writing of pixels with specific 
    bits that need to be overwritten or read.
    """    
    def __init__(
        self,
        filenames,
        preprocessing=None,
        archive_type=None,
        n_channels=3,
    ):
        """
        @param filenames - Filenames of images to load 
        @param preprocessing - Preprocessing function to apply to each loaded image
        @param archive_type - Directs program to load (and, by default, write) data
                              in an archive format (.tar.gz,.tar.xz,.zip,.7z)
        @param n_channels - How many channels to use
        """
        if preprocessing is None:
            preprocessing = lambda x: x

        self.n_channels = n_channels
            
        self.preprocessing=preprocessing
        self.archive_type = archive_type
            
        if archive_type is not None:
            self.load_archive(filenames)
        else:
            self.filenames = filenames
            self.images = [
                preprocessing(np.asarray(Image.open(fn)))[:,:,:self.n_channels] for
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
            self.image_end_indices[:-1],
            0,
            0,
            axis=0
        )

        self.n_images = len(self.image_shapes)
        self.flattened = None
        
        # flatten
        self.flatten_images()

    def load_archive(self, filenames):
        if isinstance(filenames, list):
            fn = filenames[0]
        else:
            fn = filenames
        if self.archive_type == 'gz':
            with tarfile.open(fn, 'r:gz') as f:
                self.images = [
                    self.preprocessing(np.asarray(Image.open(f.extractfile(member))))
                    for member in f
                ]
        elif self.archive_type == 'tar':
            with tarfile.open(fn, 'r') as f:
                self.images = [
                    self.preprocessing(np.asarray(Image.open(f.extractfile(member))))
                    for member in f
                ]                
        elif self.archive_type == 'xz':
            with tarfile.open(fn, 'r:xz') as f:
                self.images = [
                    self.preprocessing(np.asarray(Image.open(f.extractfile(member))))
                    for member in f
                ]
        elif self.archive_type == 'bz2':
            with tarfile.open(fn, 'r:bz2') as f:
                self.images = [
                    self.preprocessing(np.asarray(Image.open(f.extractfile(member))))
                    for member in f
                ]                
        elif self.archive_type == 'zip':
            with zipfile.ZipFile(fn, 'r') as zf:
                self.images = [
                    self.preprocessing(np.asarray(np.asarray(Image.open(zf.open(member)))))
                    for member in zf.filelist
                ]
        elif self.archive_type == '7z':
            raise NotImplementedError('Archive type 7z not implemented yet')
        else:
            raise NotImplementedError(f'Archive type {self.archive_type} not implemented!')
        

    def write_archive(
        self,
        filename,
        archive_type,
        output_path='',
        overwrite=True,
        archive_filenames = None,
    ):
        output_fn = os.path.join(output_path,filename)
        if archive_filenames is None:
            archive_filenames = self.filenames
        elif isinstance(archive_filenames, (types.FunctionType,partial)):
            archive_filenames = [archive_filenames(fn) for fn in self.filenames]
        
        if os.path.isfile(output_fn):
            if overwrite:
                os.remove(output_fn)
            else:
                raise FileExistsError(f'File {output_fn} already exists!')
            
        if self.flattened:
            self.reconstitute_images()

        if archive_type == 'zip':
            with zipfile.ZipFile(output_fn, 'w') as zf:
                for img, fn in zip(self.images, archive_filenames):
                    img_io = io.BytesIO()
                    to_image(img).save(img_io,format='PNG')
                    img_io.seek(0)
                    img_bytes = img_io.read()
                    zf.writestr(fn, img_bytes)
        elif archive_type == 'tar':
            with tarfile.open(output_fn, 'w') as f:
                for img, fn in zip(self.images, archive_filenames):
                    tinfo = tarfile.TarInfo(fn)
                    img_io = io.BytesIO()
                    to_image(img).save(img_io, format='PNG')
                    img_io.seek(0)
                    n_bytes = len(img_io.read())
                    img_io.seek(0)
                    tinfo.size=n_bytes
                    f.addfile(tinfo, img_io)
        elif archive_type == 'xz':
            with tarfile.open(output_fn, 'w:xz') as f:
                for img, fn in zip(self.images, archive_filenames):
                    tinfo = tarfile.TarInfo(fn)
                    img_io = io.BytesIO()
                    to_image(img).save(img_io, format='PNG')
                    img_io.seek(0)
                    n_bytes = len(img_io.read())
                    img_io.seek(0)
                    tinfo.size=n_bytes
                    f.addfile(tinfo, img_io)
        elif archive_type == 'gz':
            with tarfile.open(output_fn, 'w:gz') as f:
                for img, fn in zip(self.images, archive_filenames):
                    tinfo = tarfile.TarInfo(fn)
                    img_io = io.BytesIO()
                    to_image(img).save(img_io, format='PNG')
                    img_io.seek(0)
                    n_bytes = len(img_io.read())
                    img_io.seek(0)
                    tinfo.size=n_bytes
                    f.addfile(tinfo, img_io)
        elif archive_type == 'bz2':
            with tarfile.open(output_fn, 'w:bz2') as f:
                for img, fn in zip(self.images, archive_filenames):
                    tinfo = tarfile.TarInfo(fn)
                    img_io = io.BytesIO()
                    to_image(img).save(img_io, format='PNG')
                    img_io.seek(0)
                    n_bytes = len(img_io.read())
                    img_io.seek(0)
                    tinfo.size=n_bytes
                    f.addfile(tinfo, img_io)
        elif archive_type == '7z':
            raise NotImplementedError('Archive type 7z not implemented yet')
        else:
            raise NotImplementedError(f'Archive type {self.archive_type} not implemented')

    def reconstitute_images(self):
        """
        converts images back to original dimensions
        """
        self.flattened = False
        self.images = [
            self.images[
                self.image_start_indices[i]:self.image_end_indices[i]
            ].reshape(self.image_shapes[i])
            for i in range(self.n_images)
        ]

    def flatten_images(self):
        """
        converts images to long, 2-dimensional array
        """
        self.flattened = True
        self.images = np.concatenate([
            img.reshape((np.prod(img.shape[:2]), img.shape[2]))
            for img in self.images
        ])
    
    def write(self, output_filenames,output_path=''):
        if self.flattened:
            print('Reconstituting images...')
            self.reconstitute_images()
        if isinstance(output_filenames, types.FunctionType):
            output_filenames = [output_filenames(fn) for fn in self.filenames]
        for i, fn in enumerate(output_filenames):
            img = Image.fromarray(np.uint8(self.images[i]))
            img.save(os.path.join(output_path, fn))

    def read_pixels(
        self,
        indices,
        trunc_pattern=None,
    ):

        if not trunc_pattern:
            # literally read pixels
            return self.images[indices]
        else:
            trunc_pattern = np.asarray(trunc_pattern)
            # read bits of pixels, convert to bytes
            # this is the multiplier for each index as the exponent of 2
            truncsum = np.insert(
                np.cumsum(trunc_pattern,axis=1),0,0,axis=1
            )[:,:trunc_pattern.shape[1]]
            #print(truncsum)
            return np.sum(
                np.power(2, truncsum) * 
                np.mod(
                    self.images[indices],
                    np.power(2,trunc_pattern)
                ),
                axis=1
            )

    def write_pixels(
        self,
        indices,
        value,
        trunc_pattern=None,
    ):

        if not trunc_pattern:
            self.images[indices] = value
        else:
            if isinstance(value, list):
                value = np.asarray(value)
            trunc_pattern = np.asarray(trunc_pattern)            
            # this is the multiplier for each index as the exponent of 2
            initial_truncsum = np.cumsum(trunc_pattern,axis=1)
            truncsum = np.insert(
                initial_truncsum,0,0,axis=1
            )[:,:trunc_pattern.shape[1]]            
            # raw values
            raw_pixels = self.read_pixels(indices)
            trunc_powers = np.power(2,trunc_pattern)
            value = np.repeat(
                value[:,np.newaxis],
                trunc_pattern.shape[1],
                axis=1
            )
            print('V')
            print(value)
            print('I')
            print(initial_truncsum)
            print('T')
            print(truncsum)
            # convert value to array
            pixel_value = (
                # for each value,
                # MOD across the exponentiated pattern (go with highest bit for each)
                # THEN BIT SHIFT across cumulative bit count (start w/0)
                np.right_shift(
                    np.mod(value, np.power(2,initial_truncsum)),
                    truncsum
                )
            )
            
            # overwrite bytes
            self.images[indices] = (
                # first truncate
                self.images[indices] -
                np.mod(self.images[indices], trunc_powers) +
                pixel_value
            )

    def has_same_images(self, x, can_change_self=True):
        if self.flattened and not x.flattened:
            if can_change_self:
                self.reconstitute_images()
            else:
                raise ValueError('Flattened array cannot be compared to image array')
        elif not self.flattened and x.flattened:
            if can_change_self:
                self.flatten_images()
            else:
                raise ValueError('Cannot compare image array to flattened array')
        return elemwise_equals(self.images,x.images)
    

def test_image_handler():
    TEST_PATH='test_images'
    filenames = [
        os.path.join(
            TEST_PATH,
            fn
        ) for fn in [
            'br_stripes_uneven.png',
            'kittywhisper.jpg',
            'sigma_balls_crop.png'
        ]
    ]

        
    print('Initializing')
    ih = ImageHandler(
        filenames
    )
    N_PIXELS = 16000
    indices = []
    gp = index_generators.crypto_generator(
        b'A' * 32,
        mod=ih.max_image_index,
        unique=True
    )
    pixels = [next(gp) for _ in range(N_PIXELS)]

    # get channel patterns

    if False:
        # 1-byte values
        gc = index_generators.crypto_generator(
            b'Ah' * 16,
            mod=3,
        )
        patterns = [
            [
                [3,3,2],
                [3,2,3],
                [2,3,3],
            ][next(gc)]
            for _ in range(N_PIXELS)
        ]
        # generate values
        gv = index_generators.crypto_generator(
            b'\x00' * 32,
            mod=256,
        )
        values = [next(gv) for _ in range(N_PIXELS)]
    else:
        # 2-byte values
        gc = index_generators.crypto_generator(
            b'Ah' * 16,
            mod=18,
        )        
        patterns = [
            [
                [6,6,4],
                [6,4,6],
                [4,6,6],
                [7,6,5],
                [7,5,6],
                [6,5,7],
                [6,7,5],
                [5,7,6],
                [5,6,7],
                [7,7,2],
                [7,2,7],
                [2,7,7],
                [3,7,5],
                [3,5,7],
                [5,3,7],
                [5,7,3],
                [7,3,5],
                [7,5,3],
            ][next(gc)]
            for _ in range(N_PIXELS)
        ]
        # generate values
        gv = index_generators.crypto_generator(
            b'\x00' * 32,
            mod=256 ** 2,
        )
        values = [next(gv) for _ in range(N_PIXELS)]        



    # let's read the pixels
    print('READ')
    print(ih.read_pixels(pixels))
    print(ih.read_pixels(pixels, trunc_pattern=patterns))

    # let's do some writing...
    ih.write_pixels(
        indices = pixels,
        value=values,
        trunc_pattern=patterns
    )

    ih.write(output_filenames = lambda x: x.replace('.','_modified.'))

    # archive write test
    ARCHIVE_DIRECTORY='test_archives'
    os.makedirs(ARCHIVE_DIRECTORY,exist_ok=True)

    # zip
    #write
    ih.write_archive(
        'test_archive.zip',
        archive_type='zip',
        output_path=ARCHIVE_DIRECTORY,
        overwrite=True,
        archive_filenames=partial(png_fn,tracker=set())
    )

    # read & check
    zip_ih = ImageHandler(
        filenames=os.path.join(
            ARCHIVE_DIRECTORY,
            'test_archive.zip'
        ),
        archive_type='zip'
    )
    print('ZIP CHECK: %s' % zip_ih.has_same_images(ih))

    # tar
    for fmt in ['tar','gz','xz','bz2']:
        print(fmt)
        if fmt == 'tar':
            fmt2 = fmt
        else:
            fmt2 = 'tar.' + fmt
            
        ih.write_archive(
            f'test_archive.{fmt2}',
            archive_type=fmt,
            output_path=ARCHIVE_DIRECTORY,
            overwrite=True,
            archive_filenames=partial(png_fn,tracker=set())
        )

        # read & check
        new_ih = ImageHandler(
            filenames=os.path.join(
                ARCHIVE_DIRECTORY,
                f'test_archive.{fmt2}'
            ),
            archive_type=fmt
        )
        print(f'{fmt.upper()} CHECK: %s' % new_ih.has_same_images(ih))    

    

if __name__=='__main__':
    test_image_handler()
    
