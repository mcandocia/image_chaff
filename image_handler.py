"""
Contains methods to open images and write data to 
and read data from specific bits of the color 
channels.

Can read and write different archive formats for efficiency.

Designed to be used for cryptographic chaffing

"""
import base64
from collections import Counter
from copy import deepcopy
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
from utility import chunk_seq
from utility import grab
import zipfile

ARCHIVE_SUFFIXES = ['.tar','.zip','.tar.gz','.tar.xz','.tar.bz2',]

ARCHIVE_SUFFIX_REGEX = re.compile('(%s)' % '|'.join(
    [x.replace('.',r'\.') + '$' for x in ARCHIVE_SUFFIXES]
), flags=re.IGNORECASE)

def pngify(x):
    return re.sub(r'\..*','.png',x)

def is_archive(x):
    return ARCHIVE_SUFFIX_REGEX.search(x)

def get_archive_type(x):
    if is_archive(x):
        return re.sub(r'.*\.','',x).lower()
    else:
        return None

def trunc_filename(x):
    return os.path.split(x)[1]


def png_fn(x, tracker=None):
    x = re.sub(r'\..*?$','.png',x)
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
        pre_flatten=True,
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
        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]        
            
        if archive_type is not None:
            self.load_archive(filenames)
        else:
            self.filenames = filenames
            self.images = [
                preprocessing(np.asarray(Image.open(fn)))[:,:,:self.n_channels] for
                fn in filenames
            ]
            
        self.calculate_image_stats()
        self.flattened = None
        
        # flatten
        if pre_flatten:
            self.flatten_images()
        else:
            self.flattened = False

    def calculate_image_stats(self):
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

    def __add__(self, x):
        if x is None:
            new_ih = deepcopy(self)
            new_ih.flatten_images()
        if not isinstance(x, self.__class__):
            raise TypeError(f'x must be of type {self.__class__}, not {x.__class__}')
        if self.flattened:
            self.reconstitute_images()
        if x.flattened:
            x.reconstitute_images()

        # create copy so that references aren't left dangling if
        # self is later partially changed
        new_self = deepcopy(self)
        new_self.images += x.images
        new_self.filenames += x.filenames
        new_self.calculate_image_stats()

        new_self.flatten_images()
        
        return new_self

    def __radd__(self,x):
        if x is None:
            new_ih = deepcopy(self)
            new_ih.flatten_images()
            return new_ih
        return self + x
            

    def load_archive(self, filenames):
        if isinstance(filenames, list):
            fn = filenames[0]
        else:
            fn = filenames
        if self.archive_type == 'gz':
            with tarfile.open(fn, 'r:gz') as f:
                self.filenames, self.images = zip(*[
                    (trunc_filename(member.name),
                    self.preprocessing(np.asarray(Image.open(f.extractfile(member))))
                    )
                    for member in f
                ])
        elif self.archive_type == 'tar':
            with tarfile.open(fn, 'r') as f:
                self.filenames, self.images = zip(*[
                    (trunc_filename(member.name),
                    self.preprocessing(np.asarray(Image.open(f.extractfile(member))))
                    )
                    for member in f
                ])
        elif self.archive_type == 'xz':
            with tarfile.open(fn, 'r:xz') as f:
                self.filenames, self.images = zip(*[
                    (trunc_filename(member.name),
                    self.preprocessing(np.asarray(Image.open(f.extractfile(member))))
                    )
                    for member in f
                ])
        elif self.archive_type == 'bz2':
            with tarfile.open(fn, 'r:bz2') as f:
                self.filenames, self.images = zip(*[
                    (trunc_filename(member.name),
                    self.preprocessing(np.asarray(Image.open(f.extractfile(member))))
                    )
                    for member in f
                ])
        elif self.archive_type == 'zip':
            with zipfile.ZipFile(fn, 'r') as zf:
                self.filenames, self.images = zip(*[
                    (trunc_filename(member.filename),
                    self.preprocessing(np.asarray(Image.open(zf.open(member))))
                    )
                    for member in zf.filelist
                ])
        elif self.archive_type == '7z':
            raise NotImplementedError('Archive type 7z not implemented yet')
        else:
            raise NotImplementedError(f'Archive type {self.archive_type} not implemented!')


    def write_archive(
        self,
        filename,
        archive_type=None,
        output_path='',
        overwrite=True,
        archive_filenames = None,
    ):
        """
        writes contents of files to an archive in tarball or zip format

        @param filename - filename of output archive
        @param archive_type - type of archive to create; will guess 
         from filename if not specified
        @param output_path - output directory to write to
        @param overwrite - Will overwrite output if it already exists
        @archive_filenames - filenames of individual images to name within archive;
         can be a function that processes input filenames; defaults to input filenames
        """
        output_fn = os.path.join(output_path,filename)
        if archive_type is None:
            if not is_archive(filename):
                raise ValueError(f'archive type of {filename} cannot be determined!')
            archive_type = get_archive_type(filename)
        if archive_filenames is None:
            archive_filenames = [
                fn if isinstance(fn, str)
                else (
                    index_generators.md5_hex(bytes(str(fn), encoding='ascii'))
                )[:-2] + '.png'
                for fn in self.filenames
            ]
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
                #print([img.shape for img in self.images])
                #print(archive_filenames)
                for img, fn in zip(self.images, archive_filenames):
                    #print('ZIP ITER')
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
        if not self.flattened:
            return None
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
        if self.flattened:
            return None
        self.flattened = True
        self.images = np.concatenate([
            img.reshape((np.prod(img.shape[:2]), img.shape[2]))
            for img in self.images
        ])
    
    def write(
        self,
        output_filenames=None,
        output_path='',
        pngify_filenames=False
    ):
        """
        write images to specified filenames and path

        @param output_filenames - list of filenames to write to, or
         function to modify input filenames; will default to input filenames if not specified

        @ param output_path - directory to save to
        """
        if pngify_filenames:
            if isinstance(output_filenames, str):
                output_filenames = pngify(output_filenames)
            else:
                output_filenames = [pngify(output_filenames) for fn in output_filenames]
                
        if not output_filenames:
            archive_filenames = [
                fn if isinstance(fn, str)
                else (
                    index_generators.md5_hex(bytes(str(fn), encoding='ascii'))
                )[:-2] + '.png'
                for fn in self.filenames
            ]            
        if self.flattened:
            #print('Reconstituting images...')
            self.reconstitute_images()
        if isinstance(output_filenames, types.FunctionType):
            output_filenames = [output_filenames(fn) for fn in self.filenames]
        elif isinstance(output_filenames, str):
            output_filenames = [output_filenames]
        for i, fn in enumerate(output_filenames):
            img = Image.fromarray(np.uint8(self.images[i]))
            img.save(os.path.join(output_path, fn))

    def read_pixels(
        self,
        indices,
        trunc_pattern=None,
    ):

        if trunc_pattern is None:
            # literally read pixels
            return self.images[indices]
        else:
            if not isinstance(trunc_pattern, np.ndarray):
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

        if trunc_pattern is None:
            self.images[indices] = value
        else:
            if isinstance(value, list):
                value = np.asarray(value)
            if not isinstance(trunc_pattern, np.ndarray):                
                trunc_pattern = np.asarray(trunc_pattern)
            # this is the multiplier for each index as the exponent of 2
            #print(trunc_pattern)
            initial_truncsum = np.cumsum(trunc_pattern,axis=1)
            truncsum = np.insert(
                initial_truncsum,0,0,axis=1
            )[:,:trunc_pattern.shape[1]]
            #print(truncsum)
            # raw values
            raw_pixels = self.read_pixels(indices)
            trunc_powers = np.power(2,trunc_pattern)
            value = np.repeat(
                value[:,np.newaxis],
                trunc_pattern.shape[1],
                axis=1
            )
            # convert value to array
            #print(value.shape)
            #print(initial_truncsum.shape)
            #print(truncsum.shape)
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

    def __str__(self):
        return (
            f'ImageHandler object with {self.n_images} images, archive type {self.archive_type}, '
            f'and max image index of {self.max_image_index}. Filenames length: {len(self.filenames)}'
        )

    def __repr__(self):
        return self.__str__()

    def generate_noise(
        self,
        noise_ratio,
        patterns = [(3,3,2),(3,2,3),(2,3,3)],
        seed = b'\x00' * 16,
    ):
        originally_flat = self.flattened
        if not originally_flat:
            self.flatten_images()

        n_pixels = int(noise_ratio * self.max_image_index)
        gp = index_generators.crypto_generator(
            seed,
            mod=self.max_image_index,
            unique=True
        )
        gc = index_generators.crypto_generator(
            seed + b'\x00',
            mod = len(patterns)
        )
        gv = index_generators.crypto_generator(
            seed + b'\x00' * 2,
            mod=256
        )
        for i, n_subpixels in enumerate(chunk_seq(n_pixels)):
            pixels = [next(gp) for _ in range(n_subpixels)]
            channels = [patterns[next(gc)] for _ in range(n_subpixels)]
            values = [next(gv) for _ in range(n_subpixels)]
            self.write_pixels(
                indices=pixels,
                value=values,
                trunc_pattern=channels
            )
            
        # unflatten
        if not originally_flat:
            self.reconstitute_images()

    def create_15bit_channel_hash(self, *, bit_shift_override=3):
        """
        create a hash based on the most significant 5 bits of each channel
        this can be used to create a type of unique but extractable salt
        to avoid the same password generating the same pattern

        @param bit_shift_override - If you want to get a hash with a different set of bits,
                                    this can be modified, but it does not have use thus far
                                    in this code (setting this to 0 can quickly compare a large
                                    collection of images/sets of images)
        """
        originally_flattened = self.flattened
        if not originally_flattened:
            self.flatten_images()

        img_15b = np.right_shift(self.images, bit_shift_override)
        hash_val = b'X'
        # iterates through each pixel
        for i in range(self.images.shape[0]):
            new_val = b'|'.join([bytes(str(x), encoding='ascii') for x in img_15b[i]])
            hash_val = index_generators.md5_bytes(hash_val + new_val)

        #print(self.image_shapes)
        #print(b'HASH VALUE: ' + hash_val)


        if not originally_flattened:
            self.reconstitute_images()

        return hash_val

    def add_noise_to_first_bits(self, n_bits):

        if n_bits == 0:
            return 
        
        channel_generator = index_generators.create_channel_generator(
            os.urandom(16),
        )
        pixel_generator = self.create_pixel_generator()
        
        noise_generator = index_generators.crypto_generator(
            os.urandom(16),
            mod=256
        )

        if not isinstance(n_bits, int) and n_bits < 1:
            n_bits = int(np.round(n_bits * ih.max_image_index))

        for i, n_subpixels in enumerate(chunk_seq(n_bits)):
            pixels = grab(pixel_generator, n_subpixels)
            channels = grab(channel_generator, n_subpixels)
            values = grab(noise_generator, n_subpixels)
            self.write_pixels(
                indices = pixels,
                value = values,
                trunc_pattern=channels
            )

    def add_noise_to_nth_bit(self, n_bits, n=4):
        """
        adds noise to single channel, ranging from 1 through 8

        @param n_bits - Number of bits to write
        @param n - Channel index

        """
        if n_bits == 0:
            return
        
        originally_flattened = self.flattened
        if not originally_flattened:
            self.flatten_images()        

        bits = os.urandom(n_bits)
        pixel_generator = self.create_pixel_generator()
        noise_generator = index_generators.crypto_generator(
            os.urandom(16),
            mod=2,
        )

        channel_selector = index_generators.crypto_generator(
            os.urandom(16),
            mod=3
        )

        IDX_POW = np.power(2, n-1)

        for i, n_subpixels in enumerate(chunk_seq(n_bits)):
            channels = grab(channel_selector, n_subpixels)
            pixels = grab(pixel_generator, n_subpixels)
            values = np.asarray(grab(noise_generator, n_subpixels))

            # write by removing first N bits,
            # then re-adding first N-1 bits,
            # then add the bit values to the Nth bit
            self.images[pixels,channels] = (
                self.images[pixels,channels] -
                np.mod(self.images[pixels,channels], IDX_POW * 2) +
                np.mod(self.images[pixels,channels], IDX_POW)  +
                IDX_POW * values
            )

        if not originally_flattened:
            self.reconstitute_images()

    def create_pixel_generator(
        self,
        initial_value=None,
        initial_value_n_bits=16,
        *args,
        **kwargs
    ):
        if initial_value is None:
            initial_value = os.urandom(initial_value_n_bits)
        return index_generators.crypto_generator(
            initial_value,
            *args,
            mod=self.max_image_index,
            **kwargs
        )

        

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
    #N_PIXELS = 1667200
    N_PIXELS = 16672
    indices = []
    gp = index_generators.crypto_generator(
        b'A' * 32,
        mod=ih.max_image_index,
        unique=True
    )
    pixels = [next(gp) for _ in range(N_PIXELS)]

    # get channel patterns

    if True:
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
            b'\x00' * 32 + os.urandom(16),
            mod=256,
        )
        values = [next(gv) for _ in range(N_PIXELS)]
        from collections import Counter
        print('Max Channel Count: %s' % np.max(list(Counter(pixels).values())))
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

    read_values = ih.read_pixels(
        indices = pixels,
        trunc_pattern = patterns
    )

    if not np.all(read_values == values):
        print('write/read mismatch')
        diffs = [
            (i,x,y,c) for i, (x,y,c) in enumerate(zip(values,read_values,patterns)) if x != y
        ]
        if len(diffs) > 64:
            print(diffs[:32])
            print(diffs[-32:])
        else:
            print(diffs)
        print(f'# mismatches: {len(diffs)}')
        from collections import Counter
        hamming_dists = Counter([bin(x[1] ^ x[2]).count('1')for x in diffs])
        print(f'hamming dists: {hamming_dists}')
        raise ValueError('read/write mismatch!')
              
    

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
    
