import argparse
from Crypto.Cipher import AES
import datetime
import gc
from getpass import getpass
import io
import numpy as np
from PIL import Image
import os
import re
import struct

import encryption
from image_handler import ImageHandler
from image_handler import ARCHIVE_SUFFIX_REGEX, is_archive, get_archive_type
from index_generators import create_channel_generator
from index_generators import crypto_generator
from index_generators import md5_bytes
from signatures import Verifier
from utility import chunk_seq
from utility import grab

from encryption import HEADER_STRUCT_FORMAT

# TODO: test and debug >_<

HEADER_IDENTIFIER=b'WHAT IS IN HERE?'

CHANNEL_PATTERNS = [[3,3,2],[3,2,3],[2,3,3]]

MAIN_CHUNK_SIZE = 2 ** 18

VERBOSE_LOG=False
def verbose_log(x,**kwargs):
    if VERBOSE_LOG:
        tsp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{tsp}] - {x}', **kwargs)


# reduces effectiveness of rainbow tables
DEFAULT_PEPPER = b'77StrangeSigmoidalSerpentineSaltSnacks'


def get_options():
    parser = argparse.ArgumentParser(
        description = 'Hide and extract encrypted information '
        'from images and archives of images'
    )

    parser.add_argument(
        'mode',
        choices=['write','read'],
        help='Which mode should this be run in?'
    )

    parser.add_argument(
        'filename',
        help='Filename to read from (read mode)/write to (write mode)',
        default='',
    )

    parser.add_argument(
        '--output-directory',
        help='Directory to store output in',
        default=''
    )

    parser.add_argument(
        '--use-disk',
        help='Stores written pixel data on disk rather than set to use disk space instead of memory at the expense of speed',
        action='store_true',
    )

    parser.add_argument(
        '-v','--verbose',
        help='Prints more output on steps.',
        action='store_true'
    )
        

    write_parser = parser.add_argument_group(
        'Write'
    )
    read_parser = parser.add_argument_group(
        'Read'
    )

    # note that these really shouldn't be changed too often
    argon2_parser = parser.add_argument_group(
        'Argon2 Params'
    )

    credential_parser = parser.add_argument_group(
        'Credentials'
    )

    # write args

    write_parser.add_argument(
        '--source-images',
        help='Source images to use when in write mode',
        nargs='*',
        required=False,
        default=[],
    )


    # format should be R,G,B,H,W,NOISE_RATIO_
    write_parser.add_argument(
        '--image-construction-params',
        default='',
        required=False,
        help='Parameters to define image generation to create images for. '
        'Format should be R1,G1,B1,H1,W1,NOISE_RATIO1:R2,G2,B2,H2,W2,NOISE_RATIO2:...',
    )

    write_parser.add_argument(
        '--source-data',
        help='Data to write to files',
        required=False,
    )

    write_parser.add_argument(
        '--message',
        help='Message to send',
        required=False,
        default='',
    )

    write_parser.add_argument(
        '--header-filename',
        default='',
        required=False,
        help='Filename to put in header'
    )

    """
    write_parser.add_argument(
        '--output-filename',
        required=False,
        help='Filename with appropriate suffix.'
    )
    """

    write_parser.add_argument(
        '--noise-ratio',
        default=0.0,
        type=float,
        help='Ratio of noise to add to initial images'
    )

    # read args

    read_parser.add_argument(
        '--read-outfile',
        help='File to write output of read image/archive to. If not '
        'provided, will default to data stored in image.',
        required=False,
        default='',
    )

    # credentials
    credential_parser.add_argument(
        '--password-file',
        default='',
        help='File containing password to use.',
        required=False
    )

    credential_parser.add_argument(
        '--multi-password-file',
        default='',
        help='File containing many passwords to try to use. For read mode only. NOT YET IMPLEMENTED.',
        required=False,
    )

    credential_parser.add_argument(
        '--password-env',
        default='',
        help='Environmental variable password is stored in',
        required=False
    )    

    # pub key
    credential_parser.add_argument(
        '--pub-key',
        required=False,
        help='Public key path'
    )

    # priv key
    credential_parser.add_argument(
        '--priv-key',
        required=False,
        help='Private key path'
    )

    credential_parser.add_argument(
        '--priv-key-password',
        required=False,
        default='',
        help='Private key password'
    )

    credential_parser.add_argument(
        '--priv-key-password-prompt',
        required=False,
        default='',
        help='Prompts for private key password'
    )

    credential_parser.add_argument(
        '--priv-key-password-env',
        required=False,
        default='',
        help='Environmental variable private RSA key password stored in',
    )    

    # argon2

    argon2_parser.add_argument(
        '-p',
        type=int,
        help='Degree of parallelism for argon2',
        default=2
    )

    argon2_parser.add_argument(
        '--buflen',
        type=int,
        help='Length of argon2 buffer',
        default=32
    )

    argon2_parser.add_argument(
        '--memory',
        type=int,
        help='Memory used for argon2 in KiB',
        default=2 ** 14,
    )

    argon2_parser.add_argument(
        '--rounds',
        type=int,
        help='# of rounds used for argon2',
        default=128,
    )

    argon2_parser.add_argument(
        '--salt',
        required=False,
        help='Salt for write mode (really a pepper). Uses hash of pepper by default. 16 bytes. '
        'will be padded with most significant 15 bytes of data unless suppressed',
        default=None,
    )

    parser.add_argument(
        '--no-image-salt',
        action='store_true',
        help='If not selected, will not generate salt based on most significant 15 bits of '
        'pixels for images. If selected, the same noise pattern will be present for the same '
        'password and pepper.'
    )

    parser.add_argument(
        '--salt-noise',
        type=int,
        default=128,
        help='Number of pixels to add noise to 4th bit so that salts are unique '
        'when same password is used on same images. 0 is safe if same permutation of '
        'images will not be reused.'
    )

    args = parser.parse_args()
    options = vars(args)

    if re.search(r'[^A-Za-z0-9_.-]',options['header_filename']):
        raise ValueError('{header_filename} not a valid header filename option'.format(**options))

    if options['mode'] == 'write':
        
        if not options['filename']:
            raise ValueError('output_filename needs to be defined in write mode!')
        options['write_archive_type'] = get_archive_type(options['filename'])
    
    if options['salt'] is None:
        options['salt'] = md5_bytes(DEFAULT_PEPPER)[:16] #os.urandom(16)
    else:
        options['salt'] = bytes(options['salt'], encoding='UTF-8')
        if len(options['salt']) > 16:
            options['salt'] = options['salt'][:16]
        elif len(options['salt']) < 16:
            options['salt'] += b'\x00' * (16 - len(options['salt']))

    if not options['header_filename']:
        if options['message']:
            options['header_filename'] = 'message.txt'
        elif options['source_data']:
            options['header_filename'] = options['source_data']
        else:
            options['header_filename'] = 'mystery.dat'
    
    manage_passwords(options)

    if options['verbose']:
        global VERBOSE_LOG
        VERBOSE_LOG=True

    priv_key_pass = None
    if options['priv_key_password']:
        priv_key_pass = options['priv_key_password']
    elif options['priv_key_password_env']:
        priv_key_pass = os.environ[options['priv_key_password_env']]
    elif options['priv_key_password_prompt']:
        priv_key_pass = getpass('Private Key Passphrase: ')
        
    if options['pub_key'] or options['priv_key']:
        options['verifier'] = Verifier(
            pub_key_fn = options['pub_key'],
            priv_key_fn = options['priv_key'],
            priv_key_pass = priv_key_pass
        )
    else:
        options['verifier'] = None

    options['header_identifier'] = HEADER_IDENTIFIER

    return options

def manage_passwords(options):
    if options['password_file']:
        with open(options['password_file'], 'rb') as f:
            password = f.read()
    elif options['multi_password_file']:
        with open(options['multi_password_file'], 'rb') as f:
            password = f.read().splitlines()
    elif options['password_env']:
        password = bytes(
            os.environ.get(options['password_env']),
            encoding='UTF-8'
        )
    else:
        password = getpass('Password: ')

    options['password'] = password

    return password



    
    

def run(options):
    if options['mode'] == 'read':
        verbose_log('read mode')
        run_read_mode(options)
    elif options['mode'] == 'write':
        verbose_log('write mode')
        run_write_mode(options)

def run_read_mode(options):
    # read input file data
    ih = ImageHandler(
        filenames=options['filename'],
        archive_type = get_archive_type(options['filename'])
    )
    #print(f'max idx: {ih.max_image_index}')

    if not options['no_image_salt']:
        hashed_salt = ih.create_15bit_channel_hash()
        options['salt'] += hashed_salt
    else:
        hashed_salt = b''

    # hash password
    verbose_log('key 1 hash')
    key = encryption.argon2_hash(options['password'], **options)
    #print(f'key: {key}')

    # if 1 pass
    # run decryptor
    # decrypt first three blocks (1:nonce, 2: header identifier, 3:header info)
    verbose_log('key 2 hash')
    key2 = encryption.argon2_hash(
        key,
        salt=DEFAULT_PEPPER,
        rounds=32,
        memory=2**16,
        buflen=32,
        p=2
    )
    pixel_generator = crypto_generator(
        key2,
        unique=True,
        mod=ih.max_image_index,
        disk_storage = options['use_disk'],
    )

    # nonce + tag
    N_PIXELS_STEP1 = 64
    if N_PIXELS_STEP1 >= ih.max_image_index:
        raise ValueError('Insufficient space to store data')
    verbose_log('pixels step 1')
    pixels_step1 = grab(pixel_generator, N_PIXELS_STEP1)

    verbose_log('key 3 hash')
    # generate channel pattern
    key3 = encryption.argon2_hash(
        key2,
        salt=DEFAULT_PEPPER,
        rounds=32,
        memory=2**16,
        buflen=32,
        p=2,
    )
    channel_generator = create_channel_generator(
        key3
    )
    verbose_log('step 1')
    channels_step1 = grab(channel_generator, N_PIXELS_STEP1)

    values_step1 = ih.read_pixels(
        pixels_step1,
        trunc_pattern=channels_step1
    ).astype(np.uint8).tobytes()
    nonce, tag, encrypted_identifier, encrypted_header = (
        values_step1[0:16],
        values_step1[16:32],
        values_step1[32:48],
        values_step1[48:64],
    )
    #print(nonce)
    #print(f'tag: {tag}')
    # header format:
    # 8 bytes: size of payload
    # 2 bytes: length of filename
    # 1 byte: presence of signature (0:none, 1:sha256,2:sha512)
    # 1 byte: RSA size (0:none, 1:1024, 2:2048, 3: 3072, 4: 4096)
    # 4 bytes: blank (TBD)
    cipher_d = AES.new(key, AES.MODE_EAX, nonce=nonce)
    decrypted_identifier = cipher_d.decrypt(encrypted_identifier)
    if decrypted_identifier != options['header_identifier']:
        # fast way to fail
        print(decrypted_identifier)
        raise ValueError('Header identifier does not match!')
    verbose_log('decrypting header')
    decrypted_header = cipher_d.decrypt(encrypted_header)
    #print(f'DHB: {decrypted_header}')
    data_length, header_filename_length, signature_byte, rsa_size_byte = struct.unpack(
        HEADER_STRUCT_FORMAT,        
        decrypted_header
    )    
    
    # read filename from required blocks (0-2 most likely, but could be large)
    n_filename_blocks = int(np.ceil(header_filename_length / 16))
    n_data_blocks = int(np.ceil(data_length / 16))
    rsa_size = rsa_size_byte * 128
    n_data_signature_blocks = rsa_size / 16
    cuml_pixels = N_PIXELS_STEP1
    verbose_log('parsed header')
    # read signature from required blocks (0, 16, 32, 48, 64)
    if n_data_signature_blocks > 0:
        verbose_log('step 2 start')
        N_PIXELS_STEP2 = rsa_size
        cuml_pixels += N_PIXELS_STEP2
        
        if cuml_pixels >= ih.max_image_index:
            raise ValueError('Insufficient space to store data')
        pixels_step2 = grab(pixel_generator, N_PIXELS_STEP2)
        channels_step2 = grab(channel_generator, N_PIXELS_STEP2)
        values_step2 = ih.read_pixels(
            pixels_step2,
            trunc_pattern=channels_step2
        ).astype(np.uint8).tobytes()
        data_signature = cipher_d.decrypt(values_step2)
        verbose_log('step 2 complete')
    else:
        data_signature = None

    # read filename
    if n_filename_blocks:
        verbose_log('step 3 start')
        N_PIXELS_STEP3 = n_filename_blocks * 16
        cuml_pixels += N_PIXELS_STEP3
        if cuml_pixels >= ih.max_image_index:
            raise ValueError('Insufficient space to store data')        
        pixels_step3 = grab(pixel_generator, N_PIXELS_STEP3)
        channels_step3 = grab(channel_generator, N_PIXELS_STEP3)
        values_step3 = ih.read_pixels(
            pixels_step3,
            trunc_pattern=channels_step3
        ).astype(np.uint8).tobytes()
        filename_data = cipher_d.decrypt(values_step3)
        filename = os.path.split(filename_data[:header_filename_length])[1]
        #print(filename_data)
        # 
        filename = os.path.split(filename)[1]
        filename = re.sub(r'[^A-Za-z0-9_.-]','', filename.decode('utf-8'))
        if len(filename) == '':
            raise ValueError('Filename is blank after replacing illegal characters!')
        verbose_log('step 3 end')
        
        if options['read_outfile']:
            filename = options['read_outfile']
            verbose_log('filename manually provided (forced)')
    else:
        filename = options['read_outfile']
        verbose_log('filename manually provided')



    # read data
    data = b''
    if n_data_blocks:
        verbose_log('start step 4')
        N_PIXELS_STEP4 = n_data_blocks * 16
        # chunk this
        for i, n_subpixels in enumerate(chunk_seq(N_PIXELS_STEP4)):
            verbose_log(f'step 4 iter {i}')
            #gc.collect()
            #verbose_log('GC CALL')
            cuml_pixels += n_subpixels #N_PIXELS_STEP4
            if cuml_pixels >= ih.max_image_index:
                raise ValueError('Insufficient space to store data')        
            pixels_step4 = grab(pixel_generator, n_subpixels)
            # TODO: check to see if the np.array is causing error
            channels_step4 = grab(channel_generator, n_subpixels)
            values_step4 = ih.read_pixels(
                pixels_step4,
                trunc_pattern=channels_step4
            ).astype(np.uint8).tobytes()
            data += cipher_d.decrypt(values_step4)
            
    #print(f'cuml_pixels: {cuml_pixels}')
    # tag verify
    verbose_log('verifying tag')
    try:
        verification_status = cipher_d.verify(tag)
    except Exception as e:
        print('Tag verification failed!')
        raise e # put this back after testing
    
    verbose_log('tag verified')
    # rsa signature verify
    if options['verifier']:
        verbose_log('RSA verification start')
        #print('Verify Length: %s' % len(filename_data[:header_filename_length] + data))
        #print(len(data))
        #print(header_filename_length)
        verification_status = options['verifier'].verify_message(
            filename_data[:header_filename_length] + data[:data_length],
            signature = data_signature
        )
        if not verification_status:
            raise Exception('Signature could not be verified!') # put this back after testing
        verbose_log('RSA verification complete')

    # return
    data = data[:data_length]
    output_filepath = os.path.join(
        options['output_directory'],
        filename
    )
    verbose_log('beginning file write')
    with open(output_filepath, 'wb') as f:
        f.write(data)

    verbose_log('file write complete')

    # TODO: if many pass
    # try to keep on running decryptor for first 2 blocks
    # so that header matches identifier
    # worry about implementing this later...
    

def run_write_mode(options):
    # read input images or generate source images
    file_list = []
    ih = None    
    if options['source_images']:
        verbose_log('beginning source image read')
        file_list += options['source_images']
        # check if archives
        if any([is_archive(fn) for fn in file_list]):
            verbose_log('searching for archives')
            for fn in file_list:
                archive_type = get_archive_type(fn)
                verbose_log(f'reading from archive {fn} with archive type {archive_type}')
                if ih is None:
                    ih = ImageHandler(fn, archive_type=archive_type)
                else:
                    ih = ih + ImageHandler(fn, archive_type=archive_type)
        else:
            ih = ImageHandler(file_list)

    if options['salt_noise']:
        ih.add_noise_to_nth_bit(options['salt_noise'], n=4)
        

    if options['noise_ratio']:
        #random_seed = os.urandom(16)
        noisy_pixels = int(options['noise_ratio'] * ih.max_image_index)
        verbose_log(f'generating {noisy_pixels} noisy pixels on source images')
        # create sequence generators
        ih.add_noise_to_first_bits(noisy_pixels)
        #for i, n_subpixels in enumerate(chunk_seq(noisy_pixels)):

    if options['image_construction_params']:
        verbose_log('generating source images')
        ih = generate_source_images(options['image_construction_params'], ih)

    if not options['no_image_salt']:
        hashed_salt = ih.create_15bit_channel_hash()
        options['salt'] += hashed_salt
    else:
        hashed_salt = b''        
            
    # read in data
    if options['message']:
        verbose_log('converting message to bytes')
        data = bytes(options['message'], encoding='UTF-8')
    elif options['source_data']:
        verbose_log('reading source data')
        with open(options['source_data'], 'rb') as f:
            data = f.read()
    else:
        # should log warning here probably
        verbose_log('no data provided!')
        data = b''

    # TODO: create memory-efficient  version of this code
    # create data
    verbose_log('encrypting data')
    key, encrypted_data = encryption.data_to_encrypted(data, **options)

    N_PIXELS = len(encrypted_data)
    #print(f'max image index: {ih.max_image_index}')
    #print(f'key: {key}')

    # write data to image
    # generate pixels
    verbose_log('key 2 hash')
    key2 = encryption.argon2_hash(
        key,
        salt=DEFAULT_PEPPER,
        rounds=32,
        memory=2**16,
        buflen=32,
        p=2
    )
    pixel_generator = crypto_generator(
        key2,
        unique=True,
        mod=ih.max_image_index,
        disk_storage = options['use_disk'],        
    )

    if N_PIXELS >= ih.max_image_index:
        raise ValueError('Insufficient space to store data')

    
    # generate channel pattern
    verbose_log('key 3 hash')
    key3 = encryption.argon2_hash(
        key2,
        salt=DEFAULT_PEPPER,
        rounds=32,
        memory=2**16,
        buflen=32,
        p=2,
    )
    channel_generator = create_channel_generator(
        key3,
    )
    # use subpixels
    chunk_start_idx = 0
    for i, n_subpixels in enumerate(chunk_seq(N_PIXELS)):
        verbose_log(f'writing iter {i}')
        #gc.collect()
        #verbose_log('GC CALL')        
        pixels = grab(pixel_generator, n_subpixels)

        channels = grab(channel_generator, n_subpixels)

        ih.write_pixels(
            indices = pixels,
            value=np.frombuffer(
                encrypted_data[chunk_start_idx:chunk_start_idx+n_subpixels],
                dtype=np.uint8
            ),
            trunc_pattern=channels
        )
        chunk_start_idx += n_subpixels

    # save image (in appropriate mode)
    if options['write_archive_type']:
        verbose_log('writing to archive')
        ih.write_archive(
            filename = options['filename'],
            archive_type = options['write_archive_type'],
            output_path=options['output_directory'],
        )
    else:
        verbose_log('writing to file')
        ih.write(
            output_filenames = options['filename'],
            output_path=options['output_directory'],            
        )


    #print([Image.fromarray(np.uint8(img)).getcolors() for img in ih.images])    

def generate_source_images(params, original_ih = None):
    params_list = params.split(':')
    # RGB HW NOISE_RATIO
    source_images = []
    for param in params_list:
        #print(param)
        try:
            r,g,b,h,w,nr = [
                float(x) if float(x) < 1 and float(x) != 0
                else int(x)
                for x in param.split(',')
            ]
        except ValueError as e:
            print('incorrect # of params to unpack')
            raise e
        np_img = np.zeros((h,w,3), dtype=np.uint8)
        for c, v in [(0,r),(1,g),(2,b)]:
            np_img[:,:,int(c)] = int(v)
        img_io = io.BytesIO()
        img = Image.fromarray(np.uint8(np_img))
        img.save(img_io, format='PNG')
        img_io.seek(0)
        ih = ImageHandler(img_io)
        if nr > 0:
            n_pixels = ih.max_image_index
            modded_pixels = np.uint8(np.floor(nr * n_pixels))
            # generate data
            # r
            gp = crypto_generator(
                os.urandom(16),
                mod=ih.max_image_index,
                unique=False,
                disk_storage = options['use_disk'],
            )
            gc = crypto_generator(
                os.urandom(16),
                mod=3,
                unique=False
            )
            gv = crypto_generator(
                os.urandom(16),
                mod=256
            )

            patterns = [
                CHANNEL_PATTERNS[next(gc)]
                for _ in range(modded_pixels)
            ]

            values = [
                next(gv)
                for _ in range(modded_pixels)
            ]

            pixels = [
                next(gp)
                for _ in range(modded_pixels)
            ]
            ih.write_pixels(
                pixels,
                values,
                patterns
            )

        source_images.append(ih)
    base_ih = original_ih
    for ih in source_images:
        base_ih += ih

    #print(base_ih)

    return base_ih
            
        

if __name__=='__main__':
    options = get_options()
    run(options)
