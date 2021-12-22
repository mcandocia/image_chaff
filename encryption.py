import argon2

import os
import argparse
from getpass import getpass
import numpy as np
from uuid import uuid4
#from Cryptodome.Cipher import AES
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

import struct

from index_generators import md5_bytes

HEADER_STRUCT_FORMAT='>QhBB4x'
HEADER_STRUCT_NAMES = ('payload_size','filename_length','signature_flags','rsa_flags')

def argon2_hash(x, **kwargs):
    kwargs['salt']
    converted_args = {
        'secret': x,
        'salt': kwargs['salt'],
        'time_cost': kwargs['rounds'],
        'memory_cost': kwargs['memory'],
        'parallelism': kwargs['p'],
        'hash_len': kwargs['buflen'],
        
    }
    return argon2.low_level.hash_secret_raw(
        **converted_args,
        type=argon2.low_level.Type.ID # most secure
    )


# TODO: use struct() for cleaner byte reading
def data_to_encrypted(data, **options):

    # build key
    key = argon2_hash(options['password'], **options)
    # build payload
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    #print(nonce)
    size_block = (len(data)).to_bytes(8, 'big')

    filename_bytes = bytes(options['header_filename'],encoding='utf-8')

    if options['verifier']:
        signature_byte = 1
        rsa_size = options['verifier'].priv_key.size_in_bytes()
        if rsa_size == 128:
            rsa_size_byte = 1
        elif rsa_size == 256:
            rsa_size_byte = 2
        elif rsa_size == 384:
            rsa_size_byte = 3
        elif rsa_size == 512:
            rsa_size_byte = 4
        else:
            rsa_size_byte = rsa_size // 128

        #print('Signing length: %s' % len(filename_bytes + data))
        #print(len(data))
        #print(len(filename_bytes))
        data_signature = options['verifier'].sign_message(filename_bytes + data)
        
    else:
        signature_byte = 0
        rsa_size_byte = 0
        data_signature = b''
        

    data_header_block = struct.pack(
        HEADER_STRUCT_FORMAT,
        *(
            len(data),
            len(options['header_filename']),
            signature_byte, # I think this is unnecessary, can be ignored/removed later
            rsa_size_byte
        )
    )

    #print(f'DHB: {data_header_block}')
    # first 2 16-byte blocks not encrypted
    # nonce | tag | ENCRYPTED
    # 16 | 16 |

    # encrypt header
    encrypted = b''

    encrypted += cipher.encrypt(
        options['header_identifier']
    )

    encrypted += cipher.encrypt(
        data_header_block
    )

    if data_signature:
        encrypted += cipher.encrypt(
            data_signature
        )

    # filename
    n_filename_blocks = int(np.ceil(len(options['header_filename']) / 16))
    filename_pad = b'\x00' * (n_filename_blocks * 16 - len(options['header_filename']))
    encrypted += cipher.encrypt(
        filename_bytes + filename_pad
    )

    # pad data
    
    n_data_blocks = int(np.ceil(len(data) / 16))
    data_pad = b'\x00' * (n_data_blocks * 16 - len(data))
    encrypted += cipher.encrypt(
        data + data_pad
    )

    #print('Data tail: ', (data+data_pad)[-64:])

    tag = cipher.digest()
    #print(f'tag: {tag}')    

    payload = nonce + tag + encrypted
    #print(f'Payload size: {len(payload)}')
    
    return (key, payload)

def build_decryption_cipher(nonce, **options):
    key = argon2_hash(options['password'], **options)        
    cipher = AES.new(key, AES.MODE_EAX, nonce=data)
    return cipher

# data is nonce
def decrypt_block(data, cipher=None, **options):
    return cipher.decyrpt(data)

def verify_cipher(cipher, tag):
    try:
        cipher.verify(tag)
        return True
    except Exception as e:
        print(e)
        return False
        
    

