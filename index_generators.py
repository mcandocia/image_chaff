from collections import deque
import hashlib

def md5_bytes(x):
    return hashlib.md5(x).digest()

def md5_hex(x):
    return hashlib.md5(x).hexdigest()

def sha256_bytes(x):
    return hashlib.sha256(x).digest()

def crypto_generator(
    initial_value,
    byte_width=4,
    unique=False,
    rehash_function=md5_bytes,
    byteorder='big',
    verbose=False,
    mod=None
):
    """
    Creates a deterministic sequence of numbers based on hash
    sequences. Uniqueness can be enforced. If parts of a hash 
    cannot be used due to remainder issues, then they will NOT
    be recycled, but only used to create the next hash. 

    @param initial_value - Initial value used for hashing. Preferably
     in the same form as the hashes, but it is valid as long as it
     is hashable, has a length, and is indexable with integers
    @param byte_width - How many bytes to use when returning values
    @param unique - Boolean of whether uniqueness should 
     used to skip duplicate values
    @param rehash_function - Function used to hash based on previous
     value
    @param byteorder - Byte order to read data from
    """
    
    split_vals = deque([
        initial_value[i:i + byte_width]
        for i in range(0,len(initial_value),byte_width)
        if i + byte_width <= len(initial_value)
    ])
    existing_values = set()
    current_value = initial_value
    n_skips = 0
    while True:
        while not split_vals:
            current_value = rehash_function(current_value)            
            split_vals.extend(
                [
                    current_value[i:i+byte_width]
                    for i in range(0,len(current_value),byte_width)
                    if i + byte_width <= len(current_value)                    
                ]
            )
        next_byte_value = split_vals.popleft()
        int_value = int.from_bytes(next_byte_value, byteorder=byteorder)
        if mod:
            int_value = int_value % mod
        if unique:
            if int_value in existing_values:
                n_skips +=1
                if n_skips % 1000 == 0:
                    pass#print(f'skip {n_skips}')
                continue
            else:
                existing_values.add(int_value)
        if verbose:
            print(next_byte_value)
            print(int_value)
        yield int_value
        
            
