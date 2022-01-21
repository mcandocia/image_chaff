import atexit
from collections import deque
import hashlib
import tempfile

CHANNEL_ORDER = (
    (3,3,2),
    (3,2,3),
    (2,3,3),
)

def chunks(x, n):
    for i in range(0, len(x)):
        yield x[i:i+n]

def md5_bytes(x):
    return hashlib.md5(x).digest()

def md5_hex(x):
    return hashlib.md5(x).hexdigest()

def sha256_bytes(x):
    return hashlib.sha256(x).digest()

def create_channel_generator(
    key = None,
    *args,
    **kwargs
):
    if key is None:
        key = os.urandom(16)
    generator = crypto_generator(
        key,
        *args,
        mod=3,
        **kwargs,
    )

    for elem in generator:
        yield CHANNEL_ORDER[elem]
        
    

def crypto_generator(
    initial_value,
    byte_width=4,
    unique=False,
    rehash_function=md5_bytes,
    byteorder='big',
    verbose=False,
    mod=None,
    disk_storage=False
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
    @param disk_storage - If True, uses a file on disk to store set history
     rather than a set (slower, but uses disk space instead of memory)
    """
    
    split_vals = deque([
        initial_value[i:i + byte_width]
        for i in range(0,len(initial_value),byte_width)
        if i + byte_width <= len(initial_value)
    ])
    if disk_storage:
        if not mod:
            raise ValueError(
                '"mod" must be specified if using a disk target to store set data'
            )
        existing_values = DiskSet(mod)
    else:
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
                continue
            else:
                existing_values.add(int_value)
        if verbose:
            print(next_byte_value)
            print(int_value)
        yield int_value

            
class DiskSet:
    def __init__(
        self,
        size,
    ):
        self.tmp = tempfile.NamedTemporaryFile(delete=True)
        self.size = size
        atexit.register(self.close)

        # math to write pixels in 1 MiB blocks
        block_size = 2 ** 20
        n_full_blocks = size // block_size
        remaining_bytes = size % block_size

        if n_full_blocks:
            fb = b'\x00' * block_size
            for i in range(n_full_blocks):
                self.tmp.write(fb)

        if remaining_bytes:
            self.tmp.write(remaining_bytes * b'\x00')

        self.seek(0)

    def close(self):
        self.tmp.close()

    def seek(self, i):
        self.tmp.seek(i)

    def tell(self, i):
        self.tmp.tell(i)

    def check(self, x):
        self.tmp.seek(x)
        v = self.tmp.read(1)
        return v != b'\x00'

    def add(self, x):
        self.tmp.seek(x)
        self.tmp.write(b'\x01')

    def __contains__(self, value):
        return self.check(value)


    
