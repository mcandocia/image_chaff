from argon2 import argon2_hash
from uuid import uuid4
import os
import argparse
from getpass import getpass
try:
        from Cryptodome.Cipher import AES
except ImportError:
        from Crypto.Cipher import AES
