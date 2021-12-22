from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA512
from Crypto.Hash import SHA256
from Crypto import Random
import sys

def read_file(fn):
    with open(fn, 'r') as f:
        x = f.read()
    return x

class Verifier:
    def __init__(
        self,
        pub_key=None,
        priv_key=None,
        priv_key_pass=None,
        gen_keys=False,
        priv_key_fn=None,
        pub_key_fn=None,
    ):
        if pub_key:
            self.pub_key = pub_key
        if priv_key:
            self.priv_key = priv_key
        if priv_key_fn:
            self.priv_key = RSA.importKey(read_file(priv_key_fn),passphrase=priv_key_pass)
        if pub_key_fn:
            self.pub_key = RSA.importKey(read_file(pub_key_fn))
        if gen_keys:
            self.generate_keys()

    def generate_keys(self):
        random_generator = Random.new().read
        key = RSA.generate(4096, random_generator)
        self.priv_key, self.pub_key = key, key.publickey()

    def sign_message(self, message, hash_alg=SHA512.new):
        if not self.priv_key:
            raise TypeError('priv_key needs to be defined to sign a message!')
        signer = PKCS1_v1_5.new(self.priv_key)
        digest=hash_alg()
        digest.update(message)
        return signer.sign(digest)

    def verify_message(self, message, signature, hash_alg=SHA512.new):
        signer = PKCS1_v1_5.new(self.pub_key)
        digest = hash_alg()
        digest.update(message)
        return signer.verify(digest, signature)

    def set_keys(
        self,
        *,
        pub_key=None,
        priv_key=None,
        pub_key_fn=None,
        priv_key_fn=None,
    ):
        if priv_key:
            self.priv_key = priv_key
        if pub_key:
            self.pub_key = pub_key
        if priv_key_fn:
            self.priv_key = RSA.importKey(priv_key_fn)
        if pub_key_fn:
            self.pub_key = RSA.importKey(pub_key_fn)

    def save_keys(
        self,
        *,
        pub_key_fn = None,
        priv_key_fn=None,
    ):
        if not priv_key_fn or not pub_key_fn:
            raise ValueError('Need to specify both pub_key_fn and priv_key_fn!')
        if not self.pub_key or not self.priv_key:
            raise ValueError('Need to have both public and private keys set for this object!')
        with open (f"{priv_key_fn}", "wb") as priv_key_file:
            priv_key_file.write(self.priv_key.exportKey())
            
        with open (f"{pub_key_fn}", "wb") as pub_key_file:
            pub_key_file.write(self.pub_key.exportKey())
        

def newkeys(keysize):
    random_generator = Random.new().read
    key = RSA.generate(keysize, random_generator)
    private, public = key, key.publickey()
    return public, private

def importKey(externKey):
    return RSA.importKey(externKey)

def getpublickey(priv_key):
    return priv_key.publickey()

def encrypt(message, pub_key):
    #RSA encryption protocol according to PKCS#1 OAEP
    cipher = PKCS1_OAEP.new(pub_key)
    return cipher.encrypt(message)

def decrypt(ciphertext, priv_key):
    #RSA encryption protocol according to PKCS#1 OAEP
    cipher = PKCS1_OAEP.new(priv_key)
    return cipher.decrypt(ciphertext)

def sign(message, priv_key):
    signer = PKCS1_v1_5.new(priv_key)
    digest=SHA512.new()
    digest.update(message)
    return signer.sign(digest)

def verify(message, signature, pub_key):
    signer = PKCS1_v1_5.new(pub_key)
    digest = SHA512.new()
    digest.update(message)
    return signer.verify(digest, signature)


def test():
    verifier = Verifier(gen_keys=True)

    hash_alg = SHA256.new

    message = b'This is a secret message!'
    signature = verifier.sign_message(message,hash_alg=hash_alg)
    print(signature)
    print(verifier.verify_message(message, signature, hash_alg=hash_alg))
    print(verifier.verify_message(b'whoops!', signature,hash_alg=hash_alg))
    print(len(signature))
    if len(sys.argv) > 1:
        if sys.argv[1].lower()[0] == 'y':
            verifier.save_keys(
                pub_key_fn = 'pubkey.pem',
                priv_key_fn = 'privkey.pem'
            )
            # write files
            

if __name__=='__main__':
    test()


