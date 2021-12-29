# Cryptographic Image Chaffing

## Introduction

These scripts allow you to encrypt and hide data in an image/collection of images (e.g., in an archive or folder) with a password. The data is hidden in the least significant 3 bits of the image, so it will not have an effect that's usually visible to the human eye on an image.

The program can encrypt as many bytes as there are pixels in an image/collection of images, although speed degrades proportional to the inverse of the proportion of remaining pixels (e.g., 2x as slow with 1/2 remaining, 3x as slow with 1/3 remaining), so it is recommended to choose a suitable number.

It is recommended for data over a few megabytes to use `--use-disk`, which uses a temporary file on disk to store data. This slows down the speed a small bit, but dramatically reduces memory usage (the temporary disk space used is less than the memory used due).

The password uses a strong Argon2id hash that makes brute-forcing all but the easiest passwords infeasible. The hash settings can be changed, but the same ones must be used when decrypting an encrypting.

Pixels and channels are randomly selected using derivatives of the hash (more hashes), so that the patterns hidden in the image are deterministic but as close to random as can be.

## Usage

### Encryption

```
python3 image_chaff.py \
    write \
    image_with_a_secret.png \
    --output-directory=some/directory \
    --source-images image_without_a_secret.png \
    --source-data=incantation_for_world_peace.txt

```

### Decryption

```
python3 image_chaff.py \
    read \
    image_with_a_secret.png \
    --output-directory=some/directory
```

### Other Options

* `--password-file` - Specify a file that contains the password instead of being prompted for it

* `--password-env` - Specify an environmental variable that contains the password instead of being prompted for it

* `--noise-ratio` - Specify a float between 0 and 1. Uniqueness is not enforced, so no speed degredation, but there will be some overlap. For images with low noise (not recommended), this can conceal the relative size of the data hidden.

* `--use-disk` - Use a temporary file on disk instead of memory for keeping track of which pixels have already had data encoded on them. Recommended for files over a few megabytes.

* `-v` / `--verbose` - Extra printed output

* `--image-construction-params` [WRITE] - If you want to generate solid color images, possibly with noise as a quick-and-dirty way to encode data, you can specify it in the format of `R1,G1,B1,H1,W1,NOISE_RATIO1:R2,G2,B2,H2,W2,NOISE_RATIO2:...`, where the commas separate the fields within an image, and colons separate different images.

* `--message` [WRITE] - Instead of providing `--source-data`, you can send a plaintext message with this instead.

* `--header-filename` [WRITE] - This will change the filename that will be written when `read` mode is used. Note that all directories are truncated for user safety, so no extra folders will be created.

* `--noise-ratio` - This will add noise to images in the same manner as the image construction params.

* `--pub-key` - A public RSA key used for verifying the signer of a data payload.

* `--priv-key` - A private RSA key used for signing a data payload.

* `--priv-key-password` - Private key password

* `--priv-key-password-prompt` - Prompts for private key password

* `--priv-key-password-env` - Uses environmental variable for private key

#### Argon2 Parameters

* `-p` - Degree of parallelism

* `--buflen` - Buffer length used

* `--memory` - Memory used in KiB

* `--rounds` - Number of rounds

* `--salt` - A salt used. Default uses a hash of a pepper for this application.

## Best Practices

While this application was meant to be a fun exercise in steganography, if one wants to see if the encoding process makes images noticeably different, a few suggestions for image types to use:

1. AVOID images with large swaths of a single color (e.g., a black or white frame)

2. AVOID computer-generated images without a lot of noise (these usually have smooth, mostly deterministic color patterns)

3. AVOID blurry images (these tend to have less noise)

4. USE more pixels than you need. This reduces the ratio of noise.

5. USE photographs with a decent resolution.

6. Note that PNG is used for images, so other formats—e.g., ".JPEG"—images with a PNG format look a bit off.

I will be running some experiments to see if a convolutional neural network can detect images that belong to one set versus another. The noise difference is pretty small, but it might look synthetic to a proper algorithm.

You can create a CSV with histogram-like data using `check_9bits.py`. Run `python check_9bits.py --help` for information on how to run it.