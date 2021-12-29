import argparse
from collections import Counter
import csv
from PIL import Image
import numpy as np
import os
import sys
import zipfile

OUTFILE = '8bit_counts.csv'

def trunc_filename(x):
    return os.path.split(x)[1]

def open_zip_images(fn):
    with zipfile.ZipFile(fn, 'r') as zf:
        filenames, images = zip(*[
            (trunc_filename(member.filename),
            np.asarray(Image.open(zf.open(member)))[:,:,:3] % 8
            )
            for member in zf.filelist
        ])

    return filenames, images


def main(options):
    filenames = options['filenames']
    images = []
    parsed_filenames = []
    for fn in filenames:
        print(f'opening {fn}')
        if fn.endswith('.zip'):
            new_fns, new_images = open_zip_images(fn)
            images.extend(new_images)
            parsed_filenames.extend(new_fns)
        else:
            img = np.asarray(Image.open(fn)[:,:,:3]) % 8
            parsed_filenames.append(trunc_filename(fn))
            images.append(img)

    # calculate stats
    stats = {}
    for fn, img in zip(parsed_filenames, images):
        print(f'calculate {fn} image stats')
        img_freqs = get_image_freqs(img)
        stats[fn] = img_freqs

    # write to csv
    write_data(
        stats,
        options['outfile'],
        options['description'],
        options['overwrite']
    )


def write_data(stats, outfile, description='', overwrite=True):
    columns = ['description', 'filename', 'count','red','blue','green']
    # convert data
    rows = [
        (description, fn, cnt, k[0], k[1], k[2])
        for fn, file_data in stats.items()
        for k, cnt in file_data.items()
    ]
    rows.sort(key = lambda x: (x[0], x[1], x[3], x[4], x[5]))
    nl = '\n'
    if os.path.isfile(outfile) and not overwrite:
        mode = 'a'
    else:
        mode = 'w'

    with open(outfile, mode) as f:
        if mode == 'w':
            # write header
            f.write(','.join(columns) + nl)
        for row in rows:
            f.write(','.join([
                str(numtoint(x)) for x in row
            ]) + nl)

    print('done!')

def numtoint(x):
    if isinstance(x, (float)):
        return int(x)
    return x

def get_image_freqs(img):
    dims = img.shape[:2]
    # create mask
    mask = np.stack(
        (
            np.ones(dims),
            np.ones(dims) * 8,
            np.ones(dims) * 64
        ),
        axis=2
    )

    # this is the original dimensions, but a singular value
    masked_sums = np.sum(img * mask, axis=2)

    sum_counts = Counter(masked_sums.flatten())

    # convert
    rgb_counts = {
        convert_power9(k): v
        for k, v in sum_counts.items()
    }
    return rgb_counts


def convert_power9(x):
    red = x % 8
    x = (x - red) // 8
    green = x % 8
    x = (x - green) // 8
    blue = x % 8
    return (red, blue, green)
    

def get_options():
    parser = argparse.ArgumentParser(
        description='get statistics for last 8 bits of image color channels'
    )

    parser.add_argument(
        'filenames',
        nargs='+',
        help='input images filenames/zip archive filenames'
    )

    parser.add_argument(
        '--description',
        default='default',
        help='Description to add to columns. Can be used to avoid '
        'name collision with separate runs.'
    )

    parser.add_argument(
        '--outfile',
        default='default_counts.csv',
        help='Output file.'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite CSVs instead of appending'
    )

    args = parser.parse_args()

    options = vars(args)
    return options 


if __name__=='__main__':
    options = get_options()
    main(options)


