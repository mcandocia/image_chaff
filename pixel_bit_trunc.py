import argparse
from PIL import Image

def

# for testing purposes
def main(options):
    parser = argparse.ArgumentParser(
        description='Fuzzy last channels of data'
    )

    parser.add_argument(
        'filename',
        help='Input filename'
    )

    parser.add_argument(
        '--noise-ratio',
        default=1.,
        type=float,
        help='Ratio of noise'
    )

    parser.add_argument(
        '--noise-bit-pattern',
        nargs=3,
        type=int,
        default=[3,3,2],
        help='How many bits of each channel should be randomized.'
    )

    parser.add_argument(
        '--fixed-bit-pattern',
        action='store_true',
        help='Keeps the bit pattern for noise fixed'
    )

    parser.add_argument(
        '--outfile',
        required=True,
        help='Output filename'
    )
        
        

        

if __name__ == '__main__':
    options = get_options()
    main(options)
