#!/bin/bash

# this is going to be a set of tests to make sure that
# basic functionality of the scripts works
# this will be updated later to include more complex tests
# as well as diagnostics

echo "RUNNING SCRIPT TESTS"

TEST_DATA_DIRECTORY=testing_data
TEST_OUTPUT_DIRECTORY=$TEST_DATA_DIRECTORY/output
mkdir $TEST_OUTPUT_DIRECTORY

PW_FILE=$TEST_DATA_DIRECTORY/test_pass.txt

# test write single image

echo "SINGLE IMAGE WRITE/READ"

OUTPUT_1=$TEST_OUTPUT_DIRECTORY/output_1.png 
INPUT_1=$TEST_DATA_DIRECTORY/alphabet.txt
SOURCE_1=$TEST_DATA_DIRECTORY/couchy_distro.jpg

python3 image_chaff.py \
    write \
    $OUTPUT_1 \
    --source-images=$SOURCE_1 \
    --source-data=$INPUT_1 \
    --password-file=$PW_FILE \
    --use-disk \
    --noise-ratio=0.02 \
    --salt-noise=256


python3 image_chaff.py \
    read \
    $OUTPUT_1 \
    --password-file=$PW_FILE \
    --use-disk \
    --output-directory=$TEST_OUTPUT_DIRECTORY

# test 2 with archive

echo "ARCHIVE TEST"
OUTPUT_2=$TEST_OUTPUT_DIRECTORY/output_2.zip
INPUT_2=$TEST_DATA_DIRECTORY/zeroes.txt
SOURCE_2=$TEST_DATA_DIRECTORY/image_archive.zip

python3 image_chaff.py \
    write \
    $OUTPUT_2 \
    --source-images=$SOURCE_2 \
    --source-data=$INPUT_2 \
    --password-file=$PW_FILE \
    --use-disk \
    --noise-ratio=0.02 \
    --salt-noise=256


python3 image_chaff.py \
    read \
    $OUTPUT_2 \
    --password-file=$PW_FILE \
    --use-disk \
    --output-directory=$TEST_OUTPUT_DIRECTORY

# test 3 with archive
echo "FOLDER TEST"
OUTPUT_3=$TEST_OUTPUT_DIRECTORY/output_3.zip 
INPUT_3=$TEST_DATA_DIRECTORY/wonderland.txt
SOURCE_3=$TEST_DATA_DIRECTORY/image_archive.zip

python3 image_chaff.py \
    write \
    $OUTPUT_3 \
    --source-images=$SOURCE_3 \
    --source-data=$INPUT_3 \
    --password-file=$PW_FILE \
    --use-disk \
    --noise-ratio=0.02 \
    --salt-noise=256


python3 image_chaff.py \
    read \
    $OUTPUT_3 \
    --password-file=$PW_FILE \
    --use-disk \
    --output-directory=$TEST_OUTPUT_DIRECTORY



