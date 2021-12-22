#!/bin/bash

WRITE_FILENAME=hidden_message_test.zip
TEST_IMG_DIRECTORY=test_images
SOURCE_DATA_FN=a_face.png
PW_FILE=test_pw_file.txt

TEST_MODE=read

if [[ $TEST_MODE = write ]]; then
    echo "write"
# write test
python3 chaff.py \
	write \
	$WRITE_FILENAME \
	--image-construction-params 255,255,255,1000,1200,0.01:0,0,0,1000,1200,0.015 \
	--priv-key=privkey.pem \
	--pub-key=pubkey.pem \
	--source-data=$TEST_IMG_DIRECTORY/$SOURCE_DATA_FN \
	--output-filename=$TEST_IMG_DIRECTORY/$WRITE_FILENAME \
	--password-file=$PW_FILE
else
    echo "read"
# read test
python3 chaff.py \
	read \
	$TEST_IMG_DIRECTORY/$WRITE_FILENAME \
	--priv-key=privkey.pem \
	--pub-key=pubkey.pem \
	--password-file=$PW_FILE 

	
	
fi
