# image data chaffing v2

REFRESH WITH MD5 HASH = md5 hash the original and use that sequence 

RGB

1 byte can be stored per pixel {3,3,2} for RGB

SEQ_PATTERN_PIXEL():
  MONITOR: REFRESH WITH MD5 HASH WHEN EXHAUSTED
  For every 4-byte sequence:
    SELECT SEQUENCE MOD MAX_PIXELS pixel
    MAP 1 BYTE TO PIXEL

BIT_PATTERN_RGB(): 
  MONITOR: REFRESH WITH MD5 HASH WHEN EXHAUSTED
  FOR EVERY 4-byte sequence (for randomness):
    SELECT PIXEL [R,G,B][SEQUENCE MOD 3] FOR 2 BITS, REMAINING 3

ARCHIVE_PATTERN_ORDER():
  HAVE NAMES SORTED ALPHABETICALLY, USE THOSE AS INDICES
  MONITOR: REFRESH WITH MD5 HASH WHEN EXHAUSTED:
  FOR EVERY 4-byte sequence:
    SELECT FILE 

IMAGE PREPROCESSING:
 This can be used to hide the actual length of the underlying data even better
 1. TAKE RANDOM SEED
 2. TAKE NOISE RATIO (can be 1)
 3. GENERATE NOISE USING ABOVE RANDOMNESS PATTERNS
 AND/OR
 1. GENERATE NOISE
 AND/OR
 1. OTHER FILTER WITH RANDOMNESS

WRITE
 1. Enter PW
 2. Strongly Hash PW (Argon2) as HASH1
 3. Pad data with 0s
 4. Prepend data with length of data (16 bytes)
 5. AES256 encrypt data
 6. Hash HASH1 again (weaker), store as SEQ_PATTERN
 7. Hash SEQ_PATTERN (weaker), store as BIT_PATTERN
 8. Optionally hash BIT_PATTERN (weaker), store as ARCHIVE_PATTERN
 9. Take image with # of pixels <= SIZE
 9b. If multiple images are used, then concatenate indexes together according to alphabetical order.
 10. Select (x,y) pixels with index determined by SEQ_PATTERN_PIXEL()
 11. Select which pixels to truncate 3 bits from (2x) and 2 bits from (1x)  by BIT_PATTERN_RGB()
 12. Truncate end of channels, then write data.
 13. If 


READ

 1. Enter PW
 2. Strongly Hash PW (Argon2) HASH1
 3. Hash HASH1 again (weaker), store as SEQ_PATTERN
 4. Hash SEQ_PATTERN (weaker), store as BIT_PATTERN
 5. Read in bytes using method:
   6. 