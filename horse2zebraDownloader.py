'''
Horse to Zebra dataset

https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

'''

import os
import sys
from urllib.request import urlretrieve

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))

if __name__ == "__main__":
    file_name = "horse2zebra"
    os.makedirs('data',exist_ok=True)
    if os.path.isfile('data/{}.zip'.format(file_name)):
        print('{}.zip already exists.'.format(file_name))
    else:
        url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{}.zip".format(file_name)
        print('url :',url)
        print('download start')
        urlretrieve(url, 'data/{}.zip'.format(file_name), reporthook)
