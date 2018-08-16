from flask import render_template, request
from app import app
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from werkzeug import secure_filename

import cv2
import numpy as np
import os

import struct
from PIL import Image
import scipy
import scipy.misc
import scipy.cluster
from codecs import encode, decode
from colormap import rgb2hex

@app.route('/')
@app.route('/index')
def index():
    path = "app/static/uploadImages"
    imageList = list_image(path)

    path2 = "app/static/scaled_img"
    scaled_img_list = list_image(path2)
    return render_template("index.html", title = "Home", imageList = imageList, scaled_img_list = scaled_img_list)

@app.route("/uploader", methods = ['GET','POST'])
def upload_file():
    if request.method == 'POST':
        # one file
        # f = request.files['file']
        ratio = int(request.form['ratio'])
        # padColor = int(request.form['padColor'])
        redColor = int(request.form['redColor'])
        greenColor = int(request.form['greenColor'])
        blueColor = int(request.form['blueColor'])
        # f.save('app/static/uploadImages/' + secure_filename(f.filename))
        # return 'file uploaded successfully'

        # file list
        uploaded_files = request.files.getlist("file[]")
        imageList = []
        for f in uploaded_files:
            imageList.append(f.filename)
            f.save('app/static/uploadImages/' + secure_filename(f.filename))
            h_img = cv2.imread('app/static/uploadImages/' + secure_filename(f.filename)) # horizontal image
            scaled_h_img = resizeAndPad(h_img, (ratio,ratio), redColor, greenColor, blueColor)
            scaled_h_img_path = 'app/static/scaled_img/' + secure_filename(f.filename)
            cv2.imwrite(scaled_h_img_path, scaled_h_img)

        # path1 = "app/static/uploadImages"
        # imageList = list_image(path1)
        #
        # path2 = "app/static/scaled_img"
        # scaled_img_list = list_image(path2)





        # call resizeAndPad


        return render_template("index.html", title = "Home", ratio = ratio, scaled_img = f.filename, imageList = imageList, scaled_img_list = imageList)

# list all image to array
def list_image(path):


    list =  os.listdir(path)
    list2 = []
    for i in list:
        if (i.endswith(".jpg") or i.endswith(".png")):
            list2.append(i)
    return list2


# resize image
def resizeAndPad(img, size, red=0, green=0, blue=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    # if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
    #     padColor = [padColor]*3
    padColor = [blue, green, red]

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    # cv2.imshow('img-windows',scaled_img)
    # cv2.waitKey(0)
    # cv2.imwrite('test.jpg',scaled_img)

    return scaled_img


# detecte findDominatColor
def findDominatColor(path):

    NUM_CLUSTERS = 5

    print ('reading image')
    im = Image.open(path)
    im = im.resize((150, 150))      # optional, to reduce time
    # ar = np.asarray(im)
    ar = scipy.misc.fromimage(im)
    shape = ar.shape
    ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    print ('finding clusters')
    codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
    print ('cluster centres:\n', codes)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

    index_max = scipy.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    # colour = ''.join(chr(int(c)) for c in peak).encode('hex')
    colour = ''.join(chr(int(c)) for c in peak).encode()
    # colour = bytes(colour, "utf-8")
    colour = encode(colour, "hex")


    # my way to convert rgb to hex
    colour = rgb2hex(int(peak[0]), int(peak[1]), int(peak[2]))



    print ('most frequent is %s (#%s)' % (peak, str(colour)))
