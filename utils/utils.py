import re 
import cv2
import numpy as np
import os
from PIL import Image
import time
from keras.models import load_model
import keras.utils as im

def getSplitedStringByIndex(string_data, regex, index) -> str:
    """
    Function Description :
    
        getSplitedStringByIndex : mengembalikan object string yang dipilih oleh user
        dari rangkaian string yang sudah dipisah datanya. 

        Contoh Argument : (string_data = "this_dataset", regex = "_", index = 0)

        Contoh hasil : ["this"]
    """
    splited_string  = string_data.split(regex)
    res             = splited_string[index]
    return res

def getQueryImage(image_path) :
    """
    Function Description :
    
        getQueryImage : mengembalikan dua list yang berisi label dari citra
        dan list yang berisi file path dari citra. fungsi hanya menerima string path dari
        citra yang digunakan sebagai query

        Contoh Argument : (image_path = "static/queryImage/")

        Contoh hasil : 
            labels = ["rock", "rock", "paper", "scissor"]
            paths = [
                "static/queryImage/rock_1.jpg",
                "static/queryImage/rock_2.jpg",
                "static/queryImage/paper_1.jpg",
                "static/queryImage/scissor_1.jpg",
                ]
    """ 
    labels = []
    paths = []

    for data in os.listdir(image_path):
        label = getSplitedStringByIndex(data, "_", 0)
        labels.append(label)
        paths.append(os.path.join(image_path, data))
    
    return labels, paths

def getListofModel(model_path) : 
    """
    Function Description :
    
        getListofModel : mengembalikan dua list yang berisi nama dari model
        dan list yang berisi file path dari model. fungsi hanya menerima string path dari
        model yang digunakan untuk memprediksi image

        Contoh Argument : (image_path = "static/model/")

        Contoh hasil : 
            labels = ["EFFICIENTNET_model", "VGG19_model"]
            paths = [
                "static/model/EFFICIENTNET_model.h5",
                "static/model/VGG19_model.h5",
                ]
    """ 
    names = []
    paths = []

    for data in os.listdir(model_path):
        name = getSplitedStringByIndex(data, ".", 0)
        names.append(name)
        paths.append(os.path.join(model_path, data))
    
    return names, paths

def buildModel(model_path, model_list) -> list:
    """
    Function Description :
    
        buildModel : mengembalikan sebuah list yang berisi model yang siap
        digunakan untuk memprediksi image yang dimasukkan. fungsi ini menerima
        dua argumen yaitu path menuju model dan list nama model.

        Contoh Argument : (image_path = "static/model/", model_list=["EFFICIENTNET_model", "VGG19_model"])

        Contoh hasil : 
            models = [
                <keras.model>,
                <keras.model>
                ]
    """ 
    models = []
    for data in os.listdir(model_path):
        name = getSplitedStringByIndex(data, ".", 0)
        if name in model_list:
            model = load_model(os.path.join(model_path, data))
            models.append(model)
    return models

def getDifferentTime(startTime) -> float:
    """
    Function Description :
    
        getDifferentTime : mengembalikan sebuah nilai float perbedaan waktu awal 
        dan waktu akhir ketika model melakukan prediksi. perbedaan tersebut 
        berupa jumlah waktu pemrosesan dalam bentuk detik.

        Contoh Argumen : (startTime = time.time())
        
        Contoh hasil : (343.2323)
    """
    current_time    = time.time()
    different_time  = current_time - startTime
    rounded_time    = round(different_time, 4)
    return rounded_time

def preprocessImage(image):
    """
    Function Description :
    
        preprocessImage : mengembalikan sebuah array citra agar bisa diprediksi
        oleh model. citra akan dilakukan resize mengikuti besaran input 
        pada model (pada model efficentnet menggunakan (224,244)), lalu diubah menjadi array. 
        input berupa string path dari citra yang diprediksi.

        Contoh Argumen : (image = "static/queryImage/rock_1.jpg")
        
        Contoh hasil : (ndarray())
    """
    img = im.load_img(image, target_size=(224, 244))
    x = im.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    return images

def predictsImage(model_path, model_list, image_path):
    """
    Function Description :
    
        predictsImage : melakukan prediksi pada citra dengan list model yang 
        dipilih oleh pengguna. fungsi ini menerima tiga argumen yaitu model path, 
        model list dan image path. model path berisi string direktori yang menyimpan file model.
        model list berisi list nama model yang dipilih oleh pengguna. image path berisi
        string path citra yang dipilih/dimasukkan oleh pengguna. kembalian berupa list
        confidence prediksi model dan list jumlah waktu yang diperlukan model.

        Contoh Argumen : (model_path = "static/model", model_list = ["EFFICIENT_model"], image_path = "static/queryImage/rock_1.jpg")
        
        Contoh hasil : 
            predictions = [[0.999, 0.000, 0.001]]
            times = [5.9344]
    """
    image = preprocessImage(image_path)
    models = buildModel(model_path, model_list)
    predictions = []
    times = []

    for model in models:
        start = time.time()
        prediction = model.predict(image)
        total_time = getDifferentTime(start)
        predictions.append([round(prob, 3) for prob in prediction[0].tolist()])
        times.append(total_time)
    
    return predictions, times
