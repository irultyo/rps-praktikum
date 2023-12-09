# Klasifikasi Batu Gunting kertas Dengan Pretrain EfficientNetB3

Project ini menggunakan EfficientNetB3 sebagai model pretrain yang diambil menggunakan pustaka tensorflow.

**Instalasi**
python version >= 3.10
Clone repositori ini
```git clone https://github.com/irultyo/rps-praktikum.git```
Masuk ke direktori repositori
Buat sebuah venv
```python -m venv ./rpsvenv```
Aktifkan venv
- Command Prompt
```./rpesvenv/Scripts/activate.bat```
- Powershell
```./rpesvenv/Scripts/activate.ps```

Setelah itu install semua requirements
```pip install -r requirements```
Lalu jalankan flask
```python app.py```

Web dapat diakses dengan url ```localhost:5000```


**Dataset**
Project ini menggunakan dataset rock paper scissor dengan jumlah data sebanyak 2520 file. Load image menggunakan image_dataset_from_directory dari pustaka tensorflow dengan pembagian train validation 80% dan test validation 20% dengan seed 123. Dataset menggunakan label categorical sehingga label dalam bentuk one hot encoding. Image size menggunakan (224, 244) dan batch size menggunkan 128

**EfficientNet**
>Tan, M., & Le, Q. (2019, May). Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning (pp. 6105-6114). PMLR.

![N|Solid](https://www.researchgate.net/publication/353790360/figure/fig3/AS:11431281179142345@1691155534303/EfficientNet-B3-structure-aEfficientNet-B3-bMBConv6-55.png)

**Summary Model**

![N|Solid](https://i.postimg.cc/kX6xMrPh/Screenshot-2023-12-09-170515.jpg)

**Train** 
Train data dilakukan dalam 10 epoch dengan optimizer Adam dengan learning rate 0.01

**Evaluasi**
![N|Solid](https://i.postimg.cc/Qtr3WRfb/output.png)

Dari train 10 epoch, akurasi validasi sebesar 0.984127 (98.41%) dengan loss sebesar 0.051995
![N|Solid](https://i.postimg.cc/KjTFkm2J/Screenshot-2023-12-09-171745.jpg)