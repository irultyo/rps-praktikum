from flask import Flask, request, render_template

from utils import utils


CLASS_DICT          = {"Paper": 0, "Rock": 1, "Scissors": 2} # TO CHANGE
LABELS              = list(CLASS_DICT.keys())
MODEL_PATH          = "static/model/"
QUERY_IMAGE_PATH    = "static/queryImage/"
QUERY_UPLOAD_IMAGE  = "static/queryUpload/"

app = Flask(__name__) 

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/') 
def compare():
    """
        Fungsi ini digunakan untuk menampilkan halaman utama
        Terdapat 3 variabel yang nanti dimasukkan pada file html, yaitu:
            listModel   : berisi nama model yang tersimpan pada folder static/model/
            imageClass  : berisi nama kelas pada citra yang ditampilkan di halaman utama
            imageQuery  : berisi file path tiap citra yang ditampilkan di halaman utama
    """
    listModel, _                = utils.getListofModel(MODEL_PATH)
    imageClass, imageQuery      = utils.getQueryImage(QUERY_IMAGE_PATH)
    return render_template('/compare.html', listModel = listModel, imageQuery = imageQuery, imageClass=imageClass)

@app.route('/pred_comp', methods=['POST'])
def predict_compare():
    """
        Fungsi ini digunakan untuk memproses image yang dipilih untuk menghasilkan prediksi kelas.
        Terdapat 4 variabel pada fungsi ini:
            choosenModelList    : variabel ini menyimpan data model yang dipilih dari halaman sebelumnya
            getImageFile        : variabel ini menyimpan data citra yang dipilih dari halaman sebelumnya
            predictionResult    : variabel ini berisi hasil prediksi dalam bentuk list
            predictionTime      : variabel ini berisi lama waktu prediksi
    """
    choosenModelList                 = request.form.getlist('select_model') 
    getImageFile                     = request.form.get('input_image') 
    predictionResult, predictionTime = utils.predictsImage(MODEL_PATH, choosenModelList, getImageFile)  
    return render_template('/result_compare.html', labels = LABELS, probs = predictionResult, model = choosenModelList, run_time = predictionTime, img = getImageFile[7:])

@app.route('/pred_comps', methods=['POST'])
def predicts_compare():
    """
        Fungsi ini digunakan untuk memproses image yang dimasukkan pengguna untuk menghasilkan prediksi kelas.
        Terdapat 4 variabel pada fungsi ini:
            choosenModelList    : variabel ini menyimpan data model yang dipilih dari halaman sebelumnya
            getImageFile        : variabel ini menyimpan data citra yang dipilih dari halaman sebelumnya
            relocationImageFile : variabel ini digunakan untuk menyimpan path citra yang diupload pengguna
            predictionResult    : variabel ini berisi hasil prediksi dalam bentuk list
            predictionTime      : variabel ini berisi lama waktu prediksi
    """
    choosenModelList                 = request.form.getlist('select_model')
    getImageFile                     = request.files["file"]
    relocationImageFile              = QUERY_UPLOAD_IMAGE+'temp.jpg'
    getImageFile.save(relocationImageFile)
    predictionResult, predictionTime = utils.predictsImage(MODEL_PATH, choosenModelList, relocationImageFile)  
    return render_template('/result_compare.html', labels = LABELS, probs = predictionResult, model = choosenModelList, run_time = predictionTime, img = relocationImageFile[7:])

if __name__ == "__main__": 
    app.run(debug=True, host='127.0.0.1', port=5000) 
