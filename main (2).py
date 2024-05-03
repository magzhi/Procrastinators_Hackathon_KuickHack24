from flask import Flask, request, render_template
from ultralytics import YOLO

from final_check_srb import *
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'photo' not in request.files:
        return 'No file part'
    file = request.files['photo']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = 'files/' + file.filename  # Путь к сохраненному файлу
        file.save(file_path)  # Сохранение файла на сервере
        match = main1(file_path)  # Передача пути к файлу в функцию main()
        if match:
            model = YOLO(r"C:\Users\magzh\OneDrive\Desktop\HackathonDocs\FULL\runs\classify\train4\weights\best.pt")
            results = model.predict(file_path)
            if results[0].probs.top1 == 0:
                message = "This passport is legal"
            else:
                message = "This passport is fake as found by YOLO"
        else:
            message = "This passport is fake as found by OCR"
        return render_template('index.html', message=message)


if __name__ == '__main__':
    app.run(debug=True)
