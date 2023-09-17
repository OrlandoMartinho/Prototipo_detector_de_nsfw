from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

model = tf.keras.models.load_model('melhor_modelo.h5')

classes = ['Adulto', 'Nudez', 'Hentai', 'Normal', 'Sensual', 'Violento']

def fazer_previsao(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    previsao = model.predict(img)[0] * 100
    classe_predita = np.argmax(previsao)

    return classes[classe_predita], previsao

@app.route('/prever', methods=['POST'])
def prever():
  
    if 'imagem' not in request.files:
        return jsonify({'erro': 'Nenhuma imagem encontrada'})

    imagem = request.files['imagem']
    image_path = 'temp.jpg'  

    imagem.save(image_path)

    classe_predita, previsoes = fazer_previsao(image_path)

    return jsonify({
        'classe_predita': classe_predita,
        'confiancas': {classe: float(percentagem) for classe, percentagem in zip(classes, previsoes)}
    })

if __name__ == '__main__':
    app.run(debug=True)
