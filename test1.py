
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.get_default_graph()
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('test1.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1)
    x_train = train_datagen.flow_from_directory(r'C:\Users\Washifa\Desktop\D\trainset',target_size=(64,64),batch_size=32,class_mode='binary')
    x_test = test_datagen.flow_from_directory(r'C:\Users\Washifa\Desktop\D\testset',target_size=(64,64),batch_size=32,class_mode='binary')
    
    print(x_train.class_indices)
    import tensorflow as tf
    from tensorflow.python.keras.layers import Dense
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Convolution2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Flatten
    model = Sequential()
    model.add(Convolution2D(32,(3,3),input_shape = (64,64,3)))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dense(units = 128,kernel_initializer = 'uniform'))
    model.add(Dense(units = 1,kernel_initializer = 'uniform'))
    model.compile(loss='binary_crossentropy',optimizer = "adam",metrics = ["accuracy"])
    model.fit_generator(x_train,validation_data=x_test, steps_per_epoch=10)
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads\image',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        test_datagen = ImageDataGenerator(rescale = 1./255)
        vals = ['Igneous Rock type','Metamorphic Rock types','Sedimentary rock type']
        print(vals)
        test_generator = test_datagen.flow_from_directory(r'C:\Users\Washifa\Desktop\test\uploads',target_size =(64,64),class_mode ='categorical',batch_size = 32)
        print(test_generator)
        pred = model.predict_generator(test_generator)
        print(pred)
        pred=pred[0][0]
        y=str(vals[np.argmax(pred)])
        os.remove(filepath)
    return y

if __name__ == '__main__':
    app.run(debug =  False)
        
        
        
    
    
    
