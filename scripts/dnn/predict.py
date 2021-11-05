
import tensorflow as tf

model = tf.keras.models.load_model('linear_model')

for y in [70, 64, 58, 52, 46, 40, 34, 28, 22, 16, 10]:
    print(y, end='\t')
    for x in [600, 800, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
        #            RPM, dRPM, MAP, dMAP, AFR,  MAT, CLT
        features = [[x,   0,    y,   0,    14.7, 1,   80]]
        p = model.predict(features)[0][0]
        print(round((p-1)/12.2 * 1e4 / y), end='\t')
    print()
