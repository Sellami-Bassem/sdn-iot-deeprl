import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras import backend as k
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import LeakyReLU
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


subdir = "/home/lora/sdn-iot-task-scheduling/loss/"

r = lambda: np.random.randint(1, 10) 

tasks = np.array([[r(),r(),r()] for _ in range(1000)])
view_1=tasks
print("shape of Tasks:", tasks.shape)

nodes=np.array([[r(),r(),r()] for _ in range(1000)])
view_2=nodes

print("shape of Nodes:", nodes.shape)

encod_dim = [3, 5]
for encoding_dim in encod_dim:

    # Normalize the second view 1
    # view_1 = normalize(view_tfmri)
    scaler = MinMaxScaler()
    view_1 = scaler.fit_transform(view_1)

    # Second view: Normalize 
    scaler = MinMaxScaler()
    view_2 = scaler.fit_transform(view_2)
   
    """"
    Deep Multimodal Autoencoder (DMAE)
    """""

     # Inputs Shape
    input_view_1 = Input(shape=(view_1[0].shape))
    input_view_2 = Input(shape=(view_2[0].shape))


    # Encoder Model

    # First view
    encoded_1=Dense(encoding_dim*4, activation='relu', name="a")(input_view_1) # Layer 1, View 1
    encoded_1=Dense(encoding_dim*2, activation='relu', name="b")(encoded_1) # Layer 2, View 1
    print("encoded 1 shape", encoded_1.shape)

    # Second view

    encoded_2=Dense(encoding_dim*4, activation='relu', name="c")(input_view_2) # Layer 1, View 2
    encoded_2=Dense(encoding_dim*2, activation='relu', name="d")(encoded_2) # Layer 2, View 2
    print("encoded 2 shape", encoded_2.shape)

    # Shared representation with concatenation

    shared_layer = concatenate([encoded_1, encoded_2]) # Layer 3: Bottelneck layer
    print("Shared Layer", shared_layer.shape)
    output_shared_layer=Dense(encoding_dim, activation='relu', name="e")(shared_layer)
    print("Output Shared Layer", output_shared_layer.shape)


    # Decoder Model

    # Fisrt view
    #decoded_1=Dense(encoding_dim, activation='relu')(shared_layer)
    decoded_1=Dense(encoding_dim*2, activation='relu', name="f")(output_shared_layer)
    decoded_1=Dense(encoding_dim*4, activation='relu', name="g")(decoded_1)
    decoded_1=Dense(view_1[0].shape[0],  activation='linear', name="dec_1")(decoded_1)
    print("decoded_1", decoded_1.shape)

    # Second view
    #decoded_rsfmri=Dense(encoding_dim, activation='relu')(shared_layer)
    decoded_2=Dense(encoding_dim*2, activation='relu', name="i")(output_shared_layer)
    decoded_2=Dense(encoding_dim*4, activation='relu', name="j")(decoded_2)
    decoded_2=Dense(view_2[0].shape[0], activation='linear', name="dec_2")(decoded_2)
    print("decoded_2", decoded_2.shape)

    # This model maps an input to its reconstruction
    dmae= Model(inputs=[input_view_1, input_view_2], outputs=[decoded_1, decoded_2])

    dmae.summary()
    print(len(dmae.layers))
    dictionary = {v.name: i for i, v in enumerate(dmae.layers)}
    print(dictionary)
    # Seperate Encoder Model

    # this model maps an inputs to its encoded representation
    # First view
    encoder_1= Model(input_view_1, encoded_1)
    encoder_1.summary()
    # Second view
    encoder_2= Model(input_view_2, encoded_2)
    encoder_2.summary()
    # This model maps a two inputs to its bottelneck layer (shared layer)
    encoder_shared_layer= Model(inputs=[input_view_1, input_view_2], outputs=output_shared_layer)
    encoder_shared_layer.summary()


    # Separate Decoder model

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim, ))
    # retrieve the layers of the autoencoder model
    # First view
    decoder_1_layer1 = dmae.layers[-6] # Index of the first layer (after bottelneck layer)
    decoder_1_layer2 = dmae.layers[-4]
    decoder_1_layer3 = dmae.layers[-2]
    # create the decoder model
    decoder_1 = Model(encoded_input, decoder_1_layer3(decoder_1_layer2(decoder_1_layer1(encoded_input))))
    decoder_1.summary()

    # Second view

    decoder_2_layer1 = dmae.layers[-5]
    decoder_2_layer2 = dmae.layers[-3]
    decoder_2_layer3 = dmae.layers[-1]
    # create the decoder model
    decoder_2 = Model(encoded_input, decoder_2_layer3(decoder_2_layer2(decoder_2_layer1(encoded_input))))
    decoder_2.summary()


    # Train our model
    batch_size=8000
    epochs=30

    #optimizer='sgd', loss='mse'


    dmae.compile(optimizer='sgd', loss='mse')
    log_dir = './log/'
    tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    history = dmae.fit([view_1, view_2], [view_1, view_2],
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=[tbCallBack])
    # list all data in history
    print(history.history.keys())
    #fig = plt.gcf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss (MSE)')
    plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    plt.legend(['train'], loc='upper left')
    #plt.show()
    plt.savefig('MSE_dim_{}'.format(encoding_dim))
    plt.close()

    # Save the results weights
    encoder_shared_layer.save('encoder_weights_dim_{}.h5'.format(encoding_dim))
    encoder_1.save('encoder_1_weights_dim_{}.h5'.format(encoding_dim))
    encoder_2.save('encoder_2_weights_dim_{}.h5'.format(encoding_dim))
    decoder_1.save('decoder_1_weights_dim_{}.h5'.format(encoding_dim))
    decoder_2.save('ecoder_2_weights_dim_{}.h5'.format(encoding_dim))
    dmae.save('dmae_weights_{}.h5'.format(encoding_dim))
    #
