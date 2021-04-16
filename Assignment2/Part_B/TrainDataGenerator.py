from tensorflow.keras.preprocessing.image import ImageDataGenerator
 
train_generator = ImageDataGenerator(rescale=1./255,validation_split=0.1).flow_from_directory(
                                                    '/content/drive/MyDrive/Colab Notebooks/inaturalist_12K/train',
                                                     target_size=(224, 224),
                                                     batch_size=3000,
                                                     class_mode='binary',
                                                     subset='training',
                                                     seed=123)
val_generator = ImageDataGenerator(rescale=1./255,validation_split=0.1).flow_from_directory(
                                                    '/content/drive/MyDrive/Colab Notebooks/inaturalist_12K/train',
                                                     target_size=(224, 224),
                                                     batch_size=200,
                                                     class_mode='binary',
                                                     subset='validation',
                                                     seed=123)
