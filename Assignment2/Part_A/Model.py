class MyCNN(object):
  def __init__(self,Dropout,Layer1,Layer2,Layer3,Layer4,Layer5,bn,augmentation):
     self.Dropout=Dropout
     self.Layer1=Layer1
     self.Layer2=Layer2
     self.Layer3=Layer3
     self.Layer4=Layer4
     self.Layer5=Layer5
     self.bn=bn
     self.augmentation=augmentation
  def train(self):
    input_shape=(224,224,3)
   
    Layer=[self.Layer1,self.Layer2,self.Layer3,self.Layer4,self.Layer5]
    
    model =models.Sequential()
    model.add(layers.Conv2D(Layer[0], (3, 3), input_shape=input_shape))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    if(self.bn=='Yes'):
      model.add(layers.BatchNormalization())


    model.add(layers.Conv2D(Layer[1], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    if(self.bn=='Yes'):
      model.add(layers.BatchNormalization())


    model.add(layers.Conv2D(Layer[2], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    if(self.bn=='Yes'):
      model.add(layers.BatchNormalization())


    model.add(layers.Conv2D(Layer[3], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    if(self.bn=='Yes'):
      model.add(layers.BatchNormalization())


    model.add(layers.Conv2D(Layer[4], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(self.Dropout))

    model.add(layers.Dense(10,activation='softmax'))  
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
  
    if(self.augmentation=='Yes'):
      model.fit(X_at,y_at, epochs=15,validation_data=(x_av, y_av),callbacks=[WandbCallback(validation_data=(x_av, y_av))])
    if(self.augmentation=='No'):
      model.fit(X_train,y_train, epochs=15,validation_data=(X_val, y_val),callbacks=[WandbCallback(validation_data=(X_val, y_val))])
