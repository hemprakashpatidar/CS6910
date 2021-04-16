class MyCNNPre(object):
  def __init__(self,model_name):
     self.model_name=model_name

  def train(self):
    if(self.model_name=='ResNet50'):
      model_a = ResNet50(weights='imagenet')
      bi=model_a.layers[0].input
      bo=model_a.layers[-2].output
      fo=layers.Dense(10)(bo)
      model=keras.Model(inputs=bi,outputs=fo)
      opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
      model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


      model.fit(X_train,y_train, epochs=10,validation_data=(X_val, y_val))
      
    if(self.model_name=='InceptionV3'):
      model_a = InceptionV3(weights='imagenet')
      bi=model_a.layers[0].input
      bo=model_a.layers[-2].output
      fo=layers.Dense(10)(bo)
      model=keras.Model(inputs=bi,outputs=fo)
      opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
      model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


      model.fit(X_traini,y_traini, epochs=10,validation_data=(X_vali, y_vali))
    if(self.model_name=='Xception'):
      model_a = Xception(weights='imagenet')
      bi=model_a.layers[0].input
      bo=model_a.layers[-2].output
      fo=layers.Dense(10)(bo)
      model=keras.Model(inputs=bi,outputs=fo)
      opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
      model.compile(optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


      model.fit(X_traini,y_traini, epochs=10,validation_data=(X_vali, y_vali))
      
