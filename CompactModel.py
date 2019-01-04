#coding=utf-8
import os
import tensorflow as tf
import cv2
import numpy as np
from keras.models import Model  
from keras.layers import Dense,Flatten,Input  
from keras.layers import Conv2D,MaxPooling2D,Dropout,BatchNormalization  
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img 

import keras.backend.tensorflow_backend as KTF 
config = tf.ConfigProto() 
config.gpu_options.allow_growth=True #不全部占满显存, 按需分配 
sess = tf.Session(config=config) 
KTF.set_session(sess) 


# def IlluminationAug(im):
# 	aug = []
# 	aug.append(cv2.euqalizeHist(im))

label_path = "/home/DataSet/RAF/BasicEmotion/EmoLabel/list_patition_label.txt"
img_path = "/home/DataSet/RAF/BasicEmotion/image/aligned/images"

x_train = []
y_train = []
x_test = []
y_test = []
with open(label_path) as f:
	for i in f:
		i = i.strip()
		name,cls = i.split(" ")
		a,b = name.split(".")
		a +="_aligned."
		name = a+b
		img = load_img(os.path.join(img_path,name))
		#use the gray
		img = img.convert("L")
		img = img_to_array(img)
		if i.split("_")[0]=="train":
			x_train.append(img)
			#x_train.append(cv2.equalizeHist(img))
			#y_train.append(int(cls)-1)
			y_train.append(int(cls)-1)
		else:
			x_test.append(img)
			#x_test.append(cv2.equalizeHist(img))
			#y_test.append(int(cls)-1)
			y_test.append(int(cls)-1)

x_train = np.array(x_train).reshape(-1,100,100,1)
x_test = np.array(x_test).reshape(-1,100,100,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input = Input((100,100,1))
x = Conv2D(16,(5,5),padding='same',activation='relu')(input)
x = BatchNormalization()(x)
x = Conv2D(16,(5,5),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Conv2D(32,(5,5),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
x = Conv2D(32,(5,5),padding='same',activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
x = Dense(64,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)
x = Dense(64,activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)
x = Dense(7,activation='softmax')(x)
model = Model(inputs=input,outputs=x)
model.summary()

#对训练进行监控，随时保存测试集上loss最小的
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD

#当评价指标不在提升时，减少学习率
#reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
checkpoint1 = ModelCheckpoint(filepath="lrdefault_gray_best_loss_model.h5", 
	monitor='val_loss',
	verbose=1,
	save_best_only='True',
	mode='auto',
	period=1)

checkpoint2 = ModelCheckpoint(filepath="lrdefault_gray_best_acc_model.h5", 
	monitor='val_acc',
	verbose=1,
	save_best_only='True',
	mode='auto',
	period=1)

callback_lists=[checkpoint1,checkpoint2]

#sgd = SGD(lr=0.001, momentum=0.9, decay=0.001, nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer="sgd",metrics=['accuracy'])
hist = model.fit(x_train,y_train, batch_size=32, epochs=50,validation_data=(x_test,y_test), verbose=1, shuffle=True,callbacks=callback_lists)

###画ACC曲线部分########
from matplotlib import pyplot as plt
val_max_indx=np.argmax(hist.history['val_acc'])#max value index
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('harbor-model-accuracy')
#point the max val_acc#####
plt.plot(val_max_indx,np.max(hist.history['val_acc']),'ks')
show_max='['+str(val_max_indx)+', '+str(round(np.max(hist.history['val_acc']),4))+']'
plt.annotate(show_max,xytext=(val_max_indx,np.max(hist.history['val_acc'])),xy=(val_max_indx,np.max(hist.history['val_acc'])))
###########################
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig('./accuracyVSepoch2.png')
plt.show()

val_min_indx=np.argmin(hist.history['val_loss'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('harbor-model-loss')
#point the min val_loss#####
plt.plot(val_min_indx,np.min(hist.history['val_loss']),'ks')
show_min='['+str(val_min_indx)+', '+str(round(np.min(hist.history['val_loss']),4))+']'
plt.annotate(show_min,xytext=(val_min_indx,np.min(hist.history['val_loss'])),xy=(val_min_indx,np.min(hist.history['val_loss'])))
############################
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.tight_layout()
plt.savefig('./lossVSepoch2.png')
plt.show()


#######save the train log dict###
f = open('log.txt','w')
f.write(str(hist.history))
f.close()
###read the dict from txt file###
f = open('log.txt','r')
a = f.read()
dict_name = eval(a)
f.close()
#################################


#####predict and draw the prob bar#########
test_path = "/home/DataSet/RAF/mytest"
fns = os.listdir(test_path)
for fn in fns:
	img = load_img(os.path.join(test_path,fn))
	img = img.resize((100,100))
	img = np.int16(img_to_array(img))
	plt.figure(figsize=(15,10))
	plt.subplot(121)
	plt.imshow(img)
	img = img.convert("L")
	img = img.reshape((1,)+img.shape)
	pred = model.predict(img)
	#plt.title(label[pred])
	plt.subplot(122)
	plt.bar(["Surprise","Fear","Disgust","Happiness","Sadness","Anger","Neutral"],pred[0])
	plt.show()


