import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 讀取訓練資料
with open("ReutersCorn-train.json", "r", encoding='UTF-8') as trf:
    training_set = json.load(trf)

# 讀取測試資料
with open("ReutersCorn-test.json", "r", encoding='UTF-8') as trf2:
    test_set = json.load(trf2)

# 提取訓練資料的文本和標籤
train_examples = [d['text'].replace('\n', ' ') for d in training_set]
train_labels = [d['class'] for d in training_set]

# 提取測試資料的文本和標籤
val_examples = [d['text'].replace('\n', ' ') for d in test_set]
val_labels = [d['class'] for d in test_set]

# 使用 LabelEncoder 對標籤進行編碼
le = LabelEncoder()
le.fit(train_labels + val_labels)
train_labels_enc = le.transform(train_labels)
val_labels_enc = le.transform(val_labels)

# 轉換為獨熱編碼形式
num_classes = len(le.classes_)
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels_enc, num_classes=num_classes)
val_labels_one_hot = tf.keras.utils.to_categorical(val_labels_enc, num_classes=num_classes)

# 使用 TensorFlow Hub 的嵌入層
embed_url = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embed_url, input_shape=[], dtype=tf.string, trainable=True)

# 建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=[], dtype=tf.string))
model.add(hub_layer)
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  # 使用 softmax 激活函數

model.summary()

# 編譯模型
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=opt,
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=[tf.metrics.CategoricalAccuracy()])

# 訓練模型
history = model.fit(np.array(train_examples),  # 將文本轉換為 NumPy 陣列
                    train_labels_one_hot,
                    epochs=40,
                    batch_size=512,
                    validation_data=(np.array(val_examples), val_labels_one_hot),
                    verbose=1)

# 繪製訓練過程中 loss 變化及 accuracy 變化
history_dict = history.history

acc = history_dict['categorical_accuracy']  # 使用 categorical_accuracy 來取代 accuracy
val_acc = history_dict['val_categorical_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc) + 1)

# 繪製 loss 變化
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # 清除圖表

# 繪製 accuracy 變化
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
