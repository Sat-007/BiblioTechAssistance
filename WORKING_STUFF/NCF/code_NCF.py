import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')
#%matplotlib inline

import tensorflow.keras as tf



ratings_df = pd.read_csv("C:/Users/kssan/OneDrive/Desktop/FINALPROJJ/WORKING_STUFF/NCF/ratings_ncf.csv",encoding='cp1252')
books_df = pd.read_csv("C:/Users/kssan/OneDrive/Desktop/FINALPROJJ/WORKING_STUFF/NCF/books_ncf.csv",encoding='utf-8')




print(ratings_df.shape)
print(ratings_df.user_id.nunique())
print(ratings_df.book_id.nunique())
ratings_df.isna().sum()



from sklearn.model_selection import train_test_split

Xtrain, Xtest = train_test_split(ratings_df, test_size=0.2, random_state=1)
print(f"Shape of train data: {Xtrain.shape}")
print(f"Shape of test data: {Xtest.shape}")


#-------------------------------CREATING A NEURAL NETWORK--------------------------------------------------


nbook_id = ratings_df.book_id.nunique()
nuser_id = ratings_df.user_id.nunique()


#Book input network
input_books = tf.layers.Input(shape=[1])
embed_books = tf.layers.Embedding(nbook_id + 1,15)(input_books)
books_out = tf.layers.Flatten()(embed_books)

#user input network
input_users = tf.layers.Input(shape=[1])
embed_users = tf.layers.Embedding(nuser_id + 1,15)(input_users)
users_out = tf.layers.Flatten()(embed_users)

conc_layer = tf.layers.Concatenate()([books_out, users_out])
x = tf.layers.Dense(128, activation='relu')(conc_layer)
x_out = x = tf.layers.Dense(1, activation='relu')(x)

model = tf.Model([input_books, input_users], x_out)

opt = tf.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mean_squared_error')




model.summary()





hist = model.fit([Xtrain.book_id, Xtrain.user_id], Xtrain.rating, batch_size=64, epochs=3, verbose=1,validation_data=([Xtest.book_id, Xtest.user_id], Xtest.rating))




hist.history


train_loss = hist.history['loss']
val_loss = hist.history['val_loss']





plt.plot(train_loss, color='b', label='Train Loss')
plt.title("Train Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Training Error")
plt.legend()
plt.savefig("loss.png")
#plt.show()

plt.plot(train_loss, color='r', label='Train Loss')
plt.plot(val_loss, color='b', label='Validation Loss')
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.savefig("loss.png")
plt.show()

# plt.plot(val_loss, color='r', label='Validation Loss')
# plt.xlabel("Epochs")
# plt.ylabel(" Validation Error")
# plt.title("Train and Validation Loss Curve")
# plt.legend()
# plt.savefig("loss.png")
#plt.show()





model.save('model')

#-------------------------------------------Embeddings---------------------------------------------------



books_df_copy = books_df.copy()
books_df_copy = books_df_copy.set_index("book_id")
books_df_copy.head(2)




b_id =list(ratings_df.book_id.unique())
b_id.remove(10000)




book_arr = np.array(b_id)
user = np.array([53424 for i in range(len(b_id))])
pred = model.predict([book_arr, user])


pred




pred = pred.reshape(-1)
pred_ids = (-pred).argsort()[0:50]
pred_ids



print(ratings_df.user_id.nunique())




print(books_df.iloc[pred_ids])



books_df.iloc[pred_ids].to_csv("C:/Users/kssan/OneDrive/Desktop/FINALPROJJ/WORKING_STUFF/NCF/results_ncf_3.csv")#columns=["image_url","authors","title","genres"])



















