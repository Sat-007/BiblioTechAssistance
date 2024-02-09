import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Inputting the books!!
books_df = pd.read_csv('C:/Users/kssan/OneDrive/Desktop/Final_proj_scrape/data/books_scribd.csv')
#print(books_df)


# inputting the ratings csv file!!
ratings_df = pd.read_csv('C:/Users/kssan/OneDrive/Desktop/Final_proj_scrape/data/ratings-scribd.csv')
#print(ratings_df.head())


# Formatting and collection of the data
print('The Number of Books in Dataset', len(books_df))

#indexing to avoid indexing errors!!

books_df['List Index'] = books_df.index
#print(books_df.head())


# merging both ratings and books csv files based on bookid!!
merged_df = books_df.merge(ratings_df, on='book_id')
#print(merged_df)


# grouping up the users based on their userids!!
user_Group = merged_df.groupby('user_id')
#print(user_Group.head())


amountOfUsedUsers = 1000

# creating the training list!!
t_X = []

# For every user!!
for userID, curUser in user_Group:
    temp = [0]*len(books_df)       #temp list that stores ratings!!


    for num, movie in curUser.iterrows(): #for each book individually!!

        temp[movie['List Index']] = movie['rating']/5.0 

    # Appending the t_X list with the temporary list!!
    t_X.append(temp)

    if amountOfUsedUsers == 0:
        break
    amountOfUsedUsers -= 1
print(t_X)


# RBM!!!!!!!!!!!!!!!!!!!!!!!


# Setting model parameters!!
hiddenUnits = 50  
visibleUnits = len(books_df)  
vb = tf.placeholder(tf.float32, [visibleUnits])  # Number of unique books!!
hb = tf.placeholder(tf.float32, [hiddenUnits])  # Number of features the model is going to learn!!
W = tf.placeholder(tf.float32, [visibleUnits, hiddenUnits])  # Weight Matrix!!

# Phase 1: Input Processing!!
v0 = tf.placeholder("float", [None, visibleUnits])
_h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)  # Visible layer activation!!
h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))  # Gibb's Sampling!!

# Phase 2: Reconstruction!!
_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)  # Hidden layer activation!!
v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)



# Learning rate!!
alpha = 1.0

# Create the gradients!!
w_pos_grad = tf.matmul(tf.transpose(v0), h0)
w_neg_grad = tf.matmul(tf.transpose(v1), h1)

# Calculate the Contrastive Divergence to maximize!!
CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])

# Create methods to update the weights and biases!!
update_w = W + alpha * CD
update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

# Set the error function, here we use Mean Absolute Error Function!!
err = v0 - v1
err_sum = tf.reduce_mean(err*err)

# Current weight!!
cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Current visible unit biases!!
cur_vb = np.zeros([visibleUnits], np.float32)

# Current hidden unit biases!!
cur_hb = np.zeros([hiddenUnits], np.float32)

# Previous weight!!
prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)

# Previous visible unit biases!!
prv_vb = np.zeros([visibleUnits], np.float32)

# Previous hidden unit biases!!
prv_hb = np.zeros([hiddenUnits], np.float32)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Train by using EPOCH VALUES!!
epochs = 15
batchsize = 100
errors = []
for i in range(epochs):
    for start, end in zip(range(0, len(t_X), batchsize), range(batchsize, len(t_X), batchsize)):
        batch = t_X[start:end]
        cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        cur_hb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
        prv_w = cur_w
        prv_vb = cur_vb
        prv_hb = cur_hb
    errors.append(sess.run(err_sum, feed_dict={v0: t_X, W: cur_w, vb: cur_vb, hb: cur_hb}))
    print(errors[-1])
plt.plot(errors)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()  #PLOTS A GRAPH WITH A CURVE!!



# Select the input User!!
inputUser = [t_X[1]]

# Feeding in the User and Reconstructing the input!!
hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
feed = sess.run(hh0, feed_dict={v0: inputUser, W: prv_w, hb: prv_hb})
rec = sess.run(vv1, feed_dict={hh0: feed, W: prv_w, vb: prv_vb})

#Recommending the books!!

scored_books_df_50 = books_df
scored_books_df_50["Recommendation Score"] = rec[0]
sorted_score = scored_books_df_50.sort_values(["Recommendation Score"], ascending=False)
print(sorted_score.head(30))

sorted_score.to_csv('C:/Users/kssan/OneDrive/Desktop/Final_proj_scrape/data/output_result.csv')





'''

# Find the mock user's UserID from the data
print(merged_df.iloc[50])  # Result you get is UserID 150

# Find all books the mock user has watched before
books_df_50 = merged_df[merged_df['user_id'] == 150]
print(books_df_50.head())

""" Merge all books that our mock users has watched with predicted scores based on his historical data: """

# Merging books_df with ratings_df by MovieID
merged_df_50 = scored_books_df_50.merge(books_df_50, on='book_id', how='outer')

# Dropping unnecessary columns
merged_df_50 = merged_df_50.drop('List Index_y', axis=1).drop('user_id', axis=1)

# Sort and take a look at first 20 rows
print(merged_df_50.sort_values(['Recommendation Score'], ascending=False).head(20))

""" There are some books the user has not watched and has high score based on our model. So, we can recommend them. """
'''