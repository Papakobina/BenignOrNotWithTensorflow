#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import Dense, Dropout


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset = pd.read_csv("../Data/cancer_classification.csv")


# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# In[7]:


sns.countplot(x='benign_0__mal_1', data=dataset)


# In[13]:


dataset.corr()['benign_0__mal_1'].sort_values()[:-1].plot(kind='bar')


# In[15]:


sns.heatmap(dataset.corr())


# In[16]:


X = dataset.drop("benign_0__mal_1", axis=1).values
y = dataset['benign_0__mal_1'].values


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[20]:


scaler = MinMaxScaler()


# In[23]:


X_train = scaler.fit_transform(X_train)


# In[24]:


X_test = scaler.transform(X_test)


# In[33]:


model = Sequential()

model.add(Dense(30, activation="relu"))
model.add(Dense(15, activation="relu"))
# sigmoid because binarary classification
# sigmoid produces outputs values between 0 or 1
model.add(Dense(1, activation="sigmoid"))


model.compile(loss='binary_crossentropy', optimizer='adam')


# In[34]:


model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test))


# In[35]:


losses = pd.DataFrame(model.history.history)


# In[36]:


losses.plot()


# In[ ]:


# throiugh the graph we can see that we are overfitting 
# this is evident because the training loss decrwases why the val_loss increases


# In[39]:


model = Sequential()

model.add(Dense(30, activation="relu"))
model.add(Dense(15, activation="relu"))
# sigmoid because binarary classification
# sigmoid produces outputs values between 0 or 1
model.add(Dense(1, activation="sigmoid"))


model.compile(loss='binary_crossentropy', optimizer='adam')


# In[38]:


early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)


# In[ ]:





# In[40]:


model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test)
         ,callbacks=[early_stop])


# In[41]:


model_loss = pd.DataFrame(model.history.history)


# In[42]:


model_loss.plot()


# In[43]:


model = Sequential()

model.add(Dense(30, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(15, activation="relu"))
model.add(Dropout(0.5))
# sigmoid because binarary classification
# sigmoid produces outputs values between 0 or 1
model.add(Dense(1, activation="sigmoid"))


model.compile(loss='binary_crossentropy', optimizer='adam')


# In[44]:


model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test)
         ,callbacks=[early_stop])


# In[45]:


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()


# In[52]:


predictions = (model.predict(X_test) > 0.5).astype("int32")


# In[53]:


print(classification_report(y_test, predictions))


# In[54]:


print(confusion_matrix(y_test, predictions))

