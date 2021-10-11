#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Setting the Environment
import sagemaker

sess = sagemaker.Session()
bucket = sess.default_bucket()
prefix = "sagemaker/DEMO-xgboost-churn"

# Define IAM role
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()


# In[2]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import sys
import time
import json
from IPython.display import display
from time import strftime, gmtime
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer


# In[3]:


# Using s3 Buckets
get_ipython().system('aws s3 cp s3://sagemaker-sample-files/datasets/tabular/synthetic/churn.txt ./')


# In[4]:


#Creating the Dataframe
churn = pd.read_csv("./churn.txt")
pd.set_option("display.max_columns", 500)
churn


# In[5]:


len(churn.columns)


# In[6]:


# Analyzing the data
for column in churn.select_dtypes(include=["object"]).columns:
    display(pd.crosstab(index=churn[column], columns="%Observations", normalize="columns"))
    
display(churn.describe())
get_ipython().magic('matplotlib inline')
hist = churn.hist(bins=30, sharey=True, figsize=(10,10))


# In[7]:


# Droping the phone column and converting area code to type object
churn = churn.drop('Phone', axis=1)
churn["Area Code"] = churn['Area Code'].astype(object)


# In[8]:


#Explore the relationship between target attribute and the features
for column in churn.select_dtypes(include = ['object']).columns:
    if column != "Churn?":
        display(pd.crosstab(index = churn[column], columns = churn["Churn?"], normalize="columns"))
        
for column in churn.select_dtypes(exclude=['object']).columns:
    print (column)
    hist = churn[[column,"Churn?"]].hist(by="Churn?", bins=30)
    plt.show()


# In[9]:


display(churn.corr())
pd.plotting.scatter_matrix(churn, figsize=(12,12))
plt.show()


# In[10]:


churn = churn.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)


# In[11]:


#Preparing the Model Training Set
model_data = pd.get_dummies(churn)
model_data


# In[12]:


model_data = pd.concat([model_data["Churn?_True."], model_data.drop(['Churn?_False.','Churn?_True.'], axis=1)],axis=1)


# In[13]:


train_data, validation_data, test_data = np.split(
    model_data.sample(frac=1, random_state=1729),
    [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
)


# In[14]:


train_data.to_csv("train.csv", header=False, index=False)
validation_data.to_csv("validation.csv", header=False, index=False)


# In[15]:


boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "train/train.csv")
).upload_file("train.csv")
boto3.Session().resource("s3").Bucket(bucket).Object(
    os.path.join(prefix, "validation/validation.csv")
).upload_file("validation.csv")


# In[16]:


# Training the model
container = sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, "latest")
display(container)


# In[17]:


s3_input_train = TrainingInput(
    s3_data="s3://{}/{}/train".format(bucket, prefix), content_type="csv"
)
s3_input_validation = TrainingInput(
    s3_data="s3://{}/{}/validation/".format(bucket, prefix), content_type="csv"
)


# In[18]:


sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path="s3://{}/{}/output".format(bucket, prefix),
    sagemaker_session=sess,
)
xgb.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.8,
    silent=0,
    objective="binary:logistic",
    num_round=100,
)

xgb.fit({"train": s3_input_train, "validation": s3_input_validation})


# In[19]:


"""
#Logistic Regression Model
container_logistic = sagemaker.image_uris.retrieve("linear-learner", boto3.Session().region_name, "latest")
display(container)
"""


# In[20]:


"""
sess = sagemaker.Session()

ll = sagemaker.estimator.Estimator(
    container_logistic,
    role,
    instance_count=1,
    instance_type="ml.m4.xlarge",
    output_path="s3://{}/{}/output".format(bucket, prefix),
    sagemaker_session=sess,
)

ll.set_hyperparameters(
   feature_dim = 74,
        mini_batch_size = 100,
        predictor_type = 'binary_classifier',
        epochs = 10,
        num_models = 32,
        loss = absolute_loss,
)

ll.fit({"train": s3_input_train, "validation": s3_input_validation})
"""


# In[24]:


#Host
xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge', serializer=CSVSerializer())


# In[25]:


test_data.head(1)


# In[36]:


xgb_predictor.predict(np.array(test_data)[1,1:]).decode('utf-8')


# In[37]:


predictions=[]
for i in range(0,test_data.shape[0]):
    predictions.append(xgb_predictor.predict(np.array(test_data)[i,1:]).decode('utf-8'))


# In[47]:


predictions = [float(i) for i in predictions]


# In[48]:


pd.crosstab(test_data['Churn?_True.'],np.round(predictions))


# In[53]:


#Batch Transform
transformer = sagemaker.transformer.Transformer(
base_transform_job_name = 'Batch-Transform',
model_name = 'xgboost-2021-10-11-07-30-13-916',
instance_count=1,
instance_type='ml.c4.xlarge',
output_path='s3://sagemaker-us-east-2-601670882059/batchoutput')

#start a transform job
transformer.transform('s3://sagemaker-us-east-2-601670882059/sagemaker/DEMO-xgboost-churn/train', content_type = 'text/csv', split_type='Line')


# In[51]:


bucket


# In[52]:


s3_input_train.config


# In[56]:


results = pd.crosstab(test_data['Churn?_True.'],np.round(predictions)).values


# In[59]:


Accuracy = (results[0][0] + results [1][1]) / sum(results.reshape(4,))


# In[60]:


Accuracy


# In[61]:


xgb_predictor.delete_endpoint()

