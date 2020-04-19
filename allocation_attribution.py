#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("attribution_allocation_student_data.csv")

df.head()


# In[49]:


# channel spend
channel_spend = pd.read_csv("channel_spend_student_data.csv")
channel_spend


# In[31]:


df.convert_TF.value_counts()


# In[16]:


df_exposure = df[['touch_1', 'touch_2', 'touch_3', 'touch_4', 'touch_5']]

df_exposure['convert'] = df['convert_TF'].astype('int')

df_exposure['id'] = np.arange(len(df_exposure))

df_exposure = pd.melt(df_exposure, id_vars=['id', 'convert'], value_vars=['touch_1', 'touch_2', 'touch_3', 'touch_4', 'touch_5'],
       var_name = 'touch', value_name ='channel').sort_values(['id'])

df_exposure = df_exposure.dropna()

df_exposure = df_exposure.groupby('channel').convert.agg(['sum','count'])

df_exposure['convert_pct'] = df_exposure['sum'] / df_exposure['count']

df_exposure


# Suggests people aren't converting from Referral or Paid Search. Shows that the ads aren't reaching as many people.

# ## Part 1: Attribution

# In[3]:


df_attribution = df[df['convert_TF'] == True].drop(['convert_TF'], axis = 1)

df_attribution['id'] = np.arange(len(df_attribution))

df_attribution = pd.melt(df_attribution, id_vars=['id','tier'], value_vars=['touch_1', 'touch_2', 'touch_3', 'touch_4', 'touch_5'],
       var_name = 'touch', value_name ='channel').sort_values(['id','touch'])

#drop nas
df_attribution =df_attribution.dropna(axis=0)

df_attribution['touch'] = df_attribution['touch'].str[-1].astype(int)

df_attribution['max_touch'] = df_attribution.groupby('id').touch.transform('max')

df_attribution['max_ind'] = np.where(df_attribution.touch == df_attribution.max_touch, 1, 0)

df_attribution.head()


# In[4]:


print("How many channels on avg exposed to: %.2f" % df_attribution.groupby('id').touch.agg('max').agg('mean'))


# In[5]:


# last interaction
df_lastint = df_attribution[df_attribution['max_ind'] == 1].channel.value_counts().reset_index().sort_values('index')

df_lastint.columns = ['channel', 'lastint_cnt']

df_lastint


# In[6]:


# last non-direct
df_attribution_nodirect = df_attribution[df_attribution['channel'] != 'direct']

df_attribution_nodirect['max_ind'] = np.where(df_attribution_nodirect.touch == df_attribution_nodirect.groupby('id').touch.transform('max'), 1, 0)

df_lastndir = df_attribution_nodirect[df_attribution_nodirect['max_ind'] == 1].channel.value_counts().reset_index().sort_values('index')

df_lastndir.columns = ['channel', 'lastndir_cnt']

df_lastndir


# In[7]:


# first interaction
df_firstint = df_attribution[df_attribution['touch'] == 1].channel.value_counts().reset_index().sort_values('index')

df_firstint.columns = ['channel', 'firstint_cnt']

df_firstint


# In[8]:


# position based

def attribution(row):
    if row['max_touch'] == 1:
        val = 1
    elif row['max_touch'] == 2:
        val = 0.5
    elif row['touch'] == 1 or row['max_ind'] == 1:
        val = 0.4
    else:
        val = 0.2 / (int(row['max_touch']) - 2)
        
    return val
    
df_attribution['posbas_cnt'] = df_attribution.apply(attribution, axis=1)

#check if I did it correctly
df_attribution['check'] = df_attribution.groupby('id').posbas_cnt.transform(sum)
print("Check error: %d" % (sum(df_attribution['check'] != 1)))

df_posbas = df_attribution.groupby('channel').posbas_cnt.agg('sum').reset_index()

df_posbas


# In[59]:


# join together

conv_attribution = df_lastint.merge(df_lastndir, on ='channel', how ='left').merge(df_firstint, on ='channel').merge(df_posbas, on ='channel')

conv_attribution = conv_attribution.fillna(0)

conv_attribution


# In[64]:


cac = conv_attribution.copy()

cac.iloc[:,1:5] = 300 / cac.iloc[:,1:5]

cac


# **Comments on CAC:**
# * A high proportion of conversions are coming from Organic Search, contrasting the performance of Paid Search. Rather than pulling money from Paid Search, it is recommended to review keyword strategy.
# 
# * Majority of conversions were from non-paid channels (Organic Search and Direct), regardless of method. While there’s no CAC associated with the channel, these conversions may be due to other channels. This suggests that there may be gaps in our ad tracking. It is recommended to review the current tracking between channels.

# ## Part 2: Allocation

# In[76]:


def marginalcac(method):
    if method == "lastint":
        df_mcac = df_attribution[df_attribution['max_ind'] == 1].groupby(['tier', 'channel']).id.agg('count').reset_index()
    elif method == "lastint_ndir":
        df_mcac = df_attribution_nodirect[df_attribution_nodirect['max_ind'] == 1].groupby(['tier','channel']).id.agg('count').reset_index()
    elif method == "firstint":
        df_mcac = df_attribution[df_attribution['touch'] == 1].groupby(['tier','channel']).id.agg('count').reset_index()
    elif method == "posbas":
        df_mcac = df_attribution.groupby(['tier','channel']).posbas_cnt.agg('sum').reset_index()
        df_mcac.columns.values[2] = 'id'
    else:
        df = None 
    
    df_mcac['channel_spend'] = df_mcac['tier'] * 50
    df_mcac['channel_spend'] = np.where(df_mcac['channel'].isin(['direct','organic_search']), 0, df_mcac['channel_spend'])
    df_mcac['marginal_ac'] = df_mcac['id'] - df_mcac.groupby(['channel'])['id'].shift(1, fill_value = 0)
    df_mcac['marginal_spend'] = df_mcac['channel_spend'] - df_mcac.groupby(['channel'])['channel_spend'].shift(1, fill_value = 0)
    df_mcac['marginal_cac'] = df_mcac['marginal_spend'] / df_mcac['marginal_ac']
    df_mcac = df_mcac.pivot_table(index = 'tier', columns = 'channel', values ='marginal_cac', aggfunc ='sum')
    
    return df_mcac


# In[77]:


marginalcac("lastint")


# In[71]:


marginalcac("lastint_ndir")


# In[72]:


marginalcac("firstint")


# In[78]:


marginalcac("posbas")

