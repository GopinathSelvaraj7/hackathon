#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
import json
import re
import tensorflow as tf
import numpy as np
import more_itertools
import json
from neo4j import GraphDatabase, Driver
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# %matplotlib inline


# In[2]:


candidate = pd.read_excel("candidate_master_data.xlsx",sheet_name='Candidate Data')
candidate.to_csv("candidate_master.csv",index=None,header=True)
cand_df = pd.read_csv("candidate_master.csv")
# cand_df.head()


# In[3]:


def skill_parsing(skill):
    '''
    Function that extracts the skills from the Skill column 
    '''
    res = json.loads(skill,strict=False)
    skill_list=res['skills']
    
    skills = []
    
    for i in range (len(skill_list)):
        if skill_list[i]['skill_type'] == 'primary':
            skills.append(skill_list[i]['skill'])
        else:
            if skill_list[i]['skill_type'] == 'secondary':
                skills.append(skill_list[i]['skill'])
    return skills

can_skills ={}
skill_dict = dict(zip(cand_df['Candidate ID'],cand_df['Skills']))

for i in range (len(skill_dict)):
    for k,v in skill_dict.items():
        values=skill_dict[k]
        skill_set=skill_parsing(values)
        can_skills.update({k:skill_set})


# In[4]:


requisition = pd.read_excel("requisition_data.xlsx",sheet_name='Requisition Data')
requisition.to_csv("requisition.csv",index=None,header=True)
req_df = pd.read_csv("requisition.csv")
req_df = req_df[req_df['Skills'].notna()]
# req_df


# In[5]:


req_skills = []
for i in range (len(req_df)):
    text = req_df['Primary Skills (Extracted)'][i]
    req_skills.append(re.split('\n,|,',text))


# In[6]:


req_dict = dict(zip(req_df['Requisition ID'],req_skills))


# In[7]:


final_df=cand_df[['Candidate ID','Current Designation','Total Experience ( in Months)', 'Education__degree']]
final_df.drop_duplicates('Candidate ID',inplace=True)


# In[8]:


can_keys=can_skills.keys()
can_values=can_skills.values()


# In[9]:


new_list=[]
str1=' '
for i in range (len(can_values)):
    s=str1.join(list(can_values)[i])
    new_list.append(s)
    str1=' '

data={'Candidate ID':can_keys, 'Can_Skills':new_list}
skill_df = pd.DataFrame.from_dict(data)


# In[10]:


final_df=final_df.merge(skill_df,left_on=['Candidate ID'], right_on=['Candidate ID'],how='left')


# In[11]:


def query(req_skill, pri_skill, arg):
    query = """
            MATCH
            (baseSkill:Skill)-[eq_rel:EQUIVALENT]->(adj_Skill:AdjacentSkill)
            where baseSkill.name in $req_skill 
            and adj_Skill.name in $pri_skill
            return 'AdjacentSkillScore' as ScoreType ,SUM(toInteger(eq_rel.value)) as Scores
            UNION
            MATCH
            (baseSkill:Skill)-[spec_rel:SPECIFIC_TO]->(subSkill:Subskill)
            where baseSkill.name in $req_skill 
            and subSkill.name in $pri_skill
            return 'SubSkillScores' as ScoreType , SUM(toInteger(spec_rel.value)) as Scores
            UNION
            MATCH
            (baseSkill:Skill)-[impe_rel:IMPLEMENTED_WITH]->(compositeSkill:Composite)
            where baseSkill.name in $req_skill  
            and compositeSkill.name in $pri_skill
            return 'CompositeSkillScore' as ScoreType, SUM(toInteger(impe_rel.value)) as Scores

           """
    query_params = {
                "req_skill": ["javascript"],
                "pri_skill": pri_skill
                }

    with open('./settings.json') as f:
        settings = json.load(f)[arg]

    driver = GraphDatabase.driver(
            settings["neo4j_url"], 
            auth=(settings["neo4j_user"], settings["neo4j_password"]),
            encrypted=False)

    with driver.session() as session:
        data = session.run(query,**query_params).data()
    return data
    


# In[12]:


tot_score=0
score_dict = {}
for k,v in can_skills.items():
    pri_skill=v
    query_result=query(req_skill='python',pri_skill=pri_skill,arg='local')
    for i in range (len(query_result)):
        tot_score=tot_score+query_result[i]['Scores']
    score_dict[k]=tot_score
    tot_score=0


# In[13]:


scr_keys=score_dict.keys()
scr_values=score_dict.values()
data={'Candidate ID':scr_keys, 'Score':scr_values}


# In[14]:


score_df = pd.DataFrame.from_dict(data)


# In[15]:


final_df=final_df.merge(score_df,left_on=['Candidate ID'], right_on=['Candidate ID'],how='left')
final_df


# In[16]:


final_df.isna().sum()


# In[17]:


# Check for columns which arent numbers
for label, content in final_df.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# In[18]:


# Turn categorical variables into numbers and fill them
for label, content in final_df.items():
    if not pd.api.types.is_numeric_dtype(content):
#         final_df[label+"_is_missing"] = pd.isnull(content)
        final_df[label] = pd.Categorical(content).codes + 1


# In[19]:


# Find the columns that contains strings
for label, content in final_df.items():
    if pd.api.types.is_string_dtype(content):
        print(label)


# In[20]:


# This will change all string values to categories
for label, content in final_df.items():
    if pd.api.types.is_string_dtype(content):
        final_df[label] = content.astype("category").cat.as_ordered()


# In[21]:


final_df.isna().sum()


# In[22]:


final_df


# In[23]:


X = final_df[['Candidate ID','Current Designation','Total Experience ( in Months)','Education__degree','Can_Skills']]
y = final_df['Score']


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[25]:


X_train_d = X_train
X_train_d = X_train_d.drop(['Candidate ID'],axis=1)
X_test_d = X_test
X_test_d = X_test_d.drop(['Candidate ID'],axis=1)


# In[26]:


from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
scalar.fit(X_train_d)
X_trans = scalar.transform(X_train_d)


# In[27]:


def generate_model():
    model = Sequential([
        Dense(32,input_shape=(4,),activation='relu'),
        Dense(16, activation='relu'),
        Dense(12, activation='relu'),
        Dense(1, activation='relu'),
    ])

    return model


# In[28]:


generate_model()


# In[29]:


model = generate_model()
model.compile(loss='mean_squared_error',
                optimizer='adam',
                metrics=['accuracy'])


# In[30]:


output=model.fit(X_trans,y_train,batch_size=10, epochs=200)


# In[31]:


print(f"Accuracy: {round(output.history['accuracy'][-1]*100)}%")


# In[32]:


# %%time
# from sklearn.ensemble import RandomForestRegressor

# model_RFR = RandomForestRegressor(n_jobs=-1)

# model_RFR.fit(X_train,y_train)


# In[33]:


# from sklearn.metrics import mean_squared_log_error, mean_absolute_error

# def rmsle(y_test,y_preds):
#     return np.sqrt(mean_squared_log_error(y_test,y_preds))

# def show_scores(model_RFR):
#     train_preds = model_RFR.predict(X_train)
#     scores = {"Training MAE": mean_absolute_error(y_train,train_preds),
#               "Training RMSLE": rmsle(y_train,train_preds)}
    
#     return scores


# In[34]:


Xnew = scalar.transform(X_test_d)
# make a prediction
ynew = model.predict(Xnew)
# for i in range (len(ynew)):
#     print(ynew[i])
#     ynew[i]=ynew[i].round()
# show the inputs and predicted outputs
# for i in range(len(Xnew)):
# 	print("Candidate ID=%s, Predicted Score=%s" % (Xnew[i], ynew[i]))


# In[35]:


y_score = []
for i in range (len(ynew)):
    y_score.append(ynew[i][0])


# In[36]:


final_score=dict(zip(X_test['Candidate ID'],y_score))


# In[37]:


for k,v in final_score.items():
    print(f'The Candidate {k} has a score of {v}')


# In[38]:


data1 = {'Candidate ID':final_score.keys(), 'Score':final_score.values()}
score_dataframe = pd.DataFrame.from_dict(data1)


# In[39]:


score_dataframe['Rank']=score_dataframe['Score'].rank(method='max',ascending=False)
score_dataframe=score_dataframe.sort_values(by='Rank')
score_dataframe


# In[44]:


# save the model to disk
model.save('final_model.h5')


# In[ ]:




