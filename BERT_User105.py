#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


from platform import python_version
import sys
sys.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET #Parse and read XML data
import tarfile #read from tarfile instead of extracting all data
import xml.etree.ElementTree as ET
from collections import Counter
import trec
import pprint as pp
import pickle
import site
site.getsitepackages()
import tarfile
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
# Matplotlib for additional customization
#import pyplot as plt
#%matplotlib inline
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import trec
import pprint as pp
import pickle
from collections import defaultdict #I'm importing this library so we have a code with less lists!!


# ## Defining some functions

# In[2]:


import re

def getGender(text):
    male = [" male ", " boy ", " man "]
    female = [" female "," girl ", " woman "]
    res_male = False
    res_female = False
    res_male = any(ele in text for ele in male)
    res_female = any(ele2 in text for ele2 in female)
    if (res_male):
        return "Male"
    if (res_female):
        return "Female"
    if not res_male and not res_female:
        return "NA"
    
def getAge(text):
    x = re.search("((\d{1,2})\syear\sold)|((\d{1,2})[-]year[-]old)|((\d{1,2})\syo)|((\d{1,2})[-]year\sold)", text)
    if(not x):
        y = re.search("((\d{1,2})\smonth\sold)|((\d{1,2})[-]month[-]old)|((\d{1,2})\syo)|((\d{1,2})[-]month\sold)", text)
        if(not y):
            age = "NA"
        else:
            age = 0
    else:
        age = (x.group(2) or x.group(4) or x.group(6) or x.group(8))
    return age


def sum_numbers(numbers):
    total = 0
    for number in numbers:
        total += number
    return total
    


# ## Extracting information about the patients

# In[3]:


import pickle

Queries = "topics-2014_2015-summary.topics"
Qrels = "qrels-clinical_trials.txt"
with open(Queries, 'r') as queries_reader:
    txt = queries_reader.read()

root = ET.fromstring(txt)

cases = {}
genders = {}
ages = {}
for query in root.iter('TOP'):
    q_num = query.find('NUM').text
    q_title = query.find('TITLE').text
    cases[q_num] = q_title
    genders[q_num] = getGender(q_title)
    ages[q_num] = getAge(q_title)

eval = trec.TrecEvaluation(cases, Qrels)


# ### ONLY RUN ONCE THE CODE BELOW, TO GET DOC_IDS.BIN, brief_titles.BIN, detailed_descriptions.BIN, brief_summaries.BIN, criterias.BIN, genders.BIN, minimum_ages.BIN, maximum_ages.BIN FILES

# In[40]:


import xml.etree.ElementTree as ET
import tarfile

tar = tarfile.open("clinicaltrials.gov-16_dec_2015.tgz", "r:gz")
i = 0
ids = []
list_brief_title = []
list_detailed_description = []
list_brief_summary = []
list_criteria = []
list_gender = []
list_minimum_age = []
list_maximum_age = []

j = 0
for tarinfo in tar:
    if tarinfo.size > 500:
        txt = tar.extractfile(tarinfo).read().decode("utf-8", "strict")
        root = ET.fromstring(txt)

        judged = False
        for doc_id in root.iter('nct_id'):
            if doc_id.text in eval.judged_docs:
                judged = True
        
        if judged is False:
            continue
        
        i = i + 1
        for brief_title in root.iter('brief_title'):
            brief = ""
            if(not (brief_title.text.strip() and not brief_title.text.isspace())):
                brief = "Empty"               
            else:
                brief = brief_title.text                   
            list_brief_title.append(brief)
            ids.append(doc_id.text)          
           
        existsDetDesc = False
        for detailed_description in root.iter('detailed_description'):
            for child in detailed_description:
                existsDetDesc = True
                detailed_desc = ""
                if(not (child.text.strip() and not child.text.isspace())):                    
                    detailed_desc = brief_title.text
                    print(detailed_desc)
                else:
                    detailed_desc = child.text.strip()
                list_detailed_description.append(detailed_desc)
        if(not existsDetDesc):
            list_detailed_description.append(brief_title.text)       
            
        existsBriefSummary = False
        for brief_summary in root.iter('brief_summary'):
            for child in brief_summary:
                existsBriefSummary = True
                brief_sum = ""
                if(not (child.text.strip() and not child.text.isspace())):                    
                    brief_sum = brief_title  .text                  
                else:
                    brief_sum = child.text.strip()
                list_brief_summary.append(brief_sum)
        if(not existsBriefSummary):
            list_brief_summary.append(brief_title.text)
        
        existsCriteria = False
        for criteria in root.iter('criteria'):
            for child in criteria:
                existsCriteria = True
                crit = ""
                if(not (child.text.strip() and not child.text.isspace())):                    
                    crit = "Undisclosed"
                    print(crit)
                else:
                    crit = child.text.strip()
                list_criteria.append(crit)
        if(not existsCriteria):
            list_criteria.append("Undisclosed")
        
        existsGender = False
        for gender in root.iter('gender'):
            existsGender = True
            genr = ""
            if(not (gender.text.strip() and not gender.text.isspace())):                    
                genr = "Both"
            else:
                genr = gender.text.strip()
            list_gender.append(genr)
        if(not existsGender):
            list_gender.append("Both")
        
        existsMinAge = False
        for minimum_age in root.iter('minimum_age'):
            existsMinAge = True
            min_a = ""
            if(not (minimum_age.text.strip() and not minimum_age.text.strip() == "N/A" and not minimum_age.text.isspace())):                    
                min_a = "0 Years"
            else:
                min_a = minimum_age.text.strip()
            list_minimum_age.append(min_a)
        if(not existsMinAge):
            list_minimum_age.append("0 Years")
            
        existsMaxAge = False
        for maximum_age in root.iter('maximum_age'):
            existsMaxAge = True
            max_a = ""
            if(not (maximum_age.text.strip() and not maximum_age.text.strip() == "N/A" and not maximum_age.text.isspace())):                    
                max_a = "100 Years"
            else:
                max_a = maximum_age.text.strip()
            list_maximum_age.append(max_a)
        if(not existsMaxAge):
            list_maximum_age.append("100 Years")

tar.close()

pickle.dump(list_brief_title, open( "brief_titles.bin", "wb" ) )
pickle.dump(ids, open( "doc_ids.bin", "wb" ) )
pickle.dump(list_detailed_description, open( "detailed_descriptions.bin", "wb" ) )
pickle.dump(list_brief_summary, open( "brief_summaries.bin", "wb" ) )
pickle.dump(list_criteria, open( "criterias.bin", "wb" ) )
pickle.dump(list_gender, open( "genders.bin", "wb" ))
pickle.dump(list_minimum_age, open( "minimum_ages.bin", "wb" ))
pickle.dump(list_maximum_age, open( "maximum_ages.bin", "wb" ))


# ### Here we have functions for VSM and LMJM that will calculate scores

# In[4]:


def VSM(corpus):
    dictionary_metrics_vsm=defaultdict(list) # defaultdict(<class 'list'>, {})
    dicqueryID = dict()
    dicpatientID = dict()
    out_scores_vsm = []
    avg_precision_11point = np.zeros(11)
    index = TfidfVectorizer(ngram_range=(1,1), analyzer='word', stop_words = None)
    index.fit(corpus)
    X = index.transform(corpus)  
    for caseid in cases:
        query = cases[caseid]
        query_tfidf = index.transform([query]) 
        doc_scores = 1 - pairwise_distances(X, query_tfidf, metric='cosine') #DOCUMENT SCORE
        query_id = [caseid] * len(ids)
        results = pd.DataFrame(list(zip(query_id, ids, doc_scores)), columns = ['caseid','_id', 'score'])
        results_ord = results.sort_values(by=['score'], ascending = False)           
            
        out_scores_vsm.append(results_ord) #We want to return out_scores_vsm
        
        '''Evaluation Metrics:'''
        [p10, recall, ap, ndcg5, mrr] = eval.eval(results_ord, caseid)
        [precision_11point, recall_11point, total_relv_ret] = eval.evalPR(results_ord, caseid)

        '''Dictionary of Metrics so we can return a full dictionary with all the metrics:'''
        dictionary_metrics_vsm['P10s:'].append(p10)
        dictionary_metrics_vsm['Recalls:'].append(recall)
        dictionary_metrics_vsm['Aps:'].append(ap)
        dictionary_metrics_vsm['Ndcg5s:'].append(ndcg5)
        dictionary_metrics_vsm['Mrrs:'].append(mrr)
        
        if (np.shape(recall_11point) != (0,)):
            avg_precision_11point = avg_precision_11point + precision_11point
        
    '''Average of each Metrics:'''
    average_P10s_vsm = np.mean(dictionary_metrics_vsm['P10s:'])
    average_Recalls_vsm = np.mean(dictionary_metrics_vsm['Recalls:'])
    average_Aps_vsm = np.mean(dictionary_metrics_vsm['Aps:'])
    average_Ndcg5s_vsm = np.mean(dictionary_metrics_vsm['Ndcg5s:'])
    average_Mrrs_vsm = np.mean(dictionary_metrics_vsm['Mrrs:'])
                
    #print(results_ord)
    return out_scores_vsm
    
def LMJM(corpus):    
    
    dictionary_metrics_lmjm=defaultdict(list) # defaultdict(<class 'list'>, {})
    avg_precision_11point_jm = np.zeros(11)      
    l=0.8
    out_scores_lmjm = []
    for caseid in cases:
        index_cv = CountVectorizer(ngram_range=(1,1), analyzer='word', stop_words = None) 
        index_cv.fit(corpus)
        X2 = index_cv.transform(corpus) #Compute corpus representation  corpus_cv
        corpus_array=X2.toarray()
        query_cv = index_cv.transform([cases[caseid]])
        qq = query_cv.toarray()[0]

        A=len((corpus))  
        aa = np.tile(qq, [A,1]) 

        prob_word_docs = corpus_array.T/np.sum(corpus_array,axis=1) # divided by doclength
        prob_word_corpus = np.sum(corpus_array, axis=0)/np.sum(corpus_array)
        prob_word_docs_query =(1-l)*(prob_word_docs.T**aa)
        prob_word_corpus_query = l*(prob_word_corpus**aa)
        docs_scores = prob_word_docs_query + prob_word_corpus_query
        final = np.prod(docs_scores, axis = 1)
        query_id = [caseid] * len(ids)
        results = pd.DataFrame(list(zip(query_id, ids, final)), columns = ['caseid','_id', 'score'])
        results_ord = results.sort_values(by=['score'], ascending = False)
        out_scores_lmjm.append(results_ord)
        
        '''Evaluation Metrics:'''
        [p10, recall, ap, ndcg5, mrr] = eval.eval(results_ord, caseid)
        [precision_11point, recall_11point, total_relv_ret] = eval.evalPR(results_ord, caseid)

        '''Dictionary of Metrics so we can return a full dictionary with all the metrics:'''
        dictionary_metrics_lmjm['P10s:'].append(p10)
        dictionary_metrics_lmjm['Recalls:'].append(recall)
        dictionary_metrics_lmjm['Aps:'].append(ap)
        dictionary_metrics_lmjm['Ndcg5s:'].append(ndcg5)
        dictionary_metrics_lmjm['Mrrs:'].append(mrr)
        
        if (np.shape(recall_11point) != (0,)):
            avg_precision_11point_jm = avg_precision_11point_jm + precision_11point
        
    '''Average of each Metrics:'''
    average_P10s_lmjm = np.mean(dictionary_metrics_lmjm['P10s:'])
    average_Recalls_lmjm = np.mean(dictionary_metrics_lmjm['Recalls:'])
    average_Aps_lmjm = np.mean(dictionary_metrics_lmjm['Aps:'])
    average_Ndcg5s_lmjm = np.mean(dictionary_metrics_lmjm['Ndcg5s:'])
    average_Mrrs_lmjm = np.mean(dictionary_metrics_lmjm['Mrrs:'])
    
    return out_scores_lmjm


# ### Upload all necessary files

# In[5]:


import pickle

brief_titles = pickle.load( open( "brief_titles.bin", "rb" ) )
detailed_descriptions = pickle.load( open( "detailed_descriptions.bin", "rb" ) )
brief_summaries = pickle.load( open( "brief_summaries.bin", "rb" ) )
criterias = pickle.load( open( "criterias.bin", "rb" ) )
ids = pickle.load( open( "doc_ids.bin", "rb" ))
list_gender = pickle.load( open( "genders.bin", "rb" ))
list_minimum_age = pickle.load( open( "minimum_ages.bin", "rb"))
list_maximum_age = pickle.load( open( "maximum_ages.bin", "rb" ))            


# ### Now get similarity scores for each query and corpus field

# In[7]:


#Formula to get score LMJM
m1=LMJM(detailed_descriptions) 
m2=LMJM(brief_summaries)
m3=LMJM(criterias)
m4=LMJM(brief_titles)

#Formula to get scores VSM
m5=VSM(detailed_descriptions) 
m6=VSM(brief_summaries)
m7=VSM(criterias)
m8=VSM(brief_titles)         
                            
pickle.dump(m1, open( "m1.bin", "wb" ) )
pickle.dump(m2, open( "m2.bin", "wb" ) )
pickle.dump(m3, open( "m3.bin", "wb" ) )
pickle.dump(m4, open( "m4.bin", "wb" ) )

pickle.dump(m5, open( "m5.bin", "wb" ) )
pickle.dump(m6, open( "m6.bin", "wb" ) )
pickle.dump(m7, open( "m7.bin", "wb" ) )
pickle.dump(m8, open( "m8.bin", "wb" ) )


# ### Load all features

# In[6]:


m1 = pickle.load( open( "m1.bin", "rb" ) )
m2 = pickle.load( open( "m2.bin", "rb" ) )
m3 = pickle.load( open( "m3.bin", "rb" ) )
m4 = pickle.load( open( "m4.bin", "rb" ) )
m5 = pickle.load( open( "m5.bin", "rb" ) )
m6 = pickle.load( open( "m6.bin", "rb" ) )
m7 = pickle.load( open( "m7.bin", "rb" ) )
m8 = pickle.load( open( "m8.bin", "rb" ) )


# ### Create a dataframe with (query_id, doc_id and rels) and 8 features 

# In[9]:


triplets=[]
qrels = open('qrels-clinical_trials.txt', 'r')
lines = qrels.readlines()
final = {}
for line in lines:
    line=line.strip()
    line=line.split('\t')
    rel = '1'
    if line[-1] != '2':
        rel = line[-1]
    _m1 = 0
    for df in m1:
        if (df.values[0]==line[0]).any():
            if(df['_id']==line[2]).any():
                df = df.set_index(['caseid', '_id'])
                _m1 = df.at[(line[0], line[2]), 'score'] #Access a single value for a row/column label pair.
                continue
            else:
                continue
    _m2 = 0
    for df in m2:
        if (df.values[0]==line[0]).any():
            if(df['_id']==line[2]).any():
                df = df.set_index(['caseid', '_id'])
                _m2 = df.at[(line[0], line[2]), 'score']
                continue
            else:
                continue
    _m3 = 0
    for df in m3:
        if (df.values[0]==line[0]).any():
            if(df['_id']==line[2]).any():
                df = df.set_index(['caseid', '_id'])
                _m3 = df.at[(line[0], line[2]), 'score']
                continue
            else
                continue
    _m4 = 0
    for df in m4:
        if (df.values[0]==line[0]).any():
            if(df['_id']==line[2]).any():
                df = df.set_index(['caseid', '_id'])
                _m4 = df.at[(line[0], line[2]), 'score']
                continue
            else:
                continue
    _m5 = 0
    for df in m5:
        if (df.values[0]==line[0]).any():
            if(df['_id']==line[2]).any():
                df = df.set_index(['caseid', '_id'])
                _m5 = df.at[(line[0], line[2]), 'score'][0]
                continue
            else:
                continue
    _m6 = 0
    for df in m6:
        if (df.values[0]==line[0]).any():
            #print("yes")
            if(df['_id']==line[2]).any():
                df = df.set_index(['caseid', '_id'])
                _m6 = df.at[(line[0], line[2]), 'score'][0]
                continue
            else:
                continue
    _m7 = 0
    for df in m7:
        if (df.values[0]==line[0]).any():
            #print("yes")
            if(df['_id']==line[2]).any():
                df = df.set_index(['caseid', '_id'])
                _m7 = df.at[(line[0], line[2]), 'score'][0]
                continue
            else:
                continue
    _m8 = 0
    for df in m8:
        if (df.values[0]==line[0]).any():
            #print("yes")
            if(df['_id']==line[2]).any():
                df = df.set_index(['caseid', '_id'])
                _m8 = df.at[(line[0], line[2]), 'score'][0]
                continue
            else:
                continue 
    triplets.append((line[0],line[2],rel,_m1,_m2, _m3,_m4,_m5,_m6, _m7,_m8))    
    
pickle.dump(triplets, open( "triplets.bin", "wb" ) )


# The error message is alright, see here: https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur

# ### Our final list of tuples

# In[7]:


import pickle
final_all = pickle.load( open( "triplets.bin", "rb" ))


# ### Save it to dataframe, so we can scale all features easier

# In[8]:


final_dataframe_all = pd.DataFrame(final_all, columns=["queryid", "docid", "rel", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"])


# In[9]:


df_x = final_dataframe_all[["f1", "f2", "f3", "f4","f5", "f6", "f7", "f8"]]
#standardize the values for each predictor variable
final_dataframe_all[["f1", "f2", "f3", "f4","f5", "f6", "f7", "f8"]] = np.log(df_x).replace([np.inf, -np.inf], 0)
final_dataframe_all
logged = final_dataframe_all


# ### Standardize our features

# In[10]:


df_l = logged[["f1", "f2", "f3", "f4","f5", "f6", "f7", "f8"]]
logged[["f1", "f2", "f3", "f4","f5", "f6", "f7", "f8"]]= (df_l-df_l.mean())/df_l.std()
logged


# ### Here we split into training and testing datasets by query 

# In[70]:


import random
casesIds = list(cases.keys())
random.shuffle(casesIds) #shuffle the cases
train_quantity = round(len(casesIds)*0.66) #get train quantity
test_quantity = len(casesIds)-train_quantity #get test quantity
train_cases = casesIds[0:train_quantity]
test_cases = casesIds[train_quantity:]

train_df = logged.loc[logged['queryid'].isin(train_cases)] #dataframe of train
test_df = logged.loc[logged['queryid'].isin(test_cases)] #dataframe of test
X = logged[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]].values
Y = logged[["rel"]].values

X_train = train_df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]].values
Y_train = train_df[["rel"]].values

X_test = test_df[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]].values
Y_test = test_df[["rel"]].values


# ### Cross validation for best C (regularization) parameter

# In[71]:


from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

c_range = list(np.arange(0.0001, 100, 0.5))
#c_best = 0.0001
max_error = 100000
for c_val in c_range:
    clf = LogisticRegression(C=c_val, random_state=0,class_weight="balanced", max_iter=10000) #.fit(training_set, y_train)
    score = cross_val_score(clf, X_train, Y_train.ravel(), cv=10, scoring='accuracy').mean()
    if(score < max_error):
        max_error=score
        c_best=c_val
    
print('Max error:', max_error)
print('Best C', c_best)

#Train the classifier with c_best
clf = LogisticRegression(C=c_best, random_state=0, class_weight="balanced",max_iter=10000).fit(X_train, Y_train.ravel())


# In[72]:


np.exp(clf.coef_)


# ## Evaluation

# In[73]:


from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import accuracy_score

predict_train = clf.predict(X_train)
predict_test = clf.predict(X_test)
predict_all = clf.predict(X)

print(accuracy_score(Y_test, predict_test))
print(accuracy_score(Y_train, predict_train))
print(accuracy_score(Y, predict_all))


# ### Training results

# In[74]:


target_names = ['N','R']
print(classification_report(Y_train,predict_train, target_names=target_names))


# In[75]:


cnf_matrix=classification_report(Y_train,predict_train)


# ### Testing results

# In[76]:


target_names = ['N','R']
print(classification_report( Y_test, predict_test, target_names=target_names))


# ### All results

# In[77]:


target_names = ['N','R']
print(classification_report( Y, predict_all, target_names=target_names))


# ### Ranking algorithm based on combined relevance (before additional filtering)

# In[78]:


temp = final_dataframe_all[["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]]
temp = temp*clf.coef_
temp['z'] = temp.sum(axis=1)
final_dataframe_all['z'] = temp['z']
final_dataframe_all['pred'] = predict_all
relevant = final_dataframe_all.loc[final_dataframe_all['pred']=='1']
top100_before = relevant.sort_values(by=['z'], ascending=False).head(100)
top100_before


# ### Ranking algorithm based on combined relevance (after filtering other criteria)

# In[79]:


# With the final_dataframe_all, loop through all the lines and check for gender and age restrictions
#print("Before ",final_dataframe_all.loc[final_dataframe_all['pred']=='1'])
count = 0
newPred = []
previousQueryId = -1
for index, row in final_dataframe_all.iterrows():
    queryId = row["queryid"]
    queryId = row["queryid"]
    if(previousQueryId == -1): #first time running, set first as previousQueryId and get patient Data
        previousQueryId = queryId
        patientAge = ages[queryId]
        patientGender = genders[queryId]
        #print("First time running")
    if(queryId != previousQueryId): #if queryId changed, get info again 
        previousQueryId = queryId
        patientAge = ages[queryId]
        patientGender = genders[queryId]
    docId = row["docid"]
    docIndex = ids.index(docId)
    docGender = list_gender[docIndex]
    docMinAge = int(list_minimum_age[docIndex].split()[0])
    docMaxAge = int(list_maximum_age[docIndex].split()[0])
    label = row["pred"]
    if(label == "1" and ( (patientAge != "NA" and not (int(patientAge)> docMinAge and int(patientAge) < docMaxAge)) or (patientGender == "Male" and docGender =="Female") or (patientGender =="Female" and docGender =="Male"))):  
        label = "0"
        count = count+1
    newPred.append(label)
    
final_dataframe_all["pred"] = newPred
#print("after" ,final_dataframe_all.loc[final_dataframe_all['pred']=='1'])
#final_dataframe_all.to_csv('final_dataframe_all_with_z_after_cleanup.csv')           
list(final_dataframe_all["rel"])
print(accuracy_score(list(final_dataframe_all["rel"]), list(final_dataframe_all["pred"])))

target_names = ['N','R']
print(classification_report( list(final_dataframe_all["rel"]), list(final_dataframe_all["pred"]), target_names=target_names))

relevant2 = final_dataframe_all.loc[final_dataframe_all['pred']=='1']
top100_after = relevant2.sort_values(by=['z'], ascending=False).head(100)


# In[81]:


top100_after = relevant2.sort_values(by=['z'], ascending=False).head(100)
top100_after
pickle.dump(top100_after, open( "top100_after.bin", "wb" ) )


# ### Model interpretation

# In[82]:


import xgboost
import shap


# In[83]:


explainer = shap.Explainer(clf, X_train, feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"])
shap_values = explainer(X_test)


# In[84]:


shap.plots.beeswarm(shap_values,order=shap_values.abs.max(0), show =False)
#plt.savefig('shap_all_ordered.png')


# In[85]:


clf.coef_


# Our exponentiated coefficients

# In[86]:


np.exp(clf.coef_)


# In[87]:


shap.plots.waterfall(shap_values[0])


# In[88]:


explainer_all = shap.Explainer(clf, X, feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"])
shap_values_all = explainer_all(X)
shap.plots.beeswarm(shap_values_all,order=shap_values.abs.max(0), show =False)


# # BERT model selected

# Imports needed, I put them here so everyone can understand they are related to BERT

# In[4]:


from bertviz import model_view, head_view
from transformers import *

import numpy as np
import pprint

# Get the interactive Tools for Matplotlib
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel
import torch


# In[5]:


def call_html():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/5.7.0/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))


# In[6]:


#model_path = 'nboost/pt-bert-base-uncased-msmarco'

model_path='healx/biomedical-slot-filling-reader-large'

#dmis-lab/biobert-large-cased-v1.1
#model_path = 'bert-base-uncased'  # NOTE: this model was used in bert training tutorial, we should try a different model

CLS_token = "[CLS]" #this is number 101
SEP_token = "[SEP]" #this is number 102


# ### Tokenizer, configuration and model saved into pickles. Do not run again!

# In[8]:


tokenizer = AutoTokenizer.from_pretrained('healx/biomedical-slot-filling-reader-large') #https://huggingface.co/healx/biomedical-dpr-ctx-encoder
pickle.dump(tokenizer,open( "tokenizer.bin", "wb" ))
config = AutoConfig.from_pretrained('healx/biomedical-slot-filling-reader-large',  output_hidden_states=True, output_attentions=True)  
pickle.dump(config,open( "config.bin", "wb" ))
model = AutoModel.from_pretrained('healx/biomedical-slot-filling-reader-large', config=config)
pickle.dump(model,open( "model.bin", "wb" ))


# ### Load all pickles

# In[7]:


detailed_descriptions = pickle.load( open( "detailed_descriptions.bin", "rb" ) ) #trial
tokenizer = pickle.load(open( "tokenizer.bin", "rb" )) 
config = pickle.load(open( "config.bin", "rb" )) 
model = pickle.load(open( "model.bin", "rb" )) 
ids = pickle.load( open( "doc_ids.bin", "rb" ))
top100_after = pickle.load( open( "top100_after.bin", "rb" ))


# ### Check distribution of individual words in our sentences

# In[8]:


sent_len = [len(i.split()) for i in detailed_descriptions]
filtered = filter(lambda score: score <= 512, sent_len)
#print(list(filtered))
pd.Series(filtered).hist(bins=10)
"""most sentences are around 20 words long"""
#plt.savefig('distribution_word_len.png')


# ### Choose one query (sentence_a) and corresponding relevant docID (sentence_b) as well as corresponding non-relevant docID (sentence_c)

# In[11]:


top100_after.head(20)


# """Best query (with highest z) was 201512 and corresponding rel docID NCT00231374, and non-relevant docID is NCT01742026"""

# In[12]:


#detailed_descriptions


# In[9]:


sentence_a=cases['201512']
sentence_a #118 characters, 16 words
#len(sentence_a.split())


# In[10]:


rel=ids.index('NCT00231374')  
sentence_b=detailed_descriptions[rel] #245 words, need to maybe shorten it?
non_rel=ids.index('NCT00097838') #chose this one as it is short sentence
sentence_c=detailed_descriptions[non_rel] #17 words
#len(sentence_b.split())
sentence_c


# In[11]:


query_rel=sentence_a+sentence_b
query_non_rel = sentence_a+sentence_c


# In[12]:


query_non_rel


# # Tokenization

# In[13]:


inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt', add_special_tokens=True, max_length=55, truncation=True)
#Print all 3 different inputs
#pprint.pprint(inputs) 

inputs_n = tokenizer.encode_plus(sentence_a, sentence_c, return_tensors='pt', add_special_tokens=True, max_length=55, truncation=True)
#Print all 3 different inputs
#pprint.pprint(inputs_n) 

#input_ids are mappings between tokens and their respective IDs. input_ids contains the integer sequences of the input sentences. The integers 101 and 102 are special tokens. We add them to both the sequences, and 0 represents the padding token.
#attention_mask contains 1's and 0's. It tells the model to pay attention to the tokens corresponding to the mask value of 1 and ignore the rest. attention mask is to prevent the model from looking at padding tokens
#'token_type_ids' shows is computed only for the last values (the ones that has 1's). used typically in a next sentence prediction tasks, where two sentences are given.


# In[18]:


#pprint.pprint(inputs)


# In[18]:


#Print our decoded sentence_a and sentence_b
print(tokenizer.decode(inputs["input_ids"][0].tolist()))
print(tokenizer.decode(inputs_n["input_ids"][0].tolist()))


# In[19]:


input_ids = inputs['input_ids']
input_ids_n = inputs_n['input_ids']
#pprint.pprint(input_ids[0].tolist())


# In[20]:


#input_ids = inputs['input_ids']
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list) #function to get actual token for an id
pprint.pprint(tokens)

input_id_list_n = input_ids_n[0].tolist() # Batch index 0
tokens_n = tokenizer.convert_ids_to_tokens(input_id_list_n) #function to get actual token for an id
pprint.pprint(tokens_n)


# to see what exactly the token ID of the any special token is

# In[21]:


tokenizer.special_tokens_map


# In[22]:


tokenizer.convert_tokens_to_ids(["[SEP]"])


# # Model inference

# In[23]:


outputs = model(**inputs) #Because tokens is a dictionary object, we can unpack them as keyword arguments through a double star.
attention = outputs[-1]
hidden_states=outputs[2] # becase we set `output_hidden_states = True` - this is a default when running a model (we have set it in configuration part), the third item will be the # hidden states from all layers.


# In[24]:


#outputs


# In[25]:


outputs_n = model(**inputs_n) #Because tokens is a dictionary object, we can unpack them as keyword arguments through a double star.
attention_n = outputs_n[-1]
hidden_states_n=outputs_n[2]


# In[26]:


outputs.keys() 


# In[27]:


print ("Number of layers in Biomedical-slot-filling-reader-large:", len(hidden_states), "  (initial embeddings + 24 BERT layers)")
layer_i = 0

print ("Number of batches Biomedical-slot-filling-reader-large:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens Biomedical-slot-filling-reader-large:", len(hidden_states[layer_i][batch_i])) #this we have set up before in order to truncate sentence_b
token_i = 0

print ("Number of hidden units Biomedical-slot-filling-reader-large:", len(hidden_states[layer_i][batch_i][token_i])) #this means we have 1024 features?


# In[28]:


hidden_states[0] #total 25 layers


# # Extract Token embeddings

# In[29]:


import torch
from transformers import AutoTokenizer, AutoModel

def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)

def get_hidden_states(encoded, token_ids_word, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
        Select only those subword token outputs that belong to our word of interest
        and average them."""
    with torch.no_grad():
        output = model(**encoded)

     # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)


def get_word_vector(sent, idx, tokenizer, model, layers):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
       that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    print(encoded)
    # get all token idxs that belong to the word of interest
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    print(token_ids_word)
    return get_hidden_states(encoded, token_ids_word, model, layers)

# Get first and last layers
layer_first = [0]
layer_last = [-1]
#layers = [0,-1]

sent = query_rel
sent_n =query_non_rel


#idx = get_word_idx(query_rel, "headache") #sentence_a is pair of both, query and doc (tokens). tokens pass to list

#word_embedding_first_layer = get_word_vector(sent, idx, tokenizer, model, layer_first) 
#word_embedding_last_layer = get_word_vector(sent, idx, tokenizer, model, layer_last)
#word_embedding_both_layers = get_word_vector(sent, idx, tokenizer, model, layers)
#array_first=np.array(word_embedding_first_layer)
#array_last=np.array(word_embedding_last_layer)



# In[30]:


query_rel


# In[31]:


# Get id's of 10 words of interest (only 4 for now)
chosen_words = ['headache', 'Nuchal', 'complains', 'physical', 'spine','dural', 'pressure', 'needle', 'spinal', 'fluid', 'cerebrospinal','medical','leakage','myelogram']
idx_chosen_words = []
words_embeddings_first_layer = []
words_embedding_last_layer = []

for word in chosen_words:
    idx = get_word_idx(query_rel, word)
    idx_chosen_words.append(idx)
    
    word_embedding_first_layer = get_word_vector(sent, idx, tokenizer, model, layer_first) #
    word_embedding_last_layer = get_word_vector(sent, idx, tokenizer, model, layer_last)
    words_embeddings_first_layer.append(word_embedding_first_layer.numpy())
    words_embedding_last_layer.append(word_embedding_last_layer.numpy())

#print(idx_chosen_words)


# In[35]:


query_non_rel


# In[36]:


# Get id's of 10 words of interest (only 4 for now)
chosen_words_n = ['headache', 'Nuchal', 'history', 'man', 'complains', 'gag','Venezuelan','physical','HIV', 'vaccine', 'review',  'Encephalitis', 'blood','testing','urine','system', 'immunogenicity']
idx_chosen_words_n = []
words_embeddings_first_layer_n = []
words_embedding_last_layer_n = []

for word in chosen_words_n:
    idx_n = get_word_idx(query_non_rel, word)
    idx_chosen_words_n.append(idx_n)
    
    word_embedding_first_layer_n = get_word_vector(sent_n, idx_n, tokenizer, model, layer_first) #
    word_embedding_last_layer_n = get_word_vector(sent_n, idx_n, tokenizer, model, layer_last)
    words_embeddings_first_layer_n.append(word_embedding_first_layer_n.numpy())
    words_embedding_last_layer_n.append(word_embedding_last_layer_n.numpy())

#print(idx_chosen_words)


# In[ ]:





# In[37]:


np.shape(words_embeddings_first_layer_n)


# In[ ]:





# In[38]:


words_embeddings_first_layer


# In[39]:


words_embedding_last_layer_n


# ### This should be correct plot to plot chosen tokens for first and last layer vectors 

# https://stats.stackexchange.com/questions/355521/i-want-to-apply-same-pca-to-different-datasets

# In[40]:


def display_pca_scatterplot2(first_layer,last_layer, words): # second argument token (10 words)

    pca = PCA(n_components=2)
    pca.fit(first_layer)
    twodim = pca.fit_transform(first_layer)
    twodim_l = pca.transform(last_layer)
    plt.figure(figsize=(20,10))
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    plt.scatter(twodim_l[:,0], twodim_l[:,1], edgecolors='k', c='b')
    
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.5, y+0.5, word)
    for word, (x,y) in zip(words, twodim_l):
        plt.text(x+0.5, y+0.5, word)
        


# In[41]:


def display_pca_scatterplot2(first_layer,last_layer, words): # second argument token (10 words)

    pca = PCA(n_components=2)
    pca.fit(first_layer)
    twodim = pca.fit_transform(first_layer)
    twodim_l = pca.transform(last_layer)
    #plt.figure(figsize=(20,10))
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    ax1.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r',  label='first layer embeddings')
    ax1.scatter(twodim_l[:,0], twodim_l[:,1], edgecolors='k', c='b', label='last layer embeddings')
    plt.legend(loc='upper left')  
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.5, y+0.5, word)
    for word, (x,y) in zip(words, twodim_l):
        plt.text(x+0.5, y+0.5, word)


# In[42]:


display_pca_scatterplot2(words_embeddings_first_layer, words_embedding_last_layer, chosen_words)
#plt.savefig('query_rel.png')


# In[43]:


display_pca_scatterplot2(words_embeddings_first_layer_n, words_embedding_last_layer_n, chosen_words)
#plt.savefig('query_nonrel.png')


# ## Plot the different embeddings for the same set of tokens from query_relevant and query_non_relevant pairs

# ### Non_relevant

# In[44]:


chosen_words_query=['man','complains','of', 'severe', 'headache','and', 'fever.','Nuchal', 'rigidity', 'physical']
idx_chosen_words_n = []
words_embeddings_first_layer_n = []
words_embedding_last_layer_n = []

for word in chosen_words_query:
    idx_n = get_word_idx(query_non_rel, word)
    idx_chosen_words_n.append(idx_n)
    
    word_embedding_first_layer_n = get_word_vector(sent_n, idx_n, tokenizer, model, layer_first) #
    word_embedding_last_layer_n = get_word_vector(sent_n, idx_n, tokenizer, model, layer_last)
    words_embeddings_first_layer_n.append(word_embedding_first_layer_n.numpy())
    words_embedding_last_layer_n.append(word_embedding_last_layer_n.numpy())


# ### Relevant

# In[45]:


chosen_words_query=['man','complains','of', 'severe', 'headache','and', 'fever.','Nuchal', 'rigidity', 'physical']
idx_chosen_words = []
words_embeddings_first_layer = []
words_embedding_last_layer = []

for word in chosen_words_query:
    idx = get_word_idx(query_rel, word)
    idx_chosen_words.append(idx)
    
    word_embedding_first_layer = get_word_vector(sent, idx, tokenizer, model, layer_first) #
    word_embedding_last_layer = get_word_vector(sent, idx, tokenizer, model, layer_last)
    words_embeddings_first_layer.append(word_embedding_first_layer.numpy())
    words_embedding_last_layer.append(word_embedding_last_layer.numpy())


# In[46]:


display_pca_scatterplot2(words_embeddings_first_layer, words_embedding_last_layer, chosen_words_query)
#plt.savefig('query_rel_same_tokens.png')


# In[47]:


display_pca_scatterplot2(words_embeddings_first_layer_n, words_embedding_last_layer_n, chosen_words_query)
#plt.savefig('query_nonrel_same_tokens.png')


# 
# 
# #### Attention visualisation
#   

# In[48]:


#attention


# Note: Here we have to remove chosen words and their attentions from attention matrix (he said to delete rows and columns of attention that we do not need).

# In[49]:


len(attention[0][0][0])


# In[50]:


idx_chosen_words
chosen_words_attention = []

for i in idx_chosen_words:
    chosen_words_attention.append(attention[0][0][0][i])
    
len(chosen_words_attention)
chosen_words_as_string = ""
for word in chosen_words_query:
    chosen_words_as_string += " "+word
print(chosen_words_as_string)
inputs_attention = tokenizer.encode(chosen_words_as_string, return_tensors='pt')
outputs_attention = model(inputs_attention)
attention_chosen_words = outputs_attention[-1]  # Output includes attention weights when output_attentions=True
tokens_chosen_words = tokenizer.convert_ids_to_tokens(inputs_attention[0]) 
head_view(attention_chosen_words, tokens_chosen_words)


# In[55]:


model_view(attention_chosen_words, tokens_chosen_words, include_layers=[0, 23])


# In[47]:


call_html()
#head_view(attention, tokens) #Note: Here we have to remove chosen words and their attentions from attention matrix (he said to delete rows and columns of attention that we do not need).
#this way we can check attention for first layer (0) and for last layer(23)


# ### Similarities between documents (query_rel and query-non_rel)

# In[48]:


from sklearn.metrics.pairwise import cosine_similarity


# We need to plot here cosine similarity (words_embedding_first_layer, words_embedding_last_layer). This will show how similarity is changed between input and output. 

# In[49]:


cos_similarity=cosine_similarity(words_embeddings_first_layer, words_embedding_last_layer) #for our chosen tokens queryrel pair
cos_similarity2=cosine_similarity(words_embeddings_first_layer_n, words_embedding_last_layer_n) #for our chosen tokens 
cos_similarity3=cosine_similarity(words_embedding_last_layer, words_embedding_last_layer_n) #for our chosen tokens comparing last layer if we pass query_rel vs query_nonrel sentences
#cos_similarity


# In[66]:


def heatmap(x_labels, y_labels, values):
    fig, ax = plt.subplots(figsize=(10,5))
    im = ax.imshow(values)
    
   # fig = plt.figure(figsize=(10,5))
   # ax1 = fig.add_subplot(111)

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))

    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10,
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", fontsize=10,
         rotation_mode="anchor")
    
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            text = ax.text(j, i, "%.2f"%values[i, j],
                           ha="center", va="center", color="black",fontsize=8)
    fig.tight_layout()
    #plt.show()
    


# In[67]:


heatmap(chosen_words_query,chosen_words_query,cos_similarity)
#plt.savefig('heat_query_rel_first_last.png')


# In[68]:


heatmap(chosen_words_query,chosen_words_query,cos_similarity2)
#plt.savefig('heat_query_non_rel_first_last.png')


# In[69]:


heatmap(chosen_words_query,chosen_words_query,cos_similarity3)
#plt.savefig('heat_query_rel_norel_last.png')


# In[77]:


chosen_words_query=['man','complains','of', 'severe', 'headache','and', 'fever.','Nuchal', 'rigidity', 'physical']


# ### Scribles

# In[ ]:





# In[ ]:




