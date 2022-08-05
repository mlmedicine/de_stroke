import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.model_selection import cross_val_score
import random
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from sklearn.svm import SVC
#import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# import sklearn.model_selection as model_selection
from imblearn.over_sampling import RandomOverSampler

#应用主题
st.set_page_config(
    page_title="ML Medicine",
    page_icon="🐇",
)
#应用标题
st.title('Machine Learning Application for Predicting One-year Death')



# conf
col1, col2, col3 = st.columns(3)
NOS = col1.selectbox("Number of stroke lesions",('Single stroke lesion','Multiple stroke lesions'))
FBG = col2.number_input('FBG (mmol/L)',value=5.3)
HBALC = col3.number_input('HBALC (%)',value=5.6)
MB = col1.number_input("MB (ng/mL)",value=97.7)
Anticoagulation = col2.selectbox("Anticoagulation",('No','Yes'))
SS = col3.selectbox("Stroke severity",('Mild stroke','Moderate to severe stroke'))


# str_to_
map = {'Left':0,'Right':1,'Bilateral':2,
       'Single stroke lesion':0,'Multiple stroke lesions':1,
       'Mild stroke':0,'Moderate to severe stroke':1,
       'Cortex':0,'Cortex-subcortex':1,'Subcortex':2,'Brainstem':3,'Cerebellum':4,
       'No':0,'Yes':1}

NOS =map[NOS]
Anticoagulation =map[Anticoagulation]
SS =map[SS]


# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
features=[  'NOS', 'FBG',  'HBALC', 'MB', 'Anticoagulation','SS']
target='Status'

#处理数据不平衡
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

#train and predict
RF = sklearn.ensemble.RandomForestClassifier(n_estimators=21,criterion='entropy',max_features='log2',max_depth=3,random_state=12)
RF.fit(X_ros, y_ros)


#读之前存储的模型

#with open('RF.pickle', 'rb') as f:
#    RF = pickle.load(f)


sp = 0.5
#figure
is_t = (RF.predict_proba(np.array([[ NOS, FBG,  HBALC, MB, Anticoagulation,SS]]))[0][1])> sp
prob = (RF.predict_proba(np.array([[NOS, FBG,  HBALC, MB, Anticoagulation,SS]]))[0][1])*1000//1/10

#st.write('is_t:',is_t,'prob is ',prob)
#st.markdown('## is_t:'+' '+str(is_t)+' prob is:'+' '+str(prob))

if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability:  '+str(prob)+'%')
   
