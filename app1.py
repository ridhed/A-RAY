import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def load_df():
    df=pd.read_csv('Students.csv')
    return df

df=load_df()			


def get_table():
    dftable = df[['gender', 'tools', 'level of education','mental health','test mode', 'score before pandemic','score after pandemic','aggregate score']].sort_values(by=['aggregate score'], ascending=False)
    return dftable

dftable = get_table()
st.markdown("### **Student Performance Analysis**")
st.markdown("")
st.dataframe(dftable) 

visualization = st.sidebar.selectbox('Select Comparison of ',('Males and Females',
'Level of Education','Digital Tools','Score Before Pandemic','Score After Pandemic'))



if visualization=='Males and Females':
  st.markdown ("## **Gender analysis**")
  fig, ax = plt.subplots(figsize=(10,5))
  sns.countplot(df['gender'], palette = 'bone')
  ax.set_title('Comparison of Males and Females', fontweight = 30)
  ax.set_xlabel('Gender')
  ax.set_ylabel('Count')
  st.pyplot(fig)



if visualization=='Level of Education':
  st.markdown ("## **Level of Education**")
  fig, ax = plt.subplots(figsize=(10, 5))
  sns.countplot(df['level of education'], palette = 'pink')
  ax.set_title('Comparison of level of education', fontweight = 30, fontsize = 20)
  ax.set_xlabel('Groups')
  ax.set_ylabel('count')
  st.pyplot(fig)



if visualization=='Digital Tools':
  st.markdown ("## **Digital Tools**")
  fig, ax = plt.subplots(figsize=(10, 5))
  sns.countplot(df['tools'], palette = 'pink')
  ax.set_title('Comparison of Digital Tools', fontweight = 30, fontsize = 20)
  ax.set_xlabel('Groups')
  ax.set_ylabel('count')
  st.pyplot(fig)

if visualization=='Score Before Pandemic':
  st.markdown ("## **Score Before Pandemic**")
  fig, ax = plt.subplots(figsize=(20, 10))
  sns.countplot(df['score before pandemic'], palette = 'RdPu')
  ax.set_title('Score Before Pandemic', fontweight = 30, fontsize = 20)
  st.pyplot(fig)


if visualization=='Score After Pandemic':
  st.markdown ("## **Score After Pandemic**")
  fig, ax = plt.subplots(figsize=(20, 10))
  sns.countplot(df['score after pandemic'], palette = 'RdPu')
  ax.set_title('Score After Pandemic', fontweight = 30, fontsize = 20)
  st.pyplot(fig)

 
