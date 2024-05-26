import streamlit as st
import pickle
st.title('IPL SCORE PREDICTION')
run = st.number_input('Run after 6th over', value= 50, placeholder='enter a value for run')
wicket = st.number_input('Wicket after 6th over', value= 2, placeholder='enter a value for wicket')
run1 = st.number_input('Run after 16th over', value= 160, placeholder='enter a value for run')
wicket1 = st.number_input('Wicket after 16th over', value= 4, placeholder='enter a value for wicket')
loaded_model = pickle.load(open('ipl_regression.sav', 'rb'))
prediction = loaded_model.predict([[run,wicket,run1,wicket1]])
st.subheader(f'predicted score for above parameter is {int(prediction)}')
st.write(int(prediction))