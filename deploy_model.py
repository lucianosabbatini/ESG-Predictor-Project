# Python libraries
import streamlit as st
from PIL import Image

# User module files
from ml import ml_environment, ml_social, ml_governance, ml_esg

def main():

    # Display sidebar menu options
    options = ['Home','Prediction Environment', 'Prediction Social', 'Prediction Governance', 'ESG Overall', 'Results', 'End']
    choice = st.sidebar.selectbox("Menu", options)

    # Display different pages based on user choice
    if choice == 'Home':
        st.title("Free ESG Calculator ")
        image = Image.open('home.jpg')
        st.image(image, width=500)
    if choice == 'Prediction Environment':
        st.title("Input data for:")
        environment_score = ml_environment()
        st.write("Based on your inputs, your Environment Score is: ", environment_score)
    elif choice == 'Prediction Social':
        st.title("Input data for:")
        social_score = ml_social()
        st.write("Based on your inputs, your Social Score is: ", social_score)
    elif choice == 'Prediction Governance':
        st.title("Input data for:")
        governance_score = ml_governance()
        st.write("Based on your inputs, your Governance Score is: ", governance_score)
    elif choice == 'ESG Overall':
        st.title("Input data for:")
        esg_overall = ml_esg()
        st.write("Based on your inputs, your Overall ESG Score is: ", esg_overall)

    if choice == 'Results':
        
        st.title("Tesla, Inc. ")
        image2 = Image.open('tesla.png')
        st.image(image2)
    if choice == 'End':
        st.balloons()

if __name__ == "__main__":
    main()



