import streamlit as st
#Page name and icon
st.set_page_config(page_title='About', page_icon='📜')
#some information about the project
st.title('About Page')
st.header('About this Project 📜')
st.write('The data source came from Kaggle website')
st.write('The used model is K Nearest Neighbors (KNN) classifier one with Default setting')
st.write('The Libraries used in this project are:')
st.write('1 - pandas for reading data')
st.write('2 - sklearn for preprocessing data and building model')
st.write('3 - joblib to save and load the model')
st.write('4 - streamlit for building the web app interface (As workframe)')

#some information about the creator
st.header('About the Creator 👨‍💻')
col1, col2 = st.columns([1, 3])
with col1:
    st.image(
        r'result/FB_IMG_1726001609431.jpg',
        width=150,
        caption='Ahmed Ghoneim'
    )
with col2:
    st.write('**Name :** Ahmed Ghoneim')
    st.write('**E-mail :** ahmedghoneim658@gmail.com')
    #social media links
    st.markdown(
        """
        <style>
            .social-icons {
                display: flex;
                gap: 15px;
                margin-top: 20px;
            }
            .social-icons img {
                width: 40px;
                height: 40px;
            }
        </style>
        
        <div class="social-icons">
            <a href="https://www.facebook.com/Ahmed Ghoniem" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/733/733547.png" alt="Facebook"/>
            </a>
            <a href="https://www.instagram.com/Ahmed Ghoniem" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/733/733558.png" alt="Instagram"/>
            </a>
            <a href="https://www.linkedin.com/in/Ahmed Ghoneim/" target="_blank">
                <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn"/>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

