import streamlit as st

#EDA Pkage
import pandas as pd 
import numpy as np 

#Utils

import os 
import joblib
# data Viz Pkgs
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import hashlib 


# ML Interpretation
import lime
import lime.lime_tabular
# passlib, bcrybt

# db
from manged_db import * 


# password
def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password, hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return flase

feature_names_best = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites',
       'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime',
       'histology']

gender_dict = {"male":1, "female":2}
feature_dict = {"No":1, "Yes":2}

def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return key

def get_fvalue(val):
    feature_dict = {"No":1, "Yes":2}
    for key, value in feature_dict.items():
        if val == key:
            return value

## Load Mdoel 
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def main():
    """Mortality prediction APP"""
    st.title("Disease Mortality Prediction App")

    menu = ["Home", "Login", "SignUp"]
    submenu = ["Plot", "Prediction", "Metrics"]
    choice  = st.sidebar.selectbox("Menu", menu)
    if choice =="Home":
        st.subheader("Home")
        st.text("What is Hepatitis ?")

    elif choice == "Login":
        username = st.sidebar.text_input("Username")  
        password = st.sidebar.text_input("Password", type = 'password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_psd = generate_hashes(password)
            result = login_user(username, verify_hashes(password, hashed_psd))
            if result:
                st.success("Welcome {}".format(username))

                activity = st.selectbox("Activity", submenu)
                if activity == "Plot":
                    st.subheader("Data Vis Plot")
                    df = pd.read_csv("data/clean_hepatitis_data.csv")
                    st.dataframe(df)

                    df['class'].value_counts().plot(kind='bar')
                    st.pyplot()

                    freq_df = pd.read_csv("data/freq_df.csv")
                    st.bar_chart(freq_df['count'])

                    if st.checkbox("Area Chart"):
                        all_columns = df.columns.to_list()
                        feat_choices = st.multiselect("Choose a Features", all_columns)
                        new_df = df[feat_choices]
                        st.area_chart(new_df)

                elif activity == "Prediction":
                    st.subheader("Prediction Analytics")


                    age = st.number_input("Age", 7, 80)
                    sex = st.radio("Sex", tuple(gender_dict.keys()))
                    steroid = st.radio("Do you take Steroids ?", tuple(feature_dict.keys()))
                    antivirals = st.radio("Do you take Antivirals ?", tuple(feature_dict.keys()))
                    fatigue = st.radio("Do you Have Fatigue ?", tuple(feature_dict.keys()))
                    spiders = st.radio('Presence of Spider Neavi', tuple(feature_dict.keys()))
                    ascites = st.selectbox("Ascites", tuple(feature_dict.keys()))
                    varices = st.selectbox("Presence of Varices", tuple(feature_dict.keys()))
                    bilirubin = st.number_input("Bilirubin Content", 0.0,8.0)
                    alk_phosphate = st.number_input("Alkaline Photophate Content", 0.0,296.0)
                    sgot = st.number_input("Sgot", 0.0, 648.0)
                    albumin = st.number_input("Albumin", 0.0,6.4)
                    protime = st.number_input("Prothorombin Time", 0.0, 100.0)
                    histology = st.selectbox("Histology", tuple(feature_dict.keys()))
                    feature_list = [age,get_value(sex,gender_dict),get_fvalue(steroid),get_fvalue(antivirals),get_fvalue(fatigue),get_fvalue(spiders),get_fvalue(ascites),get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),get_fvalue(histology)]                    
                    
                    st.write(feature_list)
                    st.write(len(feature_list))

                    pretty_result = {"age": age, "sex":sex, "steroid":steroid, "antivirals":antivirals, 'fatigue': fatigue, "spiders":spiders, "ascites":ascites, "varices": varices, "bilirubin": bilirubin, "alk_phosphate": alk_phosphate, "sgot": sgot, "albumin": albumin, "protime": protime, "histology": histology}
                    st.json(pretty_result)
                    single_sample = np.array(feature_list).reshape(1,-1)
                    model_choice = st.selectbox("Select Model", ["LR", "KNN", "DesicionTree"])
                    if st.button("Predict"):
                        if model_choice == "KNN":
                            loaded_model = load_model("models/knn_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                        elif model_choice == "LR":
                            loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                        elif model_choice == "DesicionTree":
                            loaded_model = load_model("models/decission_tree_clf_hepB_model.pkl")
                            prediction = loaded_model.predict(single_sample)
                            pred_prob = loaded_model.predict_proba(single_sample)
                        
                        st.write(prediction)
                        if prediction == 1 : 
                            st.warning("Patient Dies")
                        else:
                            st.success("Patient Lives")
                            pred_probability_score = {"Die": pred_prob[0][0]*100, "Live": pred_prob[0][1]*100}
                            st.subheader("Prediction Probability Score using {}".format(model_choice))
                            st.json(pred_probability_score)
                        
                    if st.checkbox("Interpret"):
                        if model_choice == "KNN":
                            loaded_model = load_model("models/knn_hepB_model.pkl")
                        
                        elif model_choice == "DesicionTree":
                            loaded_model = load_model("models/decission_tree_clf_hepB_model.pkl")
                            
                        else:
                            loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
                            
                            df = pd.read_csv("data/clean_hepatitis_data.csv")
                            x = df[['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']]
                            feature_names = ['age', 'sex', 'steroid', 'antivirals','fatigue','spiders', 'ascites','varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime','histology']
                            class_names = ['Die(1)','Live(2)']
                            explainer = lime.lime_tabular.LimeTabularExplainer(x.values,feature_names=feature_names, class_names=class_names,discretize_continuous=True)
                            # The Explainer Instance
                            exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba,num_features=13, top_labels=1)
                            exp.show_in_notebook(show_table=True, show_all=False)
                            # exp.save_to_file('lime_oi.html')
                            st.write(exp.as_list())
                            new_exp = exp.as_list()
                            label_limits = [i[0] for i in new_exp]
                            # st.write(label_limits)
                            label_scores = [i[1] for i in new_exp]
                            plt.barh(label_limits,label_scores)
                            st.pyplot()
                            plt.figure(figsize=(20,10))
                            fig = exp.as_pyplot_figure()
                            st.pyplot()




            else:
                st.warning("Incorrect Username/password")
    elif choice == "SignUp":
        new_username = st.text_input("User name")
        new_password = st.text_input("Password", type = "password")

        confirm_password = st.text_input("Confirm Password", type = "password")
        if new_password == confirm_password:
            st.success("Password Confirmed")
        else :
            st.warning("Passwords not the same")
        if st.button("Submit"):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_userdata(new_username, hashed_new_password)
            st.success("You have successfully created a new account")
            st.info("Login to Get Strted ")

        


if __name__ == '__main__':
    main()

