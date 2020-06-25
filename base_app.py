"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Vectorizer
news_vectorizer = open("resources/vectoriser.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown('''Online reputation is one of the most precious assets for brands.
		A bad review on social media can be costly to a company if it’s not handled 
		effectively and swiftly. Twitter sentiment analysis allows you to keep track of
		 what’s being said about your product or service on social media, and can help you
		  detect angry customers or negative mentions before they turn into a major crisis. 
		At the same time, Twitter sentiment analysis can provide interesting insights. What
		 do customers love about your brand?  What aspects get the most negative mentions?''')

		st.subheader("Raw Twitter data and label")
		st.markdown('''Let's take a look at the the raw data that is used to train whichever
					model you choose to use to predict the sentiment''')
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
			st.markdown('''The sentiment classification is defined as as follows''')
			#Image that explains the classifications
			image = Image.open(r'resources\imgs\classifications.png')
			st.image(image, use_column_width=True)

		uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
		if uploaded_file is not None:
			data = pd.read_csv(uploaded_file)
			st.write(data[['message']])
		
		if st.checkbox('Show EDA'):
			st.subheader('Counts of tweets per class')
			plt.bar([1,2,3,4], raw['sentiment'].value_counts(), color=['red', 'green', 'blue', 'orange'])
			plt.xticks([1,2,3,4], ['pro', 'news', 'neutral', 'anti'])
			plt.ylabel('Count')
			plt.xlabel('Sentiment')

			st.pyplot()

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		st.markdown("or alternatively")
		uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
		if uploaded_file is not None:
    			tweet_text = pd.read_csv(uploaded_file)['message']
					

		#Classifier selection
		Classifier = st.selectbox("Choose Classifier",['Linear SVC','Logistic regression'])
		size = st.selectbox("Choose Size of results table",[5,10,15,20,30,60])
     
     
		if st.button("Classify"):
			# Transforming user input with vectorizer


			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if Classifier =='Linear SVC':
					st.text("Using Linear SVC classifier ..")
					predictor = joblib.load(open(os.path.join("resources/linearSVC.pkl"),"rb"))
			elif Classifier == 'Logistic regression':
					st.text("Using Logistic Regression Classifeir ..")
					predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			#predictor = joblib.load(open(os.path.join("resources/linearSVC.pkl"),"rb"))
			results = []
			n = 0
			while n < len(tweet_text):	
				vect_text = tweet_cv.transform([tweet_text[n]]).toarray()
				prediction = predictor.predict(vect_text)
				results.append((tweet_text[n],prediction))
				n+=1


			df = pd.DataFrame(results,columns=['Message','Sentiment'])

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.

			#Table that tabulates the results

			#dataframe = pd.DataFrame()
			#dataframe['Text'] = tweet_text
			#dataframe['Sentiment'] = prediction
			predictions = st.table(df.head(size))

			st.success("Text Categorized as: {}".format(predictions))


			st.subheader('Counts of tweets per class')
			plt.bar([1,2,3,4], df['Sentiment'].value_counts(), color=['red', 'green', 'blue', 'orange'])
			plt.xticks([1,2,3,4], ['pro', 'news', 'neutral', 'anti'])
			plt.ylabel('Count')
			plt.xlabel('Sentiment')
			st.pyplot()

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
