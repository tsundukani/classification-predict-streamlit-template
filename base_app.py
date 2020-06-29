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
import matplotlib
import seaborn as sns
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics import accuracy_score

#plots
from plotly import graph_objs as go 
import seaborn as sns
sns.set()
#import nltk
from PIL import Image

 # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")
cleaned = pd.read_csv("resources/cleaned.csv")

# The main function where we will build the actual app
#@st.cache()
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	header_image = Image.open(r'resources/imgs/header_image.png')
	st.image(header_image, use_column_width=True)
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Information", "Machine Learning Overview", "Prediction"]
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
		#if st.checkbox('Show raw data'): # data is hidden if box is unchecked
		if st.checkbox("Preview Raw Data"):
				data_dim = st.radio("What Section of Data Do You Want to Show?", ("Head", "Tail"))
				if data_dim =="Head":
						st.text("Showing Head of Data")
						st.write(raw[['sentiment', 'message']].head())
				if data_dim =="Tail":
						st.text("Showing Tail of Data")
						st.write(raw[['sentiment', 'message']].tail())

			#st.write(raw[['sentiment', 'message']]) # will write the df to the page
				st.markdown('''The sentiment classification is defined as as follows''')
			#Image that explains the classifications
				image = Image.open(r'resources/imgs/classifications.png')
				st.image(image, use_column_width=True)
		if st.checkbox('Show EDA'):
			st.subheader('Counts of tweets per class')
			plt.bar([1,2,3,4], raw['sentiment'].value_counts(), color=['red', 'green', 'blue', 'orange'])
			plt.xticks([1,2,3,4], ['pro', 'news', 'neutral', 'anti'])
			plt.ylabel('Count')
			plt.xlabel('Sentiment')

			st.pyplot()
			c = []
			for i in range(len(cleaned['message'])):
				c.append(cleaned['message'][i])

			st.markdown('''Let us use a wordcloud to take a look at which tags/words are most common in our raw data''')


			# joining and making a complete list
			wordcloud = WordCloud(width=2000, height=1000).generate(' '.join(c))  # word cloud
			plt.figure(figsize=(30, 10))
			plt.imshow(wordcloud)
			plt.axis('off')
			st.pyplot()


			#Funnel graph showing the spread of sentiments in the raw dataframe
			clean_train_df = raw
			i = 0
			news = []
			pro = []
			neutral = []
			anti = []
			while i < len(clean_train_df):
				if clean_train_df['sentiment'].iloc[i] == 2:
					news.append(clean_train_df['message'].iloc[i])
				elif clean_train_df['sentiment'].iloc[i] == 1:
					pro.append(clean_train_df['message'].iloc[i])
				elif clean_train_df['sentiment'].iloc[i] == 0:
					neutral.append(clean_train_df['message'].iloc[i])
				else:
					anti.append(clean_train_df['message'].iloc[i])
				i += 1

			st.markdown('''Count number of Sentiments (ordered by message)''')
			temp = clean_train_df.groupby('sentiment').count()['message'].reset_index().sort_values(by='message',ascending=False)
			temp.style.background_gradient(cmap='Purples')
			st.table(temp.head())

			fig = go.Figure(go.Funnelarea(
				text = ["Pro","News", "Neutral", "Anti"],
				values = temp['message'].values,
				title = {"position": "top center", "text": "Sentiment Distribution"}
				))
			st.plotly_chart(fig)

			

	# Building out the Overiew Page
	if selection == "Machine Learning Overview":
		st.info("What is Machine Learning?")
		st.markdown('''How does this all work? How are able to classify how a person feels from
					just a simple tweet? to answer this question we first have to understand what 
					Machine Learning is. By definition, Machine learning provides computers with
					the ability to learn without being explicitly programmed.
					Let’s try to understand Machine Learning in layman terms. \nConsider you are 
					trying to toss a paper to a dustbin.''')
		image = Image.open(r'resources/imgs/dustbin.jpeg')
		st.image(image, use_column_width=True)

		st.markdown('''After first attempt you realise that you have put too much force in it.
		After second attempt you realise you are closer to target but you need to increase your
		 throw angle. What is happening here is basically after every throw we are learning something
		  and improving the end result. We are programmed to learn from our experience.
		We can do something similar with machines too. We can program a machine to learn from every
 		attempts/experiences/data-points and then improve the outcome. Let’s see paper toss example in Machine
  		and Non-Machine approach.''')

		st.subheader('''A Generic Program (Non Machine Learning)''')

		st.markdown('''In our above example, a generic program would tell computer to measure the distance and
		 angle and apply some pre-defined formula to calculate the force required. Now if you add a fan (wind force)
		  to your setup, this program will continuously miss target and won’t learn anything from it’s failed attempt.
		   To get the outcome right, you need to reprogram taking wind factor into your formula.''')

		st.subheader('''A Machine Learning Program''')

		st.markdown('''Now, for the same example a Machine Learning program would begin with a generic formula but after
		 every attempt/experience refactor it’s formula. As the formula is continuously improved using more experiences
		  (data points) the outcome too improved. You see these things into action around you in YouTube’s Video 
		   and Facebook’s News Feed Content etc''')
		st.markdown('''Another more technical definition of Machine Learning is — A computer program is said to learn from
		 experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as
		  measured by P, improves with experience E. This basically means in machine learning for any task a machine improves
		   it’s performance with its experience. This is exactly what we observed in our paper toss example.''')
		st.markdown('''You don’t need a Machine Learning algorithm to calculate a person’s age from his date of birth. But you would use a Machine 
		Learning algorithm to guess a person's age using his Music likes. For example your data would point that Led Zeppelin
		and The Doors fans are mostly 40+ and Selena Gomez fans are generally younger than 25. Machine Learning can be used in
		 literally everything around you. But it’s important to understand that does the problem really needs to be solved through
		  Machine Learning or not.''')
    		

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")

		#A table that explains the sentiment predictions
		image = Image.open(r'resources/imgs/classifications.png')
		st.image(image, use_column_width=True)


		
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")
		st.markdown("or alternatively")
		uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
		if uploaded_file is not None:
    			tweet_text = pd.read_csv(uploaded_file)
					
		if st.checkbox('Show Model Acuracy on Training data'): # Model acuracy  is hidden if box is unchecked
			st.markdown("NB:  Untick before classifying")
			#calculation and prediction of Linear SVC Model using raw data
			news_vectorizer1 = open("resources/vectoriser.pkl","rb")
			tweet_cv1 = joblib.load(news_vectorizer1)
			predictor1 = joblib.load(open(os.path.join("resources/linearSVC.pkl"),"rb"))
			results_linear = []
			n = 0
			while n < len(raw['message']):	
				vect_text1 = tweet_cv1.transform([raw['message'][n]]).toarray()
				prediction = predictor1.predict(vect_text1)
				results_linear.append(prediction)
				n+=1
			st.write("Linear SVC Model Accuracy on Raw/Training data is :",accuracy_score(raw['sentiment'].values, results_linear))

			#Calculation and prediction for Logistic regresssion model using raw data

			news_vectorizer2 = open("resources/tfidfvect.pkl","rb")
			tweet_cv2 = joblib.load(news_vectorizer2)
			predictor2 = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			results_logistic = []
			n = 0
			while n < len(raw['message']):	
				vect_text1 = tweet_cv2.transform([raw['message'][n]]).toarray()
				prediction = predictor2.predict(vect_text1)
				results_logistic.append(prediction)
				n+=1
			st.write("Logistic Regression  Model Accuracy on Raw/Training data is :",accuracy_score(raw['sentiment'].values, results_logistic))



			



		#Classifier selection
		Classifier = st.selectbox("Choose Classifier",['Linear SVC','Logistic regression'])
		size = st.selectbox("Choose Size of results table",[5,10,15,20,30,60])
     
     
		if st.button("Classify"):
			

			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if Classifier =='Linear SVC':
					st.text("Using Linear SVC classifier ..")
					# Vectorizer
					news_vectorizer = open("resources/vectoriser.pkl","rb")
					tweet_cv = joblib.load(news_vectorizer)
					predictor = joblib.load(open(os.path.join("resources/linearSVC.pkl"),"rb"))
			elif Classifier == 'Logistic regression':
					st.text("Using Linear Regression Classifeir ..")
					# Vectorizer
					news_vectorizer = open("resources/tfidfvect.pkl","rb")
					tweet_cv = joblib.load(news_vectorizer)
					predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			elif Classifier == 'Linear Regression':
				st.text("Using Logistic Regression Classifeir ..")
				# Vectorizer
				news_vectorizer = open("resources/tfidfvect.pkl","rb")
				tweet_cv = joblib.load(news_vectorizer)
				predictor = joblib.load(open(os.path.join("resources/LinearRegression.pkl"),"rb"))
			elif Classifier == 'RFC':
				st.text("Using RFC Classifeir ..")
				# Vectorizer
				news_vectorizer = open("resources/vectoriser.pkl","rb")
				tweet_cv = joblib.load(news_vectorizer)
				predictor = joblib.load(open(os.path.join("resources/RFC.pkl"),"rb"))

			#predictor = joblib.load(open(os.path.join("resources/linearSVC.pkl"),"rb"))
			results = []
			n = 0
			while n < len(tweet_text):	
				vect_text = tweet_cv.transform([tweet_text['message'][n]]).toarray()
				prediction = predictor.predict(vect_text)
				results.append((tweet_text['message'][n],prediction[0]))
				n+=1


			df = pd.DataFrame(results,columns=['Message','Sentiment'])


			#Table that tabulates the results
			predictions = st.table(df.head(size))

			st.success("Text Categorized as: {}".format(predictions))

			#Bar Graph showing the spread of sentiments in the results table
			st.subheader('Counts of tweets per class')
			plt.bar([1,2,3,4], df['Sentiment'].value_counts(), color=['red', 'green', 'blue', 'orange'])
			plt.xticks([1,2,3,4], ['pro', 'news', 'neutral', 'anti'])
			plt.ylabel('Count')
			plt.xlabel('Sentiment')
			st.pyplot()
			st.markdown('''Count number of most occuring words Table''')

			from collections import Counter
			count = Counter(" ".join(df["Message"]).split()).most_common(20)
			st.table(pd.DataFrame(count,columns=['Word','Number of Occurances']).head(size))


			#words frequency in the results dataframe
			sns.set(style="white")
			count_df = pd.DataFrame(count,columns = ['word', 'Number of Occurances'])
			# Visualising on a barplot.

			fig, ax = plt.subplots(figsize = (9, 9))
			sns.barplot(y="word", x='Number of Occurances', ax = ax, data=count_df, palette="Paired").set_title('Count number of most occuring words Graph')
			st.pyplot()
			plt.savefig(r'resources/wordcount_graph.png')


			#Funnel graph showing the spread of sentiments in the results table
			clean_train_df = tweet_text
			clean_train_df['sentiment'] = df['Sentiment']
			i = 0
			news = []
			pro = []
			neutral = []
			anti = []
			while i < len(clean_train_df):
				if clean_train_df['sentiment'].iloc[i] == 2:
					news.append(clean_train_df['message'].iloc[i])
				elif clean_train_df['sentiment'].iloc[i] == 1:
					pro.append(clean_train_df['message'].iloc[i])
				elif clean_train_df['sentiment'].iloc[i] == 0:
					neutral.append(clean_train_df['message'].iloc[i])
				else:
					anti.append(clean_train_df['message'].iloc[i])
				i += 1

			st.markdown('''Count number of Sentiments (ordered by message)''')
			temp = clean_train_df.groupby('sentiment').count()['message'].reset_index().sort_values(by='message',ascending=False)
			temp.style.background_gradient(cmap='Purples')
			st.table(temp.head())

			fig = go.Figure(go.Funnelarea(
				text = ["Pro","News", "Neutral", "Anti"],
				values = temp['message'].values,
				title = {"position": "top center", "text": "Sentiment Distribution"}
				))
			st.plotly_chart(fig)
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
