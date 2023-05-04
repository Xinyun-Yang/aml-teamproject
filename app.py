import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import firestore, storage
import os
import plotly.express as px
from PIL import Image

from datetime import datetime
import plotly.graph_objects as go
from wordcloud import WordCloud

import pickle
import tempfile
from joblib import dump, load
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import io
import json
from sklearn.feature_extraction.text import CountVectorizer


# Connecting to Google Acount
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "google-cloud-keys-bookrec.json"


storage_client = storage.Client()

# load the training data file from bucket
# bucket = storage_client.bucket('laura-goodreads')
# blob = bucket.blob('Goodreads_books_with_genres.csv')
# data_file = tempfile.gettempdir() + '/books2.csv'
# blob.download_to_filename(data_file)
# df = pd.read_csv(data_file, encoding='ISO-8859-1')

bucket = storage_client.bucket('bk-rec-model')
blob = bucket.blob('Goodreads_books_with_genres.csv')
data_file = tempfile.gettempdir() + '/Goodreads_books_with_genres.csv'
blob.download_to_filename(data_file)
df = pd.read_csv(data_file, encoding='ISO-8859-1')

st.set_page_config(page_title="Book Recommendation Generator", layout="wide")

st.title("Book Recommendation Generator")

data_and_problem_overview, model_training, model_testing, google_cloud, team = st.tabs(["Data and Problem Overview", "Modeling", "Interaction", "Google Cloud", "Team"])

with data_and_problem_overview: 
    
#    bucket = storage_client.bucket('laura-goodreads')
#    blob1 = bucket.blob('Goodreads_books_with_genres.csv')
#    data_file1 = tempfile.gettempdir() + '/books1.csv'
#    blob1.download_to_filename(data_file1)
#    df = pd.read_csv(data_file1)

    
    #downloading training data from kaggle 
    st.subheader("Problem")
    st.write("It's so hard to find a book to read with so many options! Our model makes this dilemma easier by generating book recommendations based on what type of book the user is looking for.")
    st.subheader("Training Dataset Preview")
    
    #Histogram of Book ratings
    st.subheader("Sample Posts")
    st.write(df.head(10))
    st.write("Our team found this dataset on Kaggle. It features the 100000 Best Books Ever List on the book review site Goodreads. The dataset has 12 columns and 10000 unique values.")
    st.subheader("Distribution of Average Book Ratings")
    fig = px.histogram(df, x='average_rating')
    st.plotly_chart(fig)
    st.write("Using our model, we have been able predit what books to read off of user ratings and the book description.")
    
    #10 Authors with most books rated
    st.subheader("Top 5 Authors with the highest number of books on the list")
    df_distinct = df.drop_duplicates(subset=['Title'])
    top_authors = df_distinct['Author'].value_counts()[:5]
    fig = px.bar(top_authors, x=top_authors, y=top_authors.index, orientation='h', height=600, width=900, color_discrete_sequence=[('pink', 'steelblue', 'plum', 'papayawhip', 'lightgreen')])
    fig.update_layout(title="", xaxis_title="Number of Books", yaxis_title="Authors")
    st.plotly_chart(fig)
    st.write("The graph above shows the top 10 authors with the most books in the Goodreads list.\
                 Stephen King has the most books on the list at 43. This is important for readers to know\
                 because it illustrates the portion of authors in the dataset.")
    
    #Top 10 Books with the Most Ratings
    st.subheader("Top 10 books with the Most ratings")
    df.sort_values('ratings_count', ascending = False).loc[:, 'Title':'ratings_count'].head(10).set_index('Title')
    most_rated = df.sort_values('ratings_count', ascending = False).drop_duplicates(subset=['Title']).loc[:, 'Title':'ratings_count'].head(10)
    fig2 = px.bar(most_rated, x="ratings_count", y="Title", orientation='h', height=500, width=1000, color="ratings_count")
    fig2.update_layout(title="", xaxis_title="Count", yaxis_title="Books Titles")
    # display the figure in Streamlit
    st.plotly_chart(fig2)
    st.write("The chart above shows the top 10 books with the most ratings. This means that the books listed\
                 have been rated the most in the dataset, not that these books are the most popular. This information\
                 is important because it illustrates the ratio of ratings for the most popular books, which can signify\
                 that the overall ratings for these books are most accurate.")

##################################################################################################################################################################################
with model_training:
    st.header("Book Recommendation Storytelling Chart")
    bucket = storage_client.bucket('bk-rec-data')
    blob3 = bucket.blob('BigStoryChart.png')

    # # Download the image data as bytes
    image_bytes = BytesIO()
    blob3.download_to_file(image_bytes)
    
    # Make sure that the stream is at the beginning
    image_bytes.seek(0) 

    # Verify that the downloaded data is not empty
    if image_bytes.getbuffer().nbytes == 0:
      raise ValueError("Downloaded image is empty.")
    
    # Open the image using the PIL library
    model_plan = Image.open(image_bytes)
    st.image(model_plan)
    st.write("The diagram above illustes how our team created our model and how it will be used in our app.")
    
    st.divider()
    
    st.header("Model improvement")
    st.write("")
    st.subheader("Techniques")
    code_tech = '''

    # Create categorical features

    # 1. rating_btween
    data2['rating_between'] = data2.average_rating.apply(lambda x: 
        '0_to_1' if x>=0 and x<=1 else 
        '1_to_2' if x>1 and x<=2 else 
        '2_to_3' if x>2 and x<=3 else 
        '3_to_4' if x>3 and x<=4 else 
        '4_to_5')

    # 1 if grade falls under a particular group (say 4 and 5), else 0
    rating_data = pd.get_dummies(data2['rating_between'])


    # 2. language code
    # 1 if the book is written in a particular language eg English, else 0 
    language_data = pd.get_dummies(data2['language_code'])

    # contain the values of rating_data and language_data and will also have the values of average grade and number of grades
    features = pd.concat([rating_data, 
                        language_data,
                        data2['average_rating'],
                        data2['ratings_count']], axis=1)


    # use MinMaxScaler() to scale the value to reduce the bias for some of the books that have too many features. 
    # The algorithm will find the median for all and equalize it:
    min_max_scaler = MinMaxScaler()
    features_new = min_max_scaler.fit_transform(features)


    
    # Set up vectorizer
    pattern = re.compile('(?u)\\b\\w\\w+\\b')
    en_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    def custom_tokenizer(document):
        document = document.lower() 
        doc_spacy = en_nlp(document)
        lemmas = [token.lemma_ for token in doc_spacy] 
        return [token for token in lemmas if token not in STOP_WORDS and pattern.match(token)]
    vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, stop_words='english', ngram_range=(1,2))

    # Convert the descriptions + genres to a matrix of TF-IDF features
    X = vectorizer.fit_transform(data.text)
    
    '''
    with st.expander("Show code"):       
        st.code(code_tech,language='python')


    st.divider()

    # Selectbox
    code_knn = '''
    
    # KNN model
    model = neighbors.NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
    model.fit(features_new)
    dist, idlist = model.kneighbors(X)

    # Output function
    def get_recommendations(book_name):
        book_list_name = []
        book_id = data2[data2['Title'] == book_name].index
        book_id = book_id[0]
        for new_id in idlist[book_id]:
            book_list_name.append(data.loc[new_id].Title)
        return book_list_name
    
    '''
    
    code_km = '''
    
    # K-Means model
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X)

    # print the cluster labels
    print(kmeans.labels_)

    # print the books in each cluster
    for i in range(10):
        cluster_books = titles[kmeans.labels_ == i]
        print(f'Cluster {i}: {", ".join(cluster_books)}')
    
    
    '''
    
    
    code_cs = '''
    
    # Define a function to compute the pairwise cosine similarity between the feature vectors
    def compute_cosine_similarity(df):
        # Convert the text columns to numerical representations
        cv_genre = CountVectorizer()
        genre_matrix = cv_genre.fit_transform(df['genres'].astype(str).apply(lambda x: ','.join(x.split(';'))))
        cv_author = CountVectorizer()
        author_matrix = cv_author.fit_transform(df['Author'].astype(str))
        cv_title = CountVectorizer()
        title_matrix = cv_title.fit_transform(df['Title'].astype(str))
        cv_lang = CountVectorizer()
        lang_matrix = cv_lang.fit_transform(df['language_code'].astype(str))
        cv_pub = CountVectorizer()
        pub_matrix = cv_pub.fit_transform(df['publisher'].astype(str))

        # Standardize the numerical columns
        scaler = StandardScaler()
        numerical_cols = ['average_rating', 'ratings_count', 'text_reviews_count', 'publication_year']
        numerical_data = df[numerical_cols].fillna(df[numerical_cols].median())
        numerical_data = scaler.fit_transform(numerical_data)

        # Concatenate the numerical and text representations into a single feature matrix
        feature_matrix = np.concatenate([genre_matrix.toarray(),
                                          author_matrix.toarray(),
                                          title_matrix.toarray(),
                                          lang_matrix.toarray(),
                                          pub_matrix.toarray(),
                                          numerical_data],
                                        axis=1)

        # Compute the pairwise cosine similarity between the feature vectors
        cosine_sim_matrix = cosine_similarity(feature_matrix)

        return cosine_sim_matrix

    # Compute the pairwise cosine similarity between the feature vectors
    cosine_sim_matrix = compute_cosine_similarity(df)

    # Define a function to get book recommendations
    def get_recommendations(book_title, cosine_sim_matrix, df, top_k=10):
        indices = pd.Series(df.index, index=df['Title']).drop_duplicates()
        idx = indices[book_title]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]
        book_indices = [i[0] for i in sim_scores]
        return df['Title'].iloc[book_indices].values if len(book_indices) > 0 else []

    
    '''
    st.subheader("Models")
    model_names = ['KNN', 'K-Means']

    st.write("")
    selected_model = st.selectbox('Select a model to check corresponding code', model_names)
    if selected_model == 'KNN':
        st.code(code_knn, language='python')
    elif selected_model == 'K-Means':
        st.code(code_km, language='python')        
    else:
        st.write('Please select a model.')

    st.divider()
    st.subheader("Final Model")
    st.code(code_cs, language='python')

#########################################################################################################################################################################




with model_testing:  


    #1 Load the matrix model

    bucket = storage_client.bucket('bk-rec-model') # change the bucket name
    blob = bucket.blob('matrix.pkl')
    matrix_file = tempfile.gettempdir() + '/matrix.pkl'
    blob.download_to_filename(matrix_file)
    cosine_sim_matrix = pickle.load(open(matrix_file, 'rb'))

   
    # remove rows with 0 pages
    df = df.drop(df.index[df['num_pages'] == 0])
    df = df.drop(df.index[df['ratings_count'] == 0])
    df.drop(columns=['isbn', 'isbn13', 'Book Id'], inplace=True)

    # Convert the "publication_date" column to datetime objects, handling invalid dates
    df['publication_date'] = pd.to_datetime(df['publication_date'], format='%m/%d/%Y', errors='coerce')

    # Extract the year from the datetime objects and save it as a new column
    df['publication_year'] = df['publication_date'].dt.strftime('%Y').astype(float)
    df.drop(columns = 'publication_date', inplace=True)

    st.subheader("1. Recommend books with book title")

    book_title = st.text_input("Title")
    st.caption('Please input specific book title.')
    st.caption('Example Input:   Notes from a Small Island, **notes from a small island**, The Changeling Sea, The Lord of the Rings: Weapons and Warfare')
    st.caption('Not sensitive to white spaces and upper/lowercase letters')

    def get_recommendations(book_title,df,cosine_sim_matrix,top_k=10):
        book_title = book_title.lower()
        df_lower = df.apply(lambda x: x.astype(str).str.lower())
        indices = pd.Series(df.index, index=df_lower['Title']).drop_duplicates()
        idx = indices[book_title]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        # sim_scores = np.array(sim_scores)
        # if sim_scores.ravel()[0]:
        #     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # else:
        #     sim_scores = []
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) if sim_scores else []
        sim_scores = sim_scores[1:top_k+1]
        book_indices = [i[0] for i in sim_scores]
        result = df[['Title', 'Author', 'publisher','publication_year','genres']].iloc[book_indices]
        # result = result.rename({'publisher':'Publisher', 'publication_year':'Publication Year', 'genres':'Genres'}, index = None)
        result.columns = ['Title', 'Author','Publisher', 'Year', 'Genres']
        return result if len(book_indices) > 0 else []
    
    if st.button("Recommend"):
        if book_title == "":
            st.error('Please input a book title.', icon="⚠️")
        else:
            try:
                recommendation_title = get_recommendations(book_title,df,cosine_sim_matrix,top_k=10)
                # st.write(recommendation_title)
                if isinstance(recommendation_title, str):
                    st.warning("Sorry we couldn't find any recommendations with the information you provided. Please try again with different search criteria, or enter a book title.", icon="😔")
                else:
                    recommendation_title = recommendation_title.reset_index(drop=True)
                    recommendation_title.index += 1
                    recommendation_title = recommendation_title.reset_index()
                    recommendation_title.columns = ['Index', 'Title', 'Author', 'Publisher', 'Year', 'Genres']
                    recommendation_title = recommendation_title.set_index('Index')
                    st.dataframe(recommendation_title.rename_axis('', axis='index'), use_container_width = True)
            except KeyError as e:
                st.warning('Sorry we currently do not have any recommendations for the book you entered. Please try another one.', icon="😔")

    st.divider()

    st.subheader("2. Recommend books based on reading preference")
    st.caption("Please fill out the information below. If you do not have a preference for a particular information, you may leave it blank.")

    
    def get_recommendations_p(author=None, publisher=None, genre=None, year=None,top_k=10):
    # Create a mask based on author, publisher, language, and genre
        mask = True
        if author is not None:
            mask &= df['Author'].str.contains(author)
        if publisher is not None:
            mask &= (df['publisher'] == publisher)
        if genre is not None:
            mask &= df['genres'].str.contains(genre)
        if year and year.isdigit():
            year = int(year)
            mask &= (df['publication_year'].between(year - 4, year + 4))

        # Check if the mask returns at least one row
        if mask.sum() == 0:
            return "No book"
        # Get the index of the book that matches the mask
        idx = df[mask].index[0]
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k+1]
        book_indices = [i[0] for i in sim_scores]

        result = df[['Title', 'Author', 'publisher','publication_year','genres']].iloc[book_indices]
        result.columns = ['Title', 'Author','Publisher', 'Year', 'Genres']

        return result if len(book_indices) > 0 else []

    author = st.text_input("Author")
    # genre = st.selectbox('Select a model to check corresponding code', model_names) # multiselection
    st.caption("Example Input: Euripides")
    st.write("")

    publisher = st.text_input("Publisher")
    st.caption("Example Input: Cambridge University Press / W.W. Norton & Company")
    st.write("")

    genre = st.text_input("Genre")
    st.caption("Example Input: Classic")
    st.write("")
    
    year = st.text_input("Publication Year").strip()
    st.caption("Example Input: 2003 / 1958")  
    st.caption('Not sensitive to white spaces, but the app will display a editted warning if you enter other characters. We tried the st.number_input(), but the format was not good.')
    st.write("")


    if st.button("Recommend by preference"):
        if year and year.isdigit()==0:
            st.error('Please input a valid number for "Publication Year" or leave it blank.', icon="🚨")
        else:
            recommendation_pref = get_recommendations_p(author=author, publisher=publisher, genre=genre, year=year, top_k=10)
            if isinstance(recommendation_pref, str):
                st.warning("Sorry we can not provide specific recommendations with the information provided. Please try more detailed information or use different search criteria.", icon="😔")
            else:
                recommendation_pref = recommendation_pref.reset_index(drop=True)
                recommendation_pref.index += 1
                recommendation_pref = recommendation_pref.reset_index()
                recommendation_pref.columns = ['Index', 'Title', 'Author', 'Publisher', 'Year', 'Genres']
                recommendation_pref = recommendation_pref.set_index('Index')
                st.dataframe(recommendation_pref.rename_axis('', axis='index'), use_container_width=True)






    # description = st.text_area("Description of book content you are interested in:") # with instruction

##############################################################################################################################################################################


with google_cloud:

    db = firestore.Client()
    query = db.collection(u'bookreclist').order_by(u'created', direction=firestore.Query.DESCENDING)
    posts = list(query.stream())
    docs_dict = list(map(lambda x: x.to_dict(), posts))
    rc = pd.DataFrame(docs_dict)


    # Process datetime data
    rc['created'] = pd.to_datetime(rc['created'])
    rc = rc[rc['created'].notna()]
    rc['created'] = rc['created'].apply(lambda x: int(x.timestamp()))


    # Time limit on genres
    created_end = datetime.fromtimestamp(rc.iloc[:1,:].created.values[0])
    created_start = datetime.fromtimestamp(rc.iloc[-1:,:].created.values[0])

    date_start = st.sidebar.date_input("From", value = created_start, min_value = created_start, max_value = created_end)
    date_end = st.sidebar.date_input("To", value = created_end, min_value = created_start, max_value = created_end)
    # posts_length_range = st.sidebar.slider("Posts Length", min_value=1, max_value=9999, value=[1, 9999])

    date_start_str = date_start.strftime('%Y-%m-%d')
    date_end_str = date_end.strftime('%Y-%m-%d')
    rc['date'] = rc['created'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
    rc = rc.loc[(rc.date >= date_start_str) & (rc.date <= date_end_str), :]

    st.write("")
    # Histogram
    st.header("Daily Sales")
    st.write("")
    chart, desc = st.columns([2,1])
    with chart: 
        fig = px.histogram(rc, x='date')
        st.plotly_chart(fig)
    with desc:        
        st.write("The chart on the left shows how many people used our model (or bought books from the bookstore)")



    st.divider()

    ## World Cloud ##

    # use explode() to split the recommendations list into separate rows if we need to use a dataframe with
    # each recommended title on a different row
    rc_exp = rc.explode('recommendations')
    rc_exp = rc_exp.reset_index(drop=True)   # reset the index
    # Match the books to find the genres
    rc_merged = pd.merge(rc_exp, df, left_on='recommendations', right_on='Title', how='left')

    # Select the 'Title' and 'Genres' columns and drop any duplicates
    rc_output = rc_merged[['Title', 'genres']]

    st.write("")
    st.header("Popular Genres by users")
    st.write("")
    image, desc2 = st.columns([2,1])
    with image:
      
      # Concatenate all genres into a single string separated by spaces
      genres_text = " ".join(rc_output['genres'].astype(str).str.replace(";", " "))

      # Generate the WordCloud image
      wordcloud_image = WordCloud().generate(genres_text).to_image()

      st.image(wordcloud_image)

    with desc2:
        # st.subheader("Popular Genres")
        st.write("The image on the left shows the most popular genres within certain period of time")


    "---"


    ## Sample Posts ##
    st.header("Sample Posts") 

    if st.button("Show Sample Posts"):

        placeholder = st.empty()

        with placeholder.container():
            
            st.table(rc)
            
            # for index, row in rc.iterrows():
                # text = row["title"].strip()
                # if text != "":

                    # col1, col2 = st.columns([3,1])
                    # with col1:
                        # st.write(text)
                    # with col2:          
                        # st.info(row['recommendations'])



        if st.button("Clear", type="primary"):
            placeholder.empty()



#############################################################################################################################################
with team:

    members = [
        {"name": "Meiling Ge", "description": "Meiling is a MANA candidate at Tulane University."},
        {"name": "Sara Johansen", "description": "Sara is a MANA candidate at Tulane University."},
        {"name": "May Paddor", "description": "May is a MANA candidate at Tulane University."},
        {"name": "Matthew Stuckey", "description": "Matthew is a MANA candidate at Tulane University."},
        {"name": "Xinyun Yang", "description": "Xinyun is a MANA candidate at Tulane University."}
    ]

    # Create a dictionary to hold the image data
    image_data_dict = {}

    # Loop through the image names and download the data for each image
    for i in range(1, 6):
        # Get the name of the current image
        image_name = f"member{i}.png"

        # Get the blob corresponding to the image
        blob = storage_client.bucket("bk-rec-data").blob(image_name)

        # Download the image data as bytes
        image_bytes = io.BytesIO()
        blob.download_to_file(image_bytes)

        # Make sure that the stream is at the beginning
        image_bytes.seek(0)

        # Verify that the downloaded data is not empty
        if image_bytes.getbuffer().nbytes == 0:
            st.warning(f"Downloaded image '{image_name}' is empty.")
        else:
            # Add the image data to the dictionary
            image_data_dict[f"member{i}"] = image_bytes

    st.subheader("About")
    st.write("BookWorms consulting is a new media consulting group located in New Orleans, LA.\
              Our goal is to help firms provide personalized recommendations for their consumers\
            using AI and machine learning techniqes. Read Below to learn more about our team!")
        
    st.subheader("Team")

    # Display the images and information for each team member
    for member in members:
        col = st.columns(2)[0]

        # Get the image data for this team member
        image_name = f"{member['name'].lower().replace(' ', '')}.png"
        if image_name not in image_data_dict:
            col.warning(f"No data available for {member['name']}.")
        else:
            image_data = image_data_dict[image_name]
            image = Image.open(image_data)

            # Display the image and information for this team member
            col.image(image)
            col.subheader(member['name'])
            col.write(member['description'])


