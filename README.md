# Fake-Real_News_Detection
This Machine Learning Model detects whether the news is fake or real using Logistic Regression with accuracy of 94.7%.

# TF-IDF

Term frequency-inverse document frequency is a text vectorizer that transforms the text into a usable vector. It combines 2 concepts, Term Frequency (TF) and Document Frequency (DF).

The term frequency is the number of occurrences of a specific term in a document. Term frequency indicates how important a specific term is in a document. Term frequency represents every text from the data as a matrix whose rows are the number of documents and columns are the number of distinct terms throughout all documents.

Document frequency is the number of documents containing a specific term. Document frequency indicates how common the term is.

Inverse document frequency (IDF) is the weight of a term, it aims to reduce the weight of a term if the term’s occurrences are scattered throughout all the documents. IDF can be calculated as follow:
  
![image](https://user-images.githubusercontent.com/128599179/233066883-5fb976f0-4bd5-450e-b419-76881fd340a3.png)

Where idfᵢ is the IDF score for term i, dfᵢ is the number of documents containing term i, and n is the total number of documents. The higher the DF of a term, the lower the IDF for the term. When the number of DF is equal to n which means that the term appears in all documents, the IDF will be zero, since log(1) is zero, when in doubt just put this term in the stop word list because it doesn't provide much information.
	
The TF-IDF score as the name suggests is just a multiplication of the term frequency matrix with its IDF, it can be calculated as follow:
  
![image](https://user-images.githubusercontent.com/128599179/233066928-6e29b8fb-c877-419b-a1bf-004621bd0b7d.png)

Where wᵢⱼ is TF-IDF score for term i in document j, tfᵢⱼ is term frequency for term i in document j, and idfᵢ is IDF score for term i.



# Logistic Regression

Logistic regression (or logit regression) estimates the probability of an event occurring, such as yes or no, based on a given dataset of independent variables. Since the outcome is a probability, the dependent variable is bounded betwee n 0 and 1. 
	
In logistic regression, a logit transformation is applied to the odds-that is, the probability of success divided by the probability of failure. This is also commonly known as the log odds or the natural logarithm of odds, and this logistic function is represented by the following formulas:
 
### Logit(pi) = 1/(1+ exp(-pi))



### ln(pi/(1-pi)) = Beta_0 + Beta_1*X_1 + … + B_k*K_k


In this logistic regression equation, logit(pi) is the dependent or response variable and x is the independent variable. For binary classification, a probability less than .5 will predict 0 while a probability greater than 0 will predict 1.  


# Stemming
Stemming is a natural language processing technique that is used to reduce words to their base or root form to normalize text and making it easier to process. This is done by removing prefixes, suffixes, and other word endings that may vary while preserving the stem of the word.


# Project steps 
+ ## ***Steps for a machine learning model that can determine if a piece of news was fake or real***
  
### Collection and Pre-processing of data 
Collect the dataset of news articles that are having field as title, author, text and label. Pre-process the data by cleaning the text,checking of any null values in dataset, replacing that null values with empty string and converting the text into the reduced form by the method of Stemming and converting the text into numerical features using techniques like TF-IDF.

``` 
# loading the dataset to a pandas DataFrame
news_dataset = pd.read_csv('/content/train.csv',encoding= 'unicode_escape')

# replacing the null values with empty string
news_dataset = news_dataset.fillna('')

# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

``` 
### Performing Stemming using Porter Stemmer Algorithm
The Porter Stemmer algorithm is used to remove the suffixes from an English word and obtain its stem which becomes very useful in the field of Information Retrieval.

``` 
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content
    
news_dataset['content'] = news_dataset['content'].apply(stemming)

``` 
### Conversion of textual data to numeric data using TF-IDF
TF-IDF feature converts the textual data into the numerical data.
``` 
# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

``` 
### Spliting the data into training and testing sets
Split the dataset into a training set and a testing set to evaluate the model's performance.

```  
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=2)

```  
### Training the model 
Logistic Regression algorithm is used for the classification purpose. Machine learning model is trained on the training set.

```
# Train a Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

```
### Evaluating the model
Evaluate the model's performance on the training set.

```
# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

```
Evaluate the model's performance on the testing set.

```
# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

```
### Use the model to predict new data
Once the model is trained, predict whether a news article is real or fake.

```
X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')

```
In order to pre-process the data, we first import the dataset and used the Porter Stemmer Algorithm for the Stemming and the TF-IDF vectorizer for conversion of text to numeric form. After separating the data into training and testing sets, we used the training set to build a logistic regression model. We assess the model's performance on the testing and training set both and employ it to determine if a recent news story is real or fake.


