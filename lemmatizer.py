import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
# nltk.data.path.append('/Users/neelagarwal/IDrive Sync/Cloud-Drive/For Cloud/MS DS UoR/Semester 3/Data Science Practicum/Kaggle/training_data.csv')
# nltk.download('popular')


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from nltk.corpus import stopwords
from nltk.corpus import wordnet

import cleantext
import demoji
import re
# import wordnet

lemmatizer = WordNetLemmatizer()

def tweet_cleaner(input_text):
     
    # # removal of b's in the beginning 
    # if (input_text.startswith("b'") or input_text.startswith("b\"")) and input_text.endswith("'"):
    #     input_text = input_text[2:-1]
        
    # if (input_text.startswith("b'") or input_text.startswith("b\"")) and (not(input_text.endswith("'"))):
    #     input_text = input_text[2:]

    input_text = demoji.replace(input_text,"") # removal of emoji
    input_text = " ".join(input_text.split()) # removal of extra spaces
    input_text = re.sub(r'https?:\/\/.*[\r\n]*', '', input_text) # removal of url startign with 'http'

    # input_text_list = input_text.split()
    if not input_text:
        return ""
    input_text = cleantext.clean(input_text,
                                    punct = True, # removal of punctuation
                                    lowercase = True # converting string to lowercase
                                    )
    stop_words = set(stopwords.words('english'))
    input_text = " ".join([word for word in input_text.split() if word not in stop_words]) # removal of stopwords
    input_text = " ".join([word for word in input_text.split() if len(word) >= 3]) # removal of words less than 3 characters
    return(input_text)

##Tags the words in the tweets

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return(wordnet.ADJ)
    elif nltk_tag.startswith('V'):
        return(wordnet.VERB)
    elif nltk_tag.startswith('N'):
        return(wordnet.NOUN)
    elif nltk_tag.startswith('R'):
        return(wordnet.ADV)
    else:          
        return(None)

# removal of b's in the beginning 
# def removeB(input_text):
#     if (input_text.startswith("b'") or input_text.startswith("b\"")) and input_text.endswith("'"):
#         input_text = input_text[2:-1]
        
#     if (input_text.startswith("b'") or input_text.startswith("b\"")) and (not(input_text.endswith("'"))):
#         input_text = input_text[2:]
#     return input_text

##Lemmatizes the words in tweets and returns the cleaned and lemmatized tweet
def lemmatize_tweet(tweet):
    #tokenize the tweet and find the POS tag for each token
    # return if empty string
    if not tweet:
        return "" 
    # tweet = removeB(tweet)
    tweet = tweet_cleaner(tweet) #tweet_cleaner() will be the function you will write
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(tweet))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_tweet = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_tweet.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_tweet.append(lemmatizer.lemmatize(word, tag))
    return(" ".join(lemmatized_tweet))