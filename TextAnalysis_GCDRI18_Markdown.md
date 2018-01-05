
# GCDRI Jan 2018:  Text Analysis Classification with NLTK and scikit-learn
- Jupyter Notebook by Rachel Rakov

## Welcome!  Let's get started by importing some data!
- A large collection of textual data is called a *corpus* (pluralized as *corpora*) 
- I will be using the term corpus or corpora throughout this workshop


```python
import nltk
import matplotlib
from nltk.book import *
```

## Let's take a look at some text!


```python
print (text3[:100])
```

## Common contexts
### Takes two words as an argument, returns contexts in which they appear similarly across the text (within one text)


```python
text1.common_contexts(["pretty", "very"])
#requires a list as an argument
## across the same text
```

## Practice!

### Pick another text from the texts above, corpus, look at the top 100 words, and then see if you can find words that have common contexts between them.



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
text3.common_contexts(["light", "dark"])
```

## Dispersion plots
### Used to see where particular words appear in your corpus


```python
%matplotlib notebook

text1.dispersion_plot(["Starbuck", 'whale', 'Ahab', 'Ishmael', 'sea',"death"])
#text3.dispersion_plot(["God", "fruit", "garden", "woman"])

##Shows location of word in a text
```

### Recreate your own lexical dispersion plot with one of the texts above and 5 words



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
text6.dispersion_plot(["knights", 'ARTHUR', 'ROBIN', 'grail', 'lady',"ni"])
```

## Great!  Let's count things now!


```python
#How many words are in the texts?
print (len(text1))
print (len(text6))
print "\n"


# How many times are particular words in the text?
print (text1.count("Ahab"))
print (text6.count("Arthur"))
print (text6.count("ARTHUR"))
```


```python
## How can I get percentages of words in a text?
print (100* text1.count("the")/float(len(text1)))

print (100* text3.count("the")/float(len(text3)))
```

## Frequency Distributions


```python
from nltk import FreqDist
fdist = FreqDist(text1)
#print (fdist)


```


```python
## shows how many times "whale" occurs in text
#fdist["whale"] 

#shows percentage of "whale" in the text
fdist.freq("whale")

#fdist.most_common(100)

# shows the most common token
#fdist.max()


```

### Create a frequncy distribution that shows the percentage of the word "God" in the book of Genesis (text3)


```python

```


```python

```


```python

```


```python

```


```python

```


```python
gen_fdsit = FreqDist(text3)
gen_fdsit.freq("God")
```

### A happy chart from "Natural Language Processing with Python" (Bird, Klein & Loper)
       # Example                                        #Description
      
    fdist = FreqDist(samples) 	   create a frequency distribution containing the given samples
    fdist[sample] += 1 	           increment the count for this sample
    fdist['monstrous'] 	           count of the number of times a given sample occurred
    fdist.freq('monstrous') 	   frequency of a given sample
    fdist.N() 	                   total number of samples
    fdist.most_common(n) 	       the n most common samples and their frequencies
    for sample in fdist:           iterate over the samples
    fdist.max() 	               sample with the greatest count
    fdist.tabulate() 	           tabulate the frequency distribution
    fdist.plot() 	               graphical plot of the frequency distribution
    fdist.plot(cumulative=True) 	umulative plot of the frequency distribution
    fdist1 |= fdist2 	           update fdist1 with counts from fdist2
    fdist1 < fdist2 	 test if samples in fdist1 occur less frequently than in 

## Word Tokenization


```python
from nltk import word_tokenize
```


```python
paragraph = "Far out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the Galaxy lies"\
" a small unregarded yellow sun.  Orbiting this at a distance at roughly nintey-eight million miles is an utterly "\
"insignificant little blue-green planet whose ape-descended life forms are so amazingly primitive that they still think "\
"digital watches are a pretty neat idea."
print (paragraph)
```


```python
p = word_tokenize(paragraph)
print (p)
```

## Text comparisions using Frequency Distributions


```python
from nltk.corpus import brown
```


```python
cats = brown.categories()
for i in cats:
    print (i)
```


```python
romance_sent = brown.sents(categories=["romance"])
print (romance_sent[:5])
```


```python
news = brown.words(categories=["news"])  #get all of the words from the "news" category
romance = brown.words(categories=["romance"]) # get all of the words from the "romance" category

## Build some frequency distribution!!!! 
fdist_news = FreqDist(w.lower() for w in news)
fdist_romance = FreqDist(w.lower() for w in romance)

modals = ["can", "could", "might", "may", "would", "must", "will"]

print ("word:\t news \t \t romance")
print ("_________________________________")
for m in modals:
    print (m+":,\t" "%f \t %f")  %(fdist_news.freq(m)*100, fdist_romance.freq(m)*100)

```

## Part-of-Speech (POS) tagging


```python
from nltk import pos_tag  ###part of speech tags a list
from nltk import pos_tag_sents ### pos tags sentences, rather than individual words
```


```python
paragraph = "Far out in the uncharted backwaters of the unfashionable end of the Western Spiral arm of the Galaxy lies"\
" a small unregarded yellow sun.  Orbiting this at a distance at roughly nintey-eight million miles is an utterly "\
"insignificant little blue-green planet whose ape-descended life forms are so amazingly primitive that they still think "\
"digital watches are a pretty neat idea."
p = word_tokenize(paragraph)
print (p)
```


```python
paragraph_POS = pos_tag(p)
print (paragraph_POS)
```


```python
### Because not all of these tags are intuitive ###
nltk.help.upenn_tagset("IN")
```

## Feature extraction using the Brown Corpus
#### Can we train a computer to predict whether a sentence belongs in the news corpus or the romance corpus?

### Part of speech - number of nouns in sentences
Let's start by getting all of the sentences from the news and romance categories of the Brown corpus


```python
news_sent = brown.sents(categories=["news"])
romance_sent = brown.sents(categories=["romance"])

print (len(news_sent))
print (len(romance_sent))
```

Next, let's part of speech tag each word in the sentence!

Note: We use pos_tag_sent because we are tagging sentences, rather than individual words


```python
news_pos = pos_tag_sents(news_sent)
romance_pos = pos_tag_sents(romance_sent)
```

Now let's create a function that will count how many nouns are in each sentence of the corpus


```python
def countNouns(pos_tag_sents):
    noun_count = 0
    all_noun_counts = []
    for sentence in pos_tag_sents:
        for word in sentence:
            tag = word[1]
            if tag [:2] == "NN":  ## so that we capture both singluar and plural nouns
                noun_count = noun_count+1
        all_noun_counts.append(noun_count)
        noun_count = 0
    return all_noun_counts

news_counts = countNouns(news_pos)
romance_counts = countNouns(romance_pos)

           
        
```


```python
print romance_counts[:20]
print news_counts[:20]
```

## Machine learning:  Building train and test sets
### Seperating data, getting labels, and aligning them with features


```python
import pandas as pd
import sklearn
```

## Create training and testing labels


```python
cats = ["news", "romance"] #define what categories of brown corpus we want
print (cats)
text = [brown.sents(categories=cat) for cat in cats]
test_sets_int = 500 ## specify how many test sentences we will have per category

######## create labels for test and training sets ##############
### find how many sentences there are, subtract test_sets_int for the correct number of training and testing labels
lengths = []
for i in range(len(cats)):
    start_length = len(text[i])
    print (start_length)
    length = start_length - test_sets_int
    lengths.append(length)

print (lengths)

#### concatenate the labels together #############
train_labels = ["news"]*lengths[0]+["romance"]*lengths[1]
test_labels = ["news"]*test_sets_int+["romance"]*test_sets_int

print (train_labels[:10])
print (train_labels[-10:])
```

## Create training and testing data

#### First, let's separate out the training data from the testing data


```python
#### take the first 500 count features from each dataset and use them as test - use the rest as train ######
news_values_test = news_counts[:test_sets_int]
news_values_train = news_counts[test_sets_int:]
romance_values_test = romance_counts[:test_sets_int]
romance_values_train = romance_counts[test_sets_int:]

print (len(news_values_test))
print (len(news_values_train))
print (len (romance_values_test))
print (len(romance_values_train))

```


```python
###### concatenate the lists of data together ######
train_features = news_values_train+romance_values_train
test_features = news_values_test+romance_values_test
```

## Introduction to pandas DataFrames
So new we have both train_features and train_labels, as well as test_features and test_labels.  Let's manage this data!


```python
#### create two DataFrames - one for train, one for testing ####
train_data = pd.DataFrame(train_features, columns=["number of nouns"])
test_data = pd.DataFrame(test_features, columns=["number of nouns"])
```


```python
train_data
```

Let's add our labels to our dataframe!


```python
##### you can add columns to DataFrames like this! ####
train_data["labels"] = train_labels
test_data["labels"] = test_labels
```


```python
test_data
```

## Text classification using scikit learn

#### Seperate the dataframe into data and labels, for both train and test sets


```python
####### We use the naming conventions of sklearn here ########
X_train = train_data["number of nouns"]
y_train = train_data["labels"]


X_test = test_data["number of nouns"]
y_test = test_data["labels"]
```

### Note that because we only have one feature, we need to reshape our data

##### sklearn will tell you when your data needs to be reshaped, and will tell you how to do it


```python
X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)
```


```python
print (len(train_features))
print (len(train_labels))
```

## Classification with Linear SVC

### Step 1:  Import your classifier


```python
from sklearn.svm import LinearSVC
```

### Step 2: Create an instance of your classifier 


```python
classifier = LinearSVC()
```

### Step 3: Fit, predict, and score


```python
classifier.fit(X_train,y_train)
```


```python
predictions = classifier.predict(X_test)
```


```python
for i in range(len(y_test)):
    print predictions[i], y_test[i]
```


```python
classifier.score(X_test, y_test)
```

## Evaluate your model


```python
from sklearn.metrics import confusion_matrix
```


```python
confusion_matrix(y_test, predictions)
```

# Extra materials
We may not have time to get to these things, but here are some additional materials to help you in your classification future!
- Changing classifiers
- Adding additional features

## Classification with k-nearest neighbors
What if we used the classifier we described before, k-nearest neighbors?  How well do we do?

### Remember our steps from before:


```python
from sklearn.neighbors import KNeighborsClassifier  ### import your classifier

classifier = KNeighborsClassifier(n_neighbors=3)  ### Create a new instance of your classifier 

classifier.fit(X_train, y_train)  ### fit

predictions = classifier.predict(X_test) ### predict

```


```python
classifier.score(X_test, y_test) ### score
```


```python
confusion_matrix(y_test, predictions) ### evaluate
```

## Let's add another feature - number of modal verbs in a sentence!


```python
### If any of these modal words appear in our sentences, accumulate the total for each sentence

def modals(setType):
    modals_count = 0
    modal_features = []
    modals = ["can", "could", "might", "may", "would", "must", "will"]
    for sent in setType:
        for word in modals:
            if word in sent:
                modals_count = modals_count+1
        modal_features.append(modals_count)
        modals_count = 0
    print (len(modal_features))   
    return modal_features

news_modals = modals((brown.sents(categories="news")))
romance_modals = modals((brown.sents(categories="romance")))

```


```python
print (news_modals[:30])
```

## Create training and test sets of modal features
Second verse, same as the first!


```python
###### create feature vectors of modal counts #####
news_modals_test = news_modals[:test_sets_int]
news_modals_train = news_modals[test_sets_int:]
romance_modals_test = romance_modals[:test_sets_int]
romance_modals_train = romance_modals[test_sets_int:]

print (len(news_modals_test))
print (len(news_modals_train))
print (len(romance_modals_test))
print (len(romance_modals_train))
```


```python
### concatenate the modal features #####
modal_features_train = news_modals_train+romance_modals_train
modal_features_test = news_modals_test+romance_modals_test
```


```python
print modal_features_train[:10]

```

### Adding columns to existing DataFrames
You can add columns in DataFrames by location!


```python
train_data.insert(1, "number of modals", modal_features_train)
```


```python
test_data.insert(1, "number of modals", modal_features_test)
```

### Splitting DataFrames with more than one feature
Split columns based on column order, or use the name of the column to split


```python
X_train = train_data[train_data.columns[:2]]
y_train = train_data["labels"]


X_test = test_data[test_data.columns[:2]]
y_test = test_data["labels"]
```


```python
print X_train
```

## NOTE:  Because we have more than one feature, we no longer need to reshape our data!

### Classify like before!  (No need to re-import your classifier)


```python
classifier = LinearSVC()
classifier.fit(X_train,y_train)
predictions = classifier.predict(X_test)
classifier.score(X_test, y_test)
```

### Evaluate


```python
confusion_matrix(y_test, predictions)
```

Doesn't actually improve things by much, but that also shouldn't be too much of a surprise, from when we looked at it before


```python

```


```python

```


```python

```
