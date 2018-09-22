# Tashkeela-Model
This is a diacritization model for Arabic language. This model was built/trained using the [Tashkeela](https://www.kaggle.com/linuxscout/tashkeela): the Arabic diacritization corpus on Kaggle. Arabic diacritics are often missed in Arabic scripts. This feature is a handicap for new learner to read َArabic, text to speech conversion systems, reading and semantic analysis of Arabic texts. The automatic diacritization systems are the best solution to handle this issue. But such automation needs resources as diacritized texts to train and evaluate such systems.



**NOTE:**

This model isn't the perfect; as we are going to see in a second, it has an accuracy of around 73%. So, I think it's a good start for improving and providing a state-of-the-art model for our beautiful Arabic language.



# Data

Data is a collection of Arabic vocalized texts, which covers modern and classical Arabic language. The Data contains over 75 million of fully vocalized words obtained from 97 books, structured in **text** and **XML** files.

The corpus is collected mostly from Islamic classical books, and using semi-automatic web crawling process. The Modern Standard Arabic texts crawled from the Internet represent 1.15% of the corpus, about 867,913 words, while the most part is collected from [Shamela](http://shamela.ws) Library, which represent 98.85%, with 74,762,008 words contained in 97 books. This Tashkeela data can be downloaded from the official page of the corpus from [here](https://www.kaggle.com/linuxscout/tashkeela).

After downloading the data the unzipping it, we should find the following files:

- 97 files from http://al-islam.com located at *diacritized_text* directory.
- 2 files from aljazeera located at *diacritized_text/mas/aljazeera* directory.
- 170 files from http://al-kalema.org located at *diacritized_text/mas/al-kalema.org* directory.
- 39 files from http://enfal.de located at *diacritized_text/mas/enfal.de* directory.
- 1 files manually created located at *diacritized_text/mas/manual* directory.
- 4 XML files located at *diacritized_text/mas/sulaity* directory.
- 56 files from http://diwanalarab.com located at *diacritized_text/mas/diwanalarab* directory.
- Another 20 files located in a weirdly-looking directory :).



# Preprocessing

Before getting into the NLP model, we first have to process our data to be easily digested by the model later. The preprocessing step is kinda long and requires some trial-and-error kind of approach. So, I have created a class `Preprocessor` that are used to <u>clean</u>, <u>split</u> and <u>prepare</u> the data. Let's see how we can use this calls, we can find this class in the `preprocess_data.py` file.



## preprocess()

This is the first method that I'm going to walk you through inside the `Preprocessor()` class. This method takes one argument as an input which is the  path to the data , then it does two things actually:

- Clean this text from noise like English letters/numbers, some punctuations symbols that are no good for us
- Split the data into sentences and write those sentences in another directory `self.out_dir` where each word in the sentence is written in a separate line. These sentences are separated by a newline character `\n`. Each file should contain roughly one million words in it. This function returns nothing.

```python
>>> from preprocess_data import Preprocessor
>>>
>>> p = Preprocessor()
>>> p.preprocess('./diacritized_text')
```

After running this function, we should get a new directory called `./preprocessed` inside the parent directory. Inside this `preprocessed` directory, we should get 66 files.. [1, 2, 3, ...66]. Each file should contain roughly one million words, each word in a separate line.

So, the following Arabic line `مُقَدِّمَةُ الطَّبَرِيِّ شَيْخِ الدِّينِ فَجَاءَ فِيهِ بِالْعَجَبِ الْعُجَابِ` will be turned into:

```
مُقَدِّمَةُ
الطَّبَرِيِّ
شَيْخِ
الدِّينِ
فَجَاءَ
فِيهِ
بِالْعَجَبِ
الْعُجَابِ
```



## split()

This method takes a train-test ratio as an input (20% default value) then, it splits the preprocessed data into two directories (train, test) using the given ratio. So, if we have 100 files and the given ratio is 30%, then we would have two sets:

- Train with 70 files in it.. [1, 2, 3, ... 70].
- Test with 30 files in it ..[71, 72, ... 100].

So, let's try it out:

```python
>>> from preprocess_data import Preprocessor
>>>
>>> p = Preprocessor()
>>> p.split(0.2) #20% test data
```

After running this method, a two directories are created inside the `preprocessed` directory. The first directory is `train` which contains around 53 files (80% of the data), and the second directory is `test` which contains around 13 files (20% of the data).



## remove_diacritization()

This method aims at removing any diacritization from the test files, then write the cleaned version into
another directory. We read from the `gold` directory and write the cleaned version into `test` directory. Let's see how to use it:

```python
>>> from preprocess_data import Preprocessor
>>>
>>> p = Preprocessor()
>>> p.remove_diacritization()
```

After running this method, we should see two directories have been created. The first directory is `gold` which contains the original test data, and the second is `test` which contains the cleaned words. By clean word, I mean word without any diacritization. For example, the word `مُقَدِّمَةُ` is diacritized. While the word `مقدمة` is not diacritized (clean).



# HMM

Here, we are going to discuss the NLP model that is responsible for diacritizing Arabic scripts. This model is pretty simple. HMM stands for Hidden Markov Model which is a model used for NER tagging purposes. So, I kinda dealt with this problem as a <u>character</u>-tagging problem. Let's get into the details of the model (code-wise):



## Member Variables

As we can see, this model is pretty flexible and it can work by your standards. So, here are the member variables that can be changed as you wish and get the same result.

- `self.N` is the number of n-gram character model. So, if `self.N=3` then we get a trigram character model. If `self.N=2` then we get a bigram character model.
- `self.START` is the start symbol. In other words, when the bigram character model sees a word like `مقدمة`, it actually sees it as `*مقدمة` where `*` is at the start of the word.
- `self.NULL_TAG` is the symbol that we are using to symbolize (non-diacritized) characters. 
- `self.STATES` is all the diacritizing symbol that can be used for our model. DON'T CHANGE THE WHOLE VARIABLE, JUST THE LAST ONE. These states are:
  - double damma, double fatha, double kasera.
  - damma, fatha, kasera, sukoon.
  - shadd+double damma, shadd+double fatha, shadd+double kasera.
  - shadd+damma, shadd+fatha, shadd+kasera, shadd+skoon.
- `self.data_dir` is the location path of the data.
- `self.train_dir` is the location of files to train the model upon.
- `self.test_dir` is the location of files to test our model upon.
- `self.gold_dir` is the location of files to evaluate our model upon.
- `self.predicted_dir` is the location of files the our model has managed to diacritize.

## train()

Now, let's get to the actual work. This function, as it appears from the name, is used to train our model. The whole idea behind this model is here in this function. This function simply passes over all the training files and counts how many each a sequence of characters has been followed by a certain state. Then, these counts are saved as a dictionary in a pickle file.

How training is done?! It's pretty simple actually. Given a word like `مُقَدِّمَةُ` and a trigram character model (`self.N=3`), then the model uses a function called `word_iterator()` that divide the word into two tuples letters and diacritization symbols and returns a list of these types. Let's see an example:

```python
>>> from utils import word_iterator
>>>
>>> for t in word_iterator('مُقَدِّمَةُ'):
        print(t)
('م', 'ُ')
('ق', 'َ')
('د', 'ِّ')
('م', 'َ')
('ة', 'ُ')
```

Then it uses these tuples to create a dictionary `self.character_ngram` where the keys are a tuple of three characters and a symbol, and its value is the count. So, the keys of the dictionary regarding the past word will be:

```python
(('*', '*', 'م'), ُ )
(('*', 'م', 'ق'), َ )
(('م', 'ق', 'د'), ِّ )
(('ق', 'د', 'م'), َ )
(('د', 'م', 'ة'), ُ )
```

where the first tuple is the consecutive three letters, and the second item is the symbol that the third letter got. So, the `ُ ` is on the letter `م` in the past word and so on. Then, when the same key appears again in another word, we increase the count by one.

**NOTE:**

I have created a function in the `utils.py` file that are able to transform the pickle file into text file to be parsed directly into another programming language easily.

```python
>>> from utils import turn_pickle_to_text
>>>
>>> turn_pickle_to_text('3gram_CharModel.pickle', '3gram_CharModel.txt')
```



## diacritized_word()

This method is used to diacritize a given word based on the HMM model. The input of this method is a clean word (with no diacritization) and this method returns a diacritized word based on the trained model. Let's see an example:

```python
>>> from hmm import HMM
>>>
>>> model = HMM()
>>> word = 'مقدمة'
>>> model.diacritized_word(word)
مَقَدَّمَةِ
```

As we can see, the word could be diacritized wrong...right? That's why I created another function that could evaluate our model's output, let's see how this is done:

```python
>>> from utils import evaluate_word
>>> from hmm import HMM
>>>
>>> model = HMM()
>>> word = 'مقدمة'
>>> predicted = model.diacritized_word(word)
>>> predicted
مَقَدَّمَةِ
>>> gold_word = 'مُقَدِّمَةُ'
>>> evaluate_word(gold_word, predicted)
0.4
>>> #LUCKY ME .. HAHA
```



## diacritized_data()

This method is used to diacritized the whole test set. It read the data from the member variable `self.test_dir`. This method puts the diacritized data into `self.predicted_dir` directory.



## evaluate()

This method is used to evaluate the performance of our Hidden Markov Model. It uses the diacritized files that have been created by our model which are created in `self.predicted_dir` directory and the original (true) files located at `self.test_dir` directory. This function returns two metrics of accuracy:

- word-wise accuracy = (number of correct words / number of total words).
- character-wise accuracy = (number of correct characters / number of total characters).

If the argument `analysis` is set to `True`, it would print far more than that.

After running this trigram character model upon all the test data, it got an accuracy of `0.349430` (word-wise) and an accuracy of 0.725395 (character-wise).



# Last Words (Future Work)

As we can see, the model is far from being perfect, and it depends on the statistics of characters. So, there is a lot to do. In the beginning, I stated that this model is Hidden Markov Model. Actually, I lied.. this model is just a character language model and to make it a Hidden Markov Model, we need it to combine it transition model. 

Or we can take it further and try to study the problem in order to create a generative model that could understand the mechanics of the Arabic language and produce correct diacritization.