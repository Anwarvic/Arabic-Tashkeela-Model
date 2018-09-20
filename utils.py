import os
import re
import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt


#Global Variables
START = '*'
OTHER = 'O'
#double damma, double fatha, double kasera, damma, fatha, kasera, sukoon, shadd
VOWEL_SYMBOLS = {'ٌ', 'ً', 'ٍ', 'ُ', 'َ', 'ِ', 'ْ', 'ٌّ', 'ّ'}
VOWEL_REGEX = re.compile('|'.join(VOWEL_SYMBOLS))



def word_iterator(word):
    """
    This function takes a word (discrentized or not) as an input and returns 
    a list of tuple where the first item is the character and the second
    item is the vowel_symbol. For example:
    >>> word_iterator('الْأَلْبَاب')
    [ ('ا', 'O'),
      ('ل', 'ْ'),
      ('أ', 'َ'),
      ('ل', 'ْ'), 
      ('ب', 'َ'), 
      ('ا', 'O'), 
      ('ب', 'O') ]
    As we can see, the symbol O stands for OTHER and it means that the character
    doesn't have an associated vowel symbol
    """
    output = []
    prev_char = word[0]
    for idx, char in enumerate(word[1:]):
        try:
            #first 1 because we skipped the first character
            #second 1 because it's the next character
            next_char = word[idx+1+1]
        except IndexError:#will happen with the last character only
            next_char = ''
        if char in VOWEL_SYMBOLS:
            if next_char == '' and prev_char not in VOWEL_SYMBOLS:
                output.append((prev_char, char))
            elif prev_char not in VOWEL_SYMBOLS and next_char not in VOWEL_SYMBOLS:
                output.append((prev_char, char))
            elif prev_char not in VOWEL_SYMBOLS and next_char in VOWEL_SYMBOLS:
                output.append((prev_char, char+next_char))
        else:
            #if a character wasn't diacritized
            if prev_char not in VOWEL_SYMBOLS:
                output.append((prev_char, OTHER))
            if next_char == '':
                output.append((char, OTHER))
        prev_char = char
    return output


def create_dir(name):
    """
    This function takes a string as an input. 
    It creates a directory using this 'name' if doesn't already exist
    """
    if not name:
        #if name is empty
        return
    name = str(name)
    if not os.path.isdir(name):
        os.makedirs(name)


def clean_word(word):
    """
    This function takes a word (discrentized or not) as an input and returns 
    the word itself without any discrentization.
    For example:
    >>> x = clean_word('الْأَلْبَاب')
    >>> x
    'الألباب'
    >>> type(x)
    'str'
    """
    return re.sub(VOWEL_REGEX, '', word)

def evaluate_word(gold_word, predicted_word, analysis=False):
    """
    This function evaluate two input words:
    -> gold_word: represents the true discrentization of the word
    -> predicted_word: represents the model's discrentization of the word
    Then, this function should return the accuracy which depends on the following 
    formula which is:
                 number of correct tags
     accuracy = ------------------------
                 total number of tags
    """
    correct = 0.     #number of correct tags
    total_num = 0.   #total count of tags
    gold_tags = [tag for _, tag in word_iterator(gold_word)]
    predicted_tags = [tag for _, tag in word_iterator(predicted_word)]
    assert len(gold_tags) == len(predicted_tags)
    for gold_tag, predicted_tag in zip(gold_tags, predicted_tags):
        total_num += 1
        if gold_tag == predicted_tag:
            # print(gold_tag, predicted_tag)
            correct += 1.
    if analysis:
        return correct, total_num
    else:
        return correct/total_num
        

def draw_histogram(d, filename=None):
    """
    This function takes a dictionary with certain ranges as keys 
    and the count of occurrences of these ranges as values.
    Then, it shows the histogram figure if the 'save' parameter was
    False (default value) and it saves the figure as an image if it
    was set to True.
    """
    labels, values = zip(*d.items())
    indexes = np.arange(len(labels))

    width = 1
    plt.bar(indexes, values, width=0.8, color='#C82300', edgecolor='#CD5C5C', align='edge')
    plt.xticks(indexes + width * 0.5, labels)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()



def turn_pickle_to_text(pickle_file, text_file):
    """
    This function turns the model (as a pickle file) into
    text file to be ready for being parsed in Java.
    The input pickle_file is a dictionary where:
    key: is a tuple and character.
    value: is a count
    """
    with open(pickle_file, "rb") as fin:
        d = pickle.load(fin)

    with open(text_file, "wb") as fout:
        for k, v in d.items():
            s = k[0][0]+'|'+k[0][1]+'|'+k[0][2]+'|'+k[1]+'\t'+str(v)
            fout.write(s.encode())
            fout.write('\n'.encode())





if __name__ == "__main__":
    word = 'مُقَدِّمَةُ'
    print(clean_word(word))
    # turn_pickle_to_text('3gram_CharModel.pickle', 'E:\\Career\\Courses\\Data Science\\Natural Language Processing (NLP)\\PROJECTS\\Tashkeela\\untitled\\src\\test.txt')
    # words = ['مُقَدِّمَةُ', 'الطَّبَرِيِّ', 'شَيْخِ', 'الدِّينِ', 'فَجَاءَ', 'فِيهِ', 'بِالْعَجَبِ', 'الْعُجَابِ', 'وَنَثَرَ', 'فِيهِ', 'أَلْبَابَ', 'الْأَلْبَاب']
    print(evaluate_word('الْأَلْبَاب', clean_word('الْأَلْبَاب')))
    print(evaluate_word('مُقَدِّمَةُ', clean_word('مُقَدِّمَةُ')))
    
    # N = 3
    # for word in words:
    #     print(clean_word(word))
    #     print(word)
    #     ################
    #     charsonly, tagsonly = zip(*[('*', 'O')]*(N-1)+word_iterator(word))
    #     print(charsonly)
    #     print(tagsonly)
    #     for char, tag in zip(charsonly, tagsonly):
    #         print(char, tag)
    #     #####################
    #     # for t in word_iterator(word):
    #     #     print(t)