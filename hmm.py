import os
import _pickle as pickle
from collections import defaultdict, deque

from preprocess_data import Preprocessor
from utils import *



class HMM(object):
    def __init__(self, n=3):
        """This method is used to initialize our Hidden Markov Model"""
        assert n>=2, "Expecting n>=2."
        #create member variables
        self.N = n
        self.START = '*'
        self.NULL_TAG = 'O' #tag for characters that are NOT diacritizedd
        self.STATES = {'ٌ', 'ً', 'ٍ', 'ُ', 'َ', 'ِ', 'ْ', 'ّ', 'ٌّ', 'ًّ', 'ٍّ', 'ُّ', 'َّ', 'ِّ', 'ّْ', 'O'}
        #STATES are:
        #double damma, double fatha, double kasera, damma, fatha, kasera, sukoon, 
        #shadd+double damma, shadd+double fatha, shadd+double kasera
        #shadd+damma, shadd+fatha, shadd+kasera, shadd+skoon

        #load model if saved
        try:
            with open(str(self.N)+'gram_CharModel.pickle', 'rb') as fout:
                self.character_ngram = pickle.load(fout)
            print('Done Loading trained model!!')
        except FileNotFoundError:
            #create member containers
            self.character_ngram = defaultdict(int)
        
        #create files
        self.data_dir = os.path.join(Preprocessor().out_dir)
        self.train_dir = os.path.join(self.data_dir, 'train') #contains files to train
        create_dir(self.train_dir)
        self.test_dir = os.path.join(self.data_dir, 'test') #contains files to test
        create_dir(self.test_dir)
        self.gold_dir = os.path.join(self.test_dir, 'gold') #contains files to evaluate
        create_dir(self.gold_dir)
        self.predicted_dir =  os.path.join(self.test_dir, 'predicted', str(self.N)+"gram") #contains model generated files
        create_dir(os.path.join(self.test_dir, 'predicted'))
        create_dir(self.predicted_dir)
        self.test_dir = os.path.join(self.test_dir, 'test') #contains files to evaluate
        create_dir(self.test_dir)


    def train(self):
        """
        This method is used to train our Hidden Markov Model
        It reads the file in the 'train_dir' directory.
        Then, it trains a model and saves it.
        """
        print("----- Starting Training ------")
        for filename in os.listdir(self.train_dir):
            num_errors = 0
            print("FILE:", filename)
            with open(os.path.join(self.train_dir, filename), 'rb') as fin:
                for word in fin.readlines():
                    word = word.decode().strip()
                    if word == '': #empty line
                        continue
                    try:
                        charsonly, tagsonly = zip(*word_iterator(word))
                        assert len(charsonly) == len(tagsonly)
                        d = deque(self.START*(self.N-1), maxlen=self.N-1)
                        for char, tag in zip(charsonly, tagsonly):
                            self.character_ngram[(*d, char), tag] += 1.
                            d.append(char)
                    except:
                        num_errors += 1
            with open(str(self.N)+'gram_CharModel.pickle', 'wb') as fout:
                pickle.dump(self.character_ngram, fout)
            print("\tERROR:", num_errors)


    def diacritized_word(self, word):
        """
        This method is used to diacritized a given word based on 
        the HMM model. 
        The input of this method is a clean word (with no discrentization)
        and this method returns a dicrentized word based on the trained model
        """
        #make sure that the word is with no discrentization
        assert re.search(Preprocessor().VOWEL_REGEX, word) == None
        dq = deque(self.START*(self.N-1), maxlen=self.N-1)
        out_word = ''
        for char in word:
            winning_tag = self.NULL_TAG
            top_count = 0
            for tag in self.STATES:
                if self.character_ngram[(*dq, char), tag] > top_count:
                    winning_tag = tag
                    top_count = self.character_ngram[(*dq, char), tag]
            if winning_tag != self.NULL_TAG:
                out_word += char+winning_tag
            else:
                out_word += char
            dq.append(char)
        return out_word 


    def diacritized_data(self):
        """
        This method is used to diacritized the undiacritizedd words
        in the test set.
        """
        print("----- Starting discrentization ------")
        for filename in os.listdir(self.test_dir):
            print("FILE:", filename)
            with open(os.path.join(self.test_dir, filename), 'rb') as fin, \
              open(os.path.join(self.predicted_dir, filename), 'wb') as fout:
                for word in fin.readlines():
                    word = word.decode().strip()
                    if word == '': #empty line
                        fout.write('\n'.encode())
                    else:
                        fout.write(self.diacritized_word(word).encode())
                        fout.write('\n'.encode())
        print("Done discrentizing data!!")


    def evaluate(self, analysis=False):
        """
        This method is used to evaluate the performance of our Hidden Markov Model.
        It uses the dicrentized files that have been created by our model which
        are created in 'predicted_dir' directory and the original (true) files
        located at 'test_dir' directory.
        This function returns just the accuracy of the model
        -> (number of correct words, number of total words)
        -> (number of correct characters, number of total characters)
        -> dictionary of error scores whose keys are are seperated by 0.2
           so we have a total of five keys
        """
        print("----- Starting Evaluation ------")
        num_correct_words, total_num_words = (0., 0.)
        num_correct_chars, total_num_chars = (0., 0.)
        histogram = defaultdict(int)
        for filename in os.listdir(self.predicted_dir):
            print("FILE:", filename)
            with open(os.path.join(self.gold_dir, filename), 'rb') as gold_fin, \
              open(os.path.join(self.predicted_dir, filename), 'rb') as predicted_fin:
                for gold_word, predicted_word in zip(gold_fin.readlines(), predicted_fin.readlines()):
                    gold_word = gold_word.decode().strip()
                    predicted_word = predicted_word.decode().strip()
                    if len(gold_word) <2 or len(predicted_word) <2:  #empty line
                        continue
                    total_num_words += 1.
                    correct_chars, word_chars = evaluate_word(gold_word, predicted_word, analysis=True)
                    if correct_chars == word_chars:
                        num_correct_words += 1.
                    num_correct_chars += correct_chars
                    total_num_chars += word_chars
                    #----- Handle the histogram -----
                    acc = correct_chars/word_chars
                    if acc < 0.2:
                        histogram['below 0.2'] += 1
                    elif acc < 0.4:
                        histogram['0.2:0.4'] += 1
                    elif acc<0.6:
                        histogram['0.4:0.6'] += 1
                    elif acc<0.6:
                        histogram['0.6:0.8'] += 1
                    else:
                        histogram['above 0.8'] += 1
            # break
            print('This model has got:')
            if analysis:
                print('\tCorrect words: %d out of %d' %(num_correct_words, num_correct_chars))
                print('\tCorrect characters: %d out of %d' %(num_correct_chars, total_num_chars))
                draw_histogram(histogram, filename=str(self.N)+"gram_histogram.jpg")
            print('\tAn accuracy (character-wise): %f' %(num_correct_chars/total_num_chars))
            print('\tAn accuracy (word-wise):  %f ' %(num_correct_words/total_num_words))





if __name__ == "__main__":
    hmm = HMM()
    word = 'مقدمة'
    predicted = hmm.diacritized_word(word)
    gold_word = 'مُقَدِّمَةُ'
    print(evaluate_word(gold_word, predicted))
    # hmm.train()
    # hmm.diacritized_data()
    # hmm.evaluate(analysis=True)
    # words = ['مُقَدِّمَةُ', 'الطَّبَرِيِّ', 'شَيْخِ', 'الدِّينِ', 'فَجَاءَ', 'فِيهِ', 'بِالْعَجَبِ', 'الْعُجَابِ', 'وَنَثَرَ', 'فِيهِ', 'أَلْبَابَ', 'الْأَلْبَاب']
    # for word in words:
    #     tmp = hmm.diacritized_word(clean_word(word))
    #     print(tmp)
    #     print(evaluate_word(word, tmp))
    print("done!!")
    """

    """