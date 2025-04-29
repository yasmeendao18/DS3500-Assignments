"""
Lilian Uong and Yasmeen Dao
DS3500 / warletters-NLP
A reusable extensible framework for Natural Language Processing  / Homework 3
2/23/23 / 2/28/23
"""
import argparse
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import re
import io
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud.wordcloud_cli import RegExpAction

from sankey import make_sankey
from nltk.util import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class Warletters:

    def __init__(self):
        # manage data about the different text that
        # we register with framework
        self.data = defaultdict(dict)

    def _save_results(self, label, results):
        """ Integrate parsing results into internal state
        :param label: unique label for a text file that we parsed
        :param results: the data extracted from the file as a dictionary attribute-->raw data
        """
        for k, v in results.items():
            self.data[k][label] = v

    def rem_stopwords(text, stopwords_link = ("https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b" +
                                             "5925d/raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt")):
        """
        Remove stopwords from text and creates new list containing filtered words
        :param text: text from file
        :param stopwords_link: link to stopwords
        :return: word list of filtered text
        """
        # load NLTK library stopwords
        stopwords_nltk = stopwords.words('english')

        # Load in more stopwords from the web
        stopword_list = requests.get(stopwords_link).content
        stwords = list(set(stopword_list.decode().splitlines()))

        # Add the list of stopwords from the NLTK library
        stwords += stopwords_nltk

        # list of non stopwords
        nonstopwords = str(text).lower()
        corp = re.sub('[^a-zA-Z]+', ' ', nonstopwords).strip()
        tokens = word_tokenize(corp)
        words = [tok for tok in tokens if tok not in stwords]

        return words

    @staticmethod
    def get_ngram_data(n_gram, words):
        """
        Finds the n_gram frequencies of the text and returns data as a list of tuples

        :param n_gram: value that decides number of contiguous items to get frequency of
        :param words: the list of words from the text, excluding stopwords
        :return: a list of tuples (n-gram, frequency) sorted by frequency
        """

        # Get the frequency of each n-gram
        n_gram_fq = nltk.FreqDist(ngrams(words, n_gram))

        return n_gram_fq.most_common()

    @staticmethod
    def get_results(words, n_gram_data, n_gram):
        """
        takes all results and puts into dictionary
        :param words: words in text
        :return: dictionary of text anaylsis
        """

        # create dictionary
        results = {}

        # get word counts
        wc = Counter(words)
        results['wordcount'] = wc

        txt_wd_lngths = [len(wd) for wd in words]
        results['Avg. Word Length'] = round((sum(txt_wd_lngths) / len(txt_wd_lngths)), 3)
        # Add the n_gram data to the results dictionary
        results['{n}-gram count'.format(n=str(n_gram))] = n_gram_data

        return results
    @staticmethod
    def default_parser(filename, n_gram):
        """
        Pre-Process text file
        :param filename: file to open
        :return: text file results
        """

        # open and read file
        with open(filename, encoding='utf8') as textfiles:
            files = textfiles.readlines()

            # create string for parsed text
            text = ''

            # add each line of files to text
            for line in files:
                text += line

            # remove stopwords
            words = Warletters.rem_stopwords(text)

            # Get the n-gram frequency data for the text
            n_gram_data = Warletters.get_ngram_data(n_gram, words)

            # dictionary for results
            results = Warletters.get_results(words, n_gram_data,n_gram)

        return results

    # exception from exception class
    class RegExpAction(argparse.Action):
        def __init__(self, option_strings, dest, **kwargs):
            super(RegExpAction, self).__init__(option_strings, dest, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            try:
                re.compile(values)
            except re.error as e:
                raise argparse.ArgumentError(self, 'Invalid expression: ' + str(e))
            setattr(namespace, self.dest, values)

    def load_text(self, filename, label=None, parser=None, n_gram=1):
        """ Register a document with the framework
        :param filename: file to open
        :param label: name associated with file
        :param parser: optional parameter if default parser doesn't work
        """

        if parser is None:  # do default parsing of standard .txt file
            results = Warletters.default_parser(filename,n_gram)
        else:
            results = parser(filename,n_gram)

        if label is None:
            label = filename

        # Save / integrate the data we extracted from the file
        # into the internal state of the framework
        self._save_results(label, results)

    def load_stopwords(stopwd_link = "https://gist.githubusercontent.com/rg089/35e00abf8941d72d419224cfd5b5925d\
                                    /raw/12d899b70156fd0041fa9778d657330b024b959c/stopwords.txt"):
        """
        :param stopwd_link: link to the comprhensive stopwords
        :return: stopwords list
            Gets stopwords from website and adds to NLTK library of stop words
        """

        # load in stopwords
        stopword_list = requests.get(stopwd_link).content
        stopwords = list(set(stopword_list.decode().splitlines()))

        # load in NLTK stopwords
        stopwords_nltk = stopwords.words('english')

        # add lists together
        stopwords += stopwords_nltk

        return stopwords

    @staticmethod
    def get_data_for_sankey(text_data, words_list, k):
        """
        gets data for plotting in sankey diagram
        :param text_data: the self.data
        :param words_list: list of words
        :param k: number of words from each file
        :return: dataframe with sankey data
        """

        # create dataframe
        df_text = pd.DataFrame()

        # labels for files of texts
        sources = list(text_data['Avg. Word Length'].keys())

        if words_list:
            sources = []
            targets = []
            values = []

            # update lists with data
            for w in words_list:
                for s in sources:
                    for word, count in text_data['wordcount'][s].items():
                        if w.lower() == word.lower():
                            sources.append(s)
                            targets.append(w)
                            values.append(count)
                        else:
                            pass

            df_text['sources'] = sources
        else:
            df_text['sources'] = sources * k

            # get k most common words
            k_common = []
            for s in sources:
                k_common.append(text_data['wordcount'][s].most_common(k))

            # targets and values
            targets = []
            values = []

            # update lists
            count = 0
            while count < k:
                for val in k_common:
                    targets.append(val[count][0])
                    values.append(val[count][1])
                count += 1
        # add targets and values to dataframe
        df_text['targets'] = targets
        df_text['values'] = values

        return df_text

    def wordcount_sankey(self, wordlist=None, k=5):
        """
        :param wordlist: list of words to include
        :param k: number of words each text file connects to
        :return: none
            creates sankey diagram
        """

        # get sankey dataframe
        df_sankey = Warletters.get_data_for_sankey(self.data, wordlist, k)

        if wordlist:
            insert = ''
            if len(wordlist) == 1:
                insert += str(wordlist[0].capitalize()) + ' shows up'
            else:
                for w in wordlist:
                    if wordlist.index(w) != (len(wordlist) - 1):
                        insert += str(w.capitalize()) + ','
                    else:
                        insert += 'and ' + str(w.capitalize()) +' shows up'

            title = f'The number of times {insert} in each text'
        else:
            # title for sankey
            sankey_title = 'The {kvalue} Most Common Words in {filenum} Texts and Their Frequencies in ' \
                           'Texts'

            # change title
            title = sankey_title.format(kvalue=str(k), filenum=str(len(list(self.data['Avg. Word Length'].keys()))))
        make_sankey(df_sankey, 'sources','targets',title, 'values', pad=5)

    def plot_common_words(self, colors=None, k=10):
        """
        stacked bar chart to compare most common words k

        :param colors: bar chart colors
        :param k: num of most common words
        :return:None
            produces stacked bar chart based on common words
        """
        # get the data for word counts and extract the labels
        word_counts = self.data['wordcount']
        labels = list(word_counts.keys())

        # create a dictionary that pairs each text to a tuple containing (the k most common words in the text)
        xy_dict = {}
        for label in labels:
            tup_list = word_counts[label].most_common(k)
            y_vals = []
            x_vals = []
            for tuple in tup_list:
                x_vals.append(tuple[0])
                y_vals.append(tuple[1])
            xy_dict[label] = (x_vals, y_vals)

        # list of all the words in every text no repeats
        all_words = []
        for key, value in xy_dict.items():
            for w in value[0]:
                if w not in all_words:
                    all_words.append(w)

        # get y-value for text and add to dict list
        y_vals = []
        for label in labels:
            y = []
            for w in all_words:
                if w in xy_dict[label][0]:
                    y_vals_lst = xy_dict[label][1]
                    y.append(y_vals_lst[xy_dict[label][0].index(w)])
                else:
                    y.append(0)
            y_vals.append(y)

        # plot the y values with text label and color
        # create stacked bar chart
        for idx in range(len(y_vals)):
            if idx != 0:
                list_bottom = y_vals[:idx]
                bottom = [sum(x) for x in zip(*list_bottom)]
                if colors:
                    plt.bar(all_words, y_vals[idx], bottom=np.array(bottom), label=labels[idx], color=colors[idx])
                else:
                    plt.bar(all_words, y_vals[idx], bottom=np.array(bottom), label=labels[idx])
            else:
                if colors:
                    plt.bar(all_words, y_vals[idx], label=labels[idx], color=colors[idx])
                else:
                    plt.bar(all_words, y_vals[idx], label=labels[idx])

        # customize the stacked bar plot
        plt.title('The {n} Most Common Words in Each Text and Their Frequency Count in Each Text'.format(n=k),
                  fontsize=15, wrap=True)
        plt.xticks(rotation=90)
        plt.xlabel('Most Common Words')
        ot = y_vals.pop()
        tops = [sum(x) for x in zip(*[bottom, ot])]
        plt.ylim(0, max(tops) + 5)
        plt.ylabel('Frequency Count')
        plt.legend(title='Text Source')

    def plot_ngrams_wc(self, wordcloud_file):
        """
        Creates a wordcloud using the user-given text's n_gram counts
        :param: wordcloud_file: the name of the file
        :return:

        """

        # Extract the data from self that has the n_gram counts
        for key in self.data.keys():
            if 'gram' in key:
                full_ngram_data = self.data[key]

        # Extract the n_gram counts for given file
        file_ngram_data = full_ngram_data[wordcloud_file]

        # string that includes each n_gram in the n_gram data separated by a space
        # n gram library
        ngram_txt = ''
        for tuple in file_ngram_data:
            ngrams = tuple[0]
            ngram = ''
            for word in ngrams:
                ngram += word + '_'

            ngram = ngram[:-1]
            ngram_txt += ngram + ' '

        # Plot the wordcloud and customize it
        wordcloud = WordCloud(width=2000, height=1334, random_state=1, background_color='black', colormap='Pastel1',
                              max_words=75, collocations=False, normalize_plurals=False).generate(ngram_txt)
        plt.imshow(wordcloud)
        plt.title('A Word Cloud Showing the Frequency of {n}-grams in {source}'.format(n=list(self.data.keys())[-1][0],source=wordcloud_file))
        plt.axis('off')

    def wordcloud_comparisons(self, wordcloud_file, color=None):
        """
        :param: wordcloud_file: the file label the user wants a wordcloud for
        :param: color: colors of wordcloud text
        :return: a series of subplots comparing word text for each file
            Create subplot of wordclouds using each text file
        """
        # Create a figure that will contain four plots
        plt.subplots(4, 2)
        for i in range(len(wordcloud_file)):
            plt.subplot(4, 2, i+1)
            self.plot_ngrams_wc(wordcloud_file[i])

        # Show the graphs
        plt.subplots_adjust(left=0.08,bottom=0.08,right=0.96,top=0.93,wspace=0.1,hspace=0.2)
        plt.show()