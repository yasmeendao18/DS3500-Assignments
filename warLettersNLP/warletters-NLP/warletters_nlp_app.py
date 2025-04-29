"""
Lilian Uong and Yasmeen Dao
DS3500 / warletters-NLP
A reusable extensible framework for Natural Language Processing  / Homework 3
2/23/23 / 2/28/23
"""
# import class and call functions
from warletters import Warletters
import matplotlib.pyplot as plt

def main():

    # initialize framework
    # create class object
    wl = Warletters()

    # loop through to register text files
    fileNames = ['Alexander', 'Churchill','Hazard','Isaac','Lsilbert','Sarah','UncleGeorge','William']
    # store in list
    for name in fileNames:
        wl.load_text(name +'.txt', name)

    # plot sankey -- 5 most common words in each text
    wl.wordcount_sankey(k=5)
    # plot word cloud
    wl.wordcloud_comparisons(wordcloud_file=fileNames, color=None)
    # Plot a stacked bar chart comparing the 10 most common words in each text
    wl.plot_common_words(colors=['#483D8B', 'teal', '#253342', 'cornflowerblue','purple','lightcoral','green','orange'], k=5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()