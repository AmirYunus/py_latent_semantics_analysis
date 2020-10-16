import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

wordnet_lemmatiser = WordNetLemmatizer()
book_titles = [each_line.rstrip() for each_line in open('book_titles.txt')]
stopwords = set(each_word.rstrip() for each_word in open('stopwords.txt'))

stopwords = stopwords.union({
    'introduction',
    'edition',
    'series',
    'application',
    'approach',
    'card',
    'access',
    'package',
    'plus',
    'etext',
    'brief',
    'vol',
    'fundamental',
    'guide',
    'essential',
    'printed',
    'third',
    'second',
    'fourth'
})

def tokeniser(string):
    string = string.lower()
    tokens = nltk.tokenize.word_tokenize(string)
    tokens = [each_token for each_token in tokens if len(each_token) > 2]
    tokens = [wordnet_lemmatiser.lemmatize(each_token) for each_token in tokens]
    tokens = [each_token for each_token in tokens if each_token not in stopwords]
    tokens = [each_token for each_token in tokens if not any(each_character.isdigit() for each_character in each_token)]
    return tokens

word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
error_count = 0

for each_title in book_titles:
    try:
        each_title = each_title.encode('ascii', 'ignore').decode('utf-8')
        all_titles.append(each_title)
        tokens = tokeniser(each_title)
        all_tokens.append(tokens)

        for each_token in tokens:
            if each_token not in word_index_map:
                word_index_map[each_token] = current_index
                current_index += 1
                index_word_map.append(each_token)

    except Exception as e:
        print(e)
        print(each_title)
        error_count += 1

print(f"Number of errors parsing file: {error_count}")
print(f"Number of lines in file: {len(book_titles)}")

if error_count == len(book_titles):
    print("There is no data to parse.")
    print("Program exiting.")
    exit()

def tokens_to_vector(tokens):
    vector = np.zeros(len(word_index_map))

    for each_token in tokens:
        index = word_index_map[each_token]
        vector[index] = 1

    return vector

token_length = len(all_tokens)
word_index_map_length = len(word_index_map)
vector = np.zeros((word_index_map_length, token_length))
index = 0

for each_token in all_tokens:
    vector[:, index] = tokens_to_vector(each_token)
    index += 1

def main():
    svd = TruncatedSVD().fit_transform(vector)
    plt.scatter(svd[:,0], svd[:,1])

    for each_index in range (word_index_map_length):
        plt.annotate(text=index_word_map[each_index], xy=(svd[each_index,0], svd[each_index,1]))
    
    plt.show()

if __name__ == "__main__":
    main()
