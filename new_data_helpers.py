import numpy as np
import csv
import os
import gzip
import json
# from nltk.stem.snowball import EnglishStemmer, SpanishStemmer
# from unidecode import unidecode
from datetime import datetime
#
# # es_stemmer = SpanishStemmer()
# # en_stemmer = EnglishStemmer()
#
# english_stop = "/home/xyang/tr.xiao/tr.xiao/sem2vec/CNN_unseen/stopword/stopword-english.txt"
# spanish_stop = "/home/xyang/tr.xiao/tr.xiao/sem2vec/CNN_unseen/stopword/stopword-spanish.txt"
#
#
# def load_stopword(english=False):
#     if english:
#         stopPath = english_stop
#     else:
#         stopPath = spanish_stop
#
#     stopword_list = {}
#     with open(stopPath, 'r') as f:
#         for line in f:
#             word = line.decode('utf-8').strip()
#             stopword_list[word] = ''
#     return spanish_stop
#
#
# def clean_str(string, we_vocab, english=False):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     if english:
#         stemmer = en_stemmer
#     else:
#         stemmer = es_stemmer
#     tokenizer = re.compile(r"[@#\p{Alpha}][\w]+|[A-Z]{2,}", flags=re.UNICODE)
#     string = re.sub(r"https?[\w.:/]+", " ", string.strip(), flags=re.UNICODE)
#     tokenised_string = tokenizer.findall(string)
#     for i in xrange(len(tokenised_string)):
#         tokenised_string[i] = unidecode(stemmer.stem(tokenised_string[i]))
#         if tokenised_string[i] not in we_vocab:
#             tokenised_string[i] = "<UNK>"
#     return " ".join(tokenised_string)
#
#
# def clean_unseen_str(string, english=False):
#     if english:
#         stemmer = en_stemmer
#     else:
#         stemmer = es_stemmer
#     tokenizer = re.compile(r"[@#\p{Alpha}][\w]+|[A-Z]{2,}", flags=re.UNICODE)
#     string = re.sub(r"https?[\w.:/]+", " ", string.strip(), flags=re.UNICODE)
#     tokenised_string = tokenizer.findall(string)
#     for i in xrange(len(tokenised_string)):
#         tokenised_string[i] = unidecode(stemmer.stem(tokenised_string[i]))
#     return " ".join(tokenised_string)
#
#
# def load_data_and_labels(data_file, we_vocab, n_class, english=False):
#     """
#     Loads MR polarity data from files, splits the data into words and generates labels.
#     Returns split sentences and labels.
#     """
#     # Load data from files
#     sentences = []
#     labels = []
#     label_map = {}
#     for i in xrange(n_class):
#         label_map[str(i)] = [1 if i == j else 0 for j in xrange(n_class)]
#     print label_map
#     # raise
#
#     with open(data_file,'r') as f:
#         reader = csv.DictReader(f)
#         for line in reader:
#             text = line['text'].decode('utf-8').lower()
#             label = line['vio_label']
#             sentences.append(clean_str(text, we_vocab, english))
#             labels.append(label_map[label])
#     return np.array(sentences),np.array(labels)
#
#
# def load_WE_vocab(we_vocab_path):
#     word2index = {}
#     with open(we_vocab_path, 'r') as f:
#         count = 0
#         for line in f:
#             word = line.strip()
#             word2index[word] = count
#             count+=1
#     return word2index
#
#
# def parse_date(date_detail):
#     parsed_date = datetime.strptime(date_detail, '%a %b %d %H:%M:%S +0000 %Y')
#     return parsed_date.strftime("%Y-%m-%d %H:%M:%S")
#
# def load_unseen_data(data_path, english=False):
#     """
#     Load unseen dataset for predictions
#     :param data_path:
#     :return: processed text and fake labels
#     """
#     count = 0
#     output = []
#     # print os.listdir(data_path)
#     all_list = [os.path.join(data_path, f) for f in os.listdir(data_path)]
#     files_list = [f for f in all_list if os.path.isfile(f)]
#
#     for gzf in sorted(files_list):
#         if "json.gz" in gzf:
#             try:
#                 print gzf
#                 with gzip.open(gzf, 'r') as f:
#                     for line in f:
#                         count += 1
#                         tweet = json.loads(line)
#                         text = tweet['text'].lower()
#                         tid = tweet['id_str']
#                         t_date = parse_date(tweet['created_at'])
#                         cleaned_text = clean_unseen_str(text, english)
#                         raw_text = " ".join(text.strip().split(" "))
#                         output.append((tid, cleaned_text, raw_text, t_date))
#                         if len(output) % 1000 == 0:
#                             yield output
#                             output[:] = []
#             except Exception as err:
#                 print "Exception in {}".format(gzf)
#                 print err
#
#     if len(output) > 1:
#         yield output
#
#
# def load_unseen_data_election(data_path):
#     """
#     Load unseen dataset for predictions
#     :param data_path:
#     :return: processed text and fake labels
#     """
#     count = 0
#     output = []
#     try:
#         with gzip.open(data_path, 'r') as f:
#             reader = csv.reader(f)
#             for line in reader:
#                 count += 1
#                 tid = line[0]
#                 t_date = line[1]
#                 text = line[2].decode('utf-8').lower()
#                 label = line[3]
#                 cleaned_text = clean_unseen_str(text)
#                 if int(label) == 1:
#                     output.append((tid, cleaned_text, text, t_date))
#                 if len(output) % 1000 == 0:
#                     yield output
#                     output[:] = []
#     except Exception as err:
#         print "Exception in {}".format(data_path)
#         print err
#
#     if len(output) > 1:
#         yield output

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    if len(data) % batch_size == 0:
        num_batches_per_epoch = int(len(data) / batch_size)
    else:
        num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__=="__main__":
    load_data_and_labels("", "", 6, english=False)