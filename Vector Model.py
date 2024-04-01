import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import numpy as np

from numpy.linalg import norm
import nltk
#nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


### File IO and processing

class Document(NamedTuple):
    doc_id: int
    author: List[str]
    title: List[str]
    keyword: List[str]
    abstract: List[str]

    def sections(self):
        return [self.author, self.title, self.keyword, self.abstract]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  author: {self.author}\n" +
            f"  title: {self.title}\n" +
            f"  keyword: {self.keyword}\n" +
            f"  abstract: {self.abstract}")


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i = int(line[3:])
                docs.append(defaultdict(list))
            elif re.match(r'\.\w', line):
                category = line[1]
            elif line != '':
                for word in word_tokenize(line):
                    docs[i][category].append(word.lower())

    return [Document(i + 1, d['A'], d['T'], d['K'], d['W'])
        for i, d in enumerate(docs[1:])]

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



### Term-Document Matrix

class TermWeights(NamedTuple):
    author: float
    title: float
    keyword: float
    abstract: float

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: TermWeights):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.abstract:
        vec[word] += weights.abstract
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc, doc_freqs, weights): 
    # 1b

    tfidf_vec = defaultdict(float)
    total_words = sum(len(section) for section in doc.sections())

    for word in doc.author:
        if total_words > 0 and doc_freqs[word] > 0:
            tf = doc.author.count(word) / total_words
            idf = np.log(len(doc) / doc_freqs[word])
            tfidf_vec[word] = tf * idf * weights.author

    for word in doc.keyword:
        if total_words > 0 and doc_freqs[word] > 0:
            tf = doc.keyword.count(word) / total_words
            idf = np.log(len(doc) / doc_freqs[word])
            tfidf_vec[word] = tf * idf * weights.keyword

    for word in doc.title:
        if total_words > 0 and doc_freqs[word] > 0:
            tf = doc.title.count(word) / total_words
            idf = np.log(len(doc) / doc_freqs[word])
            tfidf_vec[word] = tf * idf * weights.title

    for word in doc.abstract:
        if total_words > 0 and doc_freqs[word] > 0:
            tf = doc.abstract.count(word) / total_words
            idf = np.log(len(doc) / doc_freqs[word])
            tfidf_vec[word] = tf * idf * weights.abstract

    return dict(tfidf_vec)


def compute_boolean(doc, doc_freqs, weights):
    boolean_vec = {}
    # 1C
    # Iterate through each term in the document
    for sec in doc.sections():
        for word in sec:
            # Check if the term is present in the document frequency dictionary
            if word in doc_freqs:
                boolean_vec[word] = 1  # Term is present
            else:
                boolean_vec[word] = 0  # Term is absent
    return boolean_vec



### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))
# 2b
def dice_sim(x, y):
    intersection = len(set(x.keys()) & set(y.keys()))
    return (2 * intersection) / (len(x) + len(y)) 

def jaccard_sim(x, y):
    intersection = len(set(x.keys()) & set(y.keys()))
    union = len(set(x.keys()) | set(y.keys()))
    return intersection / union if union != 0 else 0

def overlap_sim(x, y):
    intersection = len(set(x.keys()) & set(y.keys()))
    return intersection / min(len(x), len(y))



### Precision/Recall

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    if results == 0.0:
        return 1
    if results is None:
        return 0
    relevant_count = len(relevant)
    if relevant_count == 0:
        return 0.0
    results_count = len(results)

    retrieved_relevant_count = sum(1 for doc_id in results[:int(results_count * recall)] if doc_id in relevant)
    return retrieved_relevant_count / min(relevant_count, len(results))

def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3


def norm_recall(results, relevant):
    if results is None or relevant is None:
        return 0
    num_relevant = len(relevant)
    num_results = len(results)

    
    sum_rank_relevant = 0
    for i in range(1, num_relevant + 1):
        sum_rank_relevant += results.index(i)

    
    sum_rank_all = 0
    for i in range(1, num_relevant + 1):
        sum_rank_all += i

    numerator = sum_rank_relevant - sum_rank_all
    denominator = num_relevant * (num_results - num_relevant)

    if denominator != 0:
        normalized_recall = 1 - (numerator / denominator)
    else:
        normalized_recall = 0

    return normalized_recall 


def norm_precision(results, relevant):
    if results is None or relevant is None:
        return 0
    num_relevant = len(relevant)
    num_results = len(results)

    sum_log_rank_relevant = 0
    for i in range(1, num_relevant + 1):
        index_i = results.index(i)
        if index_i != 0:
            sum_log_rank_relevant += np.log2(index_i)
    sum_log_rank_all = 0
    for i in range(1, num_relevant + 1):
        if i != 0:
            sum_log_rank_all += np.log2(i)

    numerator = sum_log_rank_relevant - sum_log_rank_all
    denominator = num_results * np.log2(num_results) - (num_results - num_relevant) * np.log2(num_results - num_relevant) - num_relevant * np.log2(num_relevant) if num_results != num_relevant and num_results - num_relevant != 0 and num_relevant != 0 else 1

    normalized_precision = (1 - (numerator / denominator)) if denominator != 0 else 0
    return normalized_precision

def mean_precision2(results, relevant):
    if results is None or relevant is None:
        return 0
    return sum(precision_at(i / 10, results, relevant) for i in range(1, 11)) / 10


### Extensions

# TODO: put any extensions here
    def simple_feedback_method(docs):
        relevant_docs = [doc for doc, feedback in user_feedback.items() if feedback == 1]
        centroid_query = np.mean([documents[doc] for doc in relevant_docs], axis=0)

        # Weighted linear combination of original query and centroid query
        alpha = 0.5  # Weight for original query
        beta = 0.5   # Weight for centroid query
        weighted_combined_query = alpha * original_query + beta * centroid_query

        return weighted_combined_query
    
    def read_query_from_keyboard():
        """Read a query from the keyboard and return it as a Document object. """
        doc_id = int(input('Enter document ID: '))
        author = input("Enter author names (comma-separated): ").split(',')
        #print(type(author))
        title = input("Enter title words (space-separated): ").split()
        keyword = input("Enter keywords (space-separated): ").split()
        abstract = input("Enter abstract words (space-separated): ").split()
        doc =  Document(doc_id=doc_id,author=author, title=title, keyword=keyword, abstract=abstract)
        print(doc)

    return doc

#Extension 1
import itertools
from collections import Counter

def generate_bigrams(docs):
    bigrams = []
    docs = remove_stopwords(docs)
    for doc in docs:
        for section in doc.sections():
            # Generate bigrams from each section
            section_bigrams = zip(section, section[1:])
            bigrams.extend(section_bigrams)
    return bigrams

def compute_bigram_frequencies(docs):
    bigrams = generate_bigrams(docs)
    bigram_freqs = Counter(bigrams)
    return bigram_freqs

def write_bigram_freqs(bigram_freqs, output_file):
    with open(output_file, 'w') as f:
        for bigram, freq in bigram_freqs.items():
            f.write(f"{bigram[0]} {bigram[1]}\t{freq}\n")


#Extension 2     

def get_total_words():
    docs = read_docs('cacm.raw')
    total_words = set()
    for doc in docs:
        for sec in doc.sections():
            for word in sec:
                total_words.add(word)
    total_words = list(total_words)
    total_words.sort()
    return total_words


def meanX(dataX):
    return np.mean(dataX,axis=0)

def svd(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)
    featValue, featVec= np.linalg.eig(covX)
    index = np.argsort(-featValue)
    if k > n:
        print("k must lower than feature number")
        return
    else:
        selectVec = np.matrix(featVec.T[index[:k]])
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return reconData, index[:k]


total_words = get_total_words()

import numpy as np

def convert(docVecs, queryVecs):
    # Convert docVecs and queryVecs to matrices
    doc_matrix = np.array([[vec.get(word, 0) for word in total_words] for vec in docVecs])
    query_matrix = np.array([[vec.get(word, 0) for word in total_words] for vec in queryVecs])

    # Combine docMatrix and queryMatrix
    combined_matrix = np.vstack([doc_matrix, query_matrix])

    # Perform SVD
    U, Sigma, VT = np.linalg.svd(combined_matrix, full_matrices=False)

    # Determine the number of retained dimensions
    k = int(len(total_words) * 0.9)  # Adjust as needed

    # Reduce dimensions
    docMatrix_reduced = np.dot(U[:len(docVecs), :k], np.diag(Sigma[:k]))
    queryMatrix_reduced = np.dot(U[len(docVecs):, :k], np.diag(Sigma[:k]))

    # Convert reduced matrices back to dictionaries
    newDocVecs = [dict(zip(total_words, row)) for row in docMatrix_reduced]
    newQueryVecs = [dict(zip(total_words, row)) for row in queryMatrix_reduced]

    return newDocVecs, newQueryVecs



### Search

def experiment():

    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

    term_funcs = {
        'tf': compute_tf,
        'tfidf': compute_tfidf,
        'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim,
        'jaccard': jaccard_sim,
        'dice': dice_sim,
        'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        sim_funcs,
        [TermWeights(author=1, title=1, keyword=1, abstract=1),
            TermWeights(author=1, title=3, keyword=4, abstract=1),
            TermWeights(author=1, title=1, keyword=1, abstract=4),
            TermWeights(author=3, title=4, keyword=3, abstract=1)] # 5b
    ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.50', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t  ')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

        metrics = []

        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights)
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query.doc_id]

            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results,rel), # missing value for p_1.0 in table 
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])

        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)),*averages, sep='\t\t')
    # print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t\t')

        return  # TODO: just for testing; remove this when printing the full table
    
def SVDExperiment():
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

    term_funcs = {
        #'tf': compute_tf,
        'tfidf': compute_tfidf
        #'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim,
        # 'jaccard': jaccard_sim,
        # 'dice': dice_sim,
        # 'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False,True],  # stem
        [False,True],  # remove stopwords
        sim_funcs,
        [TermWeights(author=3, title=4, keyword=3, abstract=1),
         TermWeights(author=1, title=1, keyword=1, abstract=4)]
    ]
    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]
        query_vectors = [term_funcs[term](query, doc_freqs, term_weights) for query in processed_queries]
        query_ids = [query.doc_id for query in processed_queries]

        doc_vectors, query_vectors = convert(doc_vectors, query_vectors)
        metrics = []

        for index, query_vec in enumerate(query_vectors):
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query_ids[index]]

            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])

        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t')



def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


def search_debug(docs, query, relevant, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    print('Query:', query)
    print('Relevant docs: ', relevant)
    print(results)
    for doc_id, score in results_with_score[:10]:
        print('Score:', score)
        print(docs[doc_id - 1])
        print()


if __name__ == '__main__':

    experiment()

    #Extension 1 Calls
    docs = read_docs('cacm.raw')
    # Compute bigram frequencies
    bigram_freqs = compute_bigram_frequencies(docs)
    # Write bigram frequencies to file
    write_bigram_freqs(bigram_freqs, 'bigram_frequencies_final.txt')

    #Extension 2 Calls
    SVDExperiment()




