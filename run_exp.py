import pyterrier as pt
import pandas as pd
import time
import sys
import os

class Indexer():

    def __init__(self, index_path, stemmer, corpus):      
        self.index_path = index_path
        
        # Get the corpus
        self.corpus = corpus
        self.indexer = pt.TRECCollectionIndexer(index_path)

        # Set the stemmer, and don't add stopwords
        index_props = None
        if stemmer=="snowball":
            index_props = {"termpipelines": "EnglishSnowballStemmer"}
        elif stemmer == "porter":
            index_props = {"termpipelines": "PorterStemmer"}
        else: # No Stemmer
            index_props = {"termpipelines": "NoOp"}
        self.indexer.setProperties(**index_props)

    def index(self):
        return self.indexer.index(self.corpus)

class Retriever():
    def __init__(self, index):
        self.index = index
        self.retr = pt.BatchRetrieve(self.index, controls = {"wmodel": "BM25"})

    def transform(self, topics):
        return self.retr.transform(topics)

    def save_result(self, res, path):
        self.retr.saveResult(res,path)
    

def gen_index(idx_path, stemmer, data_corpus):
    # Generate the indexer
    indexer = Indexer(idx_path, stemmer, data_corpus)
    # The indexing
    start_time = time.time()
    indexref = indexer.index()
    end_time = time.time()
    index_time_len = end_time - start_time
    return (indexref, index_time_len)

 
"""
dataset: string, the name of the dataset for PyTerrier to use
index_dir_name: string, name of the indexer to use
stemmer: string, stemmer type
only_retrieval: boolean, 
"""
def conduct_experiment( dataset_name, index_dir_name, stemmer, only_retrieval):

    index_time_len = None
    print("Conducting an experiment [", dataset_name, ", ", stemmer, ", ", only_retrieval, "]:")

    indexref = None
    if( not only_retrieval ):
        # Gets the corpus for the experiment
        dataset_name = dataset_name.lower()
        corpus = None
        if dataset_name == "vaswani" or dataset_name == "trec-deep-learning-docs":
            ds = pt.datasets.get_dataset(dataset_name)
            corpus = ds.get_corpus()
        else:
            print("No corpus found, Shutting down.")
            return None

        # Generated index, get path to its data.properties
        (indexref, elapsed_time) = gen_index(index_dir_name, stemmer, corpus)        
        index = pt.IndexFactory.of(indexref)
        
        # Print a nice output
        print("Time for indexing of [", dataset_name, ", ", stemmer, "]: ", str(index_time_len) )
        print(index.getCollectionStatistics().toString())


    else:
        index = pt.IndexFactory.of(index_dir_name + '/data.properties')
   

    # Get the other Info
    queries = None
    qrels = None
    if dataset_name == "vaswani":
        ds = pt.datasets.get_dataset("vaswani")
        queries = ds.get_topics()
        qrels = ds.get_qrels()
    elif dataset_name == "trec-deep-learning-docs":
        ds = pt.datasets.get_dataset("trec-deep-learning-docs")
        queries = ds.get_topics("dev")
        qrels = ds.get_qrels("dev")
    else:
        print("No queries or qrels found, Shutting down.")
        return None

    # Get the model
    BM25 = pt.BatchRetrieve(index, controls = {"wmodel": "BM25"})
    res = BM25.transform(queries)

    # Evaluate the model
    evaluation = pt.Utils.evaluate(res, qrels, metrics=["map"], perquery=False)
    print(evaluation)

    return index_time_len, evaluation


def main():
    if not pt.started():
        pt.init()

    #   Valid Stemmers: "porter", "snowball" and ""
    #   Valid Datasets: "vaswani", "trec-deep-learning-docs"

    # Args
    argv = sys.argv 

    if( len(argv) < 4 ):
        print(" Less than 4 arguements inputted, quiting the program.")
        return 1

    dataset = argv[1]
    index_loc = argv[2]
    stemmer = argv[3]
    only_retr = argv[4] == 'T'

    time_taken, evaluation = conduct_experiment(dataset, index_loc, stemmer, only_retr)


if __name__ == "__main__":
    main()
