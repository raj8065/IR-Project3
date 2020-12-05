import pyterrier as pt
import pandas as pd
import time
import sys

class Indexer():

    def __init__(self, index_path, stemmer, corpus, docno):      
        self.index_path = index_path
        
        # Get the corpus
        self.corpus = corpus
        if docno is not None:
            self.indexer = pt.DFIndexer(index_path)
            self.doc_no =docno
        else:
            self.indexer = pt.TRECCollectionIndexer(index_path)
            self.doc_no = None

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
        if self.doc_no is not None:
            return self.indexer.index(self.corpus, self.doc_no)
        return self.indexer.index(self.corpus)

class Retriever():
    def __init__(self, index):
        self.index = index
        self.retr = pt.BatchRetrieve(self.index, controls = {"wmodel": "TF_IDF"})

    def transform(self, topics):
        return self.retr.transform(topics)

    def save_result(self, res, path):
        self.retr.saveResult(res,path)
    

def gen_index(idx_path, stemmer, data_corpus, docno = None):
    indexer = Indexer(idx_path, stemmer, data_corpus, docno)
    return indexer.index()

def perform_retrieval(index, topics, save_path):
    retr = Retriever(index)
    res = retr.transform(topics)
    retr.save_result(res, save_path)
    return res

def evaluate(res, qrels, metrics, perquery):
    return pt.Utils.evaluate(res,qrels,metrics=metrics, perquery=perquery)
 
"""
dataset: string, the name of the dataset for PyTerrier to use
index_dir_name: string, name of the indexer to use
stemmer: string, stemmer type
only_retrieval: boolean, 
"""
def conduct_experiment( dataset_name, index_dir_name, stemmer, only_retrieval ):

    index_time_len = None
    print("Conducting an experiment [", dataset_name, ", ", stemmer, ", ", only_retrieval, "]:")

    indexref = None
    if( not only_retrieval ):
        # Gets the corpus for the experiment
        dataset_name = dataset_name.lower()
        corpus = None
        if dataset_name == "vaswani":
            ds = pt.datasets.get_dataset("vaswani")
            corpus = ds.get_corpus()
        elif dataset_name == "trec-deep-learning-docs":
            ds = pt.datasets.get_dataset("trec-deep-learning-docs")
            corpus = ds.get_corpus()
        else:
            print("No corpus found, Shutting down.")
            return None

        # Generate the indexer
        indexer = Indexer(index_dir_name, stemmer, corpus, None)

        # The indexing
        start_time = time.time()
        indexref = indexer.index()
        end_time = time.time()
        index = pt.IndexFactory.of(indexref)

        index_time_len = end_time - start_time
        # Print a nice output
        print("Time for indexing of [", dataset_name, ", ", stemmer, "]: ", str(index_time_len) )
        print(index.getCollectionStatistics().toString())

        # Print Index Terms to a file
        # f = open("UnStopped_Index.txt","w")
        # for idx in index.getLexicon():
        #     f.write(idx.getKey() + "\n")
        # f.close()

    # Get index
    index = None
    if( only_retrieval ):
        index = pt.IndexFactory.of( index_dir_name + '/data.properties' )
    else:
        index = pt.IndexFactory.of(indexref)

    # Get the other Info
    queries = None
    qrels = None
    if dataset_name == "vaswani":
        ds = pt.datasets.get_dataset("vaswani")
        queries = ds.get_topics()
        qrels = ds.get_qrels()
    elif dataset_name == "trec-deep-learning-docs":
        ds = pt.datasets.get_dataset("trec-deep-learning-docs")
        queries = ds.get_topics()
        qrels = ds.get_qrels()
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

    if( argv < 4 ):
        print(" Less than 4 arguements inputted, quiting the program.")
        return 1

    dataset = argv[1]
    index_loc = arg[2]
    stemmer = arg[3]
    only_retr = argv[4] == 'T'

    time_taken, evaluation = conduct_experiment(dataset, index_loc, stemmer, only_retr)

#     cisi_dataset = pd.read_csv("./res/cisi_dataframe.csv")
#     corpus = cisi_dataset['text']
#     doc_no = cisi_dataset['docno'].astype(str)
#     #cisi_s_indexref = gen_index("./res/c_snowball_index", "snowball", corpus, doc_no)
#     cisi_p_indexref = gen_index("./res/c_porter_index", "porter", corpus, doc_no)
#     cisi_s_indexref = gen_index("./res/c_snowball_index", "snowball", corpus, doc_no)

#     cisi_p_index = pt.IndexFactory.of(cisi_p_indexref)
    #cisi_s_index = pt.IndexFactory.of(cisi_s_indexref) 
    #topics =  pt.io.read_topics("./res/CISI.QRY")
    #cisi_s_res = perform_retrieval(cisi_s_index, topics, "res/cisi_s_res")
    #cisi_p_res = perform_retrieval(cisi_p_index, topics, "res/cisi_p_res")
    
#     trec_dl_dataset = pt.datasets.get_dataset("trec-deep-learning-docs")
#     indexer = Indexer("./trec_index", "snowball", trec_dl_dataset.get_corpus(), None)
#     indexref = indexer.index()
#     index = pt.IndexFactory.of(indexref)
#     print(index.getCollectionStatistics().toString())
    
    
#    trec_dl_dataset = pt.datasets.get_dataset("trec-deep-learning-docs")
#    indexer = Indexer("./trec_porter_index", "porter", trec_dl_dataset.get_corpus(), None)
#    start = time.time()
#    indexref = indexer.index()
#    end = time.time()
#    print("Time for Porter indexing:", end - start)
#    index = pt.IndexFactory.of(indexref)
#    print(index.getCollectionStatistics().toString())

if __name__ == "__main__":
    main()
