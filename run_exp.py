import pyterrier as pt
import pandas as pd
import time

class Indexer():

    def __init__(self, index_path, stemmer, corpus, docno):      
        self.index_path = index_path
        
        self.corpus = corpus
        if docno is not None:
            self.indexer = pt.DFIndexer(index_path)
            self.doc_no =docno
        else:
            self.indexer = pt.TRECCollectionIndexer(index_path)
            self.doc_no = None

        if stemmer=="snowball":
            index_properties = {"termpipelines": "Stopwords, EnglishSnowballStemmer"}
            self.indexer.setProperties(**index_properties)
        else:
            index_properties = {"termpipelines": "Stopwords, PorterStemmer"}
            self.indexer.setProperties(**index_properties)

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
  
def main():
    if not pt.started():
        pt.init()

#     dataset = pt.datasets.get_dataset("vaswani")
#     v_corpus = dataset.get_corpus()
    
#     s_ctr = 0
#     p_ctr = 0
#     for i in range(0,20):
#         start = time.time()
#         indexref = gen_index("./res/temp/v_snowball_index", "snowball", v_corpus)
#         end = time.time()
#         s_ctr = s_ctr + (end-start)
#         #print("Time for Snowball indexing:", end - start)
#         start = time.time()
#         porter_indexref = gen_index("./res/temp/v_porter_index", "porter", v_corpus)
#         end = time.time()
#         p_ctr = p_ctr + (end-start)
#         #print("Time for Porter indexing:", end - start)
#     print("Average time for Snowball:", s_ctr/20)
#     print("Average time for Porter:", p_ctr/20)
    
#     index = pt.IndexFactory.of(indexref)
#     v_porter_idx = pt.IndexFactory.of(porter_indexref)
#     topics = dataset.get_topics()
#     s_res = perform_retrieval(index,topics,"res/v_snowball_res")
#     p_res = perform_retrieval(v_porter_idx,topics,"res/v_porter_res")

#     qrels = dataset.get_qrels()
#     s_evals = evaluate(s_res,qrels, metrics=["map"], perquery=False)
#     p_evals = evaluate(p_res, qrels, metrics=["map"], perquery=False)
#     print(s_evals, p_evals)


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
    
    
    trec_dl_dataset = pt.datasets.get_dataset("trec-deep-learning-docs")
    indexer = Indexer("./trec_porter_index", "porter", trec_dl_dataset.get_corpus(), None)
    start = time.time()
    indexref = indexer.index()
    end = time.time()
    print("Time for Porter indexing:", end - start)
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().toString())

if __name__ == "__main__":
    main()