import pyterrier as pt

class Indexer():

    def __init__(self, index_path, stemmer, corpus):      
        self.index_path = index_path
        self.indexer = pt.TRECCollectionIndexer(index_path)
        self.corpus = corpus
        if stemmer=="snowball":
            index_properties = {"termpipelines": "Stopwords, EnglishSnowballStemmer"}
            self.indexer.setProperties(**index_properties)

    def index(self):
        return self.indexer.index(self.corpus)

class Retriever():
    def __init__(self, index):
        self.index = index
        self.retr = pt.BatchRetrieve(self.index, controls = {"wmodel": "TF_IDF"})

    def transform(self, topics):
        return self.retr.transform(topics)

    def save_result(self, res):
        self.retr.saveResult(res,"res/result1.res")
    
    
def main():
    if not pt.started():
        pt.init()

    dataset = pt.datasets.get_dataset("vaswani")
    indexer = Indexer("./index", "snowball", dataset.get_corpus())
    indexref = indexer.index()

    index = pt.IndexFactory.of(indexref)
    topics = dataset.get_topics()
    retr = Retriever(index)
    res = retr.transform(topics)
    
    retr.save_result(res)

if __name__ == "__main__":
    main()
