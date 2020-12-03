import pyterrier as pt
if not pt.started():
    pt.init()

dataset = pt.datasets.get_dataset("vaswani")

print("Files in vaswani corpus: %s " % dataset.get_corpus())
index_path = "./index"
indexer = pt.TRECCollectionIndexer(index_path)


index_properies = {"block.indexing":"true", "invertedfile.lexiconscanner":"pointers", "termpipelines": "Stopwords, EnglishSnowballStemmer"}
indexer.setProperties(**index_properies)

indexref = indexer.index(dataset.get_corpus())