import pyterrier as pt
import pandas as pd

if not pt.started():
    pt.init()

print(pt.datasets.list_datasets()) 

# Generate the indexer and its data directory
indexer = pt.DFIndexer("./indexes/index_tokens_all")

# Setting the indexer properties
index_properties = {"termpipelines": ""}
indexer.setProperties(**index_properties)

# Gets the database info generated by CISI_parser.py
df = pd.read_csv("./res/cisi_dataframe.csv")
index_ref = indexer.index(df['text'], docno=df['docno'].astype(str))

# Generate The index, there cannot be an ./index directory in existance whne this happens 
index = pt.IndexFactory.of(index_ref)

# Test print
print(index.getCollectionStatistics().toString())

vaswani = pt.datasets.get_dataset("vaswani")

#print("Files in vaswani corpus: %s " % vaswani.get_corpus())

TF_IDF = pt.BatchRetrieve(index, controls = {"wmodel": "TF_IDF"})
BM25 = pt.BatchRetrieve(index, controls = {"wmodel": "BM25"})
PL2 = pt.BatchRetrieve(index, controls = {"wmodel": "PL2"})

evals = pt.pipelines.Experiment([TF_IDF,BM25,PL2],vaswani.get_topics(),vaswani.get_qrels(),['map','ndcg'])

#print(evals)
