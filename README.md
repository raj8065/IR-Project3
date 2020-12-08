# IR-Project-3

## Prerequisites
This uses PyTerrier (https://github.com/terrier-org/pyterrier/) for experimental evaluation of information retrieval using three data models: 
1. Porter Stemming
2. Snowball Stemming
3. No Stemmer used.

Install and setup PyTerrier (instructions in the repository linked above) to set up PyTerrier in a Unix-based environment.



## Basic Operation
The System is run by running the run_exp.py file (whether through an IDE or "python3 run_search_engine.py").

This will either:
1. create an index in a location specified by you, 
2. load an index from a location specified by you
and perform retrieval using a PyTerrier-derived "Topics" file, and evaluate the results using a PyTerrier-derived "qrels" file.

## Example Usage
Once PyTerrier has been installed, run:

`python3 run_exp.py <name of dataset> <path-to-index> <stemmer> <retrieval-only>`
run_exp.py: File to run
name of dataset: the dataset's name. This program currently works with the datasets 'vaswani' and 'trec-deep-learning-docs'
path to index: Path to where you want the index to be stored/read from
Stemmer: Stemming algorithm you wish to use/that the index has been stored with
retrieval-only: Whether you want to perform indexing, retrieval, and evaluation, or only retrieval. Possible values: T (for the latter), and F (for the former)

An example run is:
`python3 run_exp.py trec-deep-learning-docs ./res/trec_porter_index/ porter T`
