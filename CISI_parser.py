import pandas as pd

def add_new_doc(data, doc_id):
    data['docno'].append(doc_id.strip())
    data['text'].append('')

def add_new_line(data, text):
    data['text'][-1] =  data['text'][-1] + text

def make_CISI_dataframe():
    f = open('./res/CISI.ALL')
    data = {'docno':[], 'text':[]}

    ref_sec = False
    for line in f:
        ident = line[0:2]
        if( ident == '.I' ): # An ID line
            add_new_doc(data, line[2:])
            ref_sec = False
        elif( not ref_sec ): # If it's not int the .X section and not an ID line ...
            if( ident == '.X' ): # If it is the reference section
                ref_sec = True
            elif( (ident == '.T' or ident == '.A' or ident == '.W') and line[2:].strip() == "" ):
                # If its a marker for the title, author, or body ignore it
                # We are making a document that is a concatination of these features
                continue
            else:
                add_new_line(data, line)
    df = pd.DataFrame(data)
    df.set_index('docno', inplace=True)
    return df

if __name__ == "__main__":
    df = make_CISI_dataframe()
    df.to_csv("./res/dataframe.csv")
