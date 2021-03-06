#############################################################
### Module for processing data used in the ACERPI project ###
#############################################################

import pandas as pd

#import tokenizer from nltk that is able to separate text into sentences and tokens.
from nltk import tokenize, download
download('punkt')

### This function is used in spacy generated anottated datasets. 
### It changes the dataset so the named entity tags are associated with each token instead of each document and gives a dataframe with each token.
def extract_spacy_tokens(spacy_dataframe):
    entity_id = 0

    #Loop through each document
    for index, document in spacy_dataframe.iterrows():
        entity_tokens = {}

        #In each entity on the 'spans' column we populate a dictionary with the id of the tokens and the entity they belong to
        for entity in document['spans']:
            for token_id in range(entity['token_start'], entity['token_end'] + 1, 1):
                entity_tokens[token_id] = entity_id
            entity_id += 1

        #For each token in the 'tokens' column we give them the PER tag if they belong to an entity and tag the ID of the entity they belong to. We also tag the document ID.    
        for token in document['tokens']:
            if token['id'] in entity_tokens:
                token['tag'] = 'PER'
                token['entity_id'] = entity_tokens[token['id']]
            else:
                token['tag'] = 'O'
            token['document'] = index

    tagged_tokens = spacy_dataframe['tokens'].to_numpy()
    
    ## Flatten array of lists
    ##The 'tokens' columns is currently in a format where each row contain a list of dictionaries. Since we want a single list of token we need to flatten it so that we only have a single list of dictionaries that contains the tokens from every single document of the dataset. We will not lose the document information since we added it as a key-value pair of the dictionary before.
    token_dict_list = []
    for index, dict_list in enumerate(tagged_tokens):
        token_dict_list += dict_list

    ## Now we transform the list of dictionaries into a dataframe and each column will be a single anotated token. 
    ## We only care about the tokens that were anotatted as 'PER' by the SpaCy library since we still want to clean the rest of the document and the other tokens might change.
    token_df = pd.DataFrame(token_dict_list)
    token_df = token_df[token_df['tag'] == 'PER'].drop(columns = ['start', 'end', 'id', 'document'])

    return spacy_dataframe, token_df



### This function takes a text document that contains repeated spaces, new line character and wrong punctuation that could break sentence splits and cleans it.
def clear_text(text_input_series):
    # #Remove \n character and repeated whitespaces
    clean_text = text_input_series.replace(r'(n??\.) ?','n?? ', regex=True)
    clean_text = clean_text.replace(r'\n',' ', regex=True)
    clean_text = clean_text.replace(r'[ ]+', ' ', regex = True)
    return clean_text



### This function splits a series of documents into a dataframe of phrases. 
### The resulting dataframes will contain columns with the sentence's text, the sentence's index and the original document's index.
def split_text_sentences(text_input_series):
    # #Split documents into phrases
    sentence_df = text_input_series.apply(lambda row: tokenize.sent_tokenize(row, language='portuguese'))

    # #Turn series of list of phrases into series of phrases. Index will indicate which document it belongs to for now.
    sentence_df = sentence_df.explode()
    sentence_df = pd.DataFrame(sentence_df)
    
    # Put sentece_df index as a column and the sentence id as another column
    sentence_df = sentence_df.reset_index()
    sentence_df['SentenceID'] = sentence_df.index
    sentence_df.columns = ['document', 'sentence', 'sentence_id']
    return sentence_df


# This function takes a list of spacy tokens and anotates another list of tokens based on it.
def find_labeled_entities(tokens, spacy_tokens):
    tokens['label'] = 'O'
    for entity in range(int(spacy_tokens['entity_id'].max())):
        matched_tokens = []
        entity = list(spacy_tokens['text'][spacy_tokens['entity_id'] == entity])#.iterrows()
        if entity:
            #print('\nEntitiy Token List:\n', entity)
            #print(len(entity))
            first_tokens = tokens[tokens['tokens'] == entity[0]]
            #print(first_tokens)
            first_index_list = first_tokens.index.values
            #print(first_index_list)
            #print('\nPossible entity matches:')
            for index in first_index_list:
                #print(list(tokens['tokens'].iloc[index:index + len(entity)]))
                found_token_list = list(tokens['tokens'].iloc[index:index + len(entity)])
                #print(found_token_list == entity)
                if found_token_list == entity:
                    tokens['label'].iloc[index:index + len(entity)] = 'PER'
    return tokens

def apply_iob_format(token_df):
    for token_idx in range(token_df.shape[0]):
        if token_df['Tag'].iloc[token_idx] == 'PER':
            if token_df['Tag'].iloc[token_idx-1] == 'O':
                token_df['Tag'].iloc[token_idx] = 'B-per'
            else:
                token_df['Tag'].iloc[token_idx] = 'I-per'
    return token_df