# %%
from datasets import load_dataset
from transformers import AutoTokenizer
import preprocess 
import os
import json
import labels2id as labels2id

# %%
model = 'neuralmind/bert-base-portuguese-cased' 
url = 'liaad/Propbank-BR'
dataset = load_dataset(url, "default")
fullDatasetTrain = dataset['train']
fullDatasetTest = dataset['test']

# %%
# Tratando alguns caracteres que podem dar erro na tokenizacao
newTokensTrain = preprocess.preprocess_tokens(fullDatasetTrain[0:100])
newTokensTest = preprocess.preprocess_tokens(fullDatasetTest[0:30])

# %%

def tokenize_and_align_labels (sentences,dataset):
    tokenizer = AutoTokenizer.from_pretrained(model)
    new_labels_aligned = dict ()
    new_word_ids_dict = dict ()
    new_word_ids_list = list ()


    for index, sentences_to_tokenize in enumerate(sentences):
        new_labels_aligned[index] = []

        tokenized_sentence = tokenizer(sentences_to_tokenize, 
        is_split_into_words=True, 
        padding=True)

        words_ids = tokenized_sentence.word_ids() # aqui ele eh rapido 

        #print (f"words_ids : {words_ids}, index : {index}" )
        #frames_for_this_sentence = dataset['srl_frames'][index] # aqui ele demora mto
        #print(frames_for_this_sentence)
        #print (f"words_ids: {words_ids}, frames in the sentence: {frames_for_this_sentence}")
        
        # Vou tentar percorrer o words_ids de outra forma
        prev_word_id = None
        for word_id in words_ids:
            if word_id != prev_word_id and word_id is not None:
                new_word_ids_list.append (word_id)
            else:
                pass
            prev_word_id = word_id

            new_word_ids_dict = {'word_id' : new_word_ids_list}
        print (words_ids)
        print (new_word_ids_list)

        '''
        words_ids = tokenized_sentence.word_ids()
        
        frames_for_this_sentence = dataset['srl_frames'][index]

        for frame in frames_for_this_sentence:
            prev_word_id = None
            new_labels = []
            labels = frame['frames']
            verb = frame['verb']

            for word_id in words_ids:
                if word_id != prev_word_id and word_id is not None:
                    new_labels.append(labels[word_id])
                else:
                    new_labels.append('O')
                prev_word_id = word_id
            
            ids = [labels2id.label2id.get(label, 0) for label in new_labels]
            
            # Dicionario 
            new_labels_aligned = {'index': index,
                        'verb': verb,
                        'new_labels': new_labels, 'ids' : ids,
                        'input_ids' : tokenized_sentence['input_ids'],
                        'attention_mask' : tokenized_sentence ['attention_mask']
                        }

            print (new_labels_aligned)

'''
tokenize_and_align_labels(newTokensTrain,fullDatasetTrain)