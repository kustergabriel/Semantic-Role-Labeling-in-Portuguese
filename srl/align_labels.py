# %%
from datasets import load_dataset
from transformers import AutoTokenizer
import preprocess 
import os
import json
import finetunning

# %%
model = 'neuralmind/bert-base-portuguese-cased' 
url = 'liaad/Propbank-BR'
dataset = load_dataset(url, "default")

# %%
# Tratando alguns caracteres que podem dar erro na tokenizacao
new_tokens = preprocess.preprocess_tokens(dataset['train'])

# %%

def tokenize_and_align_labels (sentences,dataset_train):
    tokenizer = AutoTokenizer.from_pretrained(model)
    new_labels_aligned = dict ()
    debug = 0

    with open ("./out/toFineTunning.txt", "a") as f:
        for index, sentences_to_tokenize in enumerate(sentences):
            debug = debug + 1
            print (debug)
            new_labels_aligned[index] = []

            tokenized_sentence = tokenizer(sentences_to_tokenize, 
                                            is_split_into_words=True)
            
            words_ids = tokenized_sentence.word_ids()

            frames_for_this_sentence = dataset_train['srl_frames'][index]

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
                
                ids = [finetunning.label2id.get(label, 0) for label in new_labels]

                json.dump(
                        {'index': index,'verb': verb,'new_labels': new_labels, 'ids' : ids}, f)
                f.write('\n')

tokenize_and_align_labels(new_tokens,dataset['train'])