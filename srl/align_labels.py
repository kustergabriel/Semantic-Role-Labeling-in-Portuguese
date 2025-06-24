# %%
from datasets import load_dataset
from transformers import AutoTokenizer
import preprocess 

# %%
model = 'neuralmind/bert-base-portuguese-cased' 
url = 'liaad/Propbank-BR'
dataset = load_dataset(url, "default")

# %%
# Tratando alguns caracteres que podem dar erro na tokenizacao
new_tokens = preprocess.preprocess_tokens(dataset['train']) 

def tokenize_and_align_labels (sentences,dataset_train):
    tokenizer = AutoTokenizer.from_pretrained(model)
    new_labels_aligned = dict ()

    for index, sentences_to_tokenize in enumerate(sentences):
        new_labels_aligned[index] = []

        tokenized_sentence = tokenizer(sentences_to_tokenize, 
                                            padding = True, 
                                            is_split_into_words=True, 
                                            return_offsets_mapping=True)
        
        words_ids = tokenized_sentence.word_ids()

        frames_for_this_sentence = dataset_train['srl_frames'][index]
        
        for frame in frames_for_this_sentence:
            prev_word_id = None
            new_labels = []
            labels = frame['frames']
            verb = frame['verb']

            for word_id in words_ids:
                if word_id is None:
                    new_labels.append('O')
                elif word_id != prev_word_id:
                    if word_id < len(labels):
                        new_labels.append(labels[word_id])
                    else:
                        new_labels.append('O')  
                else:
                    new_labels.append('O')
                prev_word_id = word_id

            new_labels_aligned[index].append({'verb': verb, 'new_labels': new_labels})

tokenize_and_align_labels(new_tokens,dataset['train'])