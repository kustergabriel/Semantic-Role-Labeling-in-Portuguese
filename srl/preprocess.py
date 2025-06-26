# %%
def preprocess_tokens (tokens):
    list_of_wrong_simbol = ['_', '«','»']
    new_tokens_list = list()

    for sentence in tokens['tokens']:
        new_tokens = ' '.join(sentence)
        for word in new_tokens:
            if word in list_of_wrong_simbol:
                new_tokens = new_tokens.replace(word, '')
        list_words = new_tokens.split()  
        formated_list = [sentence for sentence in list_words]
        
        new_tokens_list.append(formated_list)
    
    return new_tokens_list