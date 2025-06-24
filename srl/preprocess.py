def preprocess_tokens (tokens):
    list_of_wrong_letter = ['_', '«','»']
    new_tokens_list = list()

    for sentence in tokens['tokens']:
        new_tokens = ' '.join(sentence)
        for letter in new_tokens:
            if letter in list_of_wrong_letter:
                for wrong_letter in list_of_wrong_letter:
                    new_tokens = new_tokens.replace(wrong_letter, '')
        list_words = new_tokens.split()  
        formated_list = [sentence for sentence in list_words]
        new_tokens_list.append(formated_list)
    return new_tokens_list