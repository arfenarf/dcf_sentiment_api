from nlp_utils.text_processing import prepare_text, prepare_save_corpus_dict, calculate_save_model

text_data = prepare_text()
prepare_save_corpus_dict(text_data)

prepare_text(input_type = 'text', input_text="I had trouble with the computer while I was working on her teeth")


print('wait')