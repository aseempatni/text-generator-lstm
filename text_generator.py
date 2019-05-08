from text_generator import prepare

data = open('data/sample.txt').read()
predictors, label, max_sequence_len, total_words = prepare.from_data(data)
