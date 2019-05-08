from text_generator import prepare, model

data = open('data/sample.txt').read()
predictors, label, max_sequence_len, total_words = prepare.from_data(data)
model = model.learn(predictors, label, max_sequence_len, total_words)
print (model.summary())
