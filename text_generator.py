from text_generator import prepare, model, generator

data = open('data/sample.txt').read()
predictors, label, max_sequence_len, total_words, tokenizer = prepare.from_data(data)
model = model.learn(predictors, label, max_sequence_len, total_words)
print(model.summary())

generated_text = generator.generate_text("the king", 3, max_sequence_len, model, tokenizer)
print(generated_text)
