from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

# Customize training with vocab size and min frequency
tokenizer.train(files=["output_train.txt","output_val.txt"], vocab_size=45000, min_frequency=2)

tokenizer.save_model("./BPEVocab")