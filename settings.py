chinese_bert_model_name = "bert-base-chinese"
english_bert_model_name = "bert-base-uncased"
max_tokenization_length = 128
# Limit to sentence in the poem
batch_size = 16
num_train_epochs = 20
learning_rate = 3e-4  # magic learning rate for adam, lol
weight_decay = 0.01
adam_epsilon = 1e-8
fp16 = True  # half precision training
# Number of process workers to process data
num_workers = 2
# min_word_frequency = 8
# things specific to chinese text
path_to_tang_poetry_corpus = (
    "datasets/source/poetry.txt"  # relative to project root
)
path_to_tang_poetry_vocab = "datasets/output/poetry_vocab.csv"
path_to_tang_poetry_checkpoint = "models/tang_poetry.ckpt"
max_line_length = 64
disallowed_words = ["（", "）", "(", ")", "__", "《", "》", "【", "】", "[", "]"]
path_to_haiku_source = "datasets/source/haikus.csv"
path_to_haiku_corpus = "datasets/output/haiku_corpus.txt"
path_to_haiku_checkpoint = "models/haiku.ckpt"
