import numpy as np
import jieba


def pad_sentences(sentence, sequence_length, padding_word="<PAD/>"):
    if len(sentence) < sequence_length:
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
    else:
        new_sentence = sentence[:sequence_length]
    return new_sentence


def build_input_data(data_left, vocab):
    out_left = [str(vocab[word]) if word in vocab else str(vocab['<UNK/>']) for word in data_left]

    return out_left


class Gen_Data_loader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.token_stream = []

    # def transform_positive_file(self, srcfile, positive_file, wordVocab, SEQ_LENGTH):
    #     out_op = open(positive_file, "w")
    #     for line in open(srcfile):
    #         line = line.strip("\n").split("\t")
    #         sentence = line[0]
    #         sent = jieba.lcut(sentence)
    #         sent = [s.encode("utf8") for s in sent]
    #         padded_sentence = pad_sentences(sent, SEQ_LENGTH)
    #         sentence_index = build_input_data(padded_sentence, wordVocab.word2id)
    #         out_op.write(" ".join(sentence_index) + "\n")
    #     out_op.close()

    def transform_positive_file_2(self, srcfile, positive_file, wordVocab, SEQ_LENGTH):
        out_op = open(positive_file, "w")
        for line in open(srcfile):
            line=line.decode("utf8")
            line = line.strip("\n").split("\t")
            sentence = line[0]
            sent = [s.encode("utf8") for s in sentence]
            # sent = jieba.lcut(sentence)
            # sent = [s.encode("utf8") for s in sent]
            padded_sentence = pad_sentences(sent, SEQ_LENGTH)
            sentence_index = build_input_data(padded_sentence, wordVocab.word2id)
            out_op.write(" ".join(sentence_index) + "\n")
        out_op.close()

    def create_batches(self, data_file):
        self.token_stream = []
        with open(data_file, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    self.token_stream.append(parse_line)

        self.num_batch = int(len(self.token_stream) / self.batch_size)
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
