import csv

import numpy as np


class data_helper():
    def __init__(self, sequence_max_length=1024):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
        self.char_dict = {}
        self.sequence_max_length = sequence_max_length
        for i, c in enumerate(self.alphabet):
            self.char_dict[c] = i + 1

    def char2vec(self, text):
        data = np.zeros(self.sequence_max_length)
        for i in range(0, len(text)):
            #  假如超过1024个字符则返回1024个字符
            if i > self.sequence_max_length:
                return data
            elif text[i] in self.char_dict:
                data[i] = self.char_dict[text[i]]
            else:
                # 未知字符为68
                data[i] = 68
        return data

    def load_csv_file(self, filename, num_classes):
        all_data = []
        labels = []
        with open(filename) as f:
            reader = csv.DictReader(f, fieldnames=['class'], restkey='fields')
            for row in reader:
                # one_hot
                one_hot = np.zeros(num_classes)
                one_hot[int(row['class']) - 1] = 1
                labels.append(one_hot)
                # char2vec
                data = np.ones(self.sequence_max_length) * 68
                text = row['fields'][-1].lower()
                all_data.append(self.char2vec(text))
        f.close()
        return np.array(all_data), np.array(labels)

    def load_dataset(self, dataset_path):
        with open(dataset_path + "classes.txt") as f:
            classes = []
            for line in f:
                classes.append(line.strip())
        f.close()
        num_classes = len(classes)
        train_data, train_label = self.load_csv_file(dataset_path + 'train.csv', num_classes)
        test_data, test_label = self.load_csv_file(dataset_path + 'test.csv', num_classes)
        return train_data, train_label, test_data, test_label

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_bathes_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffle_data = data[shuffle_indices]
            else:
                shuffle_data = data
            for batch_num in range(num_bathes_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffle_data[start_index:end_index]
