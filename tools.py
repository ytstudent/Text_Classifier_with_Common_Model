import collections
import numpy as np


class Tools(object):
    def __init__(self):
        self.dictionary = {}

    def word_bag(self, text):

        info_dict_list = text.split(" ")
        words = info_dict_list

        count = [['UNK', -1], ["PAD", -2]]

        # All the other words will be replaced with UNK token
        count.extend(collections.Counter(words).most_common())

        # Create an ID for each word by giving the current length of the dictionary
        # And adding that item to the dictionary

        for i, _ in count:
            self.dictionary[i] = len(self.dictionary)

        data = []
        unk_count = 0
        # # Traverse through all the text we have and produce a list
        # # where each element corresponds to the ID of the word found at that index
        for i in words:
            # If word is in the dictionary use the word ID,
            # else use the ID of the special token "UNK"
            if i in self.dictionary:
                index = self.dictionary[i]
            else:
                index = self.dictionary['UNK']
                unk_count = unk_count + 1
            data.append(index)
        # print(unk_count)
        # print("data:{}".format(data))
        # # update the count variable with the number of UNK occurences
        count[0][1] = unk_count

        reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        # print(reverse_dictionary)
        # self.dictionary['PAD'] = len(dictionary)
        return words, data, self.dictionary, reverse_dictionary

    # # Make sure the dictionary is of size of the vocabulary
    # assert len(dictionary) == vocabulary_size

    def sent2array(self, str1, token_num):
        token_num = int(token_num)
        token_list = str1.split(" ")  # token_list is sentence
        num_array = []
        for token in token_list:
            if token in self.dictionary:
                num = self.dictionary[token]
            else:
                num = self.dictionary['UNK']
            num_array.append(num)
        num_array = np.array(num_array)
        # if len(num_array) == 1:
        #     print(num_array, str1)
        if len(num_array) < token_num:

            add_num_array = np.zeros(token_num - len(num_array)) + self.dictionary["PAD"]
            num_array = np.concatenate((num_array, add_num_array))
        else:
            num_array = num_array[:token_num]

        return num_array

    def shuffle_batch(self, X, y, batch_size, max_length):
        random_X = np.random.permutation(len(X))
        n_batches = len(X) // batch_size
        for batch in np.array_split(random_X, n_batches):
            X_batch, y_batch = X[batch], y[batch]
            # print(X_batch.shape)
            real_len = []
            for i in range(X_batch.shape[0]):
                try:
                    unk_index = np.argwhere(X_batch[i, :] == self.dictionary["PAD"])[0][0]
                    real_len.append(unk_index)
                except IndexError:
                    unk_index = max_length
                    real_len.append(unk_index)
            yield X_batch, y_batch, n_batches, real_len


if __name__ == "__main__":
    from type_transform import TransformType

    trans = TransformType()
    tools = Tools()

    data_all_df = trans.from_txt_to_df(r".\data\data_all.txt")
    training_df = trans.from_txt_to_df(r".\data\training_data.txt")
    testing_df = trans.from_txt_to_df(r".\data\testing_data.txt")
    info_str = " , ".join(data_all_df["information"])

    _, _, dictionary, _ = tools.word_bag(info_str)
    print(dictionary)
