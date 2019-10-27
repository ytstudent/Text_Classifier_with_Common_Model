import pandas as pd


class TransformType(object):
    def __init__(self):
        pass

    @staticmethod
    def from_txt_to_df(path):
        """
        transform intention.txt to a pd.DataFrame
        :param path: path for intention.txt
        :return:pd.DataFrame
        """
        with open(path, "r", encoding="utf-8-sig") as f:
            data = f.read()
        # print(data)
        data_list = data.strip().split("\n")
        # print(data_list)
        zs = []
        label = []
        for i in data_list:
            i_list = i.split("\t")
            if i_list != ['']:
                try:
                    zs.append(i_list[0])
                    # print(i_list[0])
                    label.append(i_list[1])
                except IndexError:
                    pass
        df = pd.DataFrame({"information": zs, "intention": label})
        return df
