from torchtext.legacy.data import Field
from torchtext.legacy.data.dataset import Dataset
from model import START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN
import yaml

"""
Defining the mapping between numeration symbols in roman numerals and their corresponding arabic numerals
"""
num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
           (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]


"""
An arabic-to-roman numeral conversion function which works by successive subtraction of the largest possible
roman numeral representation from the current (remaining) value of the arabic one
"""
def num2roman(num):
    roman = ''

    while num > 0:
        for i, r in num_map:
            while num >= i:
                roman += r
                num -= i

    return roman


"""
Configuring the torchtext Field data structure that models `tokens` in the arabic numeral dataset (i.e. strings
representing individual digits from 0-9 + 4 tokens for <start of number>, <end of number>, <padding> and <unknown token>
"""
SRC = Field(init_token=START_TOKEN,
            eos_token=END_TOKEN,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN)

"""
Configuring the torchtext Field data structure that models `tokens` in the roman numeral dataset (i.e. strings
representing numeration elements (M, D, C, L, X, V, I) + 4 tokens for <start of number>, <end of number>, <padding>
and <unknown token>
"""
TRG = Field(init_token=START_TOKEN,
            eos_token=END_TOKEN,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN)


class NumeralExample(object):
    """
    Class modeling an example entry in the numeral conversion dataset.
    The src_numeral is the string representation of the `source` arabic numeral (e.g. "2021")
    The trg_numeral is the string representation of the `target` roman numeral (e.g. "MMXXI")
    """
    def __init__(self, src_numeral, trg_numeral):
        self.src = src_numeral
        self.trg = trg_numeral
        
    def __str__(self):
        return "src: %s, trg: %s" % (self.src, self.trg)
    
    def __repr__(self):
        return self.__str__()


class NumeralsDataset(Dataset):
    """
    Creating a PyTorch Dataset class (inheriting from torchtext.legacy.data.dataset.Dataset) to hold the
    translation examples of type NumeralExample (pairs of
    """
    def __init__(self, examples):
        super(NumeralsDataset, self).__init__(examples=examples, fields=[("src", SRC), ("trg", TRG)])
    

def generate_dataset(max_num=2021):
    """
    Method that generates the pairs of (src_numeral, trg_numeral) ranging from 1 to 2021
    :param max_num:
    :return:
    """
    examples = []
    for num in range(1, max_num + 1):
        src = str(num)
        trg = num2roman(num)
        
        examples.append(NumeralExample(src_numeral=src, trg_numeral=trg))
    
    return examples


def load_configurations(config_file_path: str):
    with open(config_file_path) as config_file:
        return yaml.load(config_file)



