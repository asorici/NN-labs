from torchtext.legacy.data import Field, BucketIterator
from torch.utils.data import DataLoader
from torchtext.legacy.data.dataset import Dataset
from model import START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN

num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
           (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

def num2roman(num):

    roman = ''

    while num > 0:
        for i, r in num_map:
            while num >= i:
                roman += r
                num -= i

    return roman


def tokenize_arabic_numeral(dataset_item):
    return list(dataset_item[0])


def tokenize_roman_numeral(dataset_item):
    return list(dataset_item[1])


SRC = Field(tokenize=tokenize_arabic_numeral,
            init_token=START_TOKEN,
            eos_token=END_TOKEN,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN)

TRG = Field(tokenize=tokenize_roman_numeral,
            init_token=START_TOKEN,
            eos_token=END_TOKEN,
            pad_token=PAD_TOKEN,
            unk_token=UNK_TOKEN)


class NumeralExample(object):
    def __init__(self, src_numeral, trg_numeral):
        self.src = src_numeral
        self.trg = trg_numeral
        
    def __str__(self):
        return "src: %s, trg: %s" % (self.src, self.trg)
    
    def __repr__(self):
        return self.__str__()


class NumeralsDataset(Dataset):
    def __init__(self, examples):
        super(NumeralsDataset, self).__init__(examples=examples, fields=[("src", SRC), ("trg", TRG)])
        # self.source_numerals = sources
        # self.target_numerals = targets
     
    # def __len__(self):
    #     return len(self.source_numerals)
    #
    # def __getitem__(self, idx):
    #     return self.source_numerals[idx], self.target_numerals[idx]
    

def generate_dataset(max_num=2021):
    examples = []
    for num in range(1, max_num + 1):
        src = str(num)
        trg = num2roman(num)
        
        examples.append(NumeralExample(src_numeral=src, trg_numeral=trg))
    
    return examples



