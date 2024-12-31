import collections
import tqdm
import os
from pathlib import Path

head_directory = Path(__file__).resolve().parent.parent
# print(head_directory)
os.chdir(head_directory)

class Vocab(object):
    """
    Special tokens predefined in the vocab file are:
    -[PAD]
    -[UNK]
    -[MASK]
    -[CLS]
    -[SEP]
    """
    
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file
        self.vocab = collections.OrderedDict()
    
    def load_vocab(self):
        """Loads a vocabulary file into a dictionary"""
        if not self.vocab:
            with open(self.vocab_file, "r") as reader:
                for index, line in tqdm.tqdm(enumerate(reader.readlines())):
                    token = line.strip()
                    self.vocab[token] = index
            self.invocab = {index: token for token, index in self.vocab.items()}
            
    def to_seq(self, sentence, seq_len=20):
        sentence = sentence.split()
            
        seq = [self.vocab.get(word, self.vocab['[UNK]']) for word in sentence][:seq_len-2]
        seq = [self.vocab['[CLS]']]+seq+[self.vocab['[SEP]']]
        
        return seq
    
    def to_sentence(self, seq):
        words = [self.invocab[index] if index < len(self.invocab) 
                 else "[%d]" % index for index in seq ]
        
        return words #" ".join(words)
    

# if __init__ == "__main__":
#     vocab_obj = Vocab("bert/pretraining/vocab_file.txt")
#     vocab_obj.load_vocab()
#     seq = vocab_obj.to_seq("P10855	KC838	KC551	KC127	KC127	KC512	KC512	KC512	KC329	KC838	KC736	KC551	KC838
# ")) 
#     #[2, 10859, 19709, 19422, 18998, 18998, 19383, 19383, 19383, 19200, 19709, 19607, 19422, 19709, 3]
#     vocab_obj.to_sentence(seq)
#     #'[CLS] P10855 KC838 KC551 KC127 KC127 KC512 KC512 KC512 KC329 KC838 KC736 KC551 KC838 [SEP]'
    
                           