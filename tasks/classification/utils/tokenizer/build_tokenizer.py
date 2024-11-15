from .BPE import BPETokenizer
from .word2vec import NLTKTokenizer, NLTKTokenizerPretrainedDataset

SUPPORTED_TOKENIZERS = {
    "bpe": BPETokenizer,
    "word2vec": NLTKTokenizer,
    "nltk": NLTKTokenizerPretrainedDataset
}

def build_tokenizer(args):
    if args["tokenizer_type"] == "nltk":
        return SUPPORTED_TOKENIZERS[args["tokenizer_type"]].from_dataset(args["args"]["dataset"])
    if "pretrained_path" in args["args"]:
        if args["tokenizer_type"] in SUPPORTED_TOKENIZERS:
            return SUPPORTED_TOKENIZERS[args["tokenizer_type"]].from_pretrained(args["args"]["pretrained_path"])
        else:
            raise ValueError(f"Tokenizer type {args['tokenizer_type']} not supported")
    else:
        print("Building tokenizer from scratch")
        if args["tokenizer_type"] in SUPPORTED_TOKENIZERS:
            tokenizer = SUPPORTED_TOKENIZERS[args["tokenizer_type"]](args["args"])
            tokenizer.build_vocab()
        else:
            raise ValueError(f"Tokenizer type {args['tokenizer_type']} not supported")