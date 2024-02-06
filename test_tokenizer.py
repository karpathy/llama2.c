import sentencepiece as spm

def load_sp_model(model_path):
    """Load SentencePiece model from file."""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    return sp

def encode_sentence(sp_model, sentence):
    """Encode a sentence using SentencePiece."""
    encoded_tokens = sp_model.EncodeAsPieces(sentence)
    token_ids = sp_model.EncodeAsIds(sentence)
    return encoded_tokens, token_ids

def decode_tokens(sp_model, tokens):
    """Decode tokens using SentencePiece."""
    return sp_model.DecodePieces(tokens)

def main():
    # File path to the trained SentencePiece model
    model_path = "C:\\Users\\Borjan\\Desktop\llama2.c\\data\\tok48000.model"

    # Load SentencePiece model
    sp_model = load_sp_model(model_path)

    # Input sentence to encode and decode
    input_sentence = """
    class Foo {
    public function test(...$args) {
        var_dump($args);
    }

    public static function test2(...$args) {
        var_dump($args);
    }
}
    """

    # Print the number of words in the sentence
    num_words = len(input_sentence.split())
    print("Number of words in the sentence:", num_words)

    # Encode the input sentence
    encoded_tokens, token_ids = encode_sentence(sp_model, input_sentence)

    # Separate arrays for tokens and token IDs
    tokens_array = []
    token_ids_array = []
    for token, token_id in zip(encoded_tokens, token_ids):
        tokens_array.append(token)
        token_ids_array.append(token_id)

    # Print the arrays separately
    print("Tokens:", tokens_array)
    print("Token IDs:", token_ids_array)

    # Print the number of tokens
    num_tokens = len(encoded_tokens)
    print("Number of tokens in the sentence:", num_tokens)

    # Decode the encoded tokens
    decoded_sentence = decode_tokens(sp_model, encoded_tokens)
    print("Decoded sentence:", decoded_sentence)

if __name__ == "__main__":
    main()
