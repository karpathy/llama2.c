from sentencepiece import SentencePieceProcessor

sp_model = SentencePieceProcessor(model_file="tokenizer.model")

vocab_size = sp_model.vocab_size()
bos_id = sp_model.bos_id()
eos_id = sp_model.eos_id()
pad_id = sp_model.pad_id()
print(vocab_size, bos_id, eos_id, pad_id)

# for i in range(vocab_size): print(i, repr(sp_model.id_to_piece(i)))

token_blob = []
offsets = []
offset = 0
for i in range(vocab_size):
    t = sp_model.id_to_piece(i)
    if i == bos_id:
        t = '\n<s>\n'
    elif i == eos_id:
        t = '\n</s>\n'
    elif len(t) == 6 and t.startswith('<0x') and t.endswith('>'):
        t = chr(int(t[3:5], 16))

    t = t.replace('▁', ' ')
    t = t.encode('utf-8')
    offsets.append(offset)
    token_blob.append(t)
    offset += len(t)

# dummy token to always be able calculate token len as
# offsets[i + 1] - offsets[i]
offsets.append(offset)

with open('tokenizer.bin', 'wb') as f:
    f.write((vocab_size + 1).to_bytes(4, 'little'))

    for offset in offsets:
        f.write(offset.to_bytes(4, 'little'))
    for token in token_blob:
        f.write(token)

print('tokenizer.bin is created')
