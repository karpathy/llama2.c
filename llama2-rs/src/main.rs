use std::mem;
use std::{fs::File, io::Read};

const CONF_VALS: usize = 7;
const CONF_SIZE: usize = std::mem::size_of::<[i32; CONF_VALS]>();

#[repr(C)]
#[derive(Debug)]
struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

impl Config {
    /// Read raw bytes and force those to be our config type (which conforms to C mem layout)
    fn from_file(path: &str) -> Self {
        let mut model_bin = File::open(path).unwrap();
        let mut buffer = [0; CONF_SIZE];
        model_bin.read_exact(&mut buffer).unwrap();
        let raw_conf = unsafe { mem::transmute::<[u8; CONF_SIZE], [i32; CONF_VALS]>(buffer) };
        Self {
            dim: raw_conf[0],
            hidden_dim: raw_conf[1],
            n_layers: raw_conf[2],
            n_heads: raw_conf[3],
            n_kv_heads: raw_conf[4],
            vocab_size: raw_conf[5],
            seq_len: raw_conf[6],
        }
    }
}

struct Vocab {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}

impl Vocab {
    fn from_file(vocab_size: i32, path: &str) -> Self {
        let mut bytes = Vec::<u8>::new();
        let mut offsets = vec![0usize; 1];
        let mut vocab_bin = File::open(path).unwrap();
        let mut len = [0; 4];
        let mut val = [0; 1];
        for _ in 0..vocab_size {
            vocab_bin.read_exact(&mut len).unwrap();
            let l = unsafe { mem::transmute::<[u8; 4], i32>(len) };
            offsets.push(offsets.last().unwrap() + l as usize);
            (0..l).for_each(|_| {
                vocab_bin.read_exact(&mut val).unwrap();
                bytes.extend(val);
            });
        }

        assert_eq!(offsets.len(), vocab_size as usize + 1);

        Self { bytes, offsets }
    }

    fn get_token(&self, idx:usize) -> &str {
        let (st, en)= (self.offsets[idx], self.offsets[idx+1]);
        let b = &self.bytes[st..en];
        std::str::from_utf8(b).unwrap()

    }
}

fn main() {
    let config = Config::from_file("../out/model.bin");
    let vocab = Vocab::from_file(config.vocab_size, "../tokenizer.bin");
    
    for i in 0..100 {
        println!("[{}]: {}", i, vocab.get_token(i));
    }


}
