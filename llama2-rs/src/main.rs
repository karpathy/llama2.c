use std::mem;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

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

    fn get_token(&self, idx: usize) -> &str {
        let (st, en) = (self.offsets[idx], self.offsets[idx + 1]);
        let b = &self.bytes[st..en];
        std::str::from_utf8(b).unwrap()
    }
}

// generic placeholder
type Ty = f32;

struct TransformerWeights {
    /// (vocab_size, dim)
    token_embedding_table: Vec<Ty>,
    /// (layer, dim) rmsnorm weights
    rms_att_weight: Vec<Ty>,
    /// (layer, dim)
    rms_ffn_weight: Vec<Ty>,
    // weights for matmuls
    /// (layer, dim, dim)
    wq: Vec<Ty>,
    /// (layer, dim, dim)
    wk: Vec<Ty>,
    /// (layer, dim, dim)
    wv: Vec<Ty>,
    /// (layer, dim, dim)
    wo: Vec<Ty>,
    // weights for ffn
    /// (layer, hidden_dim, dim)
    w1: Vec<Ty>,
    /// (layer, dim, hidden_dim)
    w2: Vec<Ty>,
    /// (layer, hidden_dim, dim)
    w3: Vec<Ty>,
    // final rmsnorm
    /// (dim,)
    rms_final_weight: Vec<Ty>,
    // freq_cis for RoPE relatively positional embeddings
    /// (seq_len, dim/2)
    freq_cis_real: Vec<Ty>,
    /// (seq_len, dim/2)
    freq_cis_imag: Vec<Ty>,
}

fn _alloc_and_read(file: &mut File, size: usize) -> Vec<Ty> {
    let bytes_to_read = size * std::mem::size_of::<Ty>();
    let mut raw_w_data = vec![0; bytes_to_read];
    file.read_exact(&mut raw_w_data).unwrap();
    unsafe {
        let float_ptr = raw_w_data.as_ptr() as *const Ty;
        let data = std::slice::from_raw_parts(float_ptr, size);
        data.to_vec()
    }
}

impl TransformerWeights {
    fn read_from_file(C: &Config, path: &str) -> Self {
        let mut model_bin = File::open(path).unwrap();
        model_bin.seek(SeekFrom::Start(CONF_SIZE as u64)).unwrap();
        let mut f = |s: i32| _alloc_and_read(&mut model_bin, s as usize);
        let head_size = C.dim / C.n_heads;
        Self {
            token_embedding_table: f(C.vocab_size * C.dim),
            rms_att_weight: f(C.n_layers * C.dim),
            rms_ffn_weight: f(C.n_layers * C.dim),
            wq: f(C.n_layers * C.dim * C.dim),
            wk: f(C.n_layers * C.dim * C.dim),
            wv: f(C.n_layers * C.dim * C.dim),
            wo: f(C.n_layers * C.dim * C.dim),
            w1: f(C.dim * C.hidden_dim),
            w2: f(C.dim * C.hidden_dim),
            w3: f(C.dim * C.hidden_dim),
            rms_final_weight: f(C.dim),
            freq_cis_real: f(C.seq_len * (head_size / 2)),
            freq_cis_imag: f(C.seq_len * (head_size / 2)),
        }
    }
}

struct RunState {
    /// activation at current time stamp (dim,)
    x: Vec<Ty>,
    /// same, but inside a residual branch (dim,)
    xb: Vec<Ty>,
    /// an additional buffer just for convenience (dim,)
    xb2: Vec<Ty>,
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    hb: Vec<Ty>,
    /// buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<Ty>,
    /// query (dim,)
    q: Vec<Ty>,
    /// key (dim,)
    k: Vec<Ty>,
    /// value (dim,)
    v: Vec<Ty>,
    /// buffer for scores/attention values (seq_len,)
    att: Vec<Ty>,
    /// output logits
    logits: Vec<Ty>,
    // kv cache
    /// (layer, seq_len, dim)
    key_cache: Vec<Ty>,
    /// (layer, seq_len, dim)
    value_cache: Vec<Ty>,
}

impl RunState {
    fn init(C: &Config) -> Self {
        let f = |size: i32| vec![0 as Ty; size as usize];
        Self {
            x: f(C.dim),
            xb: f(C.dim),
            xb2: f(C.dim),
            hb: f(C.hidden_dim),
            hb2: f(C.hidden_dim),
            q: f(C.dim),
            k: f(C.dim),
            v: f(C.dim),
            att: f(C.seq_len),
            logits: f(C.vocab_size),
            key_cache: f(C.n_layers * C.seq_len * C.dim),
            value_cache: f(C.n_layers * C.seq_len * C.dim),
        }
    }
}

fn main() {
    let model_path = "../out/model.bin";
    let tokenizer_path = "../tokenizer.bin";
    let temperature = 0 as Ty;

    let config = Config::from_file(model_path);
    let vocab = Vocab::from_file(config.vocab_size, tokenizer_path);
    let weights = TransformerWeights::read_from_file(&config, model_path);
    let mut state = RunState::init(&config);
}
