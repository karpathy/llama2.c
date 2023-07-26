use std::mem;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
};

use rand::Rng;

const CONF_VALS: usize = 7;
const CONF_SIZE: usize = std::mem::size_of::<[i32; CONF_VALS]>();

#[derive(Debug, Clone, Copy)]
struct Config {
    dim: usize,
    hidden_dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    vocab_size: usize,
    seq_len: usize,
}

impl Config {
    /// Read raw bytes and force those to be our config type (which conforms to C mem layout)
    fn from_file(path: &str) -> Self {
        let mut model_bin = File::open(path).unwrap();
        let mut buffer = [0; CONF_SIZE];
        model_bin.read_exact(&mut buffer).unwrap();
        let raw_conf = unsafe { mem::transmute::<[u8; CONF_SIZE], [i32; CONF_VALS]>(buffer) };
        Self {
            dim: raw_conf[0] as usize,
            hidden_dim: raw_conf[1] as usize,
            n_layers: raw_conf[2] as usize,
            n_heads: raw_conf[3] as usize,
            n_kv_heads: raw_conf[4] as usize,
            vocab_size: raw_conf[5] as usize,
            seq_len: raw_conf[6] as usize,
        }
    }
}

struct Vocab {
    bytes: Vec<u8>,
    offsets: Vec<usize>,
}

impl Vocab {
    fn from_file(vocab_size: usize, path: &str) -> Self {
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
        let mut f = |s: usize| _alloc_and_read(&mut model_bin, s);
        let head_size = C.dim / C.n_heads;
        Self {
            token_embedding_table: f(C.vocab_size * C.dim),
            rms_att_weight: f(C.n_layers * C.dim),
            wq: f(C.n_layers * C.dim * C.dim),
            wk: f(C.n_layers * C.dim * C.dim),
            wv: f(C.n_layers * C.dim * C.dim),
            wo: f(C.n_layers * C.dim * C.dim),
            rms_ffn_weight: f(C.n_layers * C.dim),
            w1: f(C.n_layers * C.dim * C.hidden_dim),
            w2: f(C.n_layers * C.dim * C.hidden_dim),
            w3: f(C.n_layers * C.dim * C.hidden_dim),
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

fn rmsnorm(out: &mut [Ty], x: &[Ty], w: &[Ty]) {
    let ss = x.iter().fold(0f32, |init, &v| init + v * v) / (x.len() as Ty);
    let ss = (1 as Ty) / (ss + 1e-5).sqrt();
    let normed = x.iter().zip(w.iter()).map(|(xx, ww)| xx * ww * ss);
    out.iter_mut().zip(normed).for_each(|(dst, src)| *dst = src);
}

/// For now this is a matvec
/// Wx: [n, d]x[d,] -> [n,]
fn matmul(out: &mut [Ty], x: &[Ty], w: &[Ty], in_dim: usize) {
    for (row, out_elem) in w.chunks_exact(in_dim).zip(out.iter_mut()) {
        let val = row
            .iter()
            .zip(x.iter())
            .fold(0 as Ty, |acc, (&_w, &_x)| acc + _w * _x);
        *out_elem = val;
    }
}

fn _uncheked_mut_slice(s: &mut [Ty], offset: usize, size: usize) -> &mut [Ty] {
    let ptr = s.as_mut_ptr();
    unsafe {
        let st = ptr.offset(offset as isize);
        std::slice::from_raw_parts_mut(st, size)
    }
}
fn _uncheked_slice(s: &[Ty], offset: usize, size: usize) -> &[Ty] {
    let ptr = s.as_ptr();
    unsafe {
        let st = ptr.offset(offset as isize);
        std::slice::from_raw_parts(st, size)
    }
}

fn inplace_softmax(x: &mut [Ty]) {
    let max_val = x.iter().fold(Ty::NAN, |acc, &v| v.max(acc));
    let mut denom = 0 as Ty;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        denom += *v;
    }

    x.iter_mut().for_each(|v| *v /= denom);
}

fn cdf_sample(probs: &[Ty]) -> usize {
    let mut rng = rand::thread_rng();

    let r = rng.gen::<Ty>();
    let mut cdf = 0 as Ty;
    for (idx, p) in probs.iter().enumerate() {
        cdf += *p;
        if r < cdf {
            return idx;
        }
    }
    probs.len() - 1
}

impl RunState {
    fn init(C: &Config) -> Self {
        let f = |size: usize| vec![0 as Ty; size];
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

    fn qkv_for_layer(&mut self, l: usize, w: &TransformerWeights, dim: usize) {
        let wq = _uncheked_slice(&w.wq, l * dim * dim, dim * dim);
        let wk = _uncheked_slice(&w.wk, l * dim * dim, dim * dim);
        let wv = _uncheked_slice(&w.wv, l * dim * dim, dim * dim);

        matmul(&mut self.q, &self.xb, wq, dim);
        matmul(&mut self.k, &self.xb, wk, dim);
        matmul(&mut self.v, &self.xb, wv, dim);
    }

    fn cache_kv(&mut self, pos: usize, layer: usize, C: &Config) {
        let offset = layer * C.dim * C.seq_len + pos * C.dim;
        let kc = _uncheked_mut_slice(&mut self.key_cache, offset, C.dim).as_mut();
        let vc = _uncheked_mut_slice(&mut self.value_cache, offset, C.dim).as_mut();
        // TODO: make theese unsafe and remove len checks
        kc.copy_from_slice(&self.k);
        vc.copy_from_slice(&self.v);
    }

    fn rope(&mut self, pos: usize, w: &TransformerWeights, n_heads: usize, dim: usize) {
        let head_size = dim / n_heads;
        let qk_heads = self
            .q
            .chunks_exact_mut(n_heads)
            .zip(self.k.chunks_exact_mut(n_heads));

        for (q, k) in qk_heads {
            let mut cis_real = w.freq_cis_real[pos * head_size / 2..]
                .iter()
                .take(head_size / 2);
            let mut cis_imag = w.freq_cis_imag[pos * head_size / 2..]
                .iter()
                .take(head_size / 2);

            for (qq, kk) in q.chunks_exact_mut(2).zip(k.chunks_exact_mut(2)) {
                let (q0, q1) = (qq[0], qq[1]);
                let (k0, k1) = (kk[0], kk[1]);
                let fcr = cis_real.next().unwrap();
                let fci = cis_imag.next().unwrap();
                qq[0] = q0 * fcr - q1 * fci;
                qq[1] = q0 * fci + q1 * fcr;
                kk[0] = k0 * fcr - k1 * fci;
                kk[1] = k0 * fci + k1 * fcr;
            }
        }
    }

    fn attention(&mut self, pos: usize, layer: usize, C: &Config) {
        assert!(
            pos < C.seq_len,
            "Can't attend outside of initialized seq lenght"
        );

        let head_size = C.dim / C.n_heads;

        // (seq_len, dim)
        let seq_cached_keys = self
            .key_cache
            .chunks_exact(C.seq_len * C.dim)
            .skip(layer)
            .take(1)
            .next()
            .unwrap();

        let mut q_heads = self.q.chunks_exact(head_size);
        let mut xb_heads = self.xb.chunks_exact_mut(head_size);
        for h in 0..C.n_heads {
            let q = q_heads.next().unwrap();

            let mut head_k_all_pos = seq_cached_keys
                .chunks_exact(head_size)
                .skip(h)
                .step_by(C.n_heads);

            for t in 0..=pos {
                let k = head_k_all_pos.next().unwrap();
                let score = k
                    .iter()
                    .zip(q.iter())
                    .fold(0 as Ty, |acc, (_k, _q)| acc + _k * _q);
                let score = score / (head_size as Ty).sqrt();
                unsafe {
                    *self.att.get_unchecked_mut(t) = score;
                }
            }

            let mut seq_cached_vals =
                _uncheked_slice(&self.value_cache, layer * C.dim * C.seq_len, C.seq_len * C.dim)
                    .chunks_exact(head_size)
                    .skip(h)
                    .step_by(C.n_heads);
            inplace_softmax(&mut self.att[..=pos]);
            // cahced vals have head_size values in it. we need to go over all t vals and update xb
            // dst is head_size part of xb, we gonna add t (actually pos values) values into it
            let dst = xb_heads.next().unwrap();
            // clear dst
            dst.iter_mut().for_each(|v| *v = 0f32);
            // this is different from Karphaty's impl. first go over all head size values, than skip time stamp
            // Why Karphaty's inner loop does C.dim jumps?
            // this goes over time stamps
            for (vals, attn_w) in seq_cached_vals.zip(self.att.iter()).take(pos + 1) {
                
                // aggregate timestamp to xb
                for (val, dst) in vals.iter().zip(dst.iter_mut()) {
                    *dst += val * attn_w;
                }
            }
        }
        
    }



    fn ffn(&mut self, l: usize, w: &TransformerWeights, C: &Config) {
        let rms_ffn_w = _uncheked_slice(&w.rms_ffn_weight, l * C.dim, C.dim);
        // normalize after adding residual
        rmsnorm(&mut self.xb, &self.x, rms_ffn_w);
        
        let w1 = _uncheked_slice(&w.w1, C.dim * C.hidden_dim * l, C.hidden_dim * C.dim);
        let w2 = _uncheked_slice(&w.w2, C.dim * C.hidden_dim * l, C.hidden_dim * C.dim);
        let w3 = _uncheked_slice(&w.w3, C.dim * C.hidden_dim * l, C.hidden_dim * C.dim);
        matmul(&mut self.hb, &self.xb, w1, C.dim);
        matmul(&mut self.hb2, &self.xb, w3, C.dim);
        // silu on first hidden
        self.hb
            .iter_mut()
            .for_each(|v| *v = (*v)*(1 as Ty / (1 as Ty + (-*v).exp())));

        // mix branches
        self.hb
            .iter_mut()
            .zip(self.hb2.iter())
            .for_each(|(h1, &h2)| *h1 = (*h1) * h2);

        matmul(&mut self.xb, &self.hb, w2, C.hidden_dim);
    }

    fn step(&mut self, token: usize, pos: usize, w: &TransformerWeights, C: &Config) {
        // copy content row
        // TODO: mayne direct indexing w/o bound checks is faster? benchmark
        w.token_embedding_table
            .chunks_exact(C.dim)
            .skip(token)
            .take(1)
            .for_each(|src| self.x.as_mut_slice().copy_from_slice(src));


        for l in 0..C.n_layers {
            let rms_attn_w = _uncheked_slice(&w.rms_att_weight, l * C.dim, C.dim);
            rmsnorm(&mut self.xb, &self.x, rms_attn_w);
            
            self.qkv_for_layer(l, w, C.dim);
            
            self.rope(pos, w, C.n_heads, C.dim);
            self.cache_kv(pos, l, C);
            self.attention(pos, l, C);
            // self.aggregate_attention(l, pos, C);

            let wo = _uncheked_slice(&w.wo, l * C.dim * C.dim, C.dim * C.dim);
            matmul(&mut self.xb2, &self.xb, wo, C.dim);
            // post attention residual
            self.x
                .iter_mut()
                .zip(self.xb2.iter())
                .for_each(|(dst, src)| *dst += *src);

            self.ffn(l, w, C);
            // post ffn residual
            self.x
                .iter_mut()
                .zip(self.xb.iter())
                .for_each(|(dst, src)| *dst += src);
        }

        // last rmsnorm
        let ss = self.x.iter().fold(0f32, |init, &v| init + v * v) / (self.x.len() as Ty);
        let ss = (1 as Ty) / (ss + 1e-5).sqrt();
        self.x
            .iter_mut()
            .zip(w.rms_final_weight.iter())
            .for_each(|(xx, ww)| (*xx) *= ww * ss);

        matmul(&mut self.logits, &self.x, &w.token_embedding_table, C.dim);
    }
}

fn main() {
    use std::time::Instant;
    let model_path = "../out/model.bin";
    let tokenizer_path = "../tokenizer.bin";
    let temperature = 0.9 as Ty;

    let config = Config::from_file(model_path);
    let vocab = Vocab::from_file(config.vocab_size, tokenizer_path);
    let weights = TransformerWeights::read_from_file(&config, model_path);
    let mut state = RunState::init(&config);
    let mut probs = vec![0 as Ty; config.vocab_size];

    let st = Instant::now();
    let mut pos = 0;
    let mut token = 1;
    while pos < config.seq_len {
        state.step(token, pos, &weights, &config);
        let next = if temperature == 0 as Ty {
            state
                .logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(index, _)| index)
                .unwrap()
        } else {
            state
                .logits
                .iter().zip(probs.iter_mut())
                .for_each(|(logit, p)| *p = logit / temperature);
            inplace_softmax(&mut probs);
            cdf_sample(&probs)
        };
        print!("{}", vocab.get_token(next));
        pos += 1;
        token = next;
    }
    println!("");
    let ts = pos as f32 / st.elapsed().as_secs_f32();
    println!("{:.3} Tokens/Sec", ts);
}
