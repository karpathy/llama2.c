use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::mem;
use std::slice;

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
}

#[derive(Debug, Clone)]
struct TransformerWeights {
    token_embedding_table: Vec<f32>, // (vocab_size, dim)
    rms_att_weight: Vec<f32>,        // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<f32>,        // (layer, dim)
    wq: Vec<f32>,                    // (layer, dim, dim)
    wk: Vec<f32>,                    // (layer, dim, dim)
    wv: Vec<f32>,                    // (layer, dim, dim)
    wo: Vec<f32>,                    // (layer, dim, dim)
    w1: Vec<f32>,                    // (layer, hidden_dim, dim)
    w2: Vec<f32>,                    // (layer, dim, hidden_dim)
    w3: Vec<f32>,                    // (layer, hidden_dim, dim)
    rms_final_weight: Vec<f32>,      // (dim,)
    freq_cis_real: Vec<f32>,         // (seq_len, dim/2)
    freq_cis_imag: Vec<f32>,         // (seq_len, dim/2)
}

impl TransformerWeights {
    fn try_new(config: &Config, file: &mut File) -> Result<Self, Box<dyn Error>> {
        let mut weights = Self {
            token_embedding_table: vec![0.0; (config.vocab_size * config.dim) as usize],
            rms_att_weight: vec![0.0; (config.n_layers * config.dim) as usize],
            rms_ffn_weight: vec![0.0; (config.n_layers * config.dim) as usize],
            wq: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
            wk: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
            wv: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
            wo: vec![0.0; (config.n_layers * config.dim * config.dim) as usize],
            w1: vec![0.0; (config.n_layers * config.hidden_dim * config.dim) as usize],
            w2: vec![0.0; (config.n_layers * config.dim * config.hidden_dim) as usize],
            w3: vec![0.0; (config.n_layers * config.hidden_dim * config.dim) as usize],
            rms_final_weight: vec![0.0; config.dim as usize],
            freq_cis_real: vec![0.0; (config.seq_len * config.dim / 2) as usize],
            freq_cis_imag: vec![0.0; (config.seq_len * config.dim / 2) as usize],
        };

        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.token_embedding_table.as_mut_ptr() as *mut u8,
                weights.token_embedding_table.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.rms_att_weight.as_mut_ptr() as *mut u8,
                weights.rms_att_weight.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.wq.as_mut_ptr() as *mut u8,
                weights.wq.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.wk.as_mut_ptr() as *mut u8,
                weights.wk.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.wv.as_mut_ptr() as *mut u8,
                weights.wv.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.wo.as_mut_ptr() as *mut u8,
                weights.wo.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.rms_ffn_weight.as_mut_ptr() as *mut u8,
                weights.rms_ffn_weight.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.w1.as_mut_ptr() as *mut u8,
                weights.w1.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.w2.as_mut_ptr() as *mut u8,
                weights.w2.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.w3.as_mut_ptr() as *mut u8,
                weights.w3.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.rms_final_weight.as_mut_ptr() as *mut u8,
                weights.rms_final_weight.len() * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        let head_size = (config.dim / config.n_heads) as usize;
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.freq_cis_real.as_mut_ptr() as *mut u8,
                config.seq_len as usize * head_size / 2 * mem::size_of::<f32>(),
            )
        })
        .unwrap();
        file.read_exact(unsafe {
            slice::from_raw_parts_mut(
                weights.freq_cis_imag.as_mut_ptr() as *mut u8,
                config.seq_len as usize * head_size / 2 * mem::size_of::<f32>(),
            )
        })
        .unwrap();

        Ok(weights)
    }
}

#[derive(Debug, Clone)]
struct RunState {
    x: Vec<f32>,           // activation at current time stamp (dim,)
    xb: Vec<f32>,          // same, but inside a residual branch (dim,)
    xb2: Vec<f32>,         // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,          // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,         // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,           // query (dim,)
    k: Vec<f32>,           // key (dim,)
    v: Vec<f32>,           // value (dim,)
    att: Vec<f32>,         // buffer for scores/attention values (seq_len,)
    logits: Vec<f32>,      // output logits
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}

impl RunState {
    fn new(config: &Config) -> RunState {
        RunState {
            x: vec![0.0; config.dim as usize],
            xb: vec![0.0; config.dim as usize],
            xb2: vec![0.0; config.dim as usize],
            hb: vec![0.0; config.hidden_dim as usize],
            hb2: vec![0.0; config.hidden_dim as usize],
            q: vec![0.0; config.dim as usize],
            k: vec![0.0; config.dim as usize],
            v: vec![0.0; config.dim as usize],
            att: vec![0.0; config.seq_len as usize],
            logits: vec![0.0; config.vocab_size as usize],
            key_cache: vec![0.0; (config.n_layers * config.seq_len * config.dim) as usize],
            value_cache: vec![0.0; (config.n_layers * config.seq_len * config.dim) as usize],
        }
    }
}

fn accum(a: &mut [f32], b: &[f32]) {
    for (i, val) in a.iter_mut().zip(b.iter()) {
        *i += *val;
    }
}

fn rmsnorm(o: &mut [f32], x: &[f32], weight: &[f32]) {
    let size = o.len();
    // calculate sum of squares
    let mut ss = 0.0;
    for &val in x {
        ss += val * val;
    }
    ss /= size as f32;
    ss += 1e-5_f32;
    ss = 1.0 / ss.sqrt();

    // normalize and scale
    for j in 0..o.len() {
        o[j] = weight[j] * ss * x[j];
    }
}

fn sample(probabilities: &[f32]) -> usize {
    // sample index from probabilities, they must sum to 1
    let r: f32 = rand::random();
    let mut cdf = 0.0;
    for (i, &prob) in probabilities.iter().enumerate() {
        cdf += prob;
        if r < cdf {
            return i;
        }
    }
    probabilities.len() - 1 // in case of rounding errors
}

fn softmax(x: &mut [f32]) {
    // let size = size.unwrap_or(x.len());
    if x.len() == 1 {
        x[0] = 1.0;
        return;
    }

    // find max value (for numerical stability)
    let mut max_val = x[0];
    for &val in &x[1..] {
        if val > max_val {
            max_val = val;
        }
    }

    // e^x
    for val in x.iter_mut() {
        *val = (*val - max_val).exp();
    }

    // normalize
    let mut sum = 0.0;
    for &val in x.iter() {
        sum += val;
    }
    for val in x.iter_mut() {
        *val /= sum;
    }
}

fn matmul(xout: &mut [f32], x: &[f32], w: &[f32], n: usize, d: usize) {
    // W (d,n) @ x (n,) -> xout (d,)
    for i in 0..d {
        let mut val = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

fn transformer(
    token: usize,
    pos: usize,
    config: &Config,
    state: &mut RunState,
    weights: &TransformerWeights,
) {
    // a few convenience variables
    let dim = config.dim as usize;
    let hidden_dim = config.hidden_dim as usize;
    let head_size = (dim / config.n_heads as usize) as usize;

    // copy the token embedding into x
    let content_row = &weights.token_embedding_table[token * dim..(token + 1) * dim];
    state.x.copy_from_slice(content_row);

    // pluck out the "pos" row of freq_cis_real and freq_cis_imag
    let freq_cis_real_row = &weights.freq_cis_real[pos * head_size / 2..(pos + 1) * head_size / 2];
    let freq_cis_imag_row = &weights.freq_cis_imag[pos * head_size / 2..(pos + 1) * head_size / 2];

    // forward all the layers
    for l in 0..config.n_layers as usize {
        // attention rmsnorm
        rmsnorm(
            &mut state.xb,
            &state.x,
            &weights.rms_att_weight[l * dim..(l + 1) * dim],
        );

        // qkv matmuls for this position
        matmul(
            &mut state.q,
            &state.xb,
            &weights.wq[l * dim * dim..(l + 1) * dim * dim],
            dim,
            dim,
        );
        matmul(
            &mut state.k,
            &state.xb,
            &weights.wk[l * dim * dim..(l + 1) * dim * dim],
            dim,
            dim,
        );
        matmul(
            &mut state.v,
            &state.xb,
            &weights.wv[l * dim * dim..(l + 1) * dim * dim],
            dim,
            dim,
        );

        // apply RoPE rotation to the q and k vectors for each head
        for h in 0..config.n_heads as usize {
            // get the q and k vectors for this head
            let q = &mut state.q[h * head_size..(h + 1) * head_size];
            let k = &mut state.k[h * head_size..(h + 1) * head_size];

            // rotate q and k by the freq_cis_real and freq_cis_imag
            for i in (0..head_size).step_by(2) {
                let q0 = q[i];
                let q1 = q[i + 1];
                let k0 = k[i];
                let k1 = k[i + 1];
                let fcr = freq_cis_real_row[i / 2];
                let fci = freq_cis_imag_row[i / 2];
                q[i] = q0 * fcr - q1 * fci;
                q[i + 1] = q0 * fci + q1 * fcr;
                k[i] = k0 * fcr - k1 * fci;
                k[i + 1] = k0 * fci + k1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        let loff = l * config.seq_len as usize * dim; // kv cache layer offset for convenience
        let key_cache_row = &mut state.key_cache[loff + pos * dim..loff + (pos + 1) * dim];
        let value_cache_row = &mut state.value_cache[loff + pos * dim..loff + (pos + 1) * dim];
        key_cache_row.copy_from_slice(&state.k);
        value_cache_row.copy_from_slice(&state.v);

        // multihead attention. iterate over all heads
        for h in 0..config.n_heads as usize {
            // get the query vector for this head
            let q = &state.q[h * head_size..(h + 1) * head_size];

            // iterate over all timesteps, including the current one
            for t in 0..=pos {
                // get the key vector for this head and at this timestep
                let start = loff + t * dim + h * head_size;
                let k = &state.key_cache[start..start + head_size];

                // calculate the attention score as the dot product of q and k
                let mut score = 0.0;
                for i in 0..head_size {
                    score += q[i] * k[i];
                }
                score /= (head_size as f32).sqrt();
                // save the score to the attention buffer
                state.att[t] = score;
            }

            softmax(&mut state.att[..=pos]);

            // weighted sum of the values, store back into xb
            for i in 0..head_size {
                let mut val = 0.0;
                for t in 0..=pos {
                    // note bad locality
                    val += state.att[t] * state.value_cache[loff + t * dim + h * head_size + i];
                }
                state.xb[h * head_size + i] = val;
            }
        }

        // final matmul to get the output of the attention
        matmul(
            &mut state.xb2,
            &state.xb,
            &weights.wo[l * dim * dim..(l + 1) * dim * dim],
            dim,
            dim,
        );

        // residual connection back into x
        accum(&mut state.x, &state.xb2);

        // ffn rmsnorm
        rmsnorm(
            &mut state.xb,
            &state.x,
            &weights.rms_ffn_weight[l * dim..(l + 1) * dim],
        );

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(
            &mut state.hb,
            &state.xb,
            &weights.w1[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
            dim,
            hidden_dim,
        );
        matmul(
            &mut state.hb2,
            &state.xb,
            &weights.w3[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
            dim,
            hidden_dim,
        );

        // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        for val in &mut state.hb {
            *val *= 1.0 / (1.0 + (-*val).exp());
        }

        // elementwise multiply with w3(x)
        for (hb, hb2) in state.hb.iter_mut().zip(&state.hb2) {
            *hb *= *hb2;
        }

        // final matmul to get the output of the ffn
        matmul(
            &mut state.xb,
            &state.hb,
            &weights.w2[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
            hidden_dim,
            dim,
        );

        // residual connection
        accum(&mut state.x, &state.xb);
    }

    // final rmsnorm
    let temp_x = state.x.clone();
    rmsnorm(&mut state.x, &temp_x, &weights.rms_final_weight);

    // classifier into logits
    matmul(
        &mut state.logits,
        &state.x,
        &weights.token_embedding_table,
        dim,
        config.vocab_size as usize,
    );
}

fn argmax(v: &[f32]) -> usize {
    let mut max_i = 0;
    let mut max_p = v[0];
    for (i, &val) in v.iter().enumerate().skip(1) {
        if val > max_p {
            max_i = i;
            max_p = val;
        }
    }
    max_i
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let checkpoint = args
        .next()
        .expect("Usage: llama2-rust <checkpoint_file> [temperature]");
    let temperature = args
        .next()
        .map(|t| t.parse().expect("Invalid temperature"))
        .unwrap_or(0.9_f32);

    let mut file = File::open(&checkpoint).expect("Failed to open checkpoint file");
    let config: Config = bincode::deserialize_from(&mut file).expect("Failed to read config");
    let mut state = RunState::new(&config);
    let weights = TransformerWeights::try_new(&config, &mut file)?;
    let mut token = 1; // 1 = BOS token in Llama-2 sentencepiece
    dbg!(&config);

    for pos in 0..config.seq_len as usize {
        transformer(token, pos, &config, &mut state, &weights);

        // advance
        token = if temperature == 0.0_f32 {
            // greedy argmax sampling
            argmax(&state.logits)
        } else {
            // apply the temperature to the logits
            for q in 0..config.vocab_size as usize {
                state.logits[q] /= temperature;
            }

            // apply softmax to the logits to get the probabilities for next token
            softmax(&mut state.logits);

            // we now want to sample from this distribution to get the next token
            sample(&state.logits)
        };

        println!("{token}");
    }

    Ok(())
}
