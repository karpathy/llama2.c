import 'dart:convert';
import 'dart:developer';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:args/args.dart';

class Config {
  // transformer dimension
  late int dim;
  // for ffn layers
  late int hidden_dim;
  // number of layers
  late int n_layers;
  // number of query heads
  late int n_heads;
  // number of key/value heads (can be < query heads because of multiquery)
  late int n_kv_heads;
  // vocabulary size, usually 256 (byte-level)
  late int vocab_size;
  // max sequence length
  late int seq_len;

  @override
  String toString() {
    return "Config(dim: $dim, hidden_dim: $hidden_dim, n_layers: $n_layers, n_heads: $n_heads, n_kv_heads: $n_kv_heads, vocab_size: $vocab_size, seq_len: $seq_len)";
  }
}

const configByteSize = 7 * 4;

//We are using 32 bit percision floats here
class TransformerWeights {
  // token embedding table
  late Float32List token_embedding_table; // (vocab_size, dim)
  // weights for rmsnorms
  late Float32List rms_att_weight; // (layer, dim) rmsnorm weights
  late Float32List rms_ffn_weight; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  late Float32List wq; // (layer, dim, n_heads * head_size)
  late Float32List wk; // (layer, dim, n_kv_heads * head_size)
  late Float32List wv; // (layer, dim, n_kv_heads * head_size)
  late Float32List wo; // (layer, n_heads * head_size, dim)
  // weights for ffn
  late Float32List w1; // (layer, hidden_dim, dim)
  late Float32List w2; // (layer, dim, hidden_dim)
  late Float32List w3; // (layer, hidden_dim, dim)
  // final rmsnorm
  late Float32List rms_final_weight; // (dim,)
  // freq_cis for RoPE relatively positional embeddings
  late Float32List freq_cis_real; // (seq_len, head_size/2)
  late Float32List freq_cis_imag; // (seq_len, head_size/2)
  // (optional) classifier weights for the logits, on the last layer
  late Float32List wcls;
}

class ProbIndex {
  double prob;
  int index;
  ProbIndex(this.prob, this.index);
}

class TokenIndex {
  String str;
  int id;
  TokenIndex(this.str, this.id);
}

class RunState {
  // current wave of activations
  late Float32List x; // activation at current time stamp (dim,)
  late Float32List xb; // same, but inside a residual branch (dim,)
  late Float32List xb2; // an additional buffer just for convenience (dim,)
  late Float32List hb; // buffer for hidden dimension in the ffn (hidden_dim,)
  late Float32List hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
  late Float32List q; // query (dim,)
  late Float32List k; // key (dim,)
  late Float32List v; // value (dim,)
  late Float32List att; // buffer for scores/attention values (n_heads, seq_len)
  late Float32List logits; // output logits
  late List<ProbIndex> probindex; // buffer used in top-p sampling
  // kv cache
  late Float32List key_cache; // (layer, seq_len, dim)
  late Float32List value_cache; // (layer, seq_len, dim)
}

initialize_run_state(RunState s, Config config) {
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (config.dim * config.n_kv_heads) ~/ config.n_heads;
  s.x = Float32List(config.dim);
  s.xb = Float32List(config.dim);
  s.xb2 = Float32List(config.dim);
  s.hb = Float32List(config.hidden_dim);
  s.hb2 = Float32List(config.hidden_dim);
  s.q = Float32List(config.dim);
  s.k = Float32List(kv_dim);
  s.v = Float32List(kv_dim);
  s.att = Float32List(config.n_heads * config.seq_len);
  s.logits = Float32List(config.vocab_size);
  s.probindex = [];
  s.key_cache = Float32List(config.n_layers * config.seq_len * kv_dim);
  s.value_cache = Float32List(config.n_layers * config.seq_len * kv_dim);
}

class Tokenizer {
  List<String> vocab;
  List<double> vocab_scores;
  Tokenizer(
    this.vocab,
    this.vocab_scores,
  );

  bpe_encode(String text, List<int> tokens, int n_tokens) {
    tokens = [];

    // First pass, combine raw tokens
    text.runes.forEach((element) {
      String decoded = utf8.decode([element]);
      if (vocab.contains(decoded)) {
        tokens.add(vocab.indexOf(decoded));
      }
    });

    // Second pass, combine bpe tokens
    while (true) {
      double best_score = -1e10;
      int best_id = -1;
      int best_index = -1;

      for (int i = 0; i < tokens.length - 1; i++) {
        String newStr = vocab[tokens[i]] + vocab[tokens[i + 1]];
        int newStrIndex = vocab.indexOf(newStr);
        if (newStrIndex != -1 && vocab_scores[newStrIndex] > best_score) {
          best_score = vocab_scores[newStrIndex];
          best_id = newStrIndex;
          best_index = i;
        }
      }

      if (best_index == -1) break;

      tokens[best_index] = best_id;
      tokens.removeAt(best_index + 1);
    }
    return tokens;
  }
}

// ----------------------------------------------------------------------------
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

int argmax(Float32List probabilities) {
  // return the index that has the highest probability
  int max_i = 0;
  double max_p = probabilities[0];
  for (int i = 1; i < probabilities.length; i++) {
    if (probabilities[i] > max_p) {
      max_i = i;
      max_p = probabilities[i];
    }
  }
  return max_i;
}

int sample(Float32List probabilities) {
  // sample index from probabilities (they must sum to 1!)
  double r = Random().nextDouble();
  double cdf = 0.0;
  for (int i = 0; i < probabilities.length; i++) {
    cdf += probabilities[i];
    if (r < cdf) return i;
  }
  return probabilities.length - 1; // in case of rounding errors
}

int sample_topp(Float32List probabilities, double topp) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".

  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // In the original llama.c they crop these out as candidates before sorting
  List<ProbIndex> probindex = [];

  double cutoff = (1.0 - topp) / (probabilities.length - 1);

  for (int i = 0; i < probabilities.length; i++) {
    if (probabilities[i] >= cutoff) {
      probindex.add(ProbIndex(probabilities[i], i));
    }
  }

  probindex.sort((a, b) => b.prob.compareTo(a.prob));

  // truncate the list where cumulative probability exceeds topp
  double cumulative_prob = 0.0;
  int last_idx =
      probindex.length - 1; // in case of rounding errors consider all elements
  for (int i = 0; i < probindex.length; i++) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  probindex.removeRange(last_idx + 1, probindex.length);

  // sample from the truncated list
  double r = new Random().nextDouble() * cumulative_prob;
  double cdf = 0.0;
  for (int i = 0; i <= last_idx; i++) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

rmsnorm(Float32List out, Float32List x, Float32List weight) {
  assert(out.length == x.length);
  assert(x.length == weight.length);
  // calculate sum of squares
  double ss = 0.0;
  x.forEach((element) {
    ss += element * element;
  });
  ss /= x.length;
  ss += 1e-5;
  ss = 1.0 / sqrt(ss); // sqr mean sum of squares

  // normalize and scale
  for (int j = 0; j < x.length; j++) {
    out[j] = weight[j] * (ss * x[j]);
  }
}

void softmax(Float32List x, int size) {
  // find max value (for numerical stability)
  double max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  double sum = 0.0;
  for (int i = 0; i < size; i++) {
    x[i] = exp(x[i] - max_val);
    sum += x[i];
  }
  // normalize
  for (int i = 0; i < size; i++) x[i] /= sum;
}

void matmul(Float32List out, Float32List x, Float32List w, int n, int d) {
  assert(out.length == d);
  assert(x.length == n);
  assert(w.length == n * d);

  // W (d,n) @ x (n,) -> xout (d,)
  // by far the most amount of time is spent inside this little function
  for (int i = 0; i < d; i++) {
    double val = 0.0;
    for (int j = 0; j < n; j++) {
      val += w[i * n + j] * x[j];
    }
    out[i] = val;
  }
}

transformer(int token, int pos, Config config, RunState state,
    TransformerWeights weights) {
  int dim = config.dim;
  int kv_dim = config.dim * config.n_kv_heads ~/ config.n_heads;
  int kv_mul = config.n_kv_heads ~/
      config.n_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim = config.hidden_dim;
  int head_size = config.dim ~/ config.n_heads;

  // copy the token embedding into x
  Float32List current_row = Float32List.sublistView(
      weights.token_embedding_table,
      token * config.dim,
      (token + 1) * config.dim);
  for (int i = 0; i < config.dim; i++) state.x[i] = current_row[i];

  // Note:  Divide by 2 here because Rope Parameters repeat after every 2 dimensions
  Float32List freq_cis_real_row = weights.freq_cis_real
      .sublist(pos * head_size ~/ 2, (pos + 1) * head_size ~/ 2);
  Float32List freq_cis_imag_row = weights.freq_cis_imag
      .sublist(pos * head_size ~/ 2, (pos + 1) * head_size ~/ 2);

  // forward all the layers
  for (int l = 0; l < config.n_layers; l++) {
    rmsnorm(
        state.xb,
        state.x,
        Float32List.sublistView(
            weights.rms_att_weight, l * dim, (l + 1) * dim));

    // qkv matmuls for this position
    // NOTE:yiming This look slike a place for lots of paralle work :thinking:
    // x = x @ wq, wq with dim * dim
    matmul(
        state.q,
        state.xb,
        Float32List.sublistView(weights.wq, l * dim * dim, (l + 1) * dim * dim),
        dim,
        dim);

    // x = x @ wk, wq with dim * kv_dim
    matmul(
        state.k,
        state.xb,
        Float32List.sublistView(
            weights.wk, l * dim * kv_dim, (l + 1) * dim * kv_dim),
        dim,
        kv_dim);

    // x = x @ wv, wq with dim * kv_dim
    matmul(
        state.v,
        state.xb,
        Float32List.sublistView(
            weights.wv, l * dim * kv_dim, (l + 1) * dim * kv_dim),
        dim,
        kv_dim);

    // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
    // https://arxiv.org/pdf/2104.09864v4.pdf
    // We are just reusing the loop for k and q distance calculation
    for (int v = 0; v < 2; v++) {
      Float32List vec =
          v == 0 ? state.q : state.k; // the vector to rotate (query or key)
      int vec_size = v == 0 ? dim : kv_dim; // the size of the vector

      // We are only rotating in a group of 2
      for (int i = 0; i < vec_size; i += 2) {
        double v0 = vec[i];
        double v1 = vec[i + 1];
        double fcr = freq_cis_real_row[(i % head_size) ~/ 2];
        double fci = freq_cis_imag_row[(i % head_size) ~/ 2];
        // See the RoPE paper for this section
        // 3.4.2 Computational efficient realization of rotary matrix multiplication
        // x1 = x1 + cos mθ_1 - x2 sin mθ_1
        vec[i] = v0 * fcr - v1 * fci;
        // x2 = x1 sin mθ_1 + x2 + cos mθ_1
        vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }

    // save key,value at this time step (pos) to our kv cache
    // offset by n_layer * seq_len * kv_dim
    int loff =
        l * config.seq_len * kv_dim; // kv cache layer offset for convenience
    // key cache = loff + pos * kv_dim
    int key_cache_row_offset = loff + pos * kv_dim;
    // save k,v into kv cache
    for (int i = 0; i < state.k.length; i++)
      state.key_cache[key_cache_row_offset + i] = state.k[i];

    for (int i = 0; i < state.v.length; i++)
      state.value_cache[key_cache_row_offset + i] = state.v[i];

    // multihead attention. iterate over all heads
    for (int h = 0; h < config.n_heads; h++) {
      // get the query vector for this head
      Float32List q =
          Float32List.sublistView(state.q, h * head_size, (h + 1) * head_size);
      // attention scores for this head
      Float32List att = Float32List.sublistView(
          state.att, h * config.seq_len, (h + 1) * config.seq_len);
      // iterate over all timesteps, including the current one
      for (int t = 0; t <= pos; t++) {
        // get the key vector for this head and at this timestep
        // kv_mul is just 1 now
        int key_cache_offset = loff +
            t * kv_dim +
            (h ~/ kv_mul) *
                head_size; // it's still offset by head size kv_dim = head_size * h!
        // but sometimes multiple head can share a key_cache
        Float32List k = Float32List.sublistView(
            state.key_cache, key_cache_offset, key_cache_offset + kv_dim);
        // calculate the attention score as the dot product of q and k
        double score = 0.0;
        for (int ll = 0; ll < head_size; ll++) {
          score += q[ll] * k[ll];
        }
        // TODO(yiming): reread the paper to understand better
        score /= sqrt(head_size);
        // save the score to the attention buffer
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      // soft max happens before attention * v
      // softmax is done on the entire attention
      // I think there's some trick in pytorch for this
      softmax(att, pos + 1);

      // Now we have calculated the weighted attention vector, it's time to apply attention value
      // weighted sum of the values, store back into xb
      // Clear out xb for the next stage
      for (int i = 0; i < head_size; i++) {
        state.xb[h * head_size + i] = 0.0;
      }

      Float32List xb_off =
          Float32List.sublistView(state.xb, h * head_size, (h + 1) * head_size);
      for (int t = 0; t <= pos; t++) {
        // get the value vector for this head and at this timestep
        int v_cache_offset = loff + t * kv_dim + (h ~/ kv_mul) * head_size;
        Float32List v = Float32List.sublistView(
            state.value_cache, v_cache_offset, v_cache_offset + head_size);
        // get the attention weight for this timestep
        double a = att[t];
        // accumulate the weighted value into xb
        for (int i = 0; i < head_size; i++) {
          xb_off[i] += a * v[i];
        }
      }
    }

    // final matmul to get the output of the attention
    // The "Aggregate output" of all the attention heads
    matmul(
        state.xb2,
        state.xb,
        Float32List.sublistView(weights.wo, l * dim * dim, (l + 1) * dim * dim),
        dim,
        dim);

    // residual connection back into x
    for (int i = 0; i < dim; i++) {
      state.x[i] += state.xb2[i];
    }

    // ffn rmsnorm
    rmsnorm(
        state.xb,
        state.x,
        Float32List.sublistView(
            weights.rms_ffn_weight, l * dim, (l + 1) * dim));

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    matmul(
        state.hb,
        state.xb,
        Float32List.sublistView(
            weights.w1, (l * dim * hidden_dim), (l + 1) * dim * hidden_dim),
        dim,
        hidden_dim);

    matmul(
        state.hb2,
        state.xb,
        Float32List.sublistView(
            weights.w3, (l * dim * hidden_dim), (l + 1) * dim * hidden_dim),
        dim,
        hidden_dim);

    // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
    for (int i = 0; i < hidden_dim; i++) {
      state.hb[i] = state.hb[i] * (1.0 / (1.0 + exp(-state.hb[i])));
    }

    // elementwise multiply with w3(x)
    // F.silu(self.w1(x)) * self.w3(x)
    for (int i = 0; i < hidden_dim; i++) {
      state.hb[i] = state.hb[i] * state.hb2[i];
    }

    // final matmul to get the output of the ffn
    // here we are reusing xb again!
    // x = self.w2(F.silu(self.w1(x)) * self.w3(x))
    matmul(
        state.xb,
        state.hb,
        Float32List.sublistView(
            weights.w2, l * dim * hidden_dim, (l + 1) * dim * hidden_dim),
        hidden_dim,
        dim);

    // residual connection
    for (int i = 0; i < dim; i++) {
      state.x[i] += state.xb[i];
    }
  }

  // final rmsnorm
  rmsnorm(state.x, state.x, weights.rms_final_weight);

  // classifier into logits
  matmul(state.logits, state.x, weights.wcls, config.dim, config.vocab_size);
}

void main(List<String> args) {
  String? checkpoint_path = "./stories15M.bin";
  String tokenizer_path = "tokenizer.bin";
  double temperature = 1.0;
  double top_p = 0.9;
  int rng_seed = 0; // seed rng with time by default
  int steps = 256; // number of steps to run for
  String? prompt = " One";

  var parser = ArgParser();
  parser.addOption(
    'checkpoint_path',
    abbr: 'c',
    callback: (value) => checkpoint_path = value,
  );
  parser.addOption('temp',
      abbr: 't',
      callback: (value) =>
          {if (value != null) temperature = double.parse(value)},
      defaultsTo: "1.0");
  parser.addOption('topp',
      abbr: 'p',
      callback: (value) => {if (value != null) top_p = double.parse(value)},
      defaultsTo: "0.9");
  parser.addOption('seed',
      abbr: 's',
      callback: (value) => {if (value != null) rng_seed = int.parse(value)},
      defaultsTo: "0");
  parser.addOption('steps',
      abbr: 'n',
      callback: (value) => {if (value != null) steps = int.parse(value)},
      defaultsTo: "256");
  parser.addOption('prompt',
      abbr: 'i',
      callback: (value) => {if (value != null) prompt = value},
      defaultsTo: "");
  parser.addOption('tokenizer_path',
      abbr: 'z',
      callback: (value) => {if (value != null) tokenizer_path = value});

  parser.parse(args);

  if (rng_seed == 0) rng_seed = Timeline.now;

  print("===========llama2.dart===========");
  print("check_point_path: $checkpoint_path");
  print("tokenizer_path: $tokenizer_path");
  print("temperature: $temperature");
  print("top_p: $top_p");
  print("rng_seed: $rng_seed");
  print("steps: $steps");
  print("prompt: $prompt");

  var config = Config();
  var weights = TransformerWeights();

  if (checkpoint_path == null) return print("No checkpoint path provided");

  print("========= Reading Weights =========");

  // Read Weights and Config from file
  {
    Uint8List checkpoint_bytes = File(checkpoint_path!).readAsBytesSync();
    print("Read ${checkpoint_bytes.length} bytes from $checkpoint_path");

    {
      // Reading Config
      Uint8List config_bytes = checkpoint_bytes.sublist(0, configByteSize);
      Int32List config_ints = config_bytes.buffer.asInt32List();
      config.dim = config_ints[0];
      config.hidden_dim = config_ints[1];
      config.n_layers = config_ints[2];
      config.n_heads = config_ints[3];
      config.n_kv_heads = config_ints[4];
      config.vocab_size = config_ints[5];
      config.seq_len = config_ints[6];
      print("Read Config: $config");
    }

    {
      bool shared_weights = config.vocab_size > 0;
      // negative vocab size is hacky way of signaling unshared weights. bit yikes.
      config.vocab_size = config.vocab_size.abs();
      // Load the weights
      int offset = 0;
      Float32List weight_floats =
          checkpoint_bytes.buffer.asFloat32List(configByteSize);

      int head_size = config.dim ~/ config.n_heads;
      weights.token_embedding_table = weight_floats.sublist(
          offset, offset + config.vocab_size * config.dim);
      offset += config.vocab_size * config.dim;
      print(
          "Read ${weights.token_embedding_table.lengthInBytes} bytes into token_embedding_table");

      weights.rms_att_weight =
          weight_floats.sublist(offset, offset + config.n_layers * config.dim);
      offset += config.n_layers * config.dim;
      print(
          "Read ${weights.rms_att_weight.lengthInBytes} bytes into rms_att_weight");

      weights.wq = weight_floats.sublist(offset,
          offset + config.n_layers * config.dim * config.n_heads * head_size);
      offset += config.n_layers * config.dim * config.n_heads * head_size;
      print("Read ${weights.wq.lengthInBytes} bytes into wq");

      weights.wk = weight_floats.sublist(
          offset,
          offset +
              config.n_layers * config.dim * config.n_kv_heads * head_size);
      offset += config.n_layers * config.dim * config.n_kv_heads * head_size;
      print("Read ${weights.wk.lengthInBytes} bytes into wk");

      weights.wv = weight_floats.sublist(
          offset,
          offset +
              config.n_layers * config.dim * config.n_kv_heads * head_size);
      offset += config.n_layers * config.dim * config.n_kv_heads * head_size;
      print("Read ${weights.wv.lengthInBytes} bytes into wv");

      weights.wo = weight_floats.sublist(offset,
          offset + config.n_layers * config.n_heads * head_size * config.dim);
      offset += config.n_layers * config.n_heads * head_size * config.dim;
      print("Read ${weights.wo.lengthInBytes} bytes into wo");

      weights.rms_ffn_weight =
          weight_floats.sublist(offset, offset + config.n_layers * config.dim);
      offset += config.n_layers * config.dim;
      print(
          "Read ${weights.rms_ffn_weight.lengthInBytes} bytes into rms_ffn_weight");

      weights.w1 = weight_floats.sublist(
          offset, offset + config.n_layers * config.hidden_dim * config.dim);
      offset += config.n_layers * config.hidden_dim * config.dim;
      print("Read ${weights.w1.lengthInBytes} bytes into w1");

      weights.w2 = weight_floats.sublist(
          offset, offset + config.n_layers * config.dim * config.hidden_dim);
      offset += config.n_layers * config.dim * config.hidden_dim;
      print("Read ${weights.w2.lengthInBytes} bytes into w2");

      weights.w3 = weight_floats.sublist(
          offset, offset + config.n_layers * config.hidden_dim * config.dim);
      offset += config.n_layers * config.hidden_dim * config.dim;
      print("Read ${weights.w3.lengthInBytes} bytes into w3");

      weights.rms_final_weight =
          weight_floats.sublist(offset, offset + config.dim);
      offset += config.dim;
      print(
          "Read ${weights.rms_final_weight.lengthInBytes} bytes into rms_final_weight");

      weights.freq_cis_real = weight_floats.sublist(
          offset, offset + config.seq_len * head_size ~/ 2);
      offset += config.seq_len * head_size ~/ 2;
      print(
          "Read ${weights.freq_cis_real.lengthInBytes} bytes into freq_cis_real");

      weights.freq_cis_imag = weight_floats.sublist(
          offset, offset + config.seq_len * head_size ~/ 2);
      offset += config.seq_len * head_size ~/ 2;
      print(
          "Read ${weights.freq_cis_imag.lengthInBytes} bytes into freq_cis_imag");

      if (shared_weights) {
        print("Read shared weights into wcls");
        weights.wcls = weights.token_embedding_table;
      } else {
        weights.wcls = weight_floats.sublist(
            offset, offset + config.vocab_size * config.dim);
        offset += config.dim;
        print("Read ${weights.wcls.lengthInBytes} bytes into wcls");
      }
    }
  }

  // clamp number of steps to supported range
  if (steps <= 0 || steps > config.seq_len) {
    steps = config.seq_len;
  }

  // read in the tokenizer .bin file
  List<Uint8List> vocab = new List.filled(
      config.vocab_size, new Uint8List(0)); // config.vocab_size;
  Float32List vocab_scores = new Float32List(config.vocab_size);
  {
    ByteData tokenizer_bytes =
        File(tokenizer_path).readAsBytesSync().buffer.asByteData(0);
    int offset = 0;
    // Not being used but read anyways
    int max_token_length = tokenizer_bytes.getUint32(offset, Endian.little);
    offset += 4;
    int next_str_length = 0;
    for (int i = 0; i < config.vocab_size; i++) {
      double score = tokenizer_bytes.getFloat32(offset, Endian.little);
      offset += 4;
      next_str_length = tokenizer_bytes.getUint32(offset, Endian.little);
      offset += 4;
      Uint8List next_chunk =
          tokenizer_bytes.buffer.asUint8List(offset, next_str_length);
      vocab_scores[i] = score;
      offset += next_str_length;
      vocab[i] = next_chunk;
    }
  }

  print("=====beginning generation=====");

  Tokenizer tokenizer;
  tokenizer =
      Tokenizer(vocab.map((e) => utf8.decode(e)).toList(), vocab_scores);

  // process the prompt, if any
  List<int> prompt_tokens = [];
  int num_prompt_tokens = 0;
  if (prompt != null) {
    prompt_tokens =
        tokenizer.bpe_encode(prompt!, prompt_tokens, num_prompt_tokens);
  }

  RunState state = RunState();

  initialize_run_state(state, config);
  // Finally!  the main loop
  // used to time our code, only initialized after first iteration
  int start = 0;
  int next; // will store the next token in the sequence
  // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
  int token = 1;
  int pos = 0; // position in the sequence

  while (pos < steps) {
    // transformer!  Run the model
    transformer(token, pos, config, state, weights);

    // advance the state state machine
    if (pos < prompt_tokens.length) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos];
    } else {
      // sample the next token
      if (temperature == 0.0) {
        // greedy argmax sampling: take the token with the highest probability
        next = argmax(state.logits);
      } else {
        // apply the temperature to the logits
        for (int q = 0; q < config.vocab_size; q++) {
          state.logits[q] /= temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(state.logits, state.logits.length);

        // we sample from this distribution to get the next token
        if (top_p <= 0 || top_p >= 1) {
          // simply sample from the predicted probability distribution
          next = sample(state.logits);
        } else {
          // top-p (nucleus) sampling, clamping the least likely tokens to zero
          next = sample_topp(state.logits, top_p);
        }
      }
    }
    pos++;

    // data-dependent terminating condition: the BOS (1) token delimits sequences
    if (next == 1) {
      break;
    }

    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    Uint8List token_str =
        (token == 1 && (vocab[next][0] == ' ')) ? vocab[next + 1] : vocab[next];

    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    String str;
    str = utf8.decode(token_str);

    // In the original llama2.c they check for a lot of special tokens, but I've only seen this token really being used
    // Being a little lazy here Hehe.
    if (str == "<0x0A>") {
      str = "\n";
    }
    stdout.write("$str");
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = DateTime.now().millisecondsSinceEpoch;
    }
  }
  stdout.write("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    int end = DateTime.now().millisecondsSinceEpoch;
    print("achieved tok/s: ${(pos - 1) / (end - start) * 1000} \n");
  }
}
