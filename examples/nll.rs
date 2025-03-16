use alpha_makemore_rust::file_reader::get_line_reader;
use alpha_makemore_rust::vocabulary::Vocabulary;
use alpha_makemore_rust::{TOKEN_END, TOKEN_START};
use alpha_micrograd_rust::value::Expr;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{distr::Uniform, prelude::Distribution, rng};
use std::error::Error;
use std::iter::once;
use std::time::Duration;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Arguments {
    #[arg(short, long, default_value = "names.txt")]
    filename: String,

    #[arg(short, long, default_value = "10000")]
    epochs: usize,

    #[arg(short, long, default_value = "0.1")]
    learning_rate: f64,

    #[arg(short, long, default_value = "5000")]
    max_examples: Option<usize>,
}

fn execute_with_progress<F, O>(msg: &str, f: F) -> O
where F: FnOnce() -> O {
    let pb = ProgressBar::new_spinner();
    pb.enable_steady_tick(Duration::from_millis(100));
    pb.set_message(msg.to_string());
    let result = f();

    pb.finish_with_message(format!("{} Done.", msg));
    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Arguments::parse();

    let (xs, ys, vocabulary) = execute_with_progress("Generating vocabulary...", || {
        std::thread::sleep(Duration::from_secs(1));
        generate_dataset(&args.filename)
    })?;

    let vocab_size = vocabulary.len();
    let xenc = execute_with_progress("One-hot encoding...", || {
        let mut x_1he = one_hot_encode(&xs, vocab_size);
        if let Some(max_examples) = args.max_examples {
            x_1he.truncate(max_examples);
        }
        x_1he
    });

    let w = execute_with_progress("Initializing weights...", || {
        let between = Uniform::new_inclusive(-1.0, 1.0).expect("Invalid range");
        let mut rng = rng();

        (0..vocab_size)
            .map(|_| between.sample(&mut rng))
            .enumerate()
            .map(|(i, v)| Expr::new_leaf(v, &format!("w_{}", i)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|v| vec![v])
            .collect::<Vec<_>>()
    });

    let logits = execute_with_progress("Calculating logits...",  || {
        mat_mul(&xenc, &w)
    });

    let probs = execute_with_progress("Calculating softmax...",  || {
        softmax(&logits)
    });

    let mut average_nll = execute_with_progress("Calculating Average NLL...", || {
        let mut average_nll: Expr = Expr::new_leaf(0.0, "average_nll");
        for i in 0..xenc.len() {
            let y = ys[i].clone();
            let p = &probs[i][y];
            let log_likelihood = p.clone().log(&format!("log_likelihood_{}", i));
            let nll = log_likelihood.clone().neg(&format!("loss_{}", i));
            average_nll = average_nll + nll;
        }
        average_nll = average_nll / xenc.len() as f64;
        average_nll
    });

    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(
        ProgressStyle
            ::with_template("{elapsed_precise}/{duration_precise} | {wide_bar} {percent}% | Epoch: {human_pos}/{human_len} | {msg} ")
            .expect("Invalid template")
    );
    pb.set_message("Training...");
    pb.enable_steady_tick(Duration::from_millis(100));
    for _ in 0..args.epochs {
        average_nll.learn(args.learning_rate);
        average_nll.recalculate();
        pb.inc(1);
        pb.set_message(format!("Loss (NLL): {:.4}", average_nll.result));
    }

    Ok(())
}

fn generate_dataset(
    filename: &str,
) -> Result<(Vec<usize>, Vec<usize>, Vocabulary), Box<dyn Error>> {
    let line_reader = get_line_reader(filename)?;
    let mut vocabulary: Vocabulary = Vocabulary::new();
    vocabulary.add_token(TOKEN_START);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for line in line_reader {
        let line = line?;

        let mut last_token = TOKEN_START;

        let tokens = line.chars().chain(once(TOKEN_END));
        for token in tokens {
            if !vocabulary.has_token(token) {
                vocabulary.add_token(token);
            }

            let ix1 = vocabulary
                .token_to_id(last_token)
                .expect(&format!("token {} not found", last_token));
            let ix2 = vocabulary
                .token_to_id(token)
                .expect(&format!("token {} not found", token));
            xs.push(ix1);
            ys.push(ix2);

            last_token = token;
        }
    }

    Ok((xs, ys, vocabulary))
}

fn one_hot_encode(values: &[usize], num_classes: usize) -> Vec<Vec<f64>> {
    values
        .iter()
        .map(|&ix| one_hot_encode_single(ix, num_classes))
        .collect()
}

fn one_hot_encode_single(value: usize, num_classes: usize) -> Vec<f64> {
    (0..num_classes)
        .map(|ix| if ix == value { 1.0 } else { 0.0 })
        .collect()
}

fn mat_mul(xs: &[Vec<f64>], w: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    let result: Vec<Vec<Expr>> = xs
        .iter()
        .map(|x| {
            w.iter()
                .map(|w_row| {
                    x.iter()
                        .zip(w_row.iter())
                        .map(|(x, w)| *x * w.clone())
                        .sum()
                })
                .collect()
        })
        .collect();
    result
}

fn softmax(logits: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    logits
        .iter()
        .map(|logit_row| {
            let exps = logit_row
                .iter()
                .enumerate()
                .map(|(i, logit)| logit.clone().exp(&format!("exp_logit_{}", i)))
                .collect::<Vec<_>>();

            let sum_exp = exps.iter().map(|e| e.clone()).sum::<Expr>();
            exps.into_iter().map(|exp| exp / sum_exp.clone()).collect()
        })
        .collect()
}
