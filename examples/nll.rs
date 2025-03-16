use alpha_makemore_rust::file_reader::get_line_reader;
use alpha_makemore_rust::vocabulary::Vocabulary;
use alpha_makemore_rust::{TOKEN_END, TOKEN_START};
use alpha_micrograd_rust::value::Expr;
use clap::Parser;
use rand::{distr::Uniform, prelude::Distribution, rng};
use std::error::Error;
use std::iter::once;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Arguments {
    #[arg(short, long, default_value = "names.txt")]
    filename: String,

    #[arg(short, long, default_value = "100")]
    epochs: usize,

    #[arg(short, long, default_value = "0.1")]
    learning_rate: f64,

    #[arg(short, long, default_value = "1000")]
    max_examples: Option<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Arguments::parse();

    print!("Generating vocabulary... ");
    let (xs, ys, vocabulary) = generate_dataset(&args.filename)?;
    println!("Done.");

    print!("One-hot encoding... ");
    let vocab_size = vocabulary.len();
    let mut xenc = one_hot_encode(&xs, vocab_size);
    if let Some(max_examples) = args.max_examples {
        xenc.truncate(max_examples);
    }
    println!("Done.");

    let between = Uniform::new_inclusive(-1.0, 1.0)?;
    let mut rng = rng();

    print!("Initializing weights... ");
    let w = (0..vocab_size)
        .map(|_| between.sample(&mut rng))
        .enumerate()
        .map(|(i, v)| Expr::new_leaf(v, &format!("w_{}", i)))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|v| vec![v])
        .collect::<Vec<_>>();
    println!("Done.");

    print!("Matrix multiplication... ");
    let logits = mat_mul(&xenc, &w);
    println!("Done.");

    print!("Softmax... ");
    let probs = softmax(&logits);
    println!("Done.");

    dbg!(probs.len(), probs[0].len());

    print!("Calculating Average NLL... ");
    let mut average_nll: Expr = Expr::new_leaf(0.0, "average_nll");
    for i in 0..xenc.len() {
        let y = ys[i].clone();
        let p = &probs[i][y];
        let log_likelihood = p.clone().log(&format!("log_likelihood_{}", i));
        let nll = log_likelihood.clone().neg(&format!("loss_{}", i));
        average_nll = average_nll + nll;
    }
    average_nll = average_nll / xenc.len() as f64;
    dbg!(&average_nll.result);
    println!("Done.");

    print!("Training... ");
    for epoch in 0..args.epochs {
        average_nll.learn(args.learning_rate);
        average_nll.recalculate();
        dbg!(epoch, &average_nll.result);
    }
    println!("Done.");

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
