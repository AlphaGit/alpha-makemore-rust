use crate::file_reader::get_line_reader;
use crate::vocabulary::Vocabulary;
use crate::{TOKEN_END, TOKEN_START};
use alpha_micrograd_rust::value::Expr;
use clap::Parser;
use indicatif::ProgressBar;
use std::error::Error;
use std::iter::once;
use std::time::Duration;

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Arguments {
    #[arg(short, long, default_value = "names.txt")]
    pub filename: String,

    #[arg(short, long, default_value = "10000")]
    pub epochs: usize,

    #[arg(short, long, default_value = "0.1")]
    pub learning_rate: f64,

    #[arg(short, long, default_value = "5000")]
    pub max_examples: Option<usize>,
}

pub fn execute_with_progress<F, O>(msg: &str, f: F) -> O
where
    F: FnOnce() -> O,
{
    let pb = ProgressBar::new_spinner();
    pb.enable_steady_tick(Duration::from_millis(100));
    pb.set_message(msg.to_string());
    let result = f();

    pb.finish_with_message(format!("{} Done.", msg));
    result
}

pub fn generate_dataset(
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

pub fn one_hot_encode(values: &[usize], num_classes: usize) -> Vec<Vec<f64>> {
    values
        .iter()
        .map(|&ix| one_hot_encode_single(ix, num_classes))
        .collect()
}

pub fn one_hot_encode_single(value: usize, num_classes: usize) -> Vec<f64> {
    (0..num_classes)
        .map(|ix| if ix == value { 1.0 } else { 0.0 })
        .collect()
}

pub fn mat_mul(xs: &[Vec<f64>], w: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
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

pub fn softmax(logits: &[Vec<Expr>]) -> Vec<Vec<Expr>> {
    logits
        .iter()
        .map(|logit_row| {
            let exps = logit_row
                .iter()
                .enumerate()
                .map(|(i, logit)| logit.clone().exp())
                .collect::<Vec<_>>();

            let sum_exp = exps.iter().map(|e| e.clone()).sum::<Expr>();
            exps.into_iter().map(|exp| exp / sum_exp.clone()).collect()
        })
        .collect()
}
