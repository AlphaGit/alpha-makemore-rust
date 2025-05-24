use alpha_micrograd_rust::value::Expr;
use alpha_makemore_rust::nll::{Arguments, execute_with_progress, generate_dataset, one_hot_encode, mat_mul, softmax};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::{distr::Uniform, prelude::Distribution, rng};
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Arguments::try_parse()?;

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
            .map(|(i, v)| Expr::new_leaf(v))
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
        let mut average_nll: Expr = Expr::new_leaf(0.0);
        for i in 0..xenc.len() {
            let y = ys[i].clone();
            let p = &probs[i][y];
            let log_likelihood = p.clone().log();
            let nll = log_likelihood.clone().neg();
            average_nll = average_nll + nll;
        }
        average_nll = average_nll / xenc.len() as f64;
        average_nll
    });

    let pb = ProgressBar::new(args.epochs as u64);
    pb.set_style(
        ProgressStyle
            ::with_template("{elapsed_precise}/{duration_precise} | {per_sec} | {wide_bar} {percent}% | Epoch: {human_pos}/{human_len} | {msg} ")
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

