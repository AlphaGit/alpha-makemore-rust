use alpha_micrograd_rust::value::Expr;
use alpha_makemore_rust::nll::{Arguments, generate_dataset, one_hot_encode, mat_mul, softmax};
use rand::{distr::Uniform, prelude::Distribution, rng};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn bench_nll_prepare() -> Expr {
    let args = Arguments {
        filename: "names.txt".to_string(),
        epochs: 10000,
        learning_rate: 0.1,
        max_examples: Some(5000),
    };

    let (xs, ys, vocabulary) = generate_dataset(&args.filename).unwrap();

    let vocab_size = vocabulary.len();
    let mut xenc = one_hot_encode(&xs, vocab_size);
    if let Some(max_examples) = args.max_examples {
        xenc.truncate(max_examples);
    }

    let between = Uniform::new_inclusive(-1.0, 1.0).expect("Invalid range");
    let mut rng = rng();

    let w = (0..vocab_size)
        .map(|_| between.sample(&mut rng))
        .enumerate()
        .map(|(i, v)| Expr::new_leaf(v, &format!("w_{}", i)))
        .collect::<Vec<_>>()
        .into_iter()
        .map(|v| vec![v])
        .collect::<Vec<_>>();

    let logits = mat_mul(&xenc, &w);

    let probs = softmax(&logits);

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
}

pub fn bench_nll_train(mut average_nll: Expr) {
    for _ in 0..10 {
        average_nll.learn(0.1);
        average_nll.recalculate();
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let average_nll = bench_nll_prepare();
    c.bench_function("train_10_0.1", |b| {
        b.iter(|| bench_nll_train(black_box(average_nll.clone())));
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
