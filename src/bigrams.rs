use crate::file_reader::get_line_reader;
use crate::vocabulary::Vocabulary;
use crate::{TOKEN_END, TOKEN_START, VOCAB_SIZE};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;

use rand::{SeedableRng, rngs::StdRng};

pub struct Bigrams {
    frequencies: Vec<Vec<i32>>,
    vocabulary: Vocabulary,
    rng: StdRng,
}

impl Bigrams {
    fn new() -> Self {
        Self {
            vocabulary: Vocabulary::new(),
            frequencies: vec![vec![0; VOCAB_SIZE]; VOCAB_SIZE],
            rng: StdRng::seed_from_u64(10),
        }
    }

    fn increment(&mut self, token1: char, token2: char) {
        let token1_id = match self.vocabulary.has_token(token1) {
            true => self
                .vocabulary
                .token_to_id(token1)
                .expect("Token not in vocabulary, but already counting frequencies."),
            false => self.vocabulary.add_token(token1),
        };

        let token2_id = match self.vocabulary.has_token(token2) {
            true => self
                .vocabulary
                .token_to_id(token2)
                .expect("Token not in vocabulary, but already counting frequencies."),
            false => self.vocabulary.add_token(token2),
        };

        assert!(
            token1_id < VOCAB_SIZE,
            "Token 1 ID out of bounds: {}",
            token1_id
        );
        assert!(
            token2_id < VOCAB_SIZE,
            "Token 2 ID out of bounds: {}",
            token2_id
        );

        let token1_freqs = &mut self.frequencies[token1_id as usize];
        token1_freqs[token2_id as usize] += 1;
    }

    pub fn print_bigrams(&self) {
        for (token1_id, token1_freqs) in self.frequencies.iter().enumerate() {
            for (token2_id, freq) in token1_freqs.iter().enumerate() {
                let token1 = self
                    .vocabulary
                    .id_to_token(token1_id)
                    .expect("Token ID out of bounds.");
                let token2 = self
                    .vocabulary
                    .id_to_token(token2_id)
                    .expect("Token ID out of bounds.");
                print!("{:1}{:1}: {:4} | ", token1, token2, freq);
            }
            println!("")
        }
    }

    fn sample_next_token(&mut self, token: char) -> char {
        let token_id = self
            .vocabulary
            .token_to_id(token)
            .expect("Token not in vocabulary.");

        let token_freqs = &self.frequencies[token_id];

        let wi = WeightedIndex::new(token_freqs.iter()).expect("No non-zero frequencies.");
        let next_token_id = wi.sample(&mut self.rng);

        return self
            .vocabulary
            .id_to_token(next_token_id)
            .expect("Sampled token ID out of bounds.");
    }

    pub fn generate_new_text(&mut self) -> String {
        let mut text = String::new();

        let mut last_token = TOKEN_START;

        loop {
            let next_token = self.sample_next_token(last_token);

            if next_token == TOKEN_END {
                break;
            }

            text.push(next_token);
            last_token = next_token;
        }

        text
    }
}

pub fn count_frequencies(filename: &str) -> Result<Bigrams, Box<dyn std::error::Error>> {
    let line_reader = get_line_reader(filename)?;

    let mut bigrams = Bigrams::new();

    for line in line_reader {
        let line = line?;

        let mut last_token = TOKEN_START;

        for token in line.chars() {
            // increment frequency
            bigrams.increment(last_token, token);
            last_token = token;
        }

        bigrams.increment(last_token, TOKEN_END);
    }

    Ok(bigrams)
}
