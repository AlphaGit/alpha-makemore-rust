const TOKEN_START: char = '.';
const TOKEN_END: char = '.';
const VOCAB_SIZE: usize = 27;

mod file_reader;
mod vocabulary;

use file_reader::get_line_reader;
use vocabulary::Vocabulary;

pub struct Bigrams {
    frequencies: Vec<Vec<i32>>,
    vocabulary: Vocabulary,
}

impl Bigrams {
    fn new() -> Self {
        Self {
            vocabulary: Vocabulary::new(),
            frequencies: vec![vec![0; VOCAB_SIZE]; VOCAB_SIZE],
        }
    }

    fn increment(&mut self, token1: char, token2: char) {
        let token1_id = match self.vocabulary.has_token(token1) {
            true => self.vocabulary.token_to_id(token1).expect("Token not in vocabulary, but already counting frequencies."),
            false => self.vocabulary.add_token(token1),
        };

        let token2_id = match self.vocabulary.has_token(token2) {
            true => self.vocabulary.token_to_id(token2).expect("Token not in vocabulary, but already counting frequencies."),
            false => self.vocabulary.add_token(token2),
        };

        assert!(token1_id < VOCAB_SIZE, "Token 1 ID out of bounds: {}", token1_id);
        assert!(token2_id < VOCAB_SIZE, "Token 2 ID out of bounds: {}", token2_id);

        let token1_freqs = &mut self.frequencies[token1_id as usize];
        token1_freqs[token2_id as usize] += 1;
    }

    pub fn print_bigrams(&self) {
        for (token1_id, token1_freqs) in self.frequencies.iter().enumerate() {
            for (token2_id, freq) in token1_freqs.iter().enumerate() {
                let token1 = self.vocabulary.id_to_token(token1_id).expect("Token ID out of bounds.");
                let token2 = self.vocabulary.id_to_token(token2_id).expect("Token ID out of bounds.");
                print!("{:1}{:1}: {:4} | ", token1, token2, freq);
            }
            println!("")
        }
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