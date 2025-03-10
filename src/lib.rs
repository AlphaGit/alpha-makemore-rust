use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Error, Lines};

fn get_line_reader(filename: &str) -> Result<Lines<BufReader<File>>, Error> {
    let file = File::open(filename)?;
    Ok(BufReader::new(file).lines())
}

const TOKEN_START: char = '.';
const TOKEN_END: char = '.';
const VOCAB_SIZE: usize = 27;

#[derive(Debug)]
struct Vocabulary {
    s_to_i: HashMap<char, i32>,
    i_to_s: HashMap<i32, char>,

    next_id: i32,
}

impl Vocabulary {
    fn new() -> Self {
        Self {
            s_to_i: HashMap::new(),
            i_to_s: HashMap::new(),
            next_id: 0,
        }
    }

    fn has_token(&self, token: char) -> bool {
        self.s_to_i.contains_key(&token)
    }

    fn token_to_id(&self, token: char) -> Option<i32> {
        self.s_to_i.get(&token).copied()
    }

    fn id_to_token(&self, id: i32) -> Option<char> {
        self.i_to_s.get(&id).copied()
    }

    fn add_token(&mut self, token: char) -> i32 {
        let id = self.s_to_i.entry(token).or_insert(self.next_id);
        if *id == self.next_id {
            self.next_id += 1;
        }
        self.i_to_s.insert(*id, token);
        *id
    }
}

#[derive(Debug)]
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

        assert!(token1_id < VOCAB_SIZE as i32, "Token 1 ID out of bounds: {}", token1_id);
        assert!(token2_id < VOCAB_SIZE as i32, "Token 2 ID out of bounds: {}", token2_id);

        let token1_freqs = &mut self.frequencies[token1_id as usize];
        token1_freqs[token2_id as usize] += 1;
    }

    pub fn print_bigrams(&self) {
        for (token1_id, token1) in self.vocabulary.i_to_s.iter() {
            for (token2_id, token2) in self.vocabulary.i_to_s.iter() {
                let freq = self.frequencies[*token1_id as usize][*token2_id as usize];
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