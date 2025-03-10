use std::collections::HashMap;

pub struct Vocabulary {
    s_to_i: HashMap<char, usize>,
    i_to_s: HashMap<usize, char>,

    next_id: usize,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self {
            s_to_i: HashMap::new(),
            i_to_s: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn has_token(&self, token: char) -> bool {
        self.s_to_i.contains_key(&token)
    }

    pub fn token_to_id(&self, token: char) -> Option<usize> {
        self.s_to_i.get(&token).copied()
    }

    pub fn id_to_token(&self, id: usize) -> Option<char> {
        self.i_to_s.get(&id).copied()
    }

    pub fn add_token(&mut self, token: char) -> usize {
        let id = self.s_to_i.entry(token).or_insert(self.next_id);
        if *id == self.next_id {
            self.next_id += 1;
        }
        self.i_to_s.insert(*id, token);
        *id
    }
}
