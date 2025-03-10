use std::fs::File;
use std::io::{BufRead, BufReader, Error, Lines};

pub fn get_line_reader(filename: &str) -> Result<Lines<BufReader<File>>, Error> {
    let file = File::open(filename)?;
    Ok(BufReader::new(file).lines())
}
