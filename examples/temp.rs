use makemore_rust::get_line_reader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filename = "names.txt";

    let mut line_reader = get_line_reader(filename)?;
    for _ in 0..10 {
        match line_reader.next() {
            Some(Ok(line)) => println!("{}", line),
            Some(Err(e)) => eprintln!("Error reading line: {}", e),
            None => break,
        }
    }

    Ok(())
}
