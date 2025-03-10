use makemore_rust::count_frequencies;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filename = "names.txt";

    let bigrams = count_frequencies(filename)?;

    bigrams.print_bigrams();

    Ok(())
}
