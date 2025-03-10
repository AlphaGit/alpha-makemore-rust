use makemore_rust::count_frequencies;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let filename = "names.txt";

    let mut bigrams = count_frequencies(filename)?;

    bigrams.print_bigrams();

    println!("\nGenerated names:");

    for _ in 0..10 {
        let new_text = bigrams.generate_new_text();
        println!("{}", new_text);
    }

    Ok(())
}
