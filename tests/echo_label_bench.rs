//! Label bucket benchmark suite (KS48, ADR-015).
//!
//! Measures the impact of label-based pre-filtering on echo latency at 100K scale.
//! Compares labels ON vs OFF on identical data to isolate the label contribution.
//!
//!     cargo test --release --test echo_label_bench -- --ignored --nocapture --test-threads=1

use shrimpk_core::EchoConfig;
use shrimpk_memory::EchoEngine;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Diverse content generator — 15 categories, realistic sentences
// ---------------------------------------------------------------------------

/// Generate a realistic memory sentence for a given category.
/// Each category has 20 template variations to avoid embedding collapse.
fn generate_memory(category_idx: usize, variation: usize) -> String {
    let templates: &[&[&str]] = &[
        // 0: career
        &[
            "I joined a new company as a senior engineer last month",
            "My manager asked me to lead the platform migration project",
            "I got promoted to tech lead after two years on the team",
            "We shipped the new microservices architecture last sprint",
            "I'm interviewing at three companies for a staff engineer role",
            "My team is hiring two backend developers this quarter",
            "I completed my performance review and got exceeds expectations",
            "We adopted Rust for our new systems programming work",
            "I mentored two junior developers on the distributed systems team",
            "The company announced a pivot to AI-first products",
            "I negotiated a remote work arrangement with my director",
            "Our sprint velocity increased 40% after the tooling upgrade",
            "I presented our architecture at the all-hands meeting",
            "We're migrating from AWS to GCP for cost optimization",
            "My quarterly goals include reducing latency by 50%",
            "I started pair programming sessions with the new hire",
            "The board approved our proposal for a new ML platform",
            "I switched from IntelliJ to VS Code for daily development",
            "We hit 99.99% uptime for the third consecutive month",
            "I'm attending a leadership workshop next Friday",
        ],
        // 1: language learning
        &[
            "I started learning Japanese on Duolingo about 30 days ago",
            "My Japanese teacher says I'm at JLPT N4 level now",
            "I practice Spanish vocabulary every morning for 20 minutes",
            "I joined a French conversation group that meets on Wednesdays",
            "I finished the Pimsleur German course level 2",
            "My Mandarin tones are improving after the pronunciation drills",
            "I'm reading my first novel in Portuguese",
            "The Italian grammar course covers subjunctive mood this week",
            "I downloaded Anki flashcards for Korean vocabulary",
            "I can now hold basic conversations in Arabic",
            "My language exchange partner is from Tokyo",
            "I watched a movie in Hindi without subtitles for the first time",
            "The Hebrew alphabet took me three weeks to memorize",
            "I'm preparing for the DELE Spanish certification exam",
            "I ordered food in Thai during my trip to Bangkok",
            "My Russian reading comprehension improved after the short stories",
            "I started learning sign language on weekends",
            "The Dutch pronunciation is easier than I expected",
            "I'm using spaced repetition for Swedish vocabulary",
            "My polyglot friend recommended the Michel Thomas method",
        ],
        // 2: fitness / exercise
        &[
            "I ran 5K this morning in 24 minutes which is a personal best",
            "I signed up for a half marathon happening in June",
            "Started doing yoga three times a week at CorePower",
            "My knee injury from running is finally healing",
            "I switched from running to cycling to reduce joint impact",
            "I completed a 30-day push-up challenge yesterday",
            "My gym added a new rock climbing wall that I love",
            "I hired a personal trainer for strength training twice a week",
            "Swimming laps in the morning has become my new routine",
            "I tracked 10,000 steps every day this month",
            "The CrossFit class was more intense than I expected",
            "I'm training for a triathlon next September",
            "My resting heart rate dropped from 72 to 62 bpm",
            "I started stretching for 15 minutes before every workout",
            "The hiking group explored a new trail in the mountains",
            "I bought a rowing machine for home workouts",
            "My plank hold time reached 3 minutes",
            "I joined an adult soccer league that plays on Saturdays",
            "The martial arts class teaches both striking and grappling",
            "I'm following a couch-to-5K program for beginners",
        ],
        // 3: cooking / food
        &[
            "I made homemade pasta from scratch for the first time",
            "I'm vegetarian and love cooking Indian food at home",
            "I've gotten into sourdough baking with a new starter",
            "My Thai curry recipe uses fresh lemongrass and galangal",
            "I started meal prepping on Sundays to save time",
            "I bought a new cast iron skillet for searing steaks",
            "The Japanese ramen recipe takes 12 hours for the broth",
            "I'm learning to make sushi rolls with proper vinegared rice",
            "My garden produces enough herbs for all my cooking",
            "I subscribed to a cooking class that covers French techniques",
            "The fermented hot sauce I made needs two more weeks",
            "I switched to plant-based milk for all my coffee drinks",
            "My grandmother's dumpling recipe is finally perfected",
            "I'm experimenting with sous vide cooking for proteins",
            "The farmers market has the best seasonal produce",
            "I made croissants that actually had proper lamination",
            "My spice collection now includes 40 different varieties",
            "I'm hosting a dinner party with a five-course tasting menu",
            "The bread machine makes fresh loaves every morning",
            "I learned to make proper espresso with my new machine",
        ],
        // 4: technology / programming
        &[
            "I've been writing Rust for systems programming projects",
            "I switched from VS Code to Neovim with a custom Lua config",
            "I deployed my first Kubernetes cluster on a home server",
            "My side project uses React and TypeScript with Next.js",
            "I contributed to an open source database project last week",
            "I'm learning about vector embeddings and similarity search",
            "The new M4 MacBook Pro has incredible compilation speed",
            "I set up a CI/CD pipeline with GitHub Actions and Docker",
            "My Raspberry Pi cluster runs a distributed messaging system",
            "I built a command-line tool that automates my daily workflow",
            "I'm experimenting with WebAssembly for browser-based tools",
            "The PostgreSQL query optimizer surprises me every time",
            "I migrated our monolith to microservices over six months",
            "My dotfiles repository has 200 stars on GitHub now",
            "I'm reading about B-tree implementations for database design",
            "The Nix package manager changed how I manage dependencies",
            "I built a real-time collaborative editor using CRDTs",
            "My blog runs on a static site generator I wrote in Go",
            "I'm benchmarking different hash map implementations in Rust",
            "The compiler optimization course covers loop vectorization",
        ],
        // 5: housing / moving
        &[
            "I moved from Oakland to San Francisco last month",
            "My new apartment in the Mission District has great light",
            "I'm looking at houses in the suburbs for more space",
            "The rent increase forced me to find a new place",
            "I finally set up my home office with a standing desk",
            "My roommate is moving out so I need a replacement",
            "I installed smart home devices throughout the apartment",
            "The neighborhood has three great coffee shops nearby",
            "I'm renovating the kitchen with new countertops",
            "My commute went from 45 minutes to 15 after the move",
            "I set up a small garden on the apartment balcony",
            "The building manager approved my request for a pet",
            "I painted the bedroom walls a calming shade of blue",
            "My new neighbors are friendly and we share meals sometimes",
            "I organized all the storage with labeled bins and shelving",
            "The apartment has a great view of the park from the window",
            "I'm saving for a down payment on a first home purchase",
            "My lease renewal came with a 5% increase this year",
            "I bought a robot vacuum that runs while I'm at work",
            "The furniture delivery arrives next Wednesday morning",
        ],
        // 6: music
        &[
            "I play acoustic guitar in the evenings mostly folk songs",
            "I started piano lessons after years of wanting to learn",
            "My band has a gig at a local venue next Saturday",
            "I'm producing electronic music using Ableton Live",
            "I joined a community choir that performs classical pieces",
            "My vinyl record collection now has over 200 albums",
            "I built a home recording studio in the spare bedroom",
            "I'm learning music theory to improve my compositions",
            "The jazz club downtown has live performances every Friday",
            "I switched from electric to acoustic guitar recently",
            "My favorite genre shifted from rock to ambient electronic",
            "I compose film scores as a side creative project",
            "The ukulele is surprisingly versatile for its size",
            "I attended a music festival with 50 bands over three days",
            "My daughter started violin lessons at age seven",
            "I'm transcribing songs by ear to improve my skills",
            "The music production course covers mixing and mastering",
            "I collaborate with artists online using shared DAW projects",
            "My synthesizer collection includes both analog and digital",
            "I practice scales for 30 minutes every morning before work",
        ],
        // 7: travel
        &[
            "I visited Tokyo last November and stayed in Shinjuku",
            "My trip to Portugal included Porto and the Algarve coast",
            "I'm planning a backpacking trip through Southeast Asia",
            "The Patagonia trek was the most challenging hike I've done",
            "I booked a flight to Iceland for the northern lights",
            "My road trip across the American Southwest took two weeks",
            "I learned to scuba dive during a vacation in Thailand",
            "The ancient temples in Cambodia were breathtaking",
            "I'm saving for a sabbatical year of world travel",
            "My favorite city so far is Barcelona for the architecture",
            "I documented my travels in a photo journal and blog",
            "The train journey from Paris to Istanbul was unforgettable",
            "I got my passport renewed for upcoming international trips",
            "My camping trip in Yellowstone had incredible wildlife",
            "I'm learning about responsible tourism and eco-travel",
            "The local food markets are my favorite part of traveling",
            "I volunteered at a wildlife sanctuary in Costa Rica",
            "My travel photography hobby started on a trip to Morocco",
            "I use travel hacking to fly business class on points",
            "The hostel in Amsterdam had the best community atmosphere",
        ],
        // 8: relationships / social
        &[
            "Jordan and I started dating after meeting at a Rust meetup",
            "My best friend moved to Berlin so we video call weekly",
            "I organized a dinner party for my closest friends",
            "My parents still live in Sacramento and I visit monthly",
            "I joined a book club that meets at the local library",
            "My partner and I adopted a rescue dog named Pixel",
            "I reconnected with college friends at a reunion last month",
            "My sister's wedding is next spring and I'm the best man",
            "I started hosting game nights every other Friday",
            "My mentor at work gave me career-changing advice",
            "I volunteer at a youth coding workshop on Saturdays",
            "My neighbors and I started a community garden project",
            "I joined a hiking group that meets every Sunday morning",
            "My therapist helped me set better personal boundaries",
            "I'm building stronger friendships by being more intentional",
            "My coworker and I started a side project together",
            "I attended a networking event for tech professionals",
            "My grandmother shares family recipes over video calls",
            "I made new friends through the rock climbing community",
            "My relationship with my siblings improved this year",
        ],
        // 9: finance / money
        &[
            "I set up a monthly budget tracking system in a spreadsheet",
            "My investment portfolio is diversified across index funds",
            "I opened a high-yield savings account for my emergency fund",
            "I'm paying off student loans with the avalanche method",
            "My side project generated its first revenue this month",
            "I maximized my 401k contributions for tax advantages",
            "I switched to a credit card with better travel rewards",
            "My financial advisor recommended rebalancing quarterly",
            "I'm saving 30% of my income for early retirement goals",
            "The tax return this year was larger than expected",
            "I started tracking net worth monthly in a spreadsheet",
            "My cryptocurrency holdings are a small percentage of total",
            "I negotiated a higher salary during the annual review",
            "I canceled subscriptions I wasn't using to save money",
            "My rental property generates passive income each month",
            "I'm learning about options trading with paper money first",
            "The financial independence community has great resources",
            "I created a will and estate plan with a lawyer",
            "My emergency fund covers six months of living expenses",
            "I donated to three charities this year for tax deduction",
        ],
        // 10: pets
        &[
            "I have a tabby cat named Mochi who is 3 years old",
            "I adopted a second cat, a black kitten named Pixel",
            "My golden retriever needs a longer walk route now",
            "The vet appointment for vaccinations is next Tuesday",
            "I built a cat tree that reaches the ceiling",
            "My fish tank has a new coral reef setup",
            "I'm training the puppy with positive reinforcement",
            "The pet insurance covers 80% of veterinary costs",
            "My parrot learned to say three new phrases this week",
            "I volunteered at the local animal shelter on weekends",
            "The automatic feeder dispenses food at set times",
            "My cat's favorite spot is the windowsill in the sun",
            "I switched to grain-free food for the dog's allergies",
            "The rabbit has a large enclosure in the backyard",
            "My reptile collection includes two geckos and a snake",
            "I'm fostering kittens until they find permanent homes",
            "The dog park near our house has an agility course",
            "My senior cat needs medication twice daily for thyroid",
            "I adopted a rescue greyhound from a racing track",
            "The aquarium maintenance takes about an hour each week",
        ],
        // 11: education / school
        &[
            "I enrolled in a master's degree program in computer science",
            "My online course about machine learning starts next week",
            "I'm studying for the AWS Solutions Architect certification",
            "The data structures class covers red-black trees tomorrow",
            "I completed a bootcamp on full-stack web development",
            "My thesis research focuses on distributed systems",
            "I'm taking evening classes to learn graphic design",
            "The professor recommended additional reading on algorithms",
            "I joined a study group for the upcoming final exams",
            "My GPA improved to 3.8 after focusing on time management",
            "I'm auditing a Stanford course on artificial intelligence",
            "The workshop on public speaking helped my presentation skills",
            "I earned a certificate in project management from PMI",
            "My lab partner and I published a research paper together",
            "I'm preparing applications for PhD programs in robotics",
            "The lecture on quantum computing was mind-blowing",
            "I completed all prerequisites for the advanced track",
            "My study abroad semester in Berlin was transformative",
            "I'm teaching a beginner programming workshop at the library",
            "The academic advisor helped me plan my course sequence",
        ],
        // 12: health / medical
        &[
            "My doctor recommended reducing caffeine intake",
            "I started meditating every morning for 10 minutes",
            "The blood test results came back normal this time",
            "I'm seeing a therapist weekly for anxiety management",
            "My sleep improved after implementing a consistent schedule",
            "I got my annual flu vaccination at the pharmacy",
            "The chiropractor helped with my lower back pain",
            "I'm tracking my water intake to stay hydrated",
            "My allergies are worse this spring than last year",
            "I started taking vitamin D supplements for the winter",
            "The eye exam showed I need a stronger prescription",
            "I'm doing physical therapy for my shoulder injury",
            "My dentist said I need a crown on the back molar",
            "I reduced screen time before bed and sleep quality improved",
            "The dermatologist prescribed a new skincare routine",
            "I'm managing stress better with breathing exercises",
            "My cholesterol levels improved after diet changes",
            "I started wearing blue light glasses for computer work",
            "The nutritionist created a meal plan for my goals",
            "I'm recovering well from the minor surgery last week",
        ],
        // 13: entertainment
        &[
            "I'm reading Designing Data-Intensive Applications right now",
            "I binge-watched the new sci-fi series on streaming",
            "My podcast playlist includes tech and true crime shows",
            "I finished reading three novels during vacation last week",
            "The new video game has an incredible open world design",
            "I subscribed to a film criticism newsletter I enjoy",
            "My book club chose a philosophy title for next month",
            "I watched a documentary about deep sea exploration",
            "The board game night featured a complex strategy game",
            "I'm playing through a classic RPG I missed as a kid",
            "My Kindle library has over 500 books accumulated",
            "I attended a live theater performance downtown",
            "The comedy special had me laughing for the whole hour",
            "I'm building a home theater with a projector setup",
            "My favorite author released a new novel this week",
            "I started a movie journal to track what I watch",
            "The escape room challenge was the best team activity",
            "I'm learning chess through online puzzles and lessons",
            "My audiobook habit fills the commute time perfectly",
            "The art exhibition at the museum was deeply inspiring",
        ],
        // 14: hobby / creative
        &[
            "I started woodworking and built my first bookshelf",
            "My photography hobby focuses on street and urban scenes",
            "I'm learning watercolor painting through online tutorials",
            "The pottery class teaches wheel throwing techniques",
            "I designed and 3D printed custom phone stands",
            "My knitting project is a sweater pattern from the 1940s",
            "I started a journal with daily sketches and observations",
            "The calligraphy set arrived and I practice every evening",
            "I'm restoring a vintage motorcycle in the garage",
            "My drone photography captured amazing aerial landscapes",
            "I learned to make candles with natural soy wax",
            "The leatherworking workshop taught me to make wallets",
            "I'm building a model train layout in the basement",
            "My digital art portfolio has grown to 50 pieces",
            "I started collecting and restoring vintage typewriters",
            "The origami book teaches increasingly complex designs",
            "I'm making jewelry from recycled metals and stones",
            "My embroidery project is a detailed botanical pattern",
            "I built a custom mechanical keyboard from components",
            "The film photography darkroom is finally set up at home",
        ],
    ];

    let cat = category_idx % templates.len();
    let var = variation % templates[cat].len();
    templates[cat][var].to_string()
}

/// Generate diverse query set — one per major category.
fn benchmark_queries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("What programming language do I use?", "technology"),
        ("What languages am I learning?", "language"),
        ("How do I exercise?", "fitness"),
        ("What do I like to cook?", "food"),
        ("Where do I live?", "housing"),
        ("What music do I play?", "music"),
        ("Where have I traveled?", "travel"),
        ("Do I have any pets?", "pets"),
        ("What am I studying?", "education"),
        ("How is my health?", "health"),
        ("What do I do for work?", "career"),
        ("Who are my friends?", "relationships"),
        ("What are my hobbies?", "hobby"),
        ("What am I reading?", "entertainment"),
        ("How do I manage my money?", "finance"),
        ("What is my exercise routine?", "fitness"),
        ("What editor do I use?", "technology"),
        ("What certifications am I pursuing?", "education"),
        ("Tell me about my partner", "relationships"),
        ("What investments do I have?", "finance"),
    ]
}

fn make_bench_config(data_dir: PathBuf, use_labels: bool) -> EchoConfig {
    EchoConfig {
        max_memories: 200_000,
        similarity_threshold: 0.15,
        max_echo_results: 10,
        ram_budget_bytes: 2_000_000_000,
        data_dir,
        embedding_dim: 384,
        use_labels,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Benchmark: Labels ON vs OFF at 100K
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires fastembed model download + 100K store time (~15 min)"]
fn label_benchmark_100k_comparison() {
    println!("\n======================================================================");
    println!("=== LABEL BUCKET BENCHMARK: 100K memories, Labels ON vs OFF ===");
    println!("======================================================================\n");

    let num_memories = 100_000;
    let num_queries = 20;
    let queries = benchmark_queries();

    // --- Run 1: Labels ON ---
    println!("--- RUN 1: Labels ON ---\n");
    let dir_on = tempdir().expect("temp dir");
    let config_on = make_bench_config(dir_on.path().to_path_buf(), true);
    let engine_on = EchoEngine::new(config_on).expect("engine init (labels ON)");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store 100K diverse memories
    println!("Storing {num_memories} diverse memories (labels ON)...");
    let store_start = Instant::now();
    rt.block_on(async {
        for i in 0..num_memories {
            let category = i % 15;
            let variation = i / 15;
            let text = generate_memory(category, variation);
            engine_on.store(&text, "bench").await.expect("store");
        }
    });
    let store_time_on = store_start.elapsed().as_secs_f64();
    println!("Stored {num_memories} in {store_time_on:.1}s (labels ON)\n");

    // Query with labels ON
    let mut latencies_on: Vec<f64> = Vec::with_capacity(num_queries * 5);
    println!("Running {} queries x 5 rounds (labels ON)...", num_queries);
    for round in 0..5 {
        for (i, (query, _category)) in queries.iter().enumerate() {
            let start = Instant::now();
            let results = rt.block_on(async { engine_on.echo(query, 5).await.expect("echo") });
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            latencies_on.push(ms);
            if round == 0 {
                println!(
                    "  Q{i:02}: {ms:6.2}ms | {query} | {} results",
                    results.len()
                );
            }
        }
    }
    latencies_on.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_on = latencies_on[latencies_on.len() / 2];
    let p95_on = latencies_on[latencies_on.len() * 95 / 100];
    let p99_on = latencies_on[latencies_on.len() * 99 / 100];

    // Clean up engine ON
    drop(rt);
    drop(engine_on);
    drop(dir_on);

    println!("\n  Labels ON:  P50={p50_on:.2}ms  P95={p95_on:.2}ms  P99={p99_on:.2}ms");

    // --- Run 2: Labels OFF ---
    println!("\n--- RUN 2: Labels OFF ---\n");
    let dir_off = tempdir().expect("temp dir");
    let config_off = make_bench_config(dir_off.path().to_path_buf(), false);
    let engine_off = EchoEngine::new(config_off).expect("engine init (labels OFF)");
    let rt2 = tokio::runtime::Runtime::new().unwrap();

    // Store same 100K memories (no labels)
    println!("Storing {num_memories} diverse memories (labels OFF)...");
    let store_start2 = Instant::now();
    rt2.block_on(async {
        for i in 0..num_memories {
            let category = i % 15;
            let variation = i / 15;
            let text = generate_memory(category, variation);
            engine_off.store(&text, "bench").await.expect("store");
        }
    });
    let store_time_off = store_start2.elapsed().as_secs_f64();
    println!("Stored {num_memories} in {store_time_off:.1}s (labels OFF)\n");

    // Query with labels OFF
    let mut latencies_off: Vec<f64> = Vec::with_capacity(num_queries * 5);
    println!("Running {} queries x 5 rounds (labels OFF)...", num_queries);
    for round in 0..5 {
        for (i, (query, _category)) in queries.iter().enumerate() {
            let start = Instant::now();
            let results = rt2.block_on(async { engine_off.echo(query, 5).await.expect("echo") });
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            latencies_off.push(ms);
            if round == 0 {
                println!(
                    "  Q{i:02}: {ms:6.2}ms | {query} | {} results",
                    results.len()
                );
            }
        }
    }
    latencies_off.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_off = latencies_off[latencies_off.len() / 2];
    let p95_off = latencies_off[latencies_off.len() * 95 / 100];
    let p99_off = latencies_off[latencies_off.len() * 99 / 100];

    drop(rt2);
    drop(engine_off);
    drop(dir_off);

    println!("\n  Labels OFF: P50={p50_off:.2}ms  P95={p95_off:.2}ms  P99={p99_off:.2}ms");

    // --- Comparison ---
    println!("\n======================================================================");
    println!("=== COMPARISON ===");
    println!("======================================================================");
    println!("                    P50         P95         P99         Store Time");
    println!(
        "  Labels ON:     {p50_on:7.2}ms    {p95_on:7.2}ms    {p99_on:7.2}ms    {store_time_on:.1}s"
    );
    println!(
        "  Labels OFF:    {p50_off:7.2}ms    {p95_off:7.2}ms    {p99_off:7.2}ms    {store_time_off:.1}s"
    );

    let speedup = if p50_on > 0.0 { p50_off / p50_on } else { 0.0 };
    println!("  Speedup:       {speedup:.1}x");
    println!("======================================================================\n");

    // Soft assertions (informational, not HARD gates — save for KS49)
    if p50_on < 4.0 {
        println!("TARGET MET: Labels ON P50 {p50_on:.2}ms < 4.0ms");
    } else {
        println!("TARGET MISSED: Labels ON P50 {p50_on:.2}ms >= 4.0ms — tuning needed");
    }

    if p50_on < p50_off {
        println!("LABELS HELP: {speedup:.1}x speedup (P50 {p50_off:.2}ms -> {p50_on:.2}ms)");
    } else {
        println!("LABELS NEUTRAL OR WORSE: P50 ON={p50_on:.2}ms vs OFF={p50_off:.2}ms");
    }
}

// ---------------------------------------------------------------------------
// Smaller benchmark for quick validation (1K memories)
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires fastembed model download"]
fn label_benchmark_1k_smoke() {
    println!("\n=== LABEL SMOKE TEST: 1K memories ===\n");

    let dir = tempdir().expect("temp dir");
    let config = make_bench_config(dir.path().to_path_buf(), true);
    let engine = EchoEngine::new(config).expect("engine init");
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Store 1K diverse memories
    rt.block_on(async {
        for i in 0..1_000 {
            let text = generate_memory(i % 15, i / 15);
            engine.store(&text, "bench").await.expect("store");
        }
    });

    // Verify labels are assigned
    let stats = rt.block_on(async { engine.stats().await });
    println!("Stored 1K memories. Stats: {} total", stats.total_memories);

    // Run a few queries
    let queries = benchmark_queries();
    for (query, category) in queries.iter().take(5) {
        let start = Instant::now();
        let results = rt.block_on(async { engine.echo(query, 5).await.expect("echo") });
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  {ms:6.2}ms | {category:12} | {query} | {} results",
            results.len()
        );
    }

    drop(rt);
    drop(engine);
    drop(dir);
    println!("\nSmoke test complete.");
}
