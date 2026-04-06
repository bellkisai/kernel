//! Semantic label generation for pre-filtered retrieval (ADR-015).
//!
//! Labels are prefixed strings (e.g., "topic:language", "entity:rust") that enable
//! O(1) inverted-index lookups before cosine similarity scoring.
//!
//! Two-tier generation:
//! - **Tier 1 (this module):** Synchronous at store time (<2ms). Prototype cosine
//!   matching + keyword extraction + rule-based classifier.
//! - **Tier 2 (consolidation):** Async background. GLiNER NER + LLM classification.

use crate::similarity;

/// Maximum labels per memory entry.
pub const MAX_LABELS_PER_ENTRY: usize = 10;

/// Default cosine similarity threshold for prototype matching.
const DEFAULT_PROTOTYPE_THRESHOLD: f32 = 0.55;

// ---------------------------------------------------------------------------
// Label Prototypes — pre-computed embeddings for semantic label assignment
// ---------------------------------------------------------------------------

/// Pre-computed prototype embeddings for label classification.
///
/// Each prototype is a rich multi-word description of a label category.
/// At store time, the memory's embedding is compared against all prototypes
/// via cosine similarity. Labels with similarity above threshold are assigned.
///
/// Rich descriptions enable **implicit label inference**: "in my history class
/// we are talking about WW2" matches the education prototype even though the
/// word "learning" never appears — because the embedding model captures the
/// semantic field, not just keywords.
pub struct LabelPrototypes {
    /// Label text for each prototype (e.g., "topic:career").
    pub labels: Vec<String>,
    /// Rich description used for embedding (e.g., "career, employment, job...").
    pub descriptions: Vec<String>,
    /// Embedding vector for each prototype's description.
    pub embeddings: Vec<Vec<f32>>,
    /// Cosine similarity threshold for assignment.
    pub threshold: f32,
}

impl LabelPrototypes {
    /// Create an uninitialized prototype bank (no embeddings yet).
    /// Call `initialize()` with an embedder to compute embeddings.
    pub fn new_empty() -> Self {
        let (labels, descriptions) = prototype_definitions();
        Self {
            labels,
            descriptions,
            embeddings: Vec::new(),
            threshold: DEFAULT_PROTOTYPE_THRESHOLD,
        }
    }

    /// Compute embeddings for all prototype descriptions using the given embedder.
    /// This is called once at `EchoEngine::new()` time.
    pub fn initialize<F>(&mut self, mut embed_fn: F)
    where
        F: FnMut(&str) -> Option<Vec<f32>>,
    {
        self.embeddings = self
            .descriptions
            .iter()
            .filter_map(|desc| embed_fn(desc))
            .collect();

        tracing::info!(
            prototypes = self.embeddings.len(),
            threshold = self.threshold,
            "Label prototypes initialized"
        );
    }

    /// Whether the prototype bank has been initialized with embeddings.
    pub fn is_initialized(&self) -> bool {
        !self.embeddings.is_empty() && self.embeddings.len() == self.labels.len()
    }

    /// Classify a memory embedding against all prototypes.
    /// Returns labels whose prototype similarity exceeds the threshold.
    pub fn classify(&self, embedding: &[f32]) -> Vec<String> {
        if !self.is_initialized() || embedding.is_empty() {
            return Vec::new();
        }

        self.labels
            .iter()
            .zip(self.embeddings.iter())
            .filter_map(|(label, proto_emb)| {
                let sim = similarity::cosine_similarity(embedding, proto_emb);
                if sim >= self.threshold {
                    Some(label.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Classify a query embedding and return matching label strings.
    /// Uses a slightly lower threshold for queries (broader recall).
    pub fn classify_query(&self, embedding: &[f32]) -> Vec<String> {
        if !self.is_initialized() || embedding.is_empty() {
            return Vec::new();
        }

        let query_threshold = self.threshold - 0.05; // slightly more permissive for queries
        self.labels
            .iter()
            .zip(self.embeddings.iter())
            .filter_map(|(label, proto_emb)| {
                let sim = similarity::cosine_similarity(embedding, proto_emb);
                if sim >= query_threshold {
                    Some(label.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Prototype definitions — rich descriptions for implicit inference
// ---------------------------------------------------------------------------

/// Returns (labels, descriptions) for all prototype categories.
fn prototype_definitions() -> (Vec<String>, Vec<String>) {
    let definitions = vec![
        // topic: subject domains
        (
            "topic:career",
            "career, employment, job, work position, company, hiring, promotion, salary, interview, resume, professional development",
        ),
        (
            "topic:language:natural",
            "language learning, studying a language, vocabulary, grammar, fluency, bilingual, \
             native speaker, accent, JLPT, Japanese, Spanish, French, German, Chinese, Korean, \
             Mandarin, Hindi, Arabic, Portuguese, Italian, Russian, Dutch, Swedish, Turkish, \
             Duolingo, Rosetta Stone, language exchange, speaking practice, translation",
        ),
        (
            "topic:language:programming",
            "programming language, coding language, software language, Rust, Python, Go, \
             JavaScript, TypeScript, Java, C++, C#, Ruby, Scala, Kotlin, Swift, Haskell, \
             Elixir, Clojure, Erlang, compiled language, interpreted language, systems programming, \
             scripting language, functional language, object-oriented, type system, framework",
        ),
        (
            "topic:education",
            "education, school, university, college, class, classroom, lecture, studying, homework, degree, academic, course, teaching, student, professor, exam, grade",
        ),
        (
            "topic:health",
            "health, medical, doctor, hospital, illness, medicine, treatment, diagnosis, symptoms, wellness, mental health, therapy",
        ),
        (
            "topic:fitness",
            "fitness, exercise, running, gym, workout, yoga, training, marathon, cycling, swimming, sports, physical activity, stretching",
        ),
        (
            "topic:housing",
            "housing, apartment, house, home, rent, mortgage, moving, relocation, neighborhood, real estate, roommate, lease",
        ),
        (
            "topic:food",
            "food, cooking, restaurant, recipe, cuisine, meal, baking, diet, nutrition, vegetarian, kitchen, chef, eating",
        ),
        (
            "topic:music",
            "music, instrument, guitar, piano, singing, concert, band, song, playlist, album, genre, musician, melody",
        ),
        (
            "topic:technology",
            "technology, programming, software, code, developer, computer, app, framework, library, algorithm, debugging, API, database, system",
        ),
        (
            "topic:tools:editor",
            "text editor, code editor, IDE, integrated development environment, Neovim, Vim, nvim, \
             VSCode, VS Code, Visual Studio Code, JetBrains, IntelliJ, WebStorm, PyCharm, Emacs, \
             Sublime Text, Atom, Helix, Zed, Nano, editor configuration, dotfiles, init.lua, vimrc",
        ),
        (
            "topic:finance",
            "finance, money, budget, savings, investment, bank, credit, debt, tax, expense, income, financial planning",
        ),
        (
            "topic:travel",
            "travel, trip, vacation, destination, flight, hotel, tourism, sightseeing, abroad, passport, backpacking, adventure",
        ),
        (
            "topic:relationships",
            "relationships, partner, dating, family, friends, social, love, marriage, breakup, connection, communication",
        ),
        (
            "topic:hobby",
            "hobby, craft, art, painting, photography, gardening, reading, collecting, woodworking, knitting, creative pursuit",
        ),
        (
            "topic:entertainment",
            "entertainment, movie, film, television, show, series, streaming, gaming, video game, book, novel, podcast",
        ),
        (
            "topic:pets",
            "pets, cat, dog, animal, veterinarian, adoption, puppy, kitten, fish, bird, pet care, feeding",
        ),
        // domain: life areas
        (
            "domain:work",
            "work, office, professional, business, meeting, deadline, project, team, manager, colleague, workplace, corporate",
        ),
        (
            "domain:life",
            "personal life, daily routine, lifestyle, home life, household, chores, errands, family time, weekend, evening",
        ),
        (
            "domain:social",
            "social, friends, gathering, party, community, event, meetup, networking, hangout, celebration",
        ),
        (
            "domain:health",
            "health, wellness, self-care, medical, doctor visit, mental health, therapy, sleep, recovery, rest",
        ),
        (
            "domain:creative",
            "creative, art, design, writing, music, craft, invention, expression, imagination, artistic",
        ),
        // action: verb concepts
        (
            "action:learning",
            "learning, studying, practicing, training, course, tutorial, skill development, taking a class, researching, reading about, lesson, workshop",
        ),
        (
            "action:building",
            "building, creating, developing, coding, constructing, making, implementing, engineering, architecting, designing",
        ),
        (
            "action:planning",
            "planning, scheduling, organizing, preparing, strategizing, roadmap, timeline, goal setting, future, agenda",
        ),
        (
            "action:moving",
            "moving, relocating, changing address, new apartment, packing, unpacking, settling in, neighborhood change",
        ),
        (
            "action:exercising",
            "exercising, running, working out, training, gym session, yoga, cycling, swimming, physical activity",
        ),
        (
            "action:leading",
            "leading, managing, supervising, directing, organizing team, delegation, mentoring, coordinating, decision making",
        ),
        // temporal: time signals
        (
            "temporal:current",
            "currently, right now, at the moment, these days, presently, today, this week, this month",
        ),
        (
            "temporal:past",
            "used to, previously, before, in the past, formerly, back then, years ago, when I was",
        ),
        (
            "temporal:future",
            "plan to, going to, will, want to, hope to, considering, thinking about, next year, someday",
        ),
        (
            "temporal:recurring",
            "every day, every week, every morning, regularly, routine, habit, always, usually, typically, each time",
        ),
        // memtype: memory classification (renamed from modality: to avoid Modality enum collision)
        (
            "memtype:preference",
            "prefer, like, love, enjoy, favorite, choice, rather, better, best, always choose, always use",
        ),
        (
            "memtype:fact",
            "fact, is, has, was born, lives, works at, name is, age is, located in, known as",
        ),
        (
            "memtype:goal",
            "goal, want to, aim, aspire, target, objective, plan to achieve, dream, ambition, working toward",
        ),
        (
            "memtype:habit",
            "habit, routine, every day, always do, regular practice, ritual, pattern, custom, tendency",
        ),
        // sentiment: emotional valence
        (
            "sentiment:positive",
            "great, love, happy, excited, wonderful, amazing, enjoy, fantastic, thrilled, grateful, perfect, awesome",
        ),
        (
            "sentiment:negative",
            "bad, hate, frustrated, annoyed, disappointed, terrible, struggle, difficult, problem, worried, stressed, upset",
        ),
    ];

    let labels: Vec<String> = definitions.iter().map(|(l, _)| l.to_string()).collect();
    let descriptions: Vec<String> = definitions.iter().map(|(_, d)| d.to_string()).collect();
    (labels, descriptions)
}

// ---------------------------------------------------------------------------
// Tier 1 label generation — combines all three methods
// ---------------------------------------------------------------------------

/// Generate Tier 1 labels for a memory.
///
/// Combines three methods:
/// 1. Prototype cosine matching (handles implicit inference)
/// 2. Rule-based temporal detection
/// 3. Simple entity extraction (capitalized words)
///
/// Returns up to MAX_LABELS_PER_ENTRY labels.
pub fn generate_tier1_labels(
    content: &str,
    embedding: &[f32],
    prototypes: &LabelPrototypes,
) -> Vec<String> {
    let mut labels: Vec<String> = Vec::new();

    // 1. Prototype cosine matching (primary — handles implicit labels)
    labels.extend(prototypes.classify(embedding));

    // 2. Rule-based temporal detection
    let lower = content.to_lowercase();
    if contains_any(
        &lower,
        &[
            "every day",
            "every week",
            "every morning",
            "every night",
            "routine",
            "always",
            "usually",
            "regularly",
        ],
    ) {
        push_unique(&mut labels, "temporal:recurring");
    }
    if contains_any(
        &lower,
        &[
            "used to",
            "previously",
            "back then",
            "in the past",
            "formerly",
            "last month",
            "last year",
            "last week",
            "last november",
            "last december",
            "last january",
            "last february",
            "last march",
            "last april",
            "last may",
            "last june",
            "last july",
            "last august",
            "last september",
            "last october",
            "visited",
            "years ago",
            "months ago",
            "weeks ago",
        ],
    ) {
        push_unique(&mut labels, "temporal:past");
    }
    if contains_any(
        &lower,
        &[
            "plan to",
            "going to",
            "want to",
            "hope to",
            "considering",
            "next year",
            "next month",
            "next week",
            "upcoming",
            "deadline",
            "filing deadline",
            "due date",
            "due by",
            "submit by",
            "expires",
            "scheduled for",
        ],
    ) || contains_future_date(&lower)
    {
        push_unique(&mut labels, "temporal:future");
    }
    if contains_any(
        &lower,
        &[
            "right now",
            "currently",
            "these days",
            "at the moment",
            "this week",
        ],
    ) {
        push_unique(&mut labels, "temporal:current");
    }

    // 2b. Rule-based action:learning detection (KS68 PT-3)
    // Supplements prototype cosine matching with explicit JLPT/language-learning keywords.
    if contains_any(
        &lower,
        &[
            "learning", "studying", "practicing", "jlpt", "fluent",
            "native speaker", "taking lessons", "course", "class",
            "hiragana", "katakana", "kanji",
        ],
    ) {
        push_unique(&mut labels, "action:learning");
    }

    // 3. Simple entity extraction — capitalized multi-char words not at sentence start
    // This is a lightweight heuristic; Tier 2 (GLiNER) will provide precise NER.
    let mut after_sentence_end = true; // first word is always sentence-start
    for word in content.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());

        // Skip sentence-start capitalization (after . ! ? or start of text)
        if after_sentence_end {
            after_sentence_end = word.ends_with('.') || word.ends_with('!') || word.ends_with('?');
            continue;
        }
        after_sentence_end = word.ends_with('.')
            || word.ends_with('!')
            || word.ends_with('?')
            || word.ends_with(".\n")
            || word.ends_with(':');

        // Strip possessive suffix ('s / 's)
        let clean = clean
            .strip_suffix("'s")
            .or_else(|| clean.strip_suffix("\u{2019}s"))
            .unwrap_or(clean);

        if clean.len() >= 2
            && clean.chars().next().is_some_and(|c| c.is_uppercase())
            && !is_common_word(clean)
        {
            let entity = format!("entity:{}", clean.to_lowercase());
            push_unique(&mut labels, &entity);
        }
    }

    // Cap at max
    labels.truncate(MAX_LABELS_PER_ENTRY);
    labels
}

/// Classify a query into label categories for the D6 merge.
///
/// Tier A: keyword pattern matching (<0.001ms)
/// Tier B: prototype cosine matching (~0.015ms per label)
/// Tier C: fallback (returns empty, existing pipeline handles it)
pub fn classify_query(
    query: &str,
    query_embedding: &[f32],
    prototypes: &LabelPrototypes,
) -> Vec<String> {
    let mut labels: Vec<String> = Vec::new();

    // Tier A: keyword-based query classification
    let lower = query.to_lowercase();
    if contains_any(&lower, &["language", "languages", "lingu"]) {
        let natural_signals = contains_any(
            &lower,
            &[
                "learning", "studying", "jlpt", "fluent", "native", "speak",
                "vocabulary", "grammar", "duolingo", "rosetta", "accent",
            ],
        );
        let programming_signals = contains_any(
            &lower,
            &[
                "prefer", "code", "program", "framework", "library", "develop",
                "compile", "script", "software", "typed",
            ],
        );
        match (natural_signals, programming_signals) {
            (true, false) => push_unique(&mut labels, "topic:language:natural"),
            (false, true) => push_unique(&mut labels, "topic:language:programming"),
            _ => {
                // Ambiguous or both — emit both, let scoring decide
                push_unique(&mut labels, "topic:language:natural");
                push_unique(&mut labels, "topic:language:programming");
            }
        }
        // Backward compat: also emit the legacy label so query_labels OR-union
        // picks up old memories that were stored before the split.
        push_unique(&mut labels, "topic:language");
    }
    if contains_any(
        &lower,
        &[
            "learn", "study", "class", "course", "school", "jlpt",
            "fluent", "practicing", "lessons",
        ],
    ) {
        push_unique(&mut labels, "action:learning");
    }
    if contains_any(
        &lower,
        &["work", "job", "career", "project", "office", "company"],
    ) {
        push_unique(&mut labels, "domain:work");
    }
    if contains_any(&lower, &["exercise", "run", "workout", "gym", "fitness"]) {
        push_unique(&mut labels, "topic:fitness");
    }
    if contains_any(&lower, &["cook", "food", "eat", "recipe", "restaurant"]) {
        push_unique(&mut labels, "topic:food");
    }
    if contains_any(&lower, &["music", "guitar", "piano", "song", "play"]) {
        push_unique(&mut labels, "topic:music");
    }
    if contains_any(&lower, &["live", "apartment", "house", "neighbor", "move"]) {
        push_unique(&mut labels, "topic:housing");
    }
    if contains_any(&lower, &["pet", "cat", "dog", "animal"]) {
        push_unique(&mut labels, "topic:pets");
    }
    if contains_any(&lower, &["prefer", "favorite", "like best", "choose"]) {
        push_unique(&mut labels, "memtype:preference");
    }
    if contains_any(&lower, &["travel", "trip", "visit", "vacation"]) {
        push_unique(&mut labels, "topic:travel");
    }
    if contains_any(
        &lower,
        &["code", "program", "tech", "software", "editor", "framework"],
    ) {
        push_unique(&mut labels, "topic:technology");
    }
    if contains_any(
        &lower,
        &[
            "editor", "ide", "coding tool", "neovim", "vim", "vscode",
            "text editor", "jetbrains", "emacs", "sublime", "helix", "zed",
        ],
    ) {
        push_unique(&mut labels, "topic:tools:editor");
    }
    if contains_any(&lower, &["read", "book", "reading"]) {
        push_unique(&mut labels, "topic:entertainment");
    }
    if contains_any(&lower, &["health", "doctor", "sick", "medical"]) {
        push_unique(&mut labels, "topic:health");
    }

    // Tier B: prototype cosine matching for queries that don't match keywords
    if labels.is_empty() {
        labels = prototypes.classify_query(query_embedding);
    }

    labels
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn contains_any(text: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| text.contains(p))
}

/// Detect explicit date patterns that imply future time reference.
///
/// Matches:
/// - "Month YYYY" (e.g., "april 2026") where month is a full name
/// - "YYYY-MM-DD" ISO dates (e.g., "2026-04-15")
///
/// We don't compare against the current date — any explicit date reference
/// paired with future-signalling context (deadline, filing, due) is enough.
/// This function is called only when the text already contains "deadline" or
/// similar keywords haven't matched, so it provides incremental coverage for
/// content like "patent filing April 2026".
fn contains_future_date(text: &str) -> bool {
    // Pattern 1: "month yyyy" where yyyy is a 4-digit year
    let months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ];
    for month in months {
        if let Some(pos) = text.find(month) {
            let after = &text[pos + month.len()..];
            // Check for " YYYY" immediately after month name
            let after = after.trim_start();
            if after.len() >= 4 && after[..4].chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
        }
    }
    // Pattern 2: "YYYY-MM-DD" ISO date
    let bytes = text.as_bytes();
    for i in 0..text.len().saturating_sub(9) {
        if bytes[i].is_ascii_digit()
            && bytes[i + 1].is_ascii_digit()
            && bytes[i + 2].is_ascii_digit()
            && bytes[i + 3].is_ascii_digit()
            && bytes[i + 4] == b'-'
            && bytes[i + 5].is_ascii_digit()
            && bytes[i + 6].is_ascii_digit()
            && bytes[i + 7] == b'-'
            && bytes[i + 8].is_ascii_digit()
            && bytes[i + 9].is_ascii_digit()
        {
            return true;
        }
    }
    false
}

fn push_unique(labels: &mut Vec<String>, label: &str) {
    if !labels.iter().any(|l| l == label) {
        labels.push(label.to_string());
    }
}

fn is_common_word(word: &str) -> bool {
    // Strip embedded apostrophes/quotes to check the base form too.
    // e.g. "I'm" → check both "I'm" and "Im"; "It's" → "It's" and "Its"
    let base: String = word.chars().filter(|c| c.is_alphanumeric()).collect();

    // Check the raw word OR the stripped base against the stoplist.
    is_stopword(word) || is_stopword(&base)
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        // Pronouns & determiners
        "I" | "Im" | "Ive" | "Ill" | "Id"
            | "The" | "A" | "An" | "My" | "We" | "They" | "He" | "She" | "It" | "Its"
            | "You" | "Your" | "Our" | "Their" | "His" | "Her"
            // Demonstratives & relatives
            | "This" | "That" | "Thats" | "These" | "Those" | "Which" | "Who" | "Whom"
            // Conjunctions & prepositions
            | "But" | "And" | "Or" | "So" | "If" | "In" | "On" | "At" | "To" | "For"
            | "Of" | "With" | "By" | "As" | "From" | "Into" | "Up" | "Out" | "Over"
            // Verbs & auxiliaries
            | "Is" | "Was" | "Are" | "Am" | "Be" | "Been" | "Being"
            | "Do" | "Did" | "Does" | "Doesnt" | "Dont" | "Didnt"
            | "Has" | "Have" | "Had" | "Hasnt" | "Havent"
            | "Can" | "Cant" | "Could" | "Couldnt"
            | "Will" | "Wont" | "Would" | "Wouldnt"
            | "Should" | "Shouldnt" | "May" | "Might" | "Must"
            | "Not" | "No" | "Yes"
            // Adverbs & fillers
            | "Just" | "Now" | "Then" | "Also" | "About" | "After" | "Before"
            | "Because" | "When" | "Where" | "How" | "What" | "Why"
            | "Here" | "Heres" | "There" | "Theres"
            | "However" | "Although" | "Though" | "While" | "Since" | "Until"
            | "Still" | "Yet" | "Already" | "Really" | "Very" | "Well"
            | "Actually" | "Basically" | "Generally" | "Usually" | "Often"
            | "Some" | "Many" | "Most" | "Each" | "Every" | "Any" | "All"
            | "Sure" | "Maybe" | "Perhaps" | "Overall" | "Certainly"
            | "First" | "Second" | "Third" | "Next" | "Last"
            | "New" | "Good" | "Great" | "Best" | "Better"
            | "Try" | "Trying" | "Think" | "Thinking" | "Like" | "Looking"
            | "Make" | "Making" | "Keep" | "Take" | "Taking" | "Get" | "Getting"
            | "Let" | "Lets"
            // Day abbreviations (from timestamp patterns like "[2023/05/20 (Sat)]")
            | "Mon" | "Tue" | "Wed" | "Thu" | "Fri" | "Sat" | "Sun"
            | "Monday" | "Tuesday" | "Wednesday" | "Thursday" | "Friday"
            | "Saturday" | "Sunday"
            // Month abbreviations
            | "Jan" | "Feb" | "Mar" | "Apr" | "Jun" | "Jul" | "Aug"
            | "Sep" | "Oct" | "Nov" | "Dec"
            | "January" | "February" | "March" | "April" | "June" | "July"
            | "August" | "September" | "October" | "November" | "December"
            // Common response starters (AI assistant patterns)
            | "Thank" | "Thanks" | "Please" | "Sorry" | "Note"
            | "Remember" | "Consider" | "Check" | "See" | "Using"
            // More contractions (stripped forms)
            | "Youre" | "Youve" | "Youll" | "Youd"
            | "Theyre" | "Theyve" | "Theyll"
            | "Were" | "Weve" | "Hes" | "Shes" | "Whats" | "Whos"
            // Additional common words that appear as false entities
            | "Congratulations" | "Additionally" | "Furthermore" | "Moreover"
            | "Specifically" | "Particularly" | "Especially" | "Absolutely"
            | "Definitely" | "Exactly" | "Probably" | "Possibly"
            | "Start" | "Started" | "Starting" | "Choose" | "Chose" | "Chosen"
            | "Use" | "Used" | "Uses" | "Time" | "Times"
            | "Yeah" | "Okay" | "Ok" | "Oh" | "Ah" | "Hmm" | "Wow"
            | "Today" | "Tomorrow" | "Yesterday" | "Tonight"
            | "Important" | "Interesting" | "Helpful" | "Useful"
            | "Need" | "Needs" | "Want" | "Wants" | "Feel" | "Feels"
            | "Say" | "Said" | "Saying" | "Know" | "Knew" | "Known"
            | "Found" | "Find" | "Finding" | "Work" | "Works" | "Working"
            | "Explore" | "Exploring" | "Include" | "Including" | "Includes"
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a mock prototype bank with pre-computed "embeddings" for testing.
    /// Uses simple synthetic vectors instead of real fastembed embeddings.
    fn mock_prototypes() -> LabelPrototypes {
        let (labels, descriptions) = prototype_definitions();
        // Create simple synthetic embeddings: each prototype gets a unique direction
        let dim = 384;
        let embeddings: Vec<Vec<f32>> = (0..labels.len())
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                // Each prototype has energy in a different region
                let start = (i * 5) % dim;
                for j in 0..10 {
                    v[(start + j) % dim] = 1.0;
                }
                // Normalize
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect();

        LabelPrototypes {
            labels,
            descriptions,
            embeddings,
            threshold: DEFAULT_PROTOTYPE_THRESHOLD,
        }
    }

    #[test]
    fn prototype_definitions_complete() {
        let (labels, descriptions) = prototype_definitions();
        assert!(
            labels.len() >= 35,
            "Should have >= 35 prototypes, got {}",
            labels.len()
        );
        assert_eq!(labels.len(), descriptions.len());

        // All labels have a prefix
        for label in &labels {
            assert!(label.contains(':'), "Label '{label}' missing prefix");
        }

        // All 7 dimensions represented
        let prefixes: Vec<&str> = vec![
            "topic:",
            "domain:",
            "action:",
            "temporal:",
            "memtype:",
            "sentiment:",
        ];
        for prefix in prefixes {
            assert!(
                labels.iter().any(|l| l.starts_with(prefix)),
                "Missing prefix '{prefix}' in prototypes"
            );
        }
    }

    #[test]
    fn tier1_rule_based_temporal_recurring() {
        let protos = mock_prototypes();
        let labels = generate_tier1_labels(
            "Every morning I run 5K before work",
            &vec![0.0; 384],
            &protos,
        );
        assert!(
            labels.iter().any(|l| l == "temporal:recurring"),
            "Should detect 'every morning' as temporal:recurring, got: {labels:?}"
        );
    }

    #[test]
    fn tier1_rule_based_temporal_past() {
        let protos = mock_prototypes();
        let labels = generate_tier1_labels("I used to live in Tokyo", &vec![0.0; 384], &protos);
        assert!(
            labels.iter().any(|l| l == "temporal:past"),
            "Should detect 'used to' as temporal:past, got: {labels:?}"
        );
    }

    #[test]
    fn tier1_temporal_past_extended_signals() {
        let protos = mock_prototypes();
        let labels = generate_tier1_labels(
            "I visited Paris last month and it was great",
            &vec![0.0; 384],
            &protos,
        );
        assert!(
            labels.iter().any(|l| l == "temporal:past"),
            "Should detect 'visited' + 'last month' as temporal:past, got: {labels:?}"
        );
    }

    #[test]
    fn tier1_temporal_past_years_ago() {
        let protos = mock_prototypes();
        let labels = generate_tier1_labels(
            "I moved to the US years ago",
            &vec![0.0; 384],
            &protos,
        );
        assert!(
            labels.iter().any(|l| l == "temporal:past"),
            "Should detect 'years ago' as temporal:past, got: {labels:?}"
        );
    }

    #[test]
    fn tier1_temporal_future_deadline() {
        let protos = mock_prototypes();
        let labels = generate_tier1_labels(
            "Patent filing deadline is April 2026",
            &vec![0.0; 384],
            &protos,
        );
        assert!(
            labels.iter().any(|l| l == "temporal:future"),
            "Should detect 'deadline' as temporal:future, got: {labels:?}"
        );
    }

    #[test]
    fn tier1_temporal_future_next_month() {
        let protos = mock_prototypes();
        let labels = generate_tier1_labels(
            "I have a conference next month",
            &vec![0.0; 384],
            &protos,
        );
        assert!(
            labels.iter().any(|l| l == "temporal:future"),
            "Should detect 'next month' as temporal:future, got: {labels:?}"
        );
    }

    #[test]
    fn tier1_temporal_future_month_year_pattern() {
        let protos = mock_prototypes();
        let labels = generate_tier1_labels(
            "ROSCon submission due april 2026",
            &vec![0.0; 384],
            &protos,
        );
        assert!(
            labels.iter().any(|l| l == "temporal:future"),
            "Should detect 'april 2026' date pattern as temporal:future, got: {labels:?}"
        );
    }

    #[test]
    fn tier1_temporal_future_iso_date() {
        let protos = mock_prototypes();
        let labels = generate_tier1_labels(
            "Patent provisional filing 2026-04-15",
            &vec![0.0; 384],
            &protos,
        );
        assert!(
            labels.iter().any(|l| l == "temporal:future"),
            "Should detect ISO date '2026-04-15' as temporal:future, got: {labels:?}"
        );
    }

    #[test]
    fn contains_future_date_month_year() {
        assert!(contains_future_date("april 2026"));
        assert!(contains_future_date("submit by november 2025"));
        assert!(!contains_future_date("april is a nice month"));
        assert!(!contains_future_date("no dates here"));
    }

    #[test]
    fn contains_future_date_iso() {
        assert!(contains_future_date("due 2026-04-15 sharp"));
        assert!(contains_future_date("2025-12-31"));
        assert!(!contains_future_date("2026-4-15")); // not zero-padded, no match
        assert!(!contains_future_date("no dates"));
    }

    #[test]
    fn tier1_entity_extraction_capitalized() {
        let protos = mock_prototypes();
        let labels = generate_tier1_labels(
            "I started learning Japanese on Duolingo",
            &vec![0.0; 384],
            &protos,
        );
        assert!(
            labels.iter().any(|l| l == "entity:japanese"),
            "Should extract 'Japanese' as entity:japanese, got: {labels:?}"
        );
        assert!(
            labels.iter().any(|l| l == "entity:duolingo"),
            "Should extract 'Duolingo' as entity:duolingo, got: {labels:?}"
        );
    }

    #[test]
    fn tier1_skips_common_words() {
        let protos = mock_prototypes();
        let labels = generate_tier1_labels("The quick Brown fox", &vec![0.0; 384], &protos);
        assert!(
            !labels.iter().any(|l| l == "entity:the"),
            "Should skip common word 'The', got: {labels:?}"
        );
    }

    #[test]
    fn tier1_labels_capped_at_max() {
        let protos = mock_prototypes();
        // Create content that would trigger many labels
        let content = "Every morning I run in Tokyo with Rust and Japanese music at Duolingo school university college class homework";
        let labels = generate_tier1_labels(content, &vec![0.0; 384], &protos);
        assert!(
            labels.len() <= MAX_LABELS_PER_ENTRY,
            "Should cap at {MAX_LABELS_PER_ENTRY}, got: {}",
            labels.len()
        );
    }

    #[test]
    fn query_classification_tier_a_keywords() {
        let protos = mock_prototypes();
        let labels = classify_query("what languages am I learning?", &vec![0.0; 384], &protos);
        assert!(
            labels.iter().any(|l| l == "topic:language:natural"),
            "Should match 'languages' + 'learning' → natural, got: {labels:?}"
        );
        assert!(
            labels.iter().any(|l| l == "action:learning"),
            "Should match 'learning' keyword, got: {labels:?}"
        );
    }

    #[test]
    fn query_language_natural_signals() {
        let protos = mock_prototypes();
        let labels =
            classify_query("What language is Sam learning?", &vec![0.0; 384], &protos);
        assert!(
            labels.iter().any(|l| l == "topic:language:natural"),
            "Should route 'learning' to natural, got: {labels:?}"
        );
        assert!(
            !labels.iter().any(|l| l == "topic:language:programming"),
            "Should NOT emit programming when 'learning' present, got: {labels:?}"
        );
    }

    #[test]
    fn query_language_programming_signals() {
        let protos = mock_prototypes();
        let labels = classify_query(
            "What programming language does Sam prefer?",
            &vec![0.0; 384],
            &protos,
        );
        assert!(
            labels.iter().any(|l| l == "topic:language:programming"),
            "Should route 'programming'+'prefer' to programming, got: {labels:?}"
        );
        assert!(
            !labels.iter().any(|l| l == "topic:language:natural"),
            "Should NOT emit natural when 'programming'+'prefer' present, got: {labels:?}"
        );
    }

    #[test]
    fn query_language_ambiguous_emits_both() {
        let protos = mock_prototypes();
        let labels =
            classify_query("What language does Sam know?", &vec![0.0; 384], &protos);
        assert!(
            labels.iter().any(|l| l == "topic:language:natural"),
            "Ambiguous should emit natural, got: {labels:?}"
        );
        assert!(
            labels.iter().any(|l| l == "topic:language:programming"),
            "Ambiguous should emit programming, got: {labels:?}"
        );
    }

    #[test]
    fn query_language_always_emits_legacy_label() {
        let protos = mock_prototypes();
        // Natural-only query
        let labels =
            classify_query("What language is Sam learning?", &vec![0.0; 384], &protos);
        assert!(
            labels.iter().any(|l| l == "topic:language"),
            "Natural query should also emit legacy topic:language, got: {labels:?}"
        );
        // Programming-only query
        let labels = classify_query(
            "What programming language does Sam prefer?",
            &vec![0.0; 384],
            &protos,
        );
        assert!(
            labels.iter().any(|l| l == "topic:language"),
            "Programming query should also emit legacy topic:language, got: {labels:?}"
        );
        // Ambiguous query
        let labels =
            classify_query("What language does Sam know?", &vec![0.0; 384], &protos);
        assert!(
            labels.iter().any(|l| l == "topic:language"),
            "Ambiguous query should also emit legacy topic:language, got: {labels:?}"
        );
    }

    #[test]
    fn query_classification_tier_a_work() {
        let protos = mock_prototypes();
        let labels = classify_query("where do I work?", &vec![0.0; 384], &protos);
        assert!(
            labels.iter().any(|l| l == "domain:work"),
            "Should match 'work' keyword, got: {labels:?}"
        );
    }

    #[test]
    fn classify_empty_embedding_returns_empty() {
        let protos = mock_prototypes();
        let labels = protos.classify(&[]);
        assert!(labels.is_empty());
    }

    #[test]
    fn classify_uninitialized_returns_empty() {
        let protos = LabelPrototypes::new_empty();
        assert!(!protos.is_initialized());
        let labels = protos.classify(&vec![0.1; 384]);
        assert!(labels.is_empty());
    }

    #[test]
    fn prototype_bank_has_rich_descriptions() {
        let (_, descriptions) = prototype_definitions();
        for desc in &descriptions {
            let word_count = desc.split(',').count();
            assert!(
                word_count >= 5,
                "Prototype description too sparse (need >= 5 terms): '{desc}'"
            );
        }
    }

    #[test]
    fn classify_query_editor_keywords_fire_tools_label() {
        let protos = mock_prototypes();
        for kw in &["neovim", "vscode", "text editor", "ide"] {
            let q = format!("what is my {kw} setup");
            let labels = classify_query(&q, &vec![0.0; 384], &protos);
            assert!(
                labels.contains(&"topic:tools:editor".to_string()),
                "Query '{q}' should produce topic:tools:editor, got {labels:?}",
            );
        }
    }

    #[test]
    fn classify_query_editor_also_emits_technology() {
        let protos = mock_prototypes();
        let labels = classify_query("what editor do I use", &vec![0.0; 384], &protos);
        assert!(
            labels.contains(&"topic:tools:editor".to_string()),
            "Should have topic:tools:editor, got {labels:?}",
        );
        assert!(
            labels.contains(&"topic:technology".to_string()),
            "Editor query should also trigger topic:technology, got {labels:?}",
        );
    }
}
