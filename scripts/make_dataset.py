"""Generate a diverse multi-hop QA dataset in the QAExample format.

Each item provides its own self-contained context, so the gold answer is always derivable
from the passages (the agent answers from context, RAG-style). We deliberately mix easy
2-hop chains, medium chains with a distractor passage, and hard chains (3-hop or with two
distractors / indirect phrasing) so that a small local model fails on a meaningful fraction
-- which is exactly what exercises the Reflexion loop and produces real failure modes.

Usage:
    python scripts/make_dataset.py --out data/multihop_eval.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# --- Curated, internally-consistent chains -------------------------------------------------
# (entity, intermediate, answer)

# person -> birth city -> river through that city
PERSON_CITY_RIVER = [
    ("Ada Lovelace", "London", "River Thames"),
    ("Marie Curie", "Warsaw", "Vistula"),
    ("Wolfgang Amadeus Mozart", "Salzburg", "Salzach"),
    ("Ludwig van Beethoven", "Bonn", "Rhine"),
    ("Galileo Galilei", "Pisa", "Arno"),
    ("Charles Darwin", "Shrewsbury", "River Severn"),
    ("William Shakespeare", "Stratford-upon-Avon", "River Avon"),
    ("Johannes Brahms", "Hamburg", "Elbe"),
    ("Karl Marx", "Trier", "Moselle"),
    ("George Frideric Handel", "Halle", "Saale"),
    ("Antonio Gaudi", "Reus", "Francoli"),
    ("Rene Descartes", "La Haye en Touraine", "Creuse"),
    ("Voltaire", "Paris", "Seine"),
    ("Johann Wolfgang von Goethe", "Frankfurt", "Main"),
    ("Niccolo Machiavelli", "Florence", "Arno"),
    ("Hans Christian Andersen", "Odense", "Odense River"),
]

# capital -> country -> bordering body of water
CAPITAL_COUNTRY_WATER = [
    ("Lima", "Peru", "Pacific Ocean"),
    ("Kathmandu", "Nepal", "no ocean (Nepal is landlocked)"),
    ("Lisbon", "Portugal", "Atlantic Ocean"),
    ("Cairo", "Egypt", "Mediterranean Sea"),
    ("Hanoi", "Vietnam", "South China Sea"),
    ("Santiago", "Chile", "Pacific Ocean"),
    ("Athens", "Greece", "Aegean Sea"),
    ("Oslo", "Norway", "North Sea"),
    ("Tokyo", "Japan", "Pacific Ocean"),
    ("Havana", "Cuba", "Caribbean Sea"),
    ("Stockholm", "Sweden", "Baltic Sea"),
    ("Tripoli", "Libya", "Mediterranean Sea"),
    ("Amman", "Jordan", "Dead Sea"),
    ("Wellington", "New Zealand", "Pacific Ocean"),
    ("Dakar", "Senegal", "Atlantic Ocean"),
    ("Manila", "Philippines", "Pacific Ocean"),
]

# country -> official language -> language family
COUNTRY_LANG_FAMILY = [
    ("Brazil", "Portuguese", "Romance"),
    ("Iran", "Persian", "Indo-Iranian"),
    ("Finland", "Finnish", "Uralic"),
    ("Hungary", "Hungarian", "Uralic"),
    ("Vietnam", "Vietnamese", "Austroasiatic"),
    ("Thailand", "Thai", "Kra-Dai"),
    ("Turkey", "Turkish", "Turkic"),
    ("Romania", "Romanian", "Romance"),
    ("Greece", "Greek", "Hellenic"),
    ("Israel", "Hebrew", "Semitic"),
    ("Poland", "Polish", "Slavic"),
    ("India", "Hindi", "Indo-Aryan"),
    ("Ethiopia", "Amharic", "Semitic"),
    ("Mongolia", "Mongolian", "Mongolic"),
    ("Wales", "Welsh", "Celtic"),
    ("Indonesia", "Indonesian", "Austronesian"),
]

# country -> highest peak -> mountain range
COUNTRY_PEAK_RANGE = [
    ("Nepal", "Mount Everest", "Himalayas"),
    ("Argentina", "Aconcagua", "Andes"),
    ("Tanzania", "Mount Kilimanjaro", "Eastern Rift mountains"),
    ("France", "Mont Blanc", "Alps"),
    ("United States", "Denali", "Alaska Range"),
    ("Russia", "Mount Elbrus", "Caucasus"),
    ("Japan", "Mount Fuji", "Fuji Volcanic Zone"),
    ("Pakistan", "K2", "Karakoram"),
    ("Spain", "Teide", "Canary Islands volcanic range"),
    ("Indonesia", "Puncak Jaya", "Sudirman Range"),
    ("Kenya", "Mount Kenya", "Central Highlands"),
    ("Italy", "Monte Bianco", "Alps"),
    ("Iran", "Mount Damavand", "Alborz"),
    ("Morocco", "Toubkal", "Atlas Mountains"),
    ("Chile", "Ojos del Salado", "Andes"),
    ("Switzerland", "Monte Rosa", "Alps"),
]

# musical work -> composer -> main instrument
WORK_COMPOSER_INSTRUMENT = [
    ("The Four Seasons", "Antonio Vivaldi", "violin"),
    ("Clair de Lune", "Claude Debussy", "piano"),
    ("The Well-Tempered Clavier", "Johann Sebastian Bach", "organ"),
    ("Trumpet Voluntary", "Jeremiah Clarke", "organ"),
    ("Rhapsody in Blue", "George Gershwin", "piano"),
    ("Hungarian Rhapsodies", "Franz Liszt", "piano"),
    ("Caprice No. 24", "Niccolo Paganini", "violin"),
    ("The Cello Suites", "Johann Sebastian Bach", "organ"),
    ("Moonlight Sonata", "Ludwig van Beethoven", "piano"),
    ("Bolero", "Maurice Ravel", "piano"),
    ("The Swan", "Camille Saint-Saens", "organ"),
    ("Goldberg Variations", "Johann Sebastian Bach", "organ"),
]

# book -> author -> author's university (where they taught/studied)
BOOK_AUTHOR_UNIVERSITY = [
    ("The Hobbit", "J. R. R. Tolkien", "Oxford University"),
    ("Lolita", "Vladimir Nabokov", "Cornell University"),
    ("A Brief History of Time", "Stephen Hawking", "University of Cambridge"),
    ("The Name of the Rose", "Umberto Eco", "University of Bologna"),
    ("On the Origin of Species", "Charles Darwin", "University of Cambridge"),
    ("The Gulag Archipelago", "Aleksandr Solzhenitsyn", "Rostov State University"),
    ("Narrative of the Life", "Frederick Douglass", "no university (self-taught)"),
    ("Things Fall Apart", "Chinua Achebe", "University of Ibadan"),
    ("Beloved", "Toni Morrison", "Howard University"),
    ("The Structure of Scientific Revolutions", "Thomas Kuhn", "Harvard University"),
    ("Cosmos", "Carl Sagan", "Cornell University"),
    ("Silent Spring", "Rachel Carson", "Johns Hopkins University"),
]

# company -> founder -> founder's birth country
COMPANY_FOUNDER_COUNTRY = [
    ("Tesla", "Elon Musk", "South Africa"),
    ("Microsoft", "Bill Gates", "United States"),
    ("IKEA", "Ingvar Kamprad", "Sweden"),
    ("Alibaba", "Jack Ma", "China"),
    ("SpaceX", "Elon Musk", "South Africa"),
    ("Spotify", "Daniel Ek", "Sweden"),
    ("Huawei", "Ren Zhengfei", "China"),
    ("Ferrari", "Enzo Ferrari", "Italy"),
    ("Chanel", "Coco Chanel", "France"),
    ("Sony", "Masaru Ibuka", "Japan"),
    ("Nintendo", "Fusajiro Yamauchi", "Japan"),
    ("LEGO", "Ole Kirk Christiansen", "Denmark"),
]

# landmark -> city -> country
LANDMARK_CITY_COUNTRY = [
    ("the Colosseum", "Rome", "Italy"),
    ("the Brandenburg Gate", "Berlin", "Germany"),
    ("the Sagrada Familia", "Barcelona", "Spain"),
    ("Christ the Redeemer", "Rio de Janeiro", "Brazil"),
    ("the Charles Bridge", "Prague", "Czech Republic"),
    ("the Blue Mosque", "Istanbul", "Turkey"),
    ("the Burj Khalifa", "Dubai", "United Arab Emirates"),
    ("the Acropolis", "Athens", "Greece"),
    ("the Little Mermaid statue", "Copenhagen", "Denmark"),
    ("Table Mountain", "Cape Town", "South Africa"),
    ("the Petronas Towers", "Kuala Lumpur", "Malaysia"),
    ("the Atomium", "Brussels", "Belgium"),
]

DISTRACTORS = [
    ("Trivia", "The Nile is the longest river in Africa and flows through Cairo."),
    ("Trivia", "The Amazon River carries more water than any other river on Earth."),
    ("Trivia", "Mount Kilimanjaro is the highest free-standing mountain in the world."),
    ("Trivia", "Latin is an extinct Italic language that influenced many Romance languages."),
    ("Trivia", "The piano was invented by Bartolomeo Cristofori around 1700."),
    ("Trivia", "The Pacific is the largest and deepest of Earth's oceans."),
    ("Trivia", "Harvard University is the oldest university in the United States."),
    ("Trivia", "Apple was co-founded by Steve Jobs and Steve Wozniak in California."),
]


def _difficulty(i: int) -> str:
    r = i % 3
    return {0: "easy", 1: "medium", 2: "hard"}[r]


def _add_distractors(rng: random.Random, ctx: list[dict], difficulty: str) -> list[dict]:
    n = {"easy": 0, "medium": 1, "hard": 2}[difficulty]
    if n:
        ctx = ctx + [{"title": t, "text": tx} for t, tx in rng.sample(DISTRACTORS, n)]
        rng.shuffle(ctx)
    return ctx


def build() -> list[dict]:
    rng = random.Random(42)
    items: list[dict] = []
    counter = 0

    def emit(question: str, gold: str, ctx: list[dict]) -> None:
        nonlocal counter
        diff = _difficulty(counter)
        items.append({
            "qid": f"mh{counter + 1:03d}",
            "difficulty": diff,
            "question": question,
            "gold_answer": gold,
            "context": _add_distractors(rng, ctx, diff),
        })
        counter += 1

    for person, city, river in PERSON_CITY_RIVER:
        emit(
            f"What river flows through the city where {person} was born?",
            river,
            [{"title": person, "text": f"{person} was born in {city}."},
             {"title": city, "text": f"The {river} flows through {city}."}],
        )
    for cap, country, water in CAPITAL_COUNTRY_WATER:
        emit(
            f"Which body of water borders the country whose capital is {cap}?",
            water,
            [{"title": cap, "text": f"{cap} is the capital city of {country}."},
             {"title": country, "text": f"{country} borders the {water}." if "no ocean" not in water else f"{country} is landlocked and borders no ocean."}],
        )
    for country, lang, family in COUNTRY_LANG_FAMILY:
        emit(
            f"Which language family does the official language of {country} belong to?",
            family,
            [{"title": country, "text": f"The official language of {country} is {lang}."},
             {"title": lang, "text": f"{lang} is a {family} language."}],
        )
    for country, peak, rng_name in COUNTRY_PEAK_RANGE:
        emit(
            f"Which mountain range contains the highest mountain in {country}?",
            rng_name,
            [{"title": country, "text": f"The highest mountain in {country} is {peak}."},
             {"title": peak, "text": f"{peak} is part of the {rng_name}."}],
        )
    for work, composer, instr in WORK_COMPOSER_INSTRUMENT:
        emit(
            f"What instrument did the composer of {work} mainly play?",
            instr,
            [{"title": work, "text": f"{work} was composed by {composer}."},
             {"title": composer, "text": f"{composer} mainly played the {instr}."}],
        )
    for book, author, uni in BOOK_AUTHOR_UNIVERSITY:
        emit(
            f"At which university did the author of {book} study or teach?",
            uni,
            [{"title": book, "text": f"{book} was written by {author}."},
             {"title": author, "text": f"{author} was associated with {uni}." if "no university" not in uni else f"{author} had no formal university education."}],
        )
    for company, founder, country in COMPANY_FOUNDER_COUNTRY:
        emit(
            f"In which country was the founder of {company} born?",
            country,
            [{"title": company, "text": f"{company} was founded by {founder}."},
             {"title": founder, "text": f"{founder} was born in {country}."}],
        )
    for landmark, city, country in LANDMARK_CITY_COUNTRY:
        emit(
            f"In which country is {landmark} located?",
            country,
            [{"title": landmark, "text": f"{landmark} is located in {city}."},
             {"title": city, "text": f"{city} is a city in {country}."}],
        )

    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/multihop_eval.json")
    args = ap.parse_args()
    items = build()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(items)} examples to {args.out}")


if __name__ == "__main__":
    main()
