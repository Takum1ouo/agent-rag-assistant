"""
Retrieval comparison experiment.

Each of the 68 Wikipedia articles has one natural-language question, labelled with the
article its answer should come from (e.g. "What is a black hole?" should hit
"Black hole.txt"). The 68 questions are run through vector-only retrieval, BM25-only
retrieval and hybrid retrieval (RRF), and we check whether the correct file appears among
each method's top-k results.

This script is the source of the vector vs BM25 vs hybrid comparison numbers.

Limitations: the evaluation set is hand-built "weak labelling", not annotated by a third
party and not cross-checked by multiple annotators; the criterion is "source file of a
retrieved chunk == labelled file", i.e. file-level correctness only, not whether the
specific most relevant passage was retrieved.

Also: the 68 articles differ a lot in topic, so at the lenient threshold k=4 (top four
results) all three methods reach 100% and cannot be told apart. That is not because they
are equally good but because these questions are too easy for retrieval (a ceiling
effect). Tightening the threshold to k=1 (the correct file must be ranked first) reveals
the differences, so the script reports both k=1 and k=4: k=1 separates the methods, and
k=4 matches the setting used by the question-answering script.

Usage (after `conda activate agent-rag`, from the project root): `python src/eval_retrieval.py`
"""

from pathlib import Path

from retrieval import Retriever

TOP_K = 4

EVAL_SET = [
    {"question": "How tall is Mount Everest?", "expected_source": "Mount Everest.txt"},
    {"question": "Why is the Amazon rainforest important for the planet?", "expected_source": "Amazon rainforest.txt"},
    {"question": "What is the Sahara desert like?", "expected_source": "Sahara.txt"},
    {"question": "Which countries does the Nile river flow through?", "expected_source": "Nile.txt"},
    {"question": "What is the Great Barrier Reef?", "expected_source": "Great Barrier Reef.txt"},
    {"question": "How was the Grand Canyon formed?", "expected_source": "Grand Canyon.txt"},
    {"question": "What is Antarctica like?", "expected_source": "Antarctica.txt"},
    {"question": "How large is the Pacific Ocean?", "expected_source": "Pacific Ocean.txt"},
    {"question": "Why was the Great Wall of China built?", "expected_source": "Great Wall of China.txt"},
    {"question": "Who designed the Eiffel Tower?", "expected_source": "Eiffel Tower.txt"},
    {"question": "What does the Statue of Liberty symbolize?", "expected_source": "Statue of Liberty.txt"},
    {"question": "What was the Colosseum used for in ancient Rome?", "expected_source": "Colosseum.txt"},
    {"question": "Who built the Taj Mahal and why?", "expected_source": "Taj Mahal.txt"},
    {"question": "What civilization built Machu Picchu?", "expected_source": "Machu Picchu.txt"},
    {"question": "Is Mount Fuji an active volcano?", "expected_source": "Mount Fuji.txt"},
    {"question": "What are the pyramids of ancient Egypt used for?", "expected_source": "Ancient Egypt.txt"},
    {"question": "How did the Roman Empire fall?", "expected_source": "Roman Empire.txt"},
    {"question": "When did World War II start and end?", "expected_source": "World War II.txt"},
    {"question": "What caused World War I?", "expected_source": "World War I.txt"},
    {"question": "What caused the French Revolution?", "expected_source": "French Revolution.txt"},
    {"question": "What changed during the Industrial Revolution?", "expected_source": "Industrial Revolution.txt"},
    {"question": "What was the Renaissance period known for?", "expected_source": "Renaissance.txt"},
    {"question": "What was the Cold War between?", "expected_source": "Cold War.txt"},
    {"question": "What caused the Great Depression?", "expected_source": "Great Depression.txt"},
    {"question": "What was traded along the Silk Road?", "expected_source": "Silk Road.txt"},
    {"question": "How many planets are in the Solar System?", "expected_source": "Solar System.txt"},
    {"question": "What is a black hole?", "expected_source": "Black hole.txt"},
    {"question": "Who discovered the law of gravity?", "expected_source": "Gravity.txt"},
    {"question": "How does photosynthesis work?", "expected_source": "Photosynthesis.txt"},
    {"question": "What is DNA made of?", "expected_source": "DNA.txt"},
    {"question": "Who proposed the theory of evolution?", "expected_source": "Evolution.txt"},
    {"question": "Who created the periodic table?", "expected_source": "Periodic table.txt"},
    {"question": "What is the Big Bang theory?", "expected_source": "Big Bang.txt"},
    {"question": "How fast is the speed of light?", "expected_source": "Speed of light.txt"},
    {"question": "How is electricity generated?", "expected_source": "Electricity.txt"},
    {"question": "What is nuclear fusion?", "expected_source": "Nuclear fusion.txt"},
    {"question": "What causes plate tectonics?", "expected_source": "Plate tectonics.txt"},
    {"question": "How does a volcano erupt?", "expected_source": "Volcano.txt"},
    {"question": "What causes an earthquake?", "expected_source": "Earthquake.txt"},
    {"question": "What causes climate change?", "expected_source": "Climate change.txt"},
    {"question": "What are the main parts of the human brain?", "expected_source": "Human brain.txt"},
    {"question": "How does the immune system protect the body?", "expected_source": "Immune system.txt"},
    {"question": "How big is a blue whale?", "expected_source": "Blue whale.txt"},
    {"question": "What do giant pandas eat?", "expected_source": "Giant panda.txt"},
    {"question": "Why are coral reefs important?", "expected_source": "Coral reef.txt"},
    {"question": "How do bees make honey?", "expected_source": "Bee.txt"},
    {"question": "What do sharks eat?", "expected_source": "Shark.txt"},
    {"question": "When did dinosaurs go extinct?", "expected_source": "Dinosaur.txt"},
    {"question": "How does the internet work?", "expected_source": "Internet.txt"},
    {"question": "Who invented the printing press?", "expected_source": "Printing press.txt"},
    {"question": "Who invented the telephone?", "expected_source": "Telephone.txt"},
    {"question": "Who invented the airplane?", "expected_source": "Airplane.txt"},
    {"question": "Who invented the automobile?", "expected_source": "Automobile.txt"},
    {"question": "Who invented the steam engine?", "expected_source": "Steam engine.txt"},
    {"question": "How do vaccines work?", "expected_source": "Vaccine.txt"},
    {"question": "How do antibiotics work?", "expected_source": "Antibiotic.txt"},
    {"question": "How does a nuclear power plant work?", "expected_source": "Nuclear power.txt"},
    {"question": "What are examples of renewable energy?", "expected_source": "Renewable energy.txt"},
    {"question": "Where did the Olympic Games originate?", "expected_source": "Olympic Games.txt"},
    {"question": "What does the World Health Organization do?", "expected_source": "World Health Organization.txt"},
    {"question": "When was the United Nations founded?", "expected_source": "United Nations.txt"},
    {"question": "Who created the Nobel Prize?", "expected_source": "Nobel Prize.txt"},
    {"question": "What did Leonardo da Vinci paint?", "expected_source": "Leonardo da Vinci.txt"},
    {"question": "What is Albert Einstein famous for?", "expected_source": "Albert Einstein.txt"},
    {"question": "What did Isaac Newton discover?", "expected_source": "Isaac Newton.txt"},
    {"question": "What theory is Charles Darwin known for?", "expected_source": "Charles Darwin.txt"},
    {"question": "What plays did William Shakespeare write?", "expected_source": "William Shakespeare.txt"},
    {"question": "Who painted the Mona Lisa?", "expected_source": "Mona Lisa.txt"},
]


def hit(chunks, expected_source: str) -> bool:
    sources = {Path(c.metadata.get("source", "")).name for c in chunks}
    return expected_source in sources


def run_eval(retriever: Retriever, method_name: str, search_fn, k: int):
    hits = 0
    misses = []
    for item in EVAL_SET:
        chunks = search_fn(item["question"], k=k)
        if hit(chunks, item["expected_source"]):
            hits += 1
        else:
            misses.append(item)
    rate = hits / len(EVAL_SET)
    print(f"{method_name:16s}  hits {hits}/{len(EVAL_SET)}  =  {rate:.1%}")
    return rate, misses


def main():
    print("Building the index (vector store + BM25); the first run may take a moment...")
    retriever = Retriever()
    print(f"\n{len(EVAL_SET)} evaluation questions\n")

    print(f"--- k=1 (strict: the correct file must rank first; best at showing differences) ---")
    vec_rate, _ = run_eval(retriever, "vector-only", retriever.vector_search, k=1)
    bm25_rate, _ = run_eval(retriever, "BM25-only", retriever.bm25_search, k=1)
    hybrid_rate, hybrid_misses = run_eval(retriever, "hybrid (RRF)", retriever.hybrid_search, k=1)
    print(f"hybrid vs vector-only: {hybrid_rate - vec_rate:+.1%}    hybrid vs BM25-only: {hybrid_rate - bm25_rate:+.1%}")

    print(f"\n--- k={TOP_K} (the setting used by query.py) ---")
    run_eval(retriever, "vector-only", retriever.vector_search, k=TOP_K)
    run_eval(retriever, "BM25-only", retriever.bm25_search, k=TOP_K)
    run_eval(retriever, "hybrid (RRF)", retriever.hybrid_search, k=TOP_K)

    print("\n--- Questions the hybrid method still misses at k=1 (if any) ---")
    for item in hybrid_misses:
        print(f"- {item['question']}  (expected source: {item['expected_source']})")


if __name__ == "__main__":
    main()
