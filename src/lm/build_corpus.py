"""Generate the KenLM training corpus: one Russian sentence per line for every number 1000–999999."""
from num2words import num2words

for n in range(1_000, 1_000_000):
    print(num2words(n, lang="ru"))
