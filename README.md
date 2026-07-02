## Opis projektu

Projekt realizuje system generowania pytań edukacyjnych na podstawie tekstu źródłowego z wykorzystaniem technik sztucznej inteligencji oraz przetwarzania języka naturalnego (NLP).

Celem systemu jest analiza i porównanie skuteczności różnych podejść do automatycznego tworzenia pytań dydaktycznych, w szczególności:

- modeli językowych (LLM – Mistral 7B via `llama.cpp`)
- ekstrakcji informacji (spaCy)
- podejścia hybrydowego łączącego oba mechanizmy

System generuje pytania w trzech formach:
- pytania otwarte
- pytania zamknięte (tak/nie)
- pytania z luką

Każde pytanie jest tworzone wyłącznie na podstawie dostarczonego tekstu i ma na celu zachowanie jednoznaczności oraz poprawności merytorycznej.

Projekt został wykorzystany w badaniu eksperymentalnym porównującym jakość generowanych pytań przy użyciu analizy statystycznej (ANOVA oraz test NIR), gdzie oceniano:
- poprawność merytoryczną odpowiedzi
- różnorodność typów pytań

---

## Architektura systemu

System składa się z dwóch głównych modułów:

### 1. `Config.py`
Plik konfiguracyjny zawierający:

- konfigurację modelu LLM (llama.cpp / GGUF)
- parametry generacji (temperatura, kontekst, tokeny)
- szablony promptów dla:
  - podejścia czysto LLM
  - podejścia hybrydowego
- model spaCy (`pl_core_news_md`)
- reguły i wzorce generowania pytań oparte o:
  - encje (osoby, organizacje, daty, miejsca)
  - części mowy (NOUN, ADJ)
- tekst źródłowy wykorzystywany w eksperymentach
- globalne parametry eksperymentu (liczba pytań, limity tokenów)

Dodatkowo zawiera funkcję `print_configuration()` umożliwiającą prezentację aktualnych ustawień systemu.

---

### 2. `Program.py`
Główny moduł aplikacji odpowiedzialny za:

#### Model LLM (GGUF / llama.cpp)
- generowanie pytań na podstawie promptów
- kontrolę długości kontekstu i tokenów
- bezpieczne zarządzanie limitem generacji

#### spaCy NLP pipeline
- ekstrakcję encji i tokenów znaczeniowych
- klasyfikację kontekstu (np. władca, naukowiec, bitwa)
- generowanie pytań na podstawie wzorców semantycznych

#### Podejście hybrydowe
- spaCy generuje surowe pytania
- LLM poprawia ich jakość i spójność
- końcowe pytania są standaryzowane i uporządkowane

#### Mechanizmy wspólne
- ekstrakcja pytań z odpowiedzi modelu
- usuwanie błędów numeracji
- kontrola długości tekstu względem okna kontekstu
- walidacja i filtrowanie wyników

## Wymagania środowiskowe

Python 3.10+
- biblioteki:
  - llama-cpp-python
  - spacy
  - model spaCy: pl_core_news_md
- model GGUF (np. użyty Mistral 7B Instruct)
