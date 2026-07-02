# ====================================================================================
#                                   SEKCJA KONFIGURACJI
#                                   Plik: Config.py
# ====================================================================================

# Konfiguracja modelu językowego (Llama-cpp-python)
# Słownik zawierający wszystkie parametry do kontroli modelu.
LLM_CONFIG = {
    # Ścieżka do pliku modelu GGUF. Upewnij się, że ścieżka jest poprawna.
    "model_path": "W:/Kobold/Mistral-7B-Instruct-v0.3.Q4_K_S.gguf",
    
    # Liczba warstw modelu do załadowania na GPU.
    # -1: Spróbuje załadować wszystkie warstwy na GPU. To najlepsza opcja, jeśli masz wystarczającą pamięć VRAM.
    # 0: Wszystkie warstwy będą działać na CPU. To rozwiązanie, gdy nie masz GPU lub gdy występuje błąd.
    # 1-32: Określona liczba warstw do załadowania na GPU.
    "n_gpu_layers": -1,
    
    # Temperatura generowania. Kontroluje losowość odpowiedzi.
    # Niższa wartość (np. 0.2) daje bardziej przewidywalne i "bezpieczne" odpowiedzi.
    # Wyższa wartość (np. 0.7) zwiększa kreatywność i różnorodność.
    "temperature": 0.6,
    
    # Maksymalna liczba tokenów w kontekście (w tym prompt + odpowiedź).
    "n_ctx": 4096
}

# Szablony promptów do generowania pytań.
PROMPTS = {
    "general": """[INST]
    Na podstawie poniższego tekstu wygeneruj dokładnie {num_questions} pytań w języku polskim, na które można odpowiedzieć JEDNOZNACZNIE I BEZPOŚREDNIO NA PODSTAWIE PONIŻSZEGO TEKSTU.

    Pytania muszą być tworzone w kolejności:
    1. otwarte  
    2. tak/nie  
    3. z luką   
    ... i tak dalej, aż do {num_questions} pytań.

    Każde pytanie:
    - wypisz w osobnej linii, numerowane (1., 2., 3., ...),
    - musi mieć oznaczenie rodzaju na początku: "(otwarte)", "(tak/nie)" lub "(z luką)",
    - ma być poprawne gramatycznie, sensowne i oparte wyłącznie na tekście,
    - nie może być parafrazą innego pytania,
    - nie może zawierać błędów składniowych ani nieistniejących informacji.

    ---

    ### Zasady tworzenia poszczególnych typów pytań:

    **1. Pytanie otwarte (oznacz "(otwarte)")**  
    Poproś o wyjaśnienie, opis, porównanie lub interpretację czegoś istotnego z tekstu.  
    Stosuj zwroty typu: „Wyjaśnij…”, „Opisz…”, „Jak przebiega…”, „Jakie znaczenie ma…”, „Dlaczego…”.

    **2. Pytanie zamknięte (oznacz "(tak/nie)")**  
    Zadaj krótkie i jednoznaczne pytanie, na które można odpowiedzieć „tak” lub „nie”.  
    Stosuj zwroty: „Czy…”, „Czy zgodnie z tekstem…”, „Czy prawdą jest, że…”.

    **3. Pytanie z luką (oznacz "(z luką)")**  
    Utwórz JEDNO poprawne zdanie OZNAJMUJĄCE, oparte na kluczowej, istotnej informacji z tekstu.  
    Następnie zastąp dokładnie JEDNO ważne i kluczowe słowo lub frazę (np. nazwę, datę, pojęcie, fakt, liczbę, nazwisko) pięcioma podkreśleniami: _____.  
    NIE dawaj luki za nieznaczące słowa (np. artikuly, prepozytywy, spójniki, itp.).
    W zdaniu ZAWSZE MUSI być jedna luka (_____).
    Nie używaj form pytających ani wielokrotnych luk.  
    Zdanie musi być w pełni poprawne gramatycznie.  
    Przykład dla pytania z luką: "(z luką) Bitwa odbyła się w _____ roku", "(z luką) Autor dzieła to _____.", "(z luką) Proces zachodzi w _____ stopniach."

    ---

    ### Tekst źródłowy:
    {text}
    
    ### Dodatkowe wskazówki (jeśli dotyczy):
    {specific_request}
    
    ---

    Wygeneruj gotową, numerowaną listę {num_questions} pytań zgodnych z powyższymi zasadami:
    [/INST]""",

    "hybrid": """[INST]
    Na podstawie poniższego tekstu wygeneruj dokładnie {num_questions} pytań w języku polskim, na które można odpowiedzieć JEDNOZNACZNIE I BEZPOŚREDNIO NA PODSTAWIE PONIŻSZEGO TEKSTU.

    Pytania muszą być tworzone w kolejności:
    1. otwarte  
    2. tak/nie  
    3. z luką   
    ... i tak dalej, aż do {num_questions} pytań.

    Każde pytanie:
    - wypisz w osobnej linii, numerowane (1., 2., 3., ...),
    - musi mieć oznaczenie rodzaju na początku: "(otwarte)", "(tak/nie)" lub "(z luką)",
    - ma być poprawne gramatycznie, sensowne i oparte wyłącznie na tekście,
    - nie może być parafrazą innego pytania,
    - nie może zawierać błędów składniowych ani nieistniejących informacji.

    ---

    ### Zasady tworzenia poszczególnych typów pytań:

    **1. Pytanie otwarte (oznacz "(otwarte)")**  
    Poproś o wyjaśnienie, opis, porównanie lub interpretację czegoś istotnego z tekstu.  
    Stosuj zwroty typu: „Wyjaśnij…”, „Opisz…”, „Jak przebiega…”, „Jakie znaczenie ma…”, „Dlaczego…”.

    **2. Pytanie zamknięte (oznacz "(tak/nie)")**  
    Zadaj krótkie i jednoznaczne pytanie, na które można odpowiedzieć „tak” lub „nie”.  
    Stosuj zwroty: „Czy…”, „Czy zgodnie z tekstem…”, „Czy prawdą jest, że…”.

    **3. Pytanie z luką (oznacz "(z luką)")**  
    Utwórz JEDNO poprawne zdanie OZNAJMUJĄCE, oparte na kluczowej, istotnej informacji z tekstu.  
    Następnie zastąp dokładnie JEDNO ważne i kluczowe słowo lub frazę (np. nazwę, datę, pojęcie, fakt, liczbę, nazwisko) pięcioma podkreśleniami: _____.  
    NIE dawaj luki za nieznaczące słowa (np. artikuly, prepozytywy, spójniki, itp.).
    W zdaniu ZAWSZE MUSI być jedna luka (_____).
    Nie używaj form pytających ani wielokrotnych luk.  
    Zdanie musi być w pełni poprawne gramatycznie.  
    Przykład dla pytania z luką: "(z luką) Bitwa odbyła się w _____ roku", "(z luką) Autor dzieła to _____.", "(z luką) Proces zachodzi w _____ stopniach."

    ---

    ### Tekst źródłowy:
    {text}

    ### Dodatkowe wskazówki (jeśli dotyczy):
    {specific_request}

    ### Frazy pomocnicze lub istotne zdania z tekstu:
    {key_sentences}

    ---

    Wygeneruj gotową, numerowaną listę {num_questions} pytań zgodnych z powyższymi zasadami:
    [/INST]"""
}

# Konfiguracja modelu ekstrakcji informacji (spaCy)
# Polski model językowy spaCy.
SPACY_MODEL = "pl_core_news_md"

# Wzorce wartościowych pytań dla spaCy - rozdzielone według typów
SPACY_QUESTION_PATTERNS = {
    "persName": {
        "open": {
            "władca": ["Opisz rolę {entity} jako przywódcy i jej wpływ na opisywane wydarzenia.", "Wyjaśnij decyzje polityczne {entity} i ich konsekwencje.", "Jakie strategie stosował/a {entity} w zarządzaniu?"],
            "naukowiec": ["Wyjaśnij wkład {entity} w rozwój wiedzy i jego znaczenie.", "Opisz innowacyjność badań {entity} i ich zastosowania.", "Jakie problemy naukowe rozwiązał/a {entity}?"],
            "pisarz": ["Opisz charakterystyczne cechy twórczości {entity}.", "Wyjaśnij tematykę i motywy w dziełach {entity}.", "Jakie innowacje literackie wprowadził/a {entity}?"],
            "wojskowy": ["Opisz strategię wojskową {entity} i jej efektywność.", "Wyjaśnij taktyki stosowane przez {entity} w opisywanych konfliktach.", "Jakie innowacje militarne wprowadził/a {entity}?"],
            "default": ["Opisz wkład {entity} w opisywane wydarzenia i jego znaczenie.", "Wyjaśnij działalność {entity} w kontekście epoki.", "Jakie cele realizował/a {entity}?"]
        },
        "yes_no": {
            "władca": ["Czy działania {entity} przyniosły trwałe zmiany polityczne?", "Czy {entity} skutecznie realizował/a swoje cele polityczne?"],
            "naukowiec": ["Czy odkrycia {entity} zrewolucjonizowały swoją dziedzinę?", "Czy badania {entity} znalazły praktyczne zastosowanie?"],
            "pisarz": ["Czy twórczość {entity} wprowadzała nowe trendy literackie?", "Czy dzieła {entity} miały wpływ na kolejne pokolenia?"],
            "default": ["Czy {entity} odegrał/a kluczową rolę w opisywanych wydarzeniach?", "Czy działalność {entity} miała długofalowe konsekwencje?"]
        }
    },
    "geogName": {
        "open": {
            "bitwa": ["Opisz strategiczne znaczenie miejsca bitwy pod {entity}.", "Wyjaśnij przebieg i konsekwencje bitwy pod {entity}.", "Jakie czynniki zadecydowały o wyniku bitwy pod {entity}?"],
            "miasto": ["Opisz funkcje i znaczenie {entity} jako ośrodka kulturalnego i gospodarczego.", "Wyjaśnij rozwój historyczny {entity} i jego wpływ na region.", "Jakie procesy społeczne zachodziły w {entity}?"],
            "region": ["Opisz charakterystyczne cechy geograficzne i klimatyczne regionu {entity}.", "Wyjaśnij znaczenie geograficzne {entity} dla opisywanych procesów.", "Jakie zjawiska naturalne kształtują region {entity}?"],
            "zjawisko_przyrodnicze": ["Opisz procesy geologiczne i ekologiczne zachodzące w {entity}.", "Wyjaśnij wpływ warunków naturalnych {entity} na opisywane zjawiska.", "Jakie unikalne ekosystemy występują w {entity}?"],
            "default": ["Opisz znaczenie {entity} w kontekście opisywanych wydarzeń.", "Wyjaśnij rolę {entity} w przedstawionej tematyce.", "Jakie procesy zachodziły w {entity}?"]
        },
        "yes_no": {
            "bitwa": ["Czy lokalizacja bitwy pod {entity} miała strategiczne znaczenie?", "Czy bitwa pod {entity} zmieniła układ sił w regionie?"],
            "miasto": ["Czy {entity} było ważnym ośrodkiem kulturalnym i gospodarczym?", "Czy rozwój {entity} wpłynął na okoliczne regiony?"],
            "region": ["Czy region {entity} charakteryzuje się unikalnymi cechami geograficznymi?", "Czy warunki naturalne {entity} wpływają na lokalną społeczność?"],
            "default": ["Czy {entity} odegrało znaczącą rolę w opisywanych procesach?", "Czy {entity} ma szczególne znaczenie w kontekście?"]
        }
    },
    "date": {
        "open": {
            "wojna": ["Opisz wydarzenia wojenne i ich konsekwencje w {entity}.", "Wyjaśnij przyczyny i przebieg konfliktu w {entity}.", "Jakie zmiany polityczne nastąpiły po wydarzeniach z {entity}?"],
            "powstanie": ["Opisz proces powstawania i rozwoju czegoś w {entity}.", "Wyjaśnij okoliczności i znaczenie wydarzeń z {entity}.", "Jakie warunki umożliwiły wydarzenia z {entity}?"],
            "default": ["Opisz kluczowe wydarzenia historyczne z {entity} i ich znaczenie.", "Wyjaśnij przyczyny i konsekwencje wydarzeń z {entity}.", "Jakie procesy kulminowały w {entity}?"]
        },
        "yes_no": {
            "wojna": ["Czy bitwa miała miejsce {entity}?", "Czy konflikt z {entity} zakończył się zwycięstwem?", "Czy w {entity} doszło do istotnej bitwy?", "Czy opisywane wydarzenie wojenne nastąpiło {entity}?"],
            "default": ["Czy opisywane wydarzenia wydarzyły się {entity}?", "Czy {entity} to data ważnego wydarzenia?", "Czy tekst wspomina o wydarzeniach z {entity}?", "Czy w {entity} nastąpiły kluczowe zmiany?"]
        }
    },
    "orgName": {
        "open": {
            "militarna": ["Opisz strukturę organizacyjną i strategię działania {entity}.", "Wyjaśnij rolę militarną {entity} i jej wpływ na opisywane wydarzenia.", "Jakie innowacje organizacyjne wprowadził/o {entity}?"],
            "powstanie": ["Opisz proces powstawania i ewolucję {entity}.", "Wyjaśnij cele i metody działania {entity}.", "Jakie warunki umożliwiły powstanie {entity}?"],
            "default": ["Wyjaśnij funkcje i znaczenie {entity} w opisywanym kontekście.", "Opisz strukturę i działalność {entity}.", "Jakie cele realizował/o {entity}?"]
        },
        "yes_no": {
            "militarna": ["Czy {entity} odegrał/o kluczową rolę w opisywanych konfliktach?", "Czy działalność militarna {entity} była skuteczna?"],
            "default": ["Czy {entity} znacząco wpłynął/o na opisywane wydarzenia?", "Czy {entity} skutecznie realizował/o swoje cele?"]
        }
    },
    "NOUN": {
        "open": {
            "default": [
                "Opisz znaczenie i funkcję '{entity}' w kontekście opisanym w tekście.",
                "Wyjaśnij proces związany z '{entity}' i jego wpływ na opisywany mechanizm.",
                "Jakie cechy lub właściwości posiada '{entity}'?"
            ]
        },
        "yes_no": {
            "default": [
                "Czy '{entity}' jest kluczowym składnikiem dla opisanego procesu?",
                "Czy tekst opisuje '{entity}' jako unikalny/wyjątkowy element?"
            ]
        }
    },
    "ADJ": {
        "open": {
            "default": [
                "Jakie są {entity} cechy opisywanego zjawiska/obiektu?",
                "Wyjaśnij, w jaki sposób {entity} stan/właściwość wpływa na proces."
            ]
        },
        "yes_no": {
            "default": [
                "Czy tekst potwierdza, że opisywane zjawisko jest {entity}?",
                "Czy {entity} jest jedyną właściwością opisywaną w tekście?"
            ]
        }
    }
}

# Dane wejściowe
# Tekst, na podstawie którego będą generowane pytania.
SOURCE_TEXT = """
Równanie kwadratowe, równanie drugiego stopnia[1][2] – rodzaj równania, w którym niewiadoma występuje w drugiej potędze i opcjonalnie też w pierwszej. Zazwyczaj równanie kwadratowe w domyśle ma jedną niewiadomą – wtedy zawsze sprowadza się do postaci:
ax^2 + bx + c = 0 ,   a ≠ 0.
Założenie a ≠ 0 oznacza, że do równań kwadratowych nie zalicza się równań liniowych. Powyższe równanie nie jest jedyną definicją równania kwadratowego o jednej niewiadomej – istnieją też definicje równoważne, ponieważ wyrażenie po lewej zawsze da się przekształcić do innej postaci.
Niewiadoma x i wielkości a, b, c mogą być liczbami rzeczywistymi (R) lub elementami dowolnej innej struktury, w której występują dodawanie i mnożenie. W tym artykule opisano głównie równania kwadratowe o zmiennych rzeczywistych. Jest to standardowy element wykształcenia matematycznego na poziomie średnim; przykładowo równania kwadratowe tego typu znalazły się w podstawie programowej polskich liceów i techników, także w zakresie podstawowym. Równania kwadratowe stosuje się między innymi w geometrii, na przykład planimetrii.
Równania kwadratowe w powyższym sensie mają uogólnienia opisane w odpowiedniej sekcji.
"""

# Parametry generowania pytań
# Liczba pytań do wygenerowania przez modele językowe.
NUM_QUESTIONS = 9

# Minimalna liczba tokenów, jakie model powinien móc wygenerować.
MIN_OUTPUT_TOKENS = 32

# Opcjonalny górny limit generacji tokenów (nie obowiązkowy)
MAX_OUTPUT_TOKENS = 1024

# Margines bezpieczeństwa, aby uniknąć przekroczenia limitu tokenów modelu.
SAFETY_MARGIN = 32

# Opcjonalne zapytania dla danych modelów
OPTIONAL_QUERY_GENERAL = "" # Przykład: "Uwzględnij pytania o postacie historyczne."
OPTIONAL_QUERY_HYBRID = ""  # Przykład: "Generuj same pytania typu 'uzupełnianie luki'."

def print_configuration():
    """Wypisuje aktualną konfigurację programu."""
    print("="*40)
    print(" " * 8 + "AKTUALNA KONFIGURACJA PROGRAMU")
    print("="*40)
    print("\n[Konfiguracja LLM]")
    print(f"Ścieżka modelu: {LLM_CONFIG['model_path']}")
    print(f"Warstwy GPU (n_gpu_layers): {LLM_CONFIG['n_gpu_layers']}")
    print(f"Temperatura: {LLM_CONFIG['temperature']}")
    print(f"Wielkość kontekstu: {LLM_CONFIG['n_ctx']}")
    print(f"MIN_OUTPUT_TOKENS: {MIN_OUTPUT_TOKENS}")
    print(f"MAX_OUTPUT_TOKENS: {MAX_OUTPUT_TOKENS}")
    print(f"\n[Konfiguracja spaCy]")
    print(f"Model: {SPACY_MODEL}")
    print(f"\n[Generowanie pytań]")
    print(f"Tekst źródłowy: '{SOURCE_TEXT.strip()[:70]}...'")
    print(f"Liczba pytań do wygenerowania: {NUM_QUESTIONS}")
    print(f"Opcjonalne zapytania dla metody GGUF: {OPTIONAL_QUERY_GENERAL}")
    print(f"Opcjonalne zapytania dla podejścia hybrydowego: {OPTIONAL_QUERY_HYBRID}")
    print("\n" + "="*40 + "\n")