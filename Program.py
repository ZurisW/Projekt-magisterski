# Program.py -- Zaktualizowana i zoptymalizowana wersja
from llama_cpp import Llama
import spacy
import time
import random
import re # Dodaj import re

# Importowanie wszystkich zmiennych i funkcji z pliku konfiguracyjnego
from Config import (
    LLM_CONFIG,
    PROMPTS,
    SPACY_MODEL,
    SOURCE_TEXT,
    NUM_QUESTIONS,
    OPTIONAL_QUERY_GENERAL,
    OPTIONAL_QUERY_HYBRID,
    SAFETY_MARGIN,
    MIN_OUTPUT_TOKENS,
    MAX_OUTPUT_TOKENS,
    SPACY_QUESTION_PATTERNS,
    print_configuration
)

class QuestionGenerator:
    def __init__(self, llm_model, spacy_model, context_window, safety_margin, max_output_tokens):
        self.llm = llm_model
        self.nlp = spacy_model
        self.context_window = context_window
        self.safety_margin = safety_margin
        self.max_output_tokens = max_output_tokens

    def _trim_text_to_fit(self, text, prompt_template_token_count, additional_content=""):
        """
        Przycinanie tekstu, aby zmieścił się w oknie kontekstu.
        """
        doc = self.nlp(text)
        sents = list(doc.sents)
        current_text = text

        additional_tokens = len(self.llm.tokenize(additional_content.encode("utf-8")))

        while True:
            temp_prompt = "Pytania: " + current_text + " "
            total_tokens = (
                len(self.llm.tokenize(temp_prompt.encode("utf-8"))) 
                + prompt_template_token_count 
                + additional_tokens
                + self.safety_margin 
                + MIN_OUTPUT_TOKENS
            )

            if total_tokens <= self.context_window:
                return current_text

            if len(sents) > 1:
                sents.pop()
                current_text = " ".join([sent.text.strip() for sent in sents])
            else:
                raise ValueError("Pojedyncze zdanie tekstu źródłowego jest zbyt długie, aby zmieścić się w oknie kontekstu.")

    def _compute_allowed_output(self, prompt_len):
        """
        Oblicza, ile tokenów można bezpiecznie wygenerować.
        """
        available = self.context_window - prompt_len - self.safety_margin
        if self.max_output_tokens:
            allowed = min(self.max_output_tokens, available)
        else:
            allowed = available
        return int(allowed)

    def _safe_completion(self, prompt):
        """
        Wywołuje create_completion z bezpiecznym max_tokens.
        """
        prompt_len = len(self.llm.tokenize(prompt.encode("utf-8")))
        allowed = self._compute_allowed_output(prompt_len)

        if allowed < MIN_OUTPUT_TOKENS:
            raise ValueError(
                f"Brak wystarczającej liczby tokenów dla odpowiedzi. "
                f"Dostępne miejsce: {self.context_window - prompt_len - self.safety_margin} tokenów. "
                f"MIN_OUTPUT_TOKENS = {MIN_OUTPUT_TOKENS}. Skróć tekst wejściowy lub zwiększ ustawienia."
            )

        # Dla metody hybrydowej usuń "Pytania:" ze stop tokenów
        stop_tokens = ["\n\n", "Pytania:", "Pytanie:"]
        if "**SUROWE PYTANIA DO POPRAWY (Wzorce dla LLM):**" in prompt:  # Wymuś bardziej precyzyjne zakończenie
            stop_tokens = ["\n\n\n", "Pytanie:"]
        
        out = self.llm.create_completion(
            prompt,
            max_tokens=allowed,
            temperature=LLM_CONFIG.get("temperature", 0.7),
            stop=stop_tokens
        )
        return out
    
    def _extract_numbered_questions(self, text, num_questions):
        """
        Wyciąga linie zaczynające się od numerów (1., 2., 3., ...) lub myślników (-), ignoruje puste linie.
        Usuwa duplikację numeracji i zwraca listę pytań do num_questions.
        """
        questions = []
        question_counter = 1
        
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            # Sprawdź czy linia zaczyna się od numeru
            is_numbered = len(line) > 2 and line[0].isdigit() and line[1] == '.'
            is_dash = line.startswith('-')
            
            if is_numbered:
                # --- ULEPSZONE USUWANIE DUPLIKACJI NUMERACJI ---
                clean_line = line
                
                # Próbuj dopasować wzorzec "X. Y. ..." lub "X. X. ..."
                try:
                    first_dot_index = line.find('.')
                    if first_dot_index != -1:
                        after_first_dot = line[first_dot_index + 1:].lstrip()
                        if after_first_dot and after_first_dot[0].isdigit():
                            second_dot_index = after_first_dot.find('.')
                            if second_dot_index != -1:
                                # Usuń drugą numerację i powtarzającą się spację
                                question_text = after_first_dot[second_dot_index + 1:].lstrip()
                                clean_line = f"{line[:first_dot_index + 1]} {question_text}"
                except:
                    clean_line = line
                # --- KONIEC ZMIANY ---
                
                # Usuń ewentualną numerację na początku, jeśli została źle sparsowana
                # Wyszukaj i usuń powtórzoną numerację (np. "1. 1. Pytanie")
                match_duplicate = re.match(r"^(\d+\.\s+)(\d+\.\s+)(.*)", clean_line)
                if match_duplicate:
                    clean_line = f"{match_duplicate.group(1)}{match_duplicate.group(3)}"
                
                # Upewnij się, że numeracja jest poprawna dla listy
                clean_line_text = clean_line.lstrip('123456789. ')
                clean_line = f"{question_counter}. {clean_line_text}"
                
                questions.append(clean_line)
                question_counter += 1
                
            elif is_dash:
                # Konwertuj myślnik na numerację
                question_text = line[1:].strip()
                clean_line = f"{question_counter}. {question_text}"
                questions.append(clean_line)
                question_counter += 1
                
            if len(questions) >= num_questions:
                break
        
        return questions[:num_questions]

        
    def generate_with_llama_cpp(self, text, num_questions):
        start_time = time.time()
        request_text = f"Uwaga: {OPTIONAL_QUERY_GENERAL}" if OPTIONAL_QUERY_GENERAL else ""
        
        
            
        prompt_template = PROMPTS["general"].format(num_questions=num_questions, text="", specific_request=request_text)
        prompt_template_token_count = len(self.llm.tokenize(prompt_template.encode("utf-8")))
        trimmed_text = self._trim_text_to_fit(text, prompt_template_token_count)
        
        prompt = PROMPTS["general"].format(num_questions=num_questions, text=trimmed_text, specific_request=request_text)
        
        out = self._safe_completion(prompt)
        response = out["choices"][0]["text"].strip()
        print("=== RAW OUTPUT ===")
        print(response)
        print("=== END RAW ===")
        questions = self._extract_numbered_questions(response, num_questions)
        
        exec_time = time.time() - start_time
        return questions[:num_questions], exec_time

    def _get_question_from_patterns(self, ent_text, pattern_key, sent_text, question_type):
        """
        Pobiera wartościowe pytanie z wzorców w Config.
        pattern_key: 'persName', 'geogName' (Ent-Label) lub 'NOUN', 'ADJ' (POS Tag).
        question_type: 'open', 'yes_no'
        """
        if pattern_key not in SPACY_QUESTION_PATTERNS:
            return None
        
        patterns = SPACY_QUESTION_PATTERNS[pattern_key].get(question_type, {})
        if not patterns:
            return None
        
        # --- Ustalanie Kontekstu ---
        context = "default"
        # Logika szczegółowego kontekstu działa tylko dla pierwotnych etykiet encji
        if pattern_key in ["persName", "geogName", "placeName", "date", "orgName"]:
            sent_lower = sent_text.lower()
            
            if pattern_key == "persName":
                if any(word in sent_lower for word in ["król", "książę", "mistrz", "dowódca", "rycerz", "władca", "cesarz"]):
                    context = "władca"
                elif any(word in sent_lower for word in ["odkrył", "wynalazł", "badał", "opracował", "teoria", "badanie"]):
                    context = "naukowiec"
                elif any(word in sent_lower for word in ["napisał", "autor", "poeta", "dzieło", "utwór", "książka"]):
                    context = "pisarz"
                elif any(word in sent_lower for word in ["dowodził", "prowadził", "kierował", "walczył", "bitwa"]):
                    context = "wojskowy"
            
            elif pattern_key in ["geogName", "placeName"]:
                if any(word in sent_lower for word in ["bitwa", "wojna", "stoczona", "walka", "konflikt"]):
                    context = "bitwa"
                elif any(word in sent_lower for word in ["miasto", "stolica", "centrum", "siedziba", "zamek"]):
                    context = "miasto"
                elif any(word in sent_lower for word in ["region", "obszar", "terytorium", "ziemia", "kraj"]):
                    context = "region"
                elif any(word in sent_lower for word in ["klimat", "zjawisko", "proces", "występuje", "ekosystem", "przyroda"]):
                    context = "zjawisko_przyrodnicze"
            
            elif pattern_key == "date":
                if any(word in sent_lower for word in ["bitwa", "bitew", "wojna", "walka", "konflikt", "stoczona", "stoczono", "bojów", "zwycięstw", "pogromu", "oblegał"]):
                    context = "wojna"
                elif any(word in sent_lower for word in ["powstał", "założył", "utworzył", "rozpoczął"]):
                    context = "powstanie"
            
            elif pattern_key == "orgName":
                if any(word in sent_lower for word in ["zakon", "rycerz", "wojsk", "armia"]):
                    context = "militarna"
                elif any(word in sent_lower for word in ["założył", "utworzył", "powstał"]):
                    context = "powstanie"
        
        # Pobierz wzorzec
        if isinstance(patterns, dict):
            pattern_list = patterns.get(context, patterns.get("default", []))
        else:
            pattern_list = patterns
        
        if not pattern_list:
            return None
        
        # Wybierz LOSOWY wzorzec dla różnorodności
        pattern = random.choice(pattern_list)
        question = pattern.format(entity=ent_text)
        
        # Dodaj oznaczenie typu NA POCZĄTKU w nawiasie
        if question_type == "open":
            question = f"(otwarte) {question}"
        elif question_type == "yes_no":
            question = f"(tak/nie) {question}"
        
        return question

    def generate_with_spacy(self, text):
        start_time = time.time()
        doc = self.nlp(text)
        
        # Cel: Równomierny podział, np. 3 otwarte, 3 tak/nie, 3 z luką (jeśli NUM_QUESTIONS = 9)
        TARGET = NUM_QUESTIONS // 3 + (1 if NUM_QUESTIONS % 3 > 0 else 0)
        
        # 1. Tworzenie puli POTENCJALNYCH ELEMENTÓW do generowania pytań (Encje + Kluczowe POS Tags)
        
        potential_items = []
        
        # A. Dodaj standardowe Encje
        for ent in doc.ents:
            # Używamy tylko tych, dla których mamy wzorce (persName, geogName, date, orgName)
            if ent.label_ in SPACY_QUESTION_PATTERNS:
                potential_items.append((ent.text, ent.label_, ent.sent.text))

        # B. Dodaj Kluczowe Tokeny (NOUN/ADJ)
        for sent in doc.sents:
            for token in sent:
                pos_tag = token.pos_
                # Sprawdź, czy POS Tag jest w naszych nowych wzorcach w Config
                if pos_tag in ["NOUN", "ADJ"] and pos_tag in SPACY_QUESTION_PATTERNS:
                    # Dodaj, jeśli token jest dłuższy niż 3 znaki i nie jest stop-wordem
                    if len(token.text) > 3 and not token.is_stop and token.text.isalpha(): # isalpha() żeby wykluczyć liczby
                        # Sprawdź, czy ten token nie jest już częścią większej encji (opcjonalna optymalizacja)
                        if not any(token.text in ent.text for ent in doc.ents if ent.label_ in SPACY_QUESTION_PATTERNS):
                            potential_items.append((token.text, pos_tag, token.sent.text))

        open_questions = []
        yes_no_questions = []
        fill_blank_questions = []
        
        # Zestawy do śledzenia użytych elementów
        used_items_for_open = set()
        used_items_for_yes_no = set()
        
        # 2. Generuj pytania OTWARTYCH i TAK/NIE na podstawie całej puli
        
        # Sortowanie puli, aby najpierw przetwarzać encje, a potem POS Tagi
        # Ent-Label: A, D, G, P, O
        # POS-Tag: N, A
        priority_map = {"persName": 1, "geogName": 1, "placeName": 1, "date": 1, "orgName": 1, "NOUN": 2, "ADJ": 3}
        temp_items = sorted(potential_items, key=lambda x: priority_map.get(x[1], 99))
        random.shuffle(temp_items) # Wprowadzamy losowość po priorytetyzacji
        
        for item_text, item_key, sent_text in temp_items:
            item_lower = item_text.lower()
            
            # Generowanie pytań otwartych
            if len(open_questions) < TARGET and item_lower not in used_items_for_open:
                q = self._get_question_from_patterns(item_text, item_key, sent_text, "open")
                if q:
                    open_questions.append(q)
                    used_items_for_open.add(item_lower)
            
            # Generowanie pytań tak/nie
            if len(yes_no_questions) < TARGET and item_lower not in used_items_for_yes_no:
                q = self._get_question_from_patterns(item_text, item_key, sent_text, "yes_no")
                if q:
                    yes_no_questions.append(q)
                    used_items_for_yes_no.add(item_lower)
            
            # Zatrzymujemy, gdy osiągniemy TARGET dla obu typów
            if len(open_questions) >= TARGET and len(yes_no_questions) >= TARGET:
                break
        
        # 3. Generuj pytania z lukami (agresywnie, do osiągnięcia TARGET)
        
        used_sentences_for_luka = set()
        
        # Przechodzenie przez zdania w losowej kolejności
        temp_sents = list(doc.sents)
        random.shuffle(temp_sents)
        
        for sent in temp_sents:
            if len(fill_blank_questions) >= TARGET:
                break
                
            sent_text = sent.text.strip()
            # Używamy tylko zdania o średniej długości
            words = sent_text.split()
            if len(words) < 6 or len(words) > 20:
                continue
            
            if sent_text in used_sentences_for_luka:
                continue
                
            # Priorytetyzacja tokenów do zamazania
            best_candidates = []
            tokens_list = list(sent)
            
            for idx, token in enumerate(tokens_list):
                # Pomijaj tokeny na początku (idx 0, 1), tokeny krótkie, stop-wordy
                if idx < 2 or len(token.text) < 3 or token.is_stop:
                    continue
                
                # Ulepszone priorytety:
                if token.pos_ in ["NUM", "PROPN"]:
                     best_candidates.append((token, 4))
                elif token.pos_ == "NOUN":
                     best_candidates.append((token, 3))
                elif token.pos_ == "VERB":
                    best_candidates.append((token, 2))
                
            
            best_candidates.sort(key=lambda x: x[1], reverse=True)
            
            if best_candidates:
                best_token = best_candidates[0][0]
                
                if best_token.pos_ not in ["DET", "ADP", "CONJ", "AUX"]:
                    # Poprawiony mechanizm luki (5x _)
                    question_text = sent_text.replace(best_token.text, "_____", 1)
                    
                    # Upewnij się, że kończy się kropką i nie zaczyna się od luki
                    question_text = question_text.rstrip('.,!?') + '.'
                    
                    if not question_text.startswith("_____"):
                        fill_blank_questions.append(f"(z luką) {question_text}")
                        used_sentences_for_luka.add(sent_text)
        
        # 4. Łączenie i mieszanie (interleaving) - WZÓR CYKLICZNY
        
        final_open = open_questions
        final_yes_no = yes_no_questions
        final_fill_blank = fill_blank_questions
        
        questions = []
        i = 0
        while len(questions) < NUM_QUESTIONS:
            added = False
            # 1. Pytanie otwarte
            if i < len(final_open):
                questions.append(final_open[i])
                added = True
            
            # 2. Pytanie tak/nie
            if len(questions) < NUM_QUESTIONS and i < len(final_yes_no):
                questions.append(final_yes_no[i])
                added = True
            
            # 3. Pytanie z luką
            if len(questions) < NUM_QUESTIONS and i < len(final_fill_blank):
                questions.append(final_fill_blank[i])
                added = True
                
            if not added and i >= max(len(final_open), len(final_yes_no), len(final_fill_blank)):
                # Zabrano wszystkie dostępne pytania
                break
            
            i += 1
        
        exec_time = time.time() - start_time
        return questions[:NUM_QUESTIONS], exec_time

    def generate_with_hybrid_approach(self, text, num_questions):
        """Metoda hybrydowa - spaCy generuje surowe pytania (6), LLM je dopracowuje i generuje cyklicznie"""
        start_time = time.time()
        doc = self.nlp(text)
        
        # Generuj większą pulę surowych pytań (np. 2x3=6) dla LLM jako bazę
        spacy_raw_questions = []
        TARGET_RAW = 2 
        
        # 1. Generowanie SUROWYCH pytań otwartych i tak/nie
        key_entities = []
        for ent in doc.ents:
            if ent.label_ in ["persName", "geogName", "placeName", "date", "orgName"]:
                key_entities.append((ent.text, ent.label_, ent.sent.text))

        temp_entities = key_entities[:] 
        random.shuffle(temp_entities)
        
        used_entities = set()
        
        for ent_text, ent_label, sent_text in temp_entities:
            ent_lower = ent_text.lower()
            if ent_lower not in used_entities:
                if len(spacy_raw_questions) < TARGET_RAW: # Pytanie otwarte
                    q = self._get_question_from_patterns(ent_text, ent_label, sent_text, "open")
                    if q:
                        spacy_raw_questions.append(q)
                
                if len(spacy_raw_questions) < 2 * TARGET_RAW: # Pytanie tak/nie
                    q = self._get_question_from_patterns(ent_text, ent_label, sent_text, "yes_no")
                    if q:
                        spacy_raw_questions.append(q)
                
                used_entities.add(ent_lower)
            if len(spacy_raw_questions) >= 2 * TARGET_RAW:
                break
        
        # 2. Generowanie SUROWYCH pytań z lukami
        used_sentences_for_luka = set()
        temp_sents = list(doc.sents)
        random.shuffle(temp_sents)
        
        for sent in temp_sents:
            if len(spacy_raw_questions) >= 3 * TARGET_RAW:
                break
            
            sent_text = sent.text.strip()
            words = sent_text.split()
            if len(words) < 6 or len(words) > 20 or sent_text in used_sentences_for_luka:
                continue
                
            best_candidates = []
            tokens_list = list(sent)
            
            for idx, token in enumerate(tokens_list):
                if idx < 2 or len(token.text) < 3 or token.is_stop:
                    continue
                if token.pos_ in ["NUM", "PROPN"]:
                     best_candidates.append((token, 4))
                elif token.pos_ == "NOUN":
                     best_candidates.append((token, 3))
            
            best_candidates.sort(key=lambda x: x[1], reverse=True)
            
            if best_candidates and best_candidates[0][0].pos_ not in ["DET", "ADP", "CONJ", "AUX"]:
                best_token = best_candidates[0][0]
                question_text = sent_text.replace(best_token.text, "_____", 1)
                question_text = question_text.rstrip('.,!?') + '.'
                
                if not question_text.startswith("_____"):
                    spacy_raw_questions.append(f"(z luką) {question_text}")
                    used_sentences_for_luka.add(sent_text)
                    
        # 3. Przygotuj prompt - z POPRAWIONĄ instrukcją dla LLM (CYKLICZNE PRZEPLATANIE)
        key_sentences = []
        for sent in doc.sents:
            if len(sent.ents) > 0 and 6 <= len(sent.text.split()) <= 18:
                key_sentences.append(sent.text.strip())
        
        key_sentences_str = "\n".join(key_sentences[:5])
        request_text = f"Uwaga: {OPTIONAL_QUERY_HYBRID}" if OPTIONAL_QUERY_HYBRID else ""
        
        if spacy_raw_questions:
            raw_q_str = "\n".join([f"- {q}" for q in spacy_raw_questions])

            request_text += (
                f"\n\nOtrzymałeś {len(spacy_raw_questions)} surowych pytań (Wzorce):\n"
                f"{raw_q_str}\n\n"
                f"\n\nBazując na podstawie tych {num_questions} pytań popraw i ulepsz je tak, aby uzyskać lepsze, bardziej LOGICZNE i JEDNOZNACZNE pytania edukacyjne bazowane WYŁĄCZNIE na tekście źródłowym. Pytania z luką muszą zastąpić dokładnie JEDNO ważne i kluczowe słowo lub frazę (np. nazwę, datę, pojęcie, fakt, liczbę, nazwisko) pięcioma podkreśleniami:_____."
            )
        
        prompt_template = PROMPTS["hybrid"].format(num_questions=num_questions, text="", specific_request=request_text, key_sentences="")
        prompt_template_token_count = len(self.llm.tokenize(prompt_template.encode("utf-8")))
        
        trimmed_text = self._trim_text_to_fit(text, prompt_template_token_count, additional_content=key_sentences_str)
        
        prompt = PROMPTS["hybrid"].format(num_questions=num_questions, text=trimmed_text, specific_request=request_text, key_sentences=key_sentences_str)
        
        out = self._safe_completion(prompt)
        response = out["choices"][0]["text"].strip()
        
        print(f"DEBUG: Model zwrócił {len(response)} znaków")
        if "finish_reason" in out["choices"][0]:
            print(f"DEBUG: Finish reason: {out['choices'][0]['finish_reason']}")
        
        print("=== RAW OUTPUT ===")
        print(response)
        print("=== END RAW ===")
            
        questions = self._extract_numbered_questions(response, num_questions)

        exec_time = time.time() - start_time
        return questions[:num_questions], exec_time

if __name__ == "__main__":
    import re 
    
    try:
        llm = Llama(
            model_path=LLM_CONFIG["model_path"],
            n_gpu_layers=LLM_CONFIG.get("n_gpu_layers", -1),
            n_ctx=LLM_CONFIG.get("n_ctx", 4096),
            verbose=True
        )
        context_window = llm.n_ctx()
        
        # Sprawdź czy model używa GPU
        gpu_info = ""
        if LLM_CONFIG.get("n_gpu_layers", 0) > 0:
            gpu_info = f" [GPU: {LLM_CONFIG['n_gpu_layers']} warstw]"
        else:
            gpu_info = " [CPU only]"
        
        print(f"Model GGUF załadowany pomyślnie ({context_window} tokenów okna kontekstu){gpu_info}")
        
    except Exception as e:
        print(f"Nie udało się załadować modelu GGUF z {LLM_CONFIG['model_path']}: {e}")
        exit()

    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        print(f"Błąd: Nie znaleziono modelu spaCy '{SPACY_MODEL}'. Użyj komendy: python -m spacy download {SPACY_MODEL}")
        exit()
    
    # Tworzenie generatora pytań
    generator = QuestionGenerator(llm, nlp, context_window, SAFETY_MARGIN, MAX_OUTPUT_TOKENS)
    
    print("\n=== Lokalne generowanie pytań ===\n")
    
    print_configuration()
    
    # Generowanie pytań metodą GGUF
    print("Metoda GGUF (Mistral 7B):")
    gguf_questions, gguf_time = generator.generate_with_llama_cpp(SOURCE_TEXT, NUM_QUESTIONS)
    for i, q in enumerate(gguf_questions, 1):
        print(f"{q}")
    print(f"Czas wykonania: {gguf_time:.2f} s\n")
    print("-" * 40 + "\n")
    
    # Generowanie pytań metodą spaCy
    print("Metoda spaCy (ekstrakcja encji):")
    spacy_questions, spacy_time = generator.generate_with_spacy(SOURCE_TEXT)
    for i, q in enumerate(spacy_questions, 1):
        # Spacy nie numeruje, więc numerujemy w pętli
        print(f"{i}. {q}") 
    print(f"Czas wykonania: {spacy_time:.2f} s\n")
    print("-" * 40 + "\n")
    
    # Generowanie pytań metodą hybrydową
    print("Podejście hybrydowe (spaCy + Mistral 7B):")
    hybrid_questions, hybrid_time = generator.generate_with_hybrid_approach(SOURCE_TEXT, NUM_QUESTIONS)
    for i, q in enumerate(hybrid_questions, 1):
        print(f"{q}")
    print(f"Czas wykonania: {hybrid_time:.2f} s")