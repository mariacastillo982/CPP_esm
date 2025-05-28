# llm_extraction.py
import os
import re
import spacy
from openai import OpenAI
from typing import List, Tuple
import ast

# Load spaCy model for sentence splitting
# Run `python -m spacy download en_core_web_sm` if you haven't already
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Downloading...")
    try:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print(f"Could not download or load spacy model: {e}")
        print("Please ensure you have 'en_core_web_sm' installed (`python -m spacy download en_core_web_sm`)")
        nlp = None


# Configure OpenAI API client
# API key is read from OPENAI_API_KEY environment variable by default
try:
    client = OpenAI()
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}. Ensure OPENAI_API_KEY environment variable is set.")
    client = None

def preprocess_text(text: str) -> List[str]:
    """
    Cleans text (lowercase, normalizes whitespace) and splits into sentences.
    """
    if not nlp:
        print("spaCy model not loaded. Cannot preprocess text.")
        # Fallback to simple splitting if spacy is unavailable
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return [s.strip() for s in text.split('.') if s.strip()]


    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
    
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences

def extract_triples_with_llm(text_segment: str, model_name: str = "gpt-3.5-turbo") -> List[Tuple[str, str, str]]:
    """
    Uses OpenAI's GPT model to extract entities and relationships as triples.
    (subject, predicate, object)
    
    Entities: CPP, UptakeMechanism, Cargo, MolecularPlayer, SubcellularTarget
    Relationships: usesMechanism, delivers, involves, targets, hasType
    """
    if not client:
        print("OpenAI client not initialized. Cannot extract triples.")
        return []
    if not text_segment:
        return []

    # This prompt is crucial and will need significant iteration and refinement.
    # Ensure the LLM is instructed to use the specific entity types and predicates.
    prompt = f"""
    From the text segment below, extract information about cell-penetrating peptides (CPPs).
    Identify entities and their relationships.

    Entities to identify and categorize:
    - CPP: Specific names of cell-penetrating peptides (e.g., Tat, Penetratin, Pep-1).
    - UptakeMechanism: Mechanisms of cellular uptake (e.g., endocytosis, macropinocytosis, direct translocation).
    - MolecularPlayer: Proteins or other molecules involved in the process (e.g., clathrin, heparan sulfate, caveolin).
    - Cargo: Molecules delivered by CPPs (e.g., siRNA, plasmid DNA, proteins, nanoparticles).
    - SubcellularTarget: Locations within the cell where CPPs or cargo are directed (e.g., nucleus, cytosol, mitochondria).

    Relationships to identify (these will be the predicates in the triples):
    - usesMechanism: Links a CPP to an UptakeMechanism it uses. (e.g., ["Tat", "usesMechanism", "endocytosis"])
    - delivers: Links a CPP to a Cargo it delivers. (e.g., ["Penetratin", "delivers", "siRNA"])
    - involves: Links an UptakeMechanism or CPP to a MolecularPlayer. (e.g., ["endocytosis", "involves", "clathrin"], ["Tat", "involves", "heparan sulfate"])
    - targets: Links a CPP or Cargo to a SubcellularTarget. (e.g., ["siRNA", "targets", "cytosol"])
    - hasType: Explicitly assigns a type to an entity. This is very important. (e.g., ["Tat", "hasType", "CPP"], ["endocytosis", "hasType", "UptakeMechanism"])

    Output the results as a Python list of 3-element tuples (subject, predicate, object).
    For example: [("Tat", "hasType", "CPP"), ("Tat", "usesMechanism", "endocytosis")]
    Ensure subjects and objects are concise entity names.
    If no relevant information is found, return an empty list [].

    Text segment:
    "{text_segment}"

    Extracted triples:
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert in bioinformatics and text mining. Your task is to extract structured information as Python-parsable lists of triples."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Lower temperature for more deterministic and structured output
            top_p=0.5,
        )
        
        content = response.choices[0].message.content.strip()
        
        if not content or content.lower() == "none":
            return []
        
        # The LLM should return a string representation of a list of tuples.
        # e.g., "[('Tat', 'hasType', 'CPP'), ('Tat', 'usesMechanism', 'endocytosis')]"
        # Use ast.literal_eval for safe parsing.
        try:
            # Find the list part of the string, e.g. if it's wrapped in explanations
            match = re.search(r'\[\s*\(.*?\)\s*(?:,\s*\(.*?\)\s*)*\]', content, re.DOTALL)
            if match:
                list_str = match.group(0)
            else:
                # If no clear list is found, try to evaluate content directly if it looks like a list
                if content.startswith("[") and content.endswith("]"):
                    list_str = content
                else: # No list-like structure found
                    print(f"Warning: LLM output does not appear to be a list of triples: {content}")
                    return []


            triples = ast.literal_eval(list_str)
            
            # Validate structure
            if not isinstance(triples, list):
                print(f"Warning: LLM output parsed, but is not a list: {triples}")
                return []
            
            valid_triples = []
            for t in triples:
                if isinstance(t, tuple) and len(t) == 3 and all(isinstance(s, str) for s in t):
                    valid_triples.append(t)
                else:
                    print(f"Warning: Invalid triple format in LLM output: {t}")
            return valid_triples
            
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing LLM output string into list of tuples: {e}\nOutput was: {content}")
            return []
            
    except Exception as e:
        print(f"Error during OpenAI API call or processing: {e}")
        return []

if __name__ == '__main__':
    # Example usage
    if not client:
        print("OpenAI client not initialized. Please set the OPENAI_API_KEY environment variable.")
    elif not nlp:
        print("spaCy model not loaded. Please ensure 'en_core_web_sm' is installed.")
    else:
        sample_text = "The Tat peptide is a well-known CPP that primarily uses endocytosis for cellular uptake. This process often involves heparan sulfate proteoglycans. Tat is capable of delivering various cargos, such as siRNA, to the cytosol or nucleus."
        
        print("Preprocessing text...")
        sentences = preprocess_text(sample_text)
        for i, sentence in enumerate(sentences):
            print(f"Sentence {i+1}: {sentence}")

        print("\nExtracting triples using LLM (requires OpenAI API key properly set)...")
        all_triples = []
        for sentence in sentences:
            if not sentence.strip(): continue
            print(f"\nProcessing sentence for LLM: \"{sentence}\"")
            triples = extract_triples_with_llm(sentence, model_name="gpt-3.5-turbo") # Use a faster model for testing
            if triples:
                print("  Extracted Triples:")
                for triple in triples:
                    print(f"    {triple}")
                all_triples.extend(triples)
            else:
                print("  No triples extracted for this sentence.")
        
        print("\n--- All extracted triples from sample text ---")
        if all_triples:
            for triple in all_triples:
                print(triple)
        else:
            print("No triples were extracted overall.")
