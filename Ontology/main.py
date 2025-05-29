# main.py
import argparse
import os
from owlready2 import World, get_ontology
from typing import List, Tuple, Dict

from fetch_pubmed import search_pubmed, fetch_abstracts
from llm_extraction import preprocess_text, extract_triples_with_llm
from ontology_builder import (
    DEFAULT_ONTO_NAMESPACE_STR,
    define_ontology_classes_and_properties,
    populate_ontology_from_triples,
    link_to_go_terms, # Example, may need more specific linking logic
    save_ontology,
    get_or_create_individual,
    sanitize_iri_component
)
from dotenv import load_dotenv

load_dotenv()
# This map helps translate LLM's entity type labels to our defined ontology class names.
# Keys are expected type labels from LLM (e.g., from "hasType" triples).
# Values are the actual class names defined in ontology_builder.py (after sanitization if any).
# Ensure these values match the class names in define_ontology_classes_and_properties.
ENTITY_TYPE_TO_ONTOLOGY_CLASS = {
    "CPP": "CPP",
    "UptakeMechanism": "UptakeMechanism",
    "Cargo": "Cargo",
    "MolecularPlayer": "MolecularPlayer",
    "SubcellularTarget": "SubcellularTarget",
    # Add other entity types if your LLM prompt defines them and they map to ontology classes
}


def run_pipeline(keyword: str, max_articles_extract: int, max_articles_fetch: int, openai_model: str, output_owl_file: str):
    """
    Runs the full ontology construction pipeline.
    """
    print(f"Starting ontology construction for keyword: '{keyword}'")

    # --- 0. Initialize Ontology ---
    print("\nStep 0: Initializing ontology...")
    # Using a new World for each pipeline run to ensure isolation if run multiple times in one session.
    world = World()
    onto = world.get_ontology(DEFAULT_ONTO_NAMESPACE_STR)
    define_ontology_classes_and_properties(onto) # Define schema
    print(f"Ontology schema defined at {onto.base_iri}")

    # --- 1. Literature Retrieval ---
    print("\nStep 1: Retrieving literature from PubMed...")
    pmids = search_pubmed(keyword, max_results=max_articles_fetch)
    if not pmids:
        print("No articles found for the keyword.")
        return
    print(f"Found {len(pmids)} PMIDs. Will fetch abstracts for them.")

    abstract_data = fetch_abstracts(pmids)
    if not abstract_data:
        print("No abstracts could be fetched.")
        return
    print(f"Fetched {len(abstract_data)} abstracts/titles.")

    # --- 2. Text Preprocessing & LLM Extraction ---
    print(f"\nStep 2: Preprocessing text and extracting triples (max {max_articles_extract} articles)...")
    
    # This map will store entity_label -> ontology_class_name
    # It's populated by "hasType" triples and used for relationship processing.
    global_entity_type_map: Dict[str, str] = {} 
    all_relationship_triples: List[Tuple[str, str, str]] = []
    
    processed_article_count = 0
    for item_idx, item in enumerate(abstract_data):
        if processed_article_count >= max_articles_extract:
            print(f"Reached max_articles_extract limit ({max_articles_extract}). Stopping further LLM processing.")
            break

        pmid = item['pmid']
        abstract = item['abstract']
        print(f"\nProcessing abstract for PMID: {pmid} (Article {item_idx + 1}/{len(abstract_data)})...")

        sentences = preprocess_text(abstract)
        if not sentences:
            print(f"  No sentences extracted from abstract PMID: {pmid}")
            continue
        
        article_triples_count = 0
        for i, sentence in enumerate(sentences):
            # Basic filter for sentence length to avoid tiny/noisy segments
            if len(sentence.split()) < 5 : # Skip sentences with less than 5 words
                continue 
            
            # print(f"  Extracting from sentence {i+1}/{len(sentences)}: \"{sentence[:80]}...\"")
            triples_from_sentence = extract_triples_with_llm(sentence, model_name=openai_model)
            
            if triples_from_sentence:
                # print(f"    Extracted {len(triples_from_sentence)} triples from sentence.")
                article_triples_count += len(triples_from_sentence)
                with onto: # Ensure ontology context for get_or_create_individual
                    for subj_label, pred_name, obj_label in triples_from_sentence:
                        # Process "hasType" triples immediately to populate global_entity_type_map
                        if pred_name.lower() == "hastype":
                            # `obj_label` is the type label from LLM (e.g., "CPP")
                            # Map it to our defined ontology class name
                            ontology_class_key = ENTITY_TYPE_TO_ONTOLOGY_CLASS.get(obj_label)
                            if ontology_class_key:
                                # Ensure the class name itself is sanitized if it's used as an IRI component
                                sanitized_ontology_class_name = sanitize_iri_component(ontology_class_key)
                                get_or_create_individual(onto, sanitized_ontology_class_name, subj_label, pmid=pmid)
                                global_entity_type_map[subj_label] = sanitized_ontology_class_name
                                # print(f"      Typed '{subj_label}' as '{sanitized_ontology_class_name}' (from type: {obj_label})")
                            else:
                                print(f"      Warning: Unknown entity type '{obj_label}' for '{subj_label}' from LLM. Skipping type assignment.")
                        else:
                            all_relationship_triples.append((subj_label, pred_name, obj_label, pmid)) # Store pmid with triple
            # else:
            #     print(f"    No triples extracted from this sentence.")
        
        if article_triples_count > 0:
            print(f"  Extracted {article_triples_count} triples in total from PMID: {pmid}.")
            processed_article_count += 1
        else:
            print(f"  No triples extracted from PMID: {pmid}.")


    if not all_relationship_triples and not global_entity_type_map:
        print("\nNo triples or type definitions were extracted from any abstract. Ontology will be schema only.")
    else:
        print(f"\nTotal relationship triples to process: {len(all_relationship_triples)}")
        print(f"Total entities typed: {len(global_entity_type_map)}")

    # --- 3. Ontology Population with Relationships ---
    print("\nStep 3: Populating ontology with relationships...")
    # Create a list of relationship triples without the PMID for the existing function
    # The PMID was used during individual creation via get_or_create_individual
    # If populate_ontology_from_triples needs pmid, it can be passed or individuals can store it.
    # The current get_or_create_individual already adds PMID to individuals.
    
    # We need to ensure individuals are created for subjects/objects of relationship triples
    # if they weren't in "hasType" triples. The populate_ontology_from_triples
    # uses the global_entity_type_map and falls back to property domain/range.
    
    # Re-structure relationship triples for populate_ontology_from_triples
    # (subj_label, pred_name, obj_label)
    # The pmid is passed to populate_ontology_from_triples to be associated with newly created individuals if any.
    # For now, let's assume pmid is associated during the initial get_or_create_individual call.
    # The populate_ontology_from_triples will call get_or_create_individual again, which is fine.
    
    # We need to pass the PMID for each triple if we want to associate it during relationship population
    # For simplicity, the current `populate_ontology_from_triples` takes one PMID for a batch.
    # This might need adjustment if triples from different PMIDs are mixed.
    # For now, we'll process relationships. The individuals should already have PMIDs if created via "hasType".
    # If new individuals are created during relationship processing, they might miss PMID unless handled.
    
    # Let's process relationship triples, assuming individuals are mostly known via global_entity_type_map
    # or can be inferred by populate_ontology_from_triples.
    # The pmid argument in populate_ontology_from_triples is for individuals created *during* that call.
    
    # To handle PMIDs correctly for relationships spanning different articles,
    # one might process triples per article, or ensure individuals store all their PMIDs.
    # The current get_or_create_individual appends PMIDs.
    
    # Create a list of (subj_label, pred_name, obj_label) for populate_ontology_from_triples
    # The PMID is already attached to individuals when they are created/typed.
    # If a relationship involves an entity not typed by "hasType", its PMID might be from this context.
    
    # For simplicity, we'll iterate and call populate_ontology_from_triples.
    # A better way might be to group triples by PMID if that context is critical for relationship creation.
    # However, entities are global.
    
    # The `populate_ontology_from_triples` function uses `get_or_create_individual` which handles PMIDs.
    # We can just pass the relationship triples.
    # The `pmid` argument to `populate_ontology_from_triples` is for new individuals created there.
    # This is tricky if `all_relationship_triples` mixes PMIDs.
    # Let's pass a generic "batch_pmid" or None, relying on individuals already having PMIDs.
    
    # Correct approach: ensure `get_or_create_individual` is robustly called for all entities
    # involved in relationships, with their correct PMIDs.
    # The current loop for "hasType" handles this for typed entities.
    # For entities only appearing in relationships, their PMID context is from that relationship's source.
    
    # Let's refine: ensure all entities in relationships are created with their PMID context
    # *before* calling populate_ontology_from_triples, if not already typed.
    
    temp_relationship_triples_for_population = []
    for subj_l, pred_n, obj_l, p_id in all_relationship_triples:
        # Ensure subj and obj individuals exist with their PMIDs, even if not typed by "hasType"
        # If they are not in global_entity_type_map, their class might be inferred by populate_ontology_from_triples
        # or they might be created as generic "Thing" if no domain/range helps.
        # This is a point of potential improvement: LLM should ideally type all entities.
        if subj_l not in global_entity_type_map:
            # Try to create as Thing, or let populate_ontology_from_triples infer
            get_or_create_individual(onto, "Thing", subj_l, pmid=p_id) # Creates as generic Thing if class unknown
        if obj_l not in global_entity_type_map:
            get_or_create_individual(onto, "Thing", obj_l, pmid=p_id)
        temp_relationship_triples_for_population.append((subj_l, pred_n, obj_l))

    populate_ontology_from_triples(onto, temp_relationship_triples_for_population, global_entity_type_map, pmid="batch_from_relationships")


    # --- 4. Optional: Link to GO terms (Example placeholder) ---
    # This would require a curated list of mappings or another LLM step for GO term identification.
    # print("\nStep 4: Linking to GO terms (example)...")
    # Example: if "endocytosis" individual exists and we know its GO term
    # endocytosis_ind = onto.search_one(label="endocytosis") # Search by label
    # if endocytosis_ind:
    #    link_to_go_terms(onto, "endocytosis", "GO:0006897")

    # --- 5. Export Ontology ---
    print("\nStep 5: Exporting ontology...")
    save_ontology(onto, output_owl_file)
    print(f"Ontology construction complete. Output saved to: {output_owl_file}")

    # Clean up the world to free resources, important if running in a loop or long session
    world.destroy()


def main():
    parser = argparse.ArgumentParser(description="Build a biological ontology from PubMed literature using an LLM.")
    parser.add_argument("keyword", type=str, help="Keyword to search PubMed (e.g., 'cell-penetrating peptides').")
    parser.add_argument("--max_articles_fetch", type=int, default=50, help="Maximum number of PubMed articles to fetch initially.")
    parser.add_argument("--max_articles_extract", type=int, default=10, help="Maximum number of fetched articles to process for LLM extraction.")
    parser.add_argument("--openai_model", type=str, default="deepseek/deepseek-r1-0528:free", help="OpenAI model to use for extraction (e.g., 'gpt-4', 'gpt-3.5-turbo').")
    parser.add_argument("--output_owl_file", type=str, default="cpp_ontology.owl", help="Filename for the output OWL ontology.")
    
    args = parser.parse_args()

    # API Key Management: Best practice is to use environment variables.
    # Check for essential environment variables
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set. Please set it before running the script.")
        return
    
    if not os.getenv("ENTREZ_EMAIL"):
        print("Warning: ENTREZ_EMAIL environment variable not set. Using a default placeholder in fetch_pubmed.py.")
        print("It's highly recommended to set your email for NCBI Entrez queries (ENTREZ_EMAIL).")

    run_pipeline(
        keyword=args.keyword,
        max_articles_fetch=args.max_articles_fetch,
        max_articles_extract=args.max_articles_extract,
        openai_model=args.openai_model,
        output_owl_file=args.output_owl_file
    )

if __name__ == "__main__":
    main()
