# ontology_builder.py
from owlready2 import *
from typing import List, Tuple, Dict
import re

# Define a namespace for our ontology
# It's good practice to use a persistent URL if you plan to publish it.
DEFAULT_ONTO_NAMESPACE_STR = "http://example.org/cpp_ontology.owl#" # Added # for IRI components
GO_NAMESPACE_STR = "http://purl.obolibrary.org/obo/"

def sanitize_iri_component(name: str) -> str:
    """
    Sanitizes a string to be a valid IRI component (e.g., for class or individual names).
    Replaces spaces and common special characters with underscores.
    Removes other non-alphanumeric characters (except underscore).
    """
    if not name:
        return "UnnamedEntity"
    # Replace spaces and hyphens with underscores
    name = name.replace(" ", "_").replace("-", "_")
    # Remove characters not suitable for IRIs (simplistic approach)
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    # Ensure it doesn't start with a number if it's a class name (owlready2 might handle this)
    if name and name[0].isdigit():
        name = "_" + name # Prepend underscore if starts with a digit
    if not name: # If all characters were removed
        return "SanitizedEntity"
    return name


def define_ontology_classes_and_properties(ontology: Ontology):
    """
    Defines the core classes and properties for the CPP ontology within the given ontology object.
    """
    with ontology:
        # Import OWL vocabulary
        owl = ontology.get_namespace("http://www.w3.org/2002/07/owl#")
        # --- Define Classes ---
        class CPP(Thing):
            label = "Cell-Penetrating Peptide"
        class UptakeMechanism(Thing):
            label = "Uptake Mechanism"
        class Cargo(Thing):
            label = "Cargo"
        class MolecularPlayer(Thing): # Renamed from Protein for broader scope
            label = "Molecular Player"
        class SubcellularTarget(Thing):
            label = "Subcellular Target"

        # --- Define Object Properties (Relationships) ---
        class usesMechanism(ObjectProperty):
            domain = [CPP]
            range = [UptakeMechanism]
            label = "uses mechanism"
        class delivers(ObjectProperty):
            domain = [CPP]
            range = [Cargo]
            label = "delivers"
        class involves(ObjectProperty):
            # This relationship can be between various entities.
            # Example: UptakeMechanism involves MolecularPlayer, or CPP involves MolecularPlayer.
            # Defining multiple domains/ranges using unions:
            domain = [UptakeMechanism | CPP]
            range = [MolecularPlayer]
            label = "involves"
        class targets(ObjectProperty):
            # Example: CPP targets SubcellularTarget, or Cargo targets SubcellularTarget.
            domain = [CPP | Cargo]
            range = [SubcellularTarget]
            label = "targets"
        
        # A generic interaction property, could be refined
        class interactsWith(ObjectProperty):
            label = "interacts with"
            # Could be symmetric or have specific domain/range if needed
            # For example, MolecularPlayer interactsWith MolecularPlayer
            domain = [MolecularPlayer | CPP]
            range = [MolecularPlayer | CPP]

        # --- Define Data Properties ---
        class hasGOTerm(DataProperty):
            label = "has GO term"
            range = [str] # Storing GO ID as a string (URL)
        
        class hasPmid(DataProperty):
            label = "has PubMed ID"
            range = [str]

    # owlready2 automatically adds classes and properties to the ontology
    # when defined within its context. No explicit return of ontology needed if modified in place.


def get_or_create_individual(ontology: Ontology, class_name_str: str, individual_label: str, pmid: str = None) -> Thing:
    """
    Gets an existing individual by label or creates a new one if it doesn't exist.
    Uses a sanitized version of the label for the IRI name.
    Adds the original label as rdfs:label.
    """
    sanitized_name = sanitize_iri_component(individual_label)
    
    # Check if class exists in the ontology
    ontology_class = getattr(ontology, sanitize_iri_component(class_name_str), None)
    if not ontology_class:
        print(f"Warning: Class '{class_name_str}' (sanitized: {sanitize_iri_component(class_name_str)}) not found in ontology. Cannot create individual '{individual_label}'.")
        return None

    # Search for an existing individual by sanitized name (IRI fragment)
    # Individuals are typically created in the ontology's namespace.
    individual_iri = ontology.base_iri + sanitized_name
    individual = ontology.world[individual_iri]

    if individual:
        # Check if it's already of the desired type or a superclass/subclass
        # This logic can be complex if dealing with multiple types for one individual.
        # For now, if it exists, we assume it's the one we want, or we might re-type it.
        # A more robust check would be `isinstance(individual, ontology_class)`.
        # If it exists but is of a different, incompatible type, this could be an issue.
        if not any(isinstance(individual, c) for c in ontology_class.mro()): # Check if individual is instance of class or its superclasses
             # If it exists but is not of the target class type, this is a conflict.
             # For simplicity, we might overwrite or log a warning.
             # Here, let's try to add the type if it's just a Thing, or warn.
             if individual.is_a == [Thing]: # if it's a generic Thing
                 individual.is_a.append(ontology_class)
             else:
                print(f"Warning: Individual '{individual_label}' (IRI: {individual_iri}) exists but is not of expected class '{class_name_str}'. Current types: {individual.is_a}")
                # Optionally, still return it if you want to add properties to existing entities regardless of precise type.
    else:
        # Create new individual
        try:
            individual = ontology_class(sanitized_name, namespace=ontology)
        except Exception as e:
            print(f"Error creating individual '{sanitized_name}' of class '{class_name_str}': {e}")
            return None
    
    # Add/update rdfs:label
    if individual_label not in individual.label:
        individual.label.append(individual_label)
    
    # Add PMID if provided
    if pmid and hasattr(ontology, "hasPmid") and pmid not in individual.hasPmid:
        individual.hasPmid.append(pmid)
        
    return individual


def populate_ontology_from_triples(ontology: Ontology, triples: List[Tuple[str, str, str]], 
                                   entity_type_map: Dict[str, str], pmid: str = None):
    """
    Populates the ontology with individuals and relationships from extracted triples.
    `entity_type_map` is crucial: it maps an entity name (subject/object of a "hasType" triple)
    to its ontology class name (e.g., "Tat" -> "CPP").
    This map should be pre-populated by processing "hasType" triples from the LLM.
    """
    with ontology:
        for subj_label, pred_name, obj_label in triples:
            
            # Sanitize predicate name to match property definition
            sanitized_pred_name = sanitize_iri_component(pred_name)
            relationship_prop = getattr(ontology, sanitized_pred_name, None)

            if not relationship_prop or not isinstance(relationship_prop, ObjectProperty):
                print(f"Warning: Predicate '{pred_name}' (sanitized: {sanitized_pred_name}) is not a defined ObjectProperty in the ontology. Skipping triple: ({subj_label}, {pred_name}, {obj_label})")
                continue

            # Determine classes of subject and object using the entity_type_map
            subj_class_str = entity_type_map.get(subj_label)
            obj_class_str = entity_type_map.get(obj_label)

            if not subj_class_str:
                # Fallback: try to infer from property domain if simple
                if relationship_prop.domain:
                    # Domain is a list of disjunctions (lists of classes).
                    # Simplistic: take first class of first disjunction.
                    try:
                        subj_class_str = relationship_prop.domain[0][0].__name__
                    except: pass # Ignore if complex or not found
                if not subj_class_str:
                    print(f"Warning: Class for subject '{subj_label}' not found in type map and cannot infer from domain of '{pred_name}'. Skipping.")
                    continue
            
            if not obj_class_str:
                # Fallback: try to infer from property range if simple
                if relationship_prop.range:
                    try:
                        obj_class_str = relationship_prop.range[0][0].__name__
                    except: pass
                if not obj_class_str:
                    print(f"Warning: Class for object '{obj_label}' not found in type map and cannot infer from range of '{pred_name}'. Skipping.")
                    continue

            subj_individual = get_or_create_individual(ontology, subj_class_str, subj_label, pmid)
            obj_individual = get_or_create_individual(ontology, obj_class_str, obj_label, pmid)

            if subj_individual and obj_individual:
                # Add relationship if not already present
                if obj_individual not in relationship_prop[subj_individual]:
                    relationship_prop[subj_individual].append(obj_individual)
                    # print(f"  Added relationship: {subj_label} ({subj_class_str}) --{pred_name}--> {obj_label} ({obj_class_str})")
                # else:
                    # print(f"  Relationship already exists: {subj_label} --{pred_name}--> {obj_label}")
            else:
                if not subj_individual:
                    print(f"  Failed to get/create subject: {subj_label} ({subj_class_str})")
                if not obj_individual:
                    print(f"  Failed to get/create object: {obj_label} ({obj_class_str})")


def link_to_go_terms(ontology: Ontology, individual_label: str, go_id: str):
    """
    Links an individual in the ontology to a Gene Ontology term using owl:sameAs.
    Also retains the hasGOTerm property for convenience.
    """
    sanitized_label_name = sanitize_iri_component(individual_label)
    individual = ontology.search_one(iri=f"*{sanitized_label_name}") # Search by IRI fragment
    
    if not individual: # Fallback to searching by rdfs:label
        individual = ontology.search_one(label=individual_label)

    if individual and go_id.startswith("GO:"):
        go_term_url = GO_NAMESPACE_STR + go_id.replace(":", "_")
        
        # 1. Add owl:sameAs assertion
        if not any(str(sameas) == go_term_url for sameas in individual.same_as):
            individual.same_as.append(go_term_url)
            print(f"Added owl:sameAs link between '{individual_label}' and GO term: {go_term_url}")
        
        # 2. Keep hasGOTerm for convenience (optional)
        if hasattr(ontology, "hasGOTerm") and go_term_url not in individual.hasGOTerm:
            individual.hasGOTerm.append(go_term_url)
    
    elif not individual:
        print(f"Warning: Individual '{individual_label}' not found for GO term linking.")
    elif not go_id.startswith("GO:"):
        print(f"Warning: Invalid GO ID format: {go_id}")

def save_ontology(ontology: Ontology, file_path: str):
    """
    Saves the ontology to an OWL/RDFXML file.
    """
    try:
        ontology.save(file=file_path, format="rdfxml")
        print(f"Ontology saved to {file_path}")
    except Exception as e:
        print(f"Error saving ontology: {e}")


if __name__ == '__main__':
    # --- 1. Setup Ontology ---
    print("Setting up a new ontology...")
    # Create a unique world for each run if testing, or reuse one.
    world = World() 
    onto = world.get_ontology(DEFAULT_ONTO_NAMESPACE_STR)
    # Clear previous ontology content if it exists from a prior run in the same world/namespace
    # For a clean run:
    # list(onto.classes()) # etc. to clear if needed, or use a new world/IRI.
    # This example assumes a fresh ontology object 'onto' each time this script runs directly.

    define_ontology_classes_and_properties(onto)
    print(f"Ontology '{onto.base_iri}' created with classes and properties.")

    # --- 2. Simulate LLM Output (Triples) ---
    # Step 2a: "hasType" triples to define entity types.
    # These are crucial for `populate_ontology_from_triples` to know the class of individuals.
    type_triples_from_llm = [
        ("Tat peptide", "hasType", "CPP"),
        ("Penetratin", "hasType", "CPP"),
        ("endocytosis", "hasType", "UptakeMechanism"),
        ("direct translocation", "hasType", "UptakeMechanism"),
        ("siRNA", "hasType", "Cargo"),
        ("clathrin", "hasType", "MolecularPlayer"),
        ("heparan sulfate", "hasType", "MolecularPlayer"),
        ("nucleus", "hasType", "SubcellularTarget"),
        ("cytosol", "hasType", "SubcellularTarget"),
    ]

    # Step 2b: Relationship triples.
    relationship_triples_from_llm = [
        ("Tat peptide", "usesMechanism", "endocytosis"),
        ("Penetratin", "usesMechanism", "direct translocation"),
        ("Tat peptide", "delivers", "siRNA"),
        ("endocytosis", "involves", "clathrin"),
        ("Tat peptide", "involves", "heparan sulfate"), # e.g., CPP binding to cell surface player
        ("siRNA", "targets", "cytosol"),
        ("Tat peptide", "targets", "nucleus"),
    ]
    
    example_pmid = "PMID:123456"

    # --- 3. Populate Ontology ---
    print("\nPopulating ontology with individuals and types...")
    # Build the entity_type_map from "hasType" triples
    current_entity_type_map = {}
    with onto: # Ensure operations are within the ontology context
        for subj_label, _, type_label in type_triples_from_llm:
            # Map LLM's type label (e.g., "CPP") to our ontology class name (e.g., "CPP")
            # Here, they are the same, but could be different.
            # The class name needs to match what's defined in `define_ontology_classes_and_properties`.
            ontology_class_name = sanitize_iri_component(type_label) # Ensure class name is sanitized if needed
            
            # Create the individual and assign its type
            ind = get_or_create_individual(onto, ontology_class_name, subj_label, pmid=example_pmid)
            if ind:
                current_entity_type_map[subj_label] = ontology_class_name # Store mapping for relationship processing
                # print(f"  Created/Typed: {subj_label} as {ontology_class_name}")

    print("\nPopulating ontology with relationships...")
    populate_ontology_from_triples(onto, relationship_triples_from_llm, current_entity_type_map, pmid=example_pmid)

    # --- 4. Link to GO terms (Example) ---
    print("\nLinking to GO terms...")
    link_to_go_terms(onto, "endocytosis", "GO:0006897")
    link_to_go_terms(onto, "clathrin", "GO:0005882") # Example: clathrin-coated vesicle membrane

    # --- 5. Save Ontology ---
    output_owl_file = "cpp_ontology_example.owl"
    print(f"\nSaving ontology to {output_owl_file}...")
    save_ontology(onto, output_owl_file)

    print("\n--- Example Ontology Construction Complete ---")
    print(f"To view, open '{output_owl_file}' in an ontology editor like Protege.")

    # --- Verification (Optional: List some individuals and their properties) ---
    print("\n--- Ontology Verification (Examples) ---")
    tat_peptide_ind = onto.search_one(label="Tat peptide")
    if tat_peptide_ind:
        print(f"\nIndividual: {tat_peptide_ind.label.first()}")
        print(f"  IRI: {tat_peptide_ind.iri}")
        print(f"  Class(es): {[c.name for c in tat_peptide_ind.is_a]}")
        if hasattr(tat_peptide_ind, "usesMechanism"):
             print(f"  usesMechanism: {[m.label.first() if m.label else m.name for m in tat_peptide_ind.usesMechanism]}")
        if hasattr(tat_peptide_ind, "delivers"):
            print(f"  delivers: {[c.label.first() if c.label else c.name for c in tat_peptide_ind.delivers]}")
        if hasattr(tat_peptide_ind, "involves"): # Check if 'involves' property exists on the individual
            related_players = getattr(tat_peptide_ind, str(onto.involves.name), []) # Access property via its name
            print(f"  involves: {[p.label.first() if p.label else p.name for p in related_players]}")
        if hasattr(tat_peptide_ind, "hasPmid") and tat_peptide_ind.hasPmid:
            print(f"  hasPmid: {tat_peptide_ind.hasPmid}")


    endocytosis_ind = onto.search_one(label="endocytosis")
    if endocytosis_ind:
        print(f"\nIndividual: {endocytosis_ind.label.first()}")
        print(f"  IRI: {endocytosis_ind.iri}")
        print(f"  Class(es): {[c.name for c in endocytosis_ind.is_a]}")
        if hasattr(endocytosis_ind, "involves"):
            related_players = getattr(endocytosis_ind, str(onto.involves.name), [])
            print(f"  involves: {[p.label.first() if p.label else p.name for p in related_players]}")
        if hasattr(endocytosis_ind, "hasGOTerm") and endocytosis_ind.hasGOTerm:
            print(f"  hasGOTerm: {endocytosis_ind.hasGOTerm}")
    
    # Destroy the world to clean up (optional, good for repeated test runs)
    # world.destroy()
