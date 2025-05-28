# fetch_pubmed.py
import os
from Bio import Entrez
from typing import List, Dict

# It's good practice to tell NCBI who you are
# Set these as environment variables or replace the default string
Entrez.email = os.getenv("ENTREZ_EMAIL", "your_email@example.com")
Entrez.api_key = os.getenv("ENTREZ_API_KEY", None) # Optional, but recommended for higher request rates

def search_pubmed(query: str, max_results: int = 20) -> List[str]:
    """
    Searches PubMed for a given query and returns a list of PMIDs.
    """
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=str(max_results))
        record = Entrez.read(handle)
        handle.close()
        return record.get("IdList", [])
    except Exception as e:
        print(f"Error searching PubMed: {e}")
        return []

def fetch_abstracts(pmids: List[str]) -> List[Dict[str, str]]:
    """
    Fetches abstracts for a list of PMIDs.
    Returns a list of dictionaries, each containing 'pmid' and 'abstract'.
    """
    if not pmids:
        return []
    
    abstracts_data = []
    try:
        # Fetch in batches to avoid issues with very long ID lists if necessary
        # For simplicity, fetching all at once here.
        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        for pubmed_article in records.get('PubmedArticle', []):
            pmid = str(pubmed_article.get('MedlineCitation', {}).get('PMID', 'Unknown_PMID'))
            article = pubmed_article.get('MedlineCitation', {}).get('Article', {})
            abstract_text = ""
            if 'Abstract' in article and article['Abstract']:
                abstract_node = article['Abstract']
                # Abstracts can be structured (list of AbstractText parts) or a single string
                if isinstance(abstract_node.get('AbstractText'), list):
                    # Handle cases where AbstractText elements might not be strings (e.g., have attributes)
                    texts = []
                    for sec in abstract_node['AbstractText']:
                        if isinstance(sec, str):
                            texts.append(sec)
                        elif hasattr(sec, 'attributes') and hasattr(sec, 'childNodes') and sec.childNodes:
                             # Attempt to get text content from complex nodes
                            texts.append(" ".join(child.nodeValue for child in sec.childNodes if child.nodeValue))
                        elif hasattr(sec, 'value'): # If it's a simple object with a value attribute
                            texts.append(str(sec.value))

                    abstract_text = " ".join(texts)
                elif isinstance(abstract_node.get('AbstractText'), str):
                    abstract_text = str(abstract_node['AbstractText'])
            
            if abstract_text:
                abstracts_data.append({"pmid": pmid, "abstract": abstract_text.strip()})
            else:
                # Try to get title if abstract is missing, as a fallback for some processing
                title = article.get('ArticleTitle', '')
                if title:
                     abstracts_data.append({"pmid": pmid, "abstract": str(title).strip() + " (Title Only)"})


    except Exception as e:
        print(f"Error fetching abstracts: {e}")
        # Potentially return partial data if some abstracts were fetched before error
        # For now, returns what's collected or empty list on major failure.

    return abstracts_data

if __name__ == '__main__':
    # Example usage
    if not Entrez.email or Entrez.email == "your_email@example.com":
        print("Please set your email for Entrez queries by setting the ENTREZ_EMAIL environment variable.")
    else:
        query = "cell-penetrating peptides mechanism"
        print(f"Searching PubMed for: {query}")
        pmids = search_pubmed(query, max_results=5)
        
        if pmids:
            print(f"Found PMIDs: {pmids}")
            print("\nFetching abstracts...")
            abstracts = fetch_abstracts(pmids)
            for item in abstracts:
                print(f"\nPMID: {item['pmid']}")
                print(f"Abstract: {item['abstract'][:300]}...") # Print first 300 chars
        else:
            print("No articles found.")
