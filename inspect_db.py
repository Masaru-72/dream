import chromadb
import json

def inspect_dream_journal():
    """
    Connects to the persistent ChromaDB and prints all stored
    documents in a human-readable JSON format.
    """
    try:
        # Connect to the SAME persistent database directory
        client = chromadb.PersistentClient(path="chroma_db_store")
        
        # Get the collection
        dream_collection = client.get_collection("dream_journal_ko")
        
        # Retrieve ALL items from the collection.
        all_dreams = dream_collection.get(include=["documents"])

        # Check if there are any dreams
        if not all_dreams or not all_dreams['ids']:
            print("The dream journal is empty.")
            return

        print(f"Found {len(all_dreams['ids'])} dreams in the journal.\n")

        # Pretty-print the data as a JSON object
        output_data = {
            "total_dreams": len(all_dreams['ids']),
            "dreams": []
        }
        for i, doc_id in enumerate(all_dreams['ids']):
            output_data["dreams"].append({
                "id": doc_id,
                "text": all_dreams['documents'][i]
            })
            
        # --- FIX ---
        # Add 'ensure_ascii=False' to correctly print characters
        # from languages other than English (like Korean, Japanese, etc.).
        print(json.dumps(output_data, indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"An error occurred while trying to inspect the database: {e}")
        print("Have you run the main application yet to create the database?")

if __name__ == "__main__":
    inspect_dream_journal()