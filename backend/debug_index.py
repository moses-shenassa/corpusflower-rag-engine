from backend.retrieval import get_client, _get_collections

def main():
    client = get_client()
    print("=== All collections ===")
    for col in client.list_collections():
        print(f"- {col.name}")

    chunks_col, docs_col = _get_collections(client)
    print("\nUsing collections:")
    print("  Chunks:", chunks_col.name)
    print("  Docs:  ", docs_col.name)

    print("\nDocs collection stats:")
    print("  count:", docs_col.count())

    peek = docs_col.peek()
    ids = peek.get("ids", [[]])[0]
    docs = peek.get("documents", [[]])[0]
    metas = peek.get("metadatas", [[]])[0]

    print("\n=== Sample docs from docs collection ===")
    for i, (doc_id, text, meta) in enumerate(zip(ids, docs, metas)):
        if i >= 5:
            break
        title = (meta or {}).get("title") or (meta or {}).get("file_name") or "Untitled"
        source = (meta or {}).get("source") or (meta or {}).get("file_path") or ""
        print(f"[{i}] id={doc_id}")
        print(f"    title:  {title}")
        if source:
            print(f"    source: {source}")
        print("    snippet:", (text or "").replace("\n", " ")[:200], "...")
        print()

if __name__ == "__main__":
    main()
