from rag_core import ask_question

while True:
    query = input("\nAsk a biomedical question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    answer, sources, context = ask_question(query)

    print("\n" + "="*60)
    print("🧠 BIOMEDICAL ANSWER")
    print("="*60)
    print(answer)
    print("="*60)

    print("\n📚 Sources:")
    for i, source in enumerate(sources):
        print(f"Source {i+1}: {source}")