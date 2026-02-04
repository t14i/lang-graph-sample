"""
LangGraph Memory Verification - Part 1: Store Basic Operations

Goal: Verify InMemoryStore basic CRUD operations
- put/get/delete/search
- namespace structure
- JSON value storage

Reference: https://langchain-ai.github.io/langgraph/concepts/memory/
"""

from langgraph.store.memory import InMemoryStore


def main():
    print("=" * 60)
    print("LangGraph Memory - Store Basic Operations")
    print("=" * 60)

    # Initialize store
    store = InMemoryStore()
    print("\n1. InMemoryStore initialized")

    # =========================================================================
    # Test 1: Basic put/get
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 1: Basic put/get")
    print("-" * 40)

    # Put data with namespace
    store.put(("users", "user_123"), "preferences", {"theme": "dark", "language": "ja"})
    print("PUT: ('users', 'user_123'), key='preferences', value={'theme': 'dark', 'language': 'ja'}")

    # Get data
    item = store.get(("users", "user_123"), "preferences")
    print(f"GET: {item}")
    print(f"  - namespace: {item.namespace}")
    print(f"  - key: {item.key}")
    print(f"  - value: {item.value}")
    print(f"  - created_at: {item.created_at}")
    print(f"  - updated_at: {item.updated_at}")

    # =========================================================================
    # Test 2: Namespace structure
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 2: Namespace structure (folder-like)")
    print("-" * 40)

    # Multiple users with same structure
    store.put(("users", "user_123"), "profile", {"name": "Tanaka", "age": 30})
    store.put(("users", "user_456"), "profile", {"name": "Suzuki", "age": 25})
    store.put(("users", "user_456"), "preferences", {"theme": "light", "language": "en"})

    # Different namespace
    store.put(("system", "config"), "api_settings", {"timeout": 30, "retries": 3})

    print("Stored data structure:")
    print("  users/")
    print("    user_123/")
    print("      - preferences")
    print("      - profile")
    print("    user_456/")
    print("      - preferences")
    print("      - profile")
    print("  system/")
    print("    config/")
    print("      - api_settings")

    # =========================================================================
    # Test 3: Search within namespace
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 3: Search within namespace")
    print("-" * 40)

    # Search all items in a namespace
    results = store.search(("users", "user_123"))
    print(f"Search ('users', 'user_123'): {len(list(results))} items found")

    # Re-search (iterator was consumed)
    results = list(store.search(("users", "user_123")))
    for item in results:
        print(f"  - {item.key}: {item.value}")

    # Search in parent namespace (all users)
    results = list(store.search(("users",)))
    print(f"\nSearch ('users',): {len(results)} items found")
    for item in results:
        print(f"  - {item.namespace}/{item.key}: {item.value}")

    # =========================================================================
    # Test 4: Update existing item
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 4: Update existing item")
    print("-" * 40)

    # Get original
    original = store.get(("users", "user_123"), "preferences")
    print(f"Before update: {original.value}")
    print(f"  updated_at: {original.updated_at}")

    # Update (put with same key)
    store.put(("users", "user_123"), "preferences", {"theme": "light", "language": "ja", "notifications": True})

    updated = store.get(("users", "user_123"), "preferences")
    print(f"After update: {updated.value}")
    print(f"  updated_at: {updated.updated_at}")

    # =========================================================================
    # Test 5: Delete item
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 5: Delete item")
    print("-" * 40)

    # Delete
    store.delete(("users", "user_456"), "preferences")
    print("DELETE: ('users', 'user_456'), key='preferences'")

    # Verify deletion
    deleted_item = store.get(("users", "user_456"), "preferences")
    print(f"GET after delete: {deleted_item}")

    # Other items still exist
    remaining = store.get(("users", "user_456"), "profile")
    print(f"Other item still exists: {remaining.value}")

    # =========================================================================
    # Test 6: Get non-existent item
    # =========================================================================
    print("\n" + "-" * 40)
    print("Test 6: Get non-existent item")
    print("-" * 40)

    result = store.get(("users", "user_999"), "nothing")
    print(f"GET non-existent: {result}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Store Basic Operations:
  - put(namespace, key, value): Save/update data
  - get(namespace, key): Retrieve data (returns None if not found)
  - delete(namespace, key): Remove data
  - search(namespace): List items in namespace

Namespace Structure:
  - Tuple-based hierarchical structure: ("level1", "level2", ...)
  - Works like folder paths
  - search() can query parent namespace to get all children

Data Model:
  - Item has: namespace, key, value, created_at, updated_at
  - value can be any JSON-serializable dict

Comparison with CrewAI:
  - CrewAI: Automatic memory management (memory=True)
  - LangGraph: Explicit API, more control but more code
""")


if __name__ == "__main__":
    main()
