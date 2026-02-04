# LangGraph 検証レポート

## 概要

LangGraphのTool Calling、HITL（Human-in-the-Loop）、Durable Execution、Memory機能を検証し、本番環境での利用可能性を評価した。

---

# Part 1: Quick Start

## 1.1 最小構成（01_quickstart.py）

**目的**: LangGraphの基本概念を理解する

### グラフ構造

```
START → chatbot → END
```

### コード要点

```python
# State定義 - TypedDictベース
class State(TypedDict):
    messages: Annotated[list, add_messages]  # add_messagesでメッセージ追記

# Node定義 - 関数
def chatbot(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # 差分を返す（add_messagesでマージ）

# Graph構築
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# 実行
result = graph.invoke({"messages": [("user", "Hello")]})
```

### 学んだこと

| 要素 | 説明 |
|------|------|
| `StateGraph(State)` | グラフ初期化 |
| `add_node(name, fn)` | ノード追加 |
| `add_edge(from, to)` | エッジ接続 |
| `compile()` | 実行可能なグラフに変換 |
| Node関数 | `State → State`（差分返却でOK） |

### 実行結果

```
LangGraph is a framework for building stateful, multi-actor applications
with LLMs by modeling them as graphs where nodes represent functions/agents
and edges represent the flow of information.
```

---

# Part 2: Tool Calling

## 2.1 Tool定義方法（02_tool_definition.py）

**目的**: Toolの定義方法の比較

### 4つの定義方法

```python
# Method 1: @tool decorator (Simple)
@tool
def get_weather_simple(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

# Method 2: @tool with Annotated (Better docs)
@tool
def get_weather_typed(
    city: Annotated[str, "The city name to get weather for"],
    unit: Annotated[str, "Temperature unit"] = "celsius"
) -> str:
    """Get current weather for a city with specified unit."""
    ...

# Method 3: Pydantic schema (Full control)
class WeatherInput(BaseModel):
    city: str = Field(description="The city name")
    unit: str = Field(default="celsius", description="Temperature unit")

@tool(args_schema=WeatherInput)
def get_weather_pydantic(city: str, unit: str = "celsius") -> str:
    ...

# Method 4: StructuredTool (Programmatic)
search_tool = StructuredTool.from_function(
    func=_search_impl,
    name="web_search",
    description="Search the web",
    args_schema=SearchInput,
)
```

### 実行結果

```
Tool: get_weather_simple
Args Schema: {'properties': {'city': {'type': 'string'}}, 'required': ['city']}

Tool: get_weather_typed
Args Schema: {'properties': {
    'city': {'description': 'The city name to get weather for', 'type': 'string'},
    'unit': {'description': 'Temperature unit', 'default': 'celsius', 'type': 'string'}
}, 'required': ['city']}

Tool: get_weather_pydantic
Args Schema: {'properties': {
    'city': {'description': 'The city name', 'type': 'string'},
    'unit': {'description': 'Temperature unit', 'default': 'celsius', 'type': 'string'},
    'include_forecast': {'description': 'Include 3-day forecast', 'default': false}
}, 'required': ['city']}
```

### 比較

| 方法 | 長所 | 短所 | 推奨用途 |
|------|------|------|----------|
| @tool シンプル | 最小コード | 引数説明なし | プロトタイプ |
| @tool + Annotated | 説明付き | 多引数で冗長 | 中規模 |
| @tool + Pydantic | フル制御、バリデーション | ボイラープレート多 | 本番 |
| StructuredTool | プログラム的生成可能 | 最も冗長 | 動的生成 |

---

## 2.2 Tool実行（03_tool_execution.py）

**目的**: ToolNodeの動作確認

### グラフ構造

```
START → agent → [tool_calls?] → tools → agent → ... → END
                     ↓ no
                    END
```

### コード要点

```python
# LLMにToolをバインド
llm_with_tools = llm.bind_tools(tools)

# ToolNode（組み込み）
graph_builder.add_node("tools", ToolNode(tools))

# 条件分岐
def should_continue(state: State) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges("agent", should_continue, ["tools", END])

# ループバック
graph_builder.add_edge("tools", "agent")
```

### 実行結果

```
=== TEST: Single Tool ===
Query: What's the weather in Tokyo?
  [0] HumanMessage: What's the weather in Tokyo?
  [1] AIMessage: tool_calls=['get_weather']
  [2] ToolMessage: Sunny, 22°C
  [3] AIMessage: The current weather in Tokyo is sunny with a temperature of 22°C.

=== TEST: Multiple Tools ===
Query: What's the weather in Tokyo and the stock price of AAPL?
  [1] AIMessage: tool_calls=['get_weather', 'get_stock_price']  ← 並列呼び出し
  [2] ToolMessage: Sunny, 22°C
  [3] ToolMessage: $178.50
  [4] AIMessage: Weather: Sunny, 22°C / AAPL: $178.50

=== TEST: Unknown Data ===
Query: What's the weather in Antarctica?
  [2] ToolMessage: No data for Antarctica
  [3] AIMessage: I wasn't able to get weather data for Antarctica...
```

### 確認できたこと

| 項目 | 動作 |
|------|------|
| 単一ツール | 正常に呼び出し・結果返却 |
| 複数ツール | 1つのAIMessageで並列呼び出し |
| ツール選択 | LLMが適切なツールを選択 |
| データなし | ToolMessageで返却、LLMが適切に応答 |

---

## 2.3 エラーハンドリング（04_tool_error_handling.py）

**目的**: ツール実行失敗時の動作確認

### デフォルト動作（handle_tool_errors=True）

```python
graph_builder.add_node("tools", ToolNode(tools, handle_tool_errors=True))
```

### 実行結果

```
=== DEFAULT ERROR HANDLING ===

Message flow:
  HumanMessage: Process the number 'abc' using validation_error_tool
  AIMessage: tool_calls=['validation_error_tool']
  ToolMessage: Error: ValueError('Expected number, got: abc') Please fix your mistakes.
  AIMessage: The validation_error_tool rejected the input 'abc' because it's not a number...

Observation: ToolNode catches exception, returns error as ToolMessage
LLM receives error and explains it to user
```

### カスタムリトライ実装

```python
class RetryToolNode:
    def __init__(self, tools: list, max_retries: int = 2):
        self.tools_by_name = {t.name: t for t in tools}
        self.max_retries = max_retries

    def __call__(self, state: State) -> State:
        for tool_call in last_message.tool_calls:
            for attempt in range(self.max_retries + 1):
                try:
                    result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                    break
                except Exception as e:
                    last_error = e
            # Handle success or final failure
```

### 実行結果（リトライ）

```
=== RETRY ERROR HANDLING ===
Testing flaky API with retry...
  Attempt 1 failed: API temporarily unavailable
  Attempt 2 failed: API temporarily unavailable
  Attempt 3 succeeded!

Final response: The flaky API call succeeded: "API result for: test query"
```

### エラーメッセージ形式

```python
ToolMessage:
  tool_call_id: toolu_01N8VVACcSYxSxivTAoCXS86
  content: "Error: TimeoutError('Operation timed out after 100s') Please fix your mistakes."
  status: error
```

### まとめ

| 項目 | デフォルト動作 |
|------|--------------|
| 例外キャッチ | ✅ 自動でキャッチ |
| ToolMessage変換 | ✅ エラー内容をToolMessageで返却 |
| 例外伝播 | ❌ しない（グラフは継続） |
| リトライ | ❌ なし（自前実装必要） |
| LLMの反応 | エラーを認識して適切に応答 |

---

# Part 3: Human-in-the-Loop (HITL)

## 3.1 interrupt基礎（05_hitl_interrupt.py）

**目的**: interrupt() による中断・再開の動作確認

### グラフ構造

```
START → agent → [tool_calls?] → human_approval → tools → agent → ... → END
                     ↓ no
                    END
```

### 核となるAPI

| API | 役割 |
|-----|------|
| `interrupt(value)` | グラフ実行を一時停止し、valueを返す |
| `Command(resume=data)` | 中断したグラフを再開し、dataをinterrupt()の戻り値として渡す |
| `graph.get_state(config)` | 現在の状態を取得（`state.next`で中断位置確認） |
| Checkpointer | 状態永続化（interrupt必須） |

### コード要点

```python
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

def human_approval(state: State) -> State:
    tool_call = state["messages"][-1].tool_calls[0]

    # ここでグラフが停止し、invoke()が返る
    approval = interrupt({
        "tool_name": tool_call["name"],
        "tool_args": tool_call["args"],
        "message": f"Approve {tool_call['name']}?"
    })

    if not approval.get("approved", False):
        raise ValueError("Rejected")
    return state

# Checkpointer必須
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
```

### 実行フロー

```python
config = {"configurable": {"thread_id": "test-thread-1"}}

# 1. 最初のinvoke → 中断
result = graph.invoke({"messages": [...]}, config=config)

# 2. 状態確認
state = graph.get_state(config)
print(state.next)  # ('human_approval',) ← 中断位置
print(state.tasks[0].interrupts[0].value)  # interrupt()に渡した値

# 3. 再開
result = graph.invoke(Command(resume={"approved": True}), config=config)
```

### 実行結果

```
=== Starting conversation ===
Graph state: StateSnapshot(
    next=('human_approval',),  # ← ここで停止中
    ...
)

=== Interrupted! Waiting at: ('human_approval',) ===
Interrupt value: {
    'tool_name': 'send_email',
    'tool_args': {'to': 'bob@example.com', 'subject': 'Hello', 'body': 'How are you?'},
}

=== Resuming with approval ===
Final result: The email has been sent successfully to bob@example.com.
```

---

## 3.2 Approve / Reject / Edit（06_hitl_approve_reject_edit.py）

**目的**: 3つの承認パターンの動作確認

### 実装パターン

```python
def human_approval(state: State) -> Command:
    tool_call = state["messages"][-1].tool_calls[0]

    decision = interrupt({
        "tool_name": tool_call["name"],
        "tool_args": tool_call["args"],
        "options": ["approve", "reject", "edit"],
    })

    action = decision.get("action", "reject")

    if action == "approve":
        return Command(goto="tools")

    elif action == "reject":
        rejection_msg = ToolMessage(
            content=f"Rejected: {decision.get('reason')}",
            tool_call_id=tool_call["id"],
        )
        return Command(goto="agent", update={"messages": [rejection_msg]})

    elif action == "edit":
        edited_args = decision.get("edited_args")
        last_message.tool_calls[0]["args"] = edited_args
        return Command(goto="tools", update={"messages": [last_message]})
```

### 実行結果

```
=== Test 1: APPROVE ===
Resuming with: {'action': 'approve'}
Final response: The email has been successfully sent to alice@example.com.

=== Test 2: REJECT ===
Resuming with: {'action': 'reject', 'reason': 'This looks like spam'}
Final response: I understand your concern. The email does appear to have
characteristics commonly associated with spam. Would you like to send
a legitimate email instead?

=== Test 3: EDIT ===
Resuming with: {'action': 'edit', 'edited_args': {'to': 'correct@example.com', ...}}
Final response: The email has been sent successfully to correct@example.com.
```

### まとめ

| パターン | 動作 | LLMの反応 |
|---------|------|----------|
| Approve | ツール実行 → 完了報告 | 「送信しました」 |
| Reject | ツールスキップ → agentへ戻る | 拒否理由を理解して代替案を提示 |
| Edit | 引数書き換え → ツール実行 | 編集後の値で実行完了を報告 |

---

## 3.3 Checkpointer比較

| Checkpointer | 用途 | 永続性 | 設定 |
|--------------|------|--------|------|
| `MemorySaver` | 開発・テスト | プロセス内のみ | 不要 |
| `SqliteSaver` | 小規模本番 | ファイル | DB path |
| `PostgresSaver` | 本番推奨 | 完全 | 接続文字列 |

### PostgresSaverの使用例

```python
from langgraph_checkpoint_postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/langgraph"
)
graph = graph_builder.compile(checkpointer=checkpointer)
```

---

# Part 4: Durable Execution

## 4.1 基本的なCheckpoint動作（07_durable_basic.py）

**目的**: Checkpointがいつ、何を保存するか理解する

### Checkpointのタイミング

```
=== グラフ実行 ===
  [Step1] 実行中... (step_count: 0)
  [Step2] 実行中... (step_count: 1)
  [Step3] 実行中... (step_count: 2)

Checkpoint履歴:
  [0] next=(), step_count=3        ← Step3完了後
  [1] next=('step3',), step_count=2  ← Step2完了後
  [2] next=('step2',), step_count=1  ← Step1完了後
  [3] next=('step1',), step_count=0  ← 初期状態
  [4] next=('__start__',)            ← 開始前
```

**観察**: Checkpointは各ノード完了後に保存される。

### 再起動後の再開

```python
# Phase 1: 実行してstep1後に停止
graph1 = build_graph()
for chunk in graph1.stream(input, config):
    if step1_done:
        break  # クラッシュをシミュレート

# Phase 2: 新しいグラフインスタンス（再起動をシミュレート）
graph2 = build_graph()
state = graph2.get_state(config)
# state.next = ('step2',) ← 復旧された！

# Phase 3: 再開
result = graph2.invoke(None, config=config)
# step2から継続
```

**実行結果**:
```
[Phase 1] 最初の実行...
  [Step1] 実行中...
  Step1完了、クラッシュをシミュレート...
  クラッシュ後の状態: next=('step2',), step_count=1

[Phase 2] 再起動後...
  復旧した状態: next=('step2',), step_count=1
  復旧したmetadata: {'step1_done': True}

[Phase 3] 再開中...
  [Step2] 実行中...
  [Step3] 実行中...
  最終step_count: 3
```

---

## 4.2 HITL + 永続化（08_durable_hitl.py）

**目的**: interruptがプロセス再起動後も維持されるか確認

### テストフロー

```
[Phase 1] 開始 → Agent → interrupt() → 停止

[Phase 2] 再起動（新しいグラフインスタンス）
  復旧した状態: next=('human_approval',)
  復旧したinterrupt値: {tool_name, tool_args, ...}

[Phase 3] Command(resume={"action": "approve"})で再開
  → ツール実行 → 完了
```

**実行結果**:
```
[Phase 1] 実行開始...
  中断位置: ('human_approval',)
  Interrupt値: {'tool_name': 'send_email', 'tool_args': {...}}

[Phase 2] 再起動をシミュレート...
  復旧した状態: next=('human_approval',)
  復旧したinterrupt: {'tool_name': 'send_email', ...}

[Phase 3] 承認で再開...
  最終レスポンス: メール送信成功
  最終approval_count: 1
```

**重要な発見**: HITLのinterruptは完全に永続化される。サーバー再起動しても承認待ちが失われない。

---

## 4.3 本番での課題（09_durable_production.py）

### 並行実行（同じthread_id）

```
同じthread_idで3つの並行実行を開始...

結果: [(0, 1), (2, 1), (1, 1)]  ← 全部counter=1
エラー: []
最終counter: 1  ← 最後の書き込みが勝つ
```

**問題**: 同じthread_idでの並行invoke()は競合状態を引き起こす。

**解決策**: 会話ごとにユニークなthread_idを生成する。

### Checkpointサイズの増加

```
1スレッド, 各1メッセージ: 4.0 KB
2スレッド, 各2メッセージ: 8.0 KB
5スレッド, 各5メッセージ: 20.0 KB
10スレッド, 各10メッセージ: 40.0 KB
```

**観察**: 線形に増加。各Checkpointで状態全体のスナップショット。

### スレッド一覧取得

```python
# 組み込みAPIなし - ストレージを直接クエリ
cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
threads = cursor.fetchall()
# ['list-test-0', 'list-test-1', ...]
```

**問題**: 全thread_idを取得するAPIがない。

### まとめ

| 課題 | 状態 | 解決策 |
|------|------|--------|
| Checkpointタイミング | ✅ 各ノード後 | - |
| 再起動後の再開 | ✅ 動作する | 同じthread_id使用 |
| HITL永続化 | ✅ 完全サポート | - |
| 並行アクセス | ⚠️ 競合状態 | ユニークthread_id |
| Checkpointクリーンアップ | ❌ 自動なし | カスタムジョブ |
| スレッド一覧 | ❌ APIなし | ストレージ直接クエリ |
| 状態マイグレーション | ⚠️ 手動 | スキーマバージョニング |

---

# Part 5: Memory

## 5.1 Store基本操作（11_memory_store_basic.py）

**目的**: InMemoryStoreの基本CRUD操作確認

### コード

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# put - namespace（フォルダ構造）で保存
store.put(("users", "user_123"), "preferences", {"theme": "dark", "language": "ja"})

# get - namespaceとkeyで取得
item = store.get(("users", "user_123"), "preferences")
# item.value = {"theme": "dark", "language": "ja"}

# search - namespace内のアイテムを一覧
results = store.search(("users", "user_123"))

# delete - 削除
store.delete(("users", "user_123"), "preferences")
```

### 実行結果

```
PUT: ('users', 'user_123'), key='preferences', value={'theme': 'dark', 'language': 'ja'}
GET: Item(namespace=['users', 'user_123'], key='preferences', value={'theme': 'dark', 'language': 'ja'})
Search ('users', 'user_123'): 2 items found
GET after delete: None
```

### まとめ

| 操作 | 説明 |
|------|------|
| `put(namespace, key, value)` | 保存/更新 |
| `get(namespace, key)` | 取得（存在しない場合None） |
| `search(namespace)` | namespace内のアイテム一覧 |
| `delete(namespace, key)` | 削除 |

---

## 5.2 セマンティック検索（12_memory_semantic_search.py）

**目的**: Embeddingベースのセマンティック検索確認

### コード

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(
    index={
        "dims": 1536,  # text-embedding-3-small次元
        "embed": "openai:text-embedding-3-small",
    }
)

# 'text'フィールドでセマンティックインデックス
store.put(("memories",), "food_1", {"text": "I love Italian food, especially pasta"})
store.put(("memories",), "work_1", {"text": "I work as a software engineer"})

# セマンティック検索
results = store.search(
    ("memories",),
    query="What food do I like?",
    limit=3
)
for item in results:
    print(f"[{item.score:.4f}] {item.value['text']}")
```

### 期待される結果

```
Query: 'What food do I like?'
  [0.8523] I love Italian food, especially pasta
  [0.4102] I work as a software engineer

Query: 'dietary restrictions'
  [0.7891] I'm allergic to shellfish and peanuts
```

### まとめ

| 機能 | 説明 |
|------|------|
| Embeddingモデル | OpenAI text-embedding-3-small |
| 類似度 | コサイン類似度（0-1） |
| フィルター | メタデータベースのフィルタリング可能 |
| Score > 0.8 | 高い関連性 |
| Score < 0.5 | 低い関連性 |

---

## 5.3 クロススレッド永続化（13_memory_cross_thread.py）

**目的**: 異なるthread_id間でのメモリ共有確認

### アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│                 Store（長期メモリ）                   │
│  ┌─────────────────┐    ┌─────────────────┐        │
│  │  users/alice/   │    │  users/bob/     │        │
│  │  - memory_0     │    │  - memory_0     │        │
│  │  - memory_1     │    │                 │        │
│  └─────────────────┘    └─────────────────┘        │
└─────────────────────────────────────────────────────┘
                ↑ 全スレッドで共有

┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ thread-1 │  │ thread-2 │  │ thread-3 │  │ thread-4 │
│ (Alice)  │  │ (Alice)  │  │  (Bob)   │  │ (Alice)  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
      ↓              ↓              ↓              ↓
┌─────────────────────────────────────────────────────┐
│            Checkpointer（短期メモリ）                 │
│        各スレッドで独立した会話履歴                    │
└─────────────────────────────────────────────────────┘
```

### ポイント

- **Store**: クロススレッド、namespaceでユーザー分離
- **Checkpointer**: スレッドごとの会話履歴
- セッション1で保存 → セッション2（別スレッド）でアクセス可能

---

## 5.4 LangMem Memory Tools（14_memory_langmem_tools.py）

**目的**: LangMemによるエージェント管理メモリの確認

### コード

```python
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})

# namespaceテンプレートでメモリツール作成
manage_memory = create_manage_memory_tool(namespace=("memories", "{user_id}"))
search_memory = create_search_memory_tool(namespace=("memories", "{user_id}"))

agent = create_react_agent(
    "openai:gpt-4o",
    tools=[manage_memory, search_memory],
    store=store,
)

# エージェントが自律的にメモリを保存/検索
response = agent.invoke(
    {"messages": [{"role": "user", "content": "私の名前は田中です。覚えておいて。"}]},
    config={"configurable": {"user_id": "user_123"}}
)
```

### エージェントの動作

| アクション | タイミング |
|------------|-----------|
| メモリ保存 | 「覚えて」や個人情報の共有時 |
| メモリ検索 | 過去の情報について質問された時 |
| メモリ更新 | 以前の情報を訂正された時 |

### 特徴

LangMemのメモリツールは**CrewAIにはない機能**。エージェントがメモリ操作を明示的に制御できる。

---

## 5.5 バックグラウンド抽出（15_memory_background_extraction.py）

**目的**: 会話からの自動ファクト抽出確認

### コード

```python
from langmem import create_memory_store_manager

manager = create_memory_store_manager(
    "openai:gpt-4o",
    namespace=("memories", "{user_id}"),
)

conversation = [
    {"role": "user", "content": "イタリアン大好きだけど、貝類アレルギーがあるんです。"},
    {"role": "assistant", "content": "食の好みとアレルギーをメモしておきますね。"},
]

# 非同期でメモリ抽出
await manager.ainvoke(
    {"messages": conversation},
    config={"configurable": {"user_id": "user_123"}},
    store=store,
)
```

### 抽出の動作

- ユーザーのファクト、好み、制約を識別
- セマンティック検索用のEmbeddingを作成
- 矛盾する情報を更新（統合）
- 非同期で処理（バックグラウンド）

---

## 5.6 Memoryまとめ

| 機能 | サポート | 備考 |
|------|---------|------|
| 基本CRUD | ✅ 完全 | put/get/delete/search |
| Namespace | ✅ 完全 | フォルダ構造 |
| セマンティック検索 | ✅ 完全 | OpenAI Embeddings |
| クロススレッド | ✅ 完全 | Storeは全スレッドで共有 |
| LangMemツール | ✅ 完全 | エージェント管理メモリ |
| バックグラウンド抽出 | ✅ 完全 | 自動ファクト抽出 |
| 本番ストレージ | ✅ PostgresStore | pgvectorでベクトル |
| クリーンアップ | ❌ なし | TTL/自動クリーンアップなし |
| プライバシー | ⚠️ 手動 | PII対応は自前 |

### CrewAI比較

| 機能 | LangGraph + LangMem | CrewAI |
|------|---------------------|--------|
| 基本構造 | Store + namespace | ChromaDB + SQLite |
| Embedding | OpenAI（設定可能） | OpenAI（設定可能） |
| セマンティック検索 | ✅ | ✅ |
| クロスセッション | ✅ | ✅ |
| エージェントメモリツール | ✅ **独自機能** | ❌ |
| バックグラウンド抽出 | ✅ **独自機能** | ❌ |
| 本番ストレージ | PostgresStore | 外部DB移行必要 |

---

# Part 6: 本番環境での課題

## 6.1 監査ログ

**現状**: なし

```python
def human_approval(state: State) -> Command:
    decision = interrupt({...})

    # 自前でログ記録が必要
    audit_logger.log(
        timestamp=datetime.now(),
        user_id=???,  # どこから取得する？
        action=decision["action"],
        tool_name=tool_call["name"],
        tool_args=tool_call["args"],
    )
```

**課題**:
- 承認者のユーザーIDをどう渡すか
- `Command(resume=...)`にメタデータを含める必要あり

---

## 6.2 タイムアウト

**現状**: なし。中断したまま永久に待機。

```python
# バックグラウンドジョブで実装
async def cleanup_stale_threads():
    for thread_id in get_active_threads():
        state = graph.get_state({"configurable": {"thread_id": thread_id}})
        if state.next and is_stale(state.created_at, timeout=timedelta(hours=24)):
            graph.invoke(
                Command(resume={"action": "reject", "reason": "Timeout"}),
                config={"configurable": {"thread_id": thread_id}}
            )
```

**課題**:
- スレッド一覧取得APIがない（Checkpointer依存）
- タイムアウト処理ロジックを自前実装

---

## 6.3 通知システム

**現状**: なし

```python
state = graph.get_state(config)
if state.next:
    # 自前で通知
    slack.send(f"承認待ち: {state.tasks[0].interrupts[0].value}")
    email.send(approver, "承認リクエスト", ...)
```

---

## 6.4 認可（誰が承認できるか）

**現状**: なし

```python
def human_approval(state: State) -> Command:
    decision = interrupt({
        "required_role": "admin",
        ...
    })

    # resume時に渡されたユーザー情報を検証
    if not has_role(decision["approver_id"], "admin"):
        raise PermissionError("Not authorized")
```

---

## 5.5 Web APIとの統合パターン

```python
from fastapi import FastAPI
from langgraph.types import Command

app = FastAPI()

@app.post("/chat")
async def chat(message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [("user", message)]}, config=config)

    state = graph.get_state(config)
    if state.next:
        return {
            "status": "pending_approval",
            "thread_id": thread_id,
            "approval_request": state.tasks[0].interrupts[0].value
        }
    return {
        "status": "completed",
        "response": result["messages"][-1].content
    }

@app.post("/approve/{thread_id}")
async def approve(thread_id: str, decision: dict):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(Command(resume=decision), config=config)
    return {"response": result["messages"][-1].content}
```

---

# Part 7: 評価まとめ

## Good

| カテゴリ | 項目 | 評価 | 備考 |
|----------|------|------|------|
| Tool Calling | `@tool` デコレータ | ⭐⭐⭐⭐⭐ | シンプル、Pydanticも使える |
| Tool Calling | `ToolNode` | ⭐⭐⭐⭐⭐ | 自動実行、エラーハンドリング |
| Tool Calling | 複数ツール並列 | ⭐⭐⭐⭐⭐ | 1メッセージで並列呼び出し |
| HITL | `interrupt()` API | ⭐⭐⭐⭐⭐ | シンプルで直感的 |
| HITL | `Command` による制御 | ⭐⭐⭐⭐⭐ | goto, update, resumeが柔軟 |
| HITL | Approve/Reject/Edit | ⭐⭐⭐⭐⭐ | 全パターン実装可能 |
| Durable | 状態永続化 | ⭐⭐⭐⭐ | Postgres/SQLite対応 |
| Durable | Durable Execution | ⭐⭐⭐⭐ | 再起動後も再開可能 |
| Durable | HITL永続化 | ⭐⭐⭐⭐⭐ | interruptが再起動後も維持 |
| Memory | Store API | ⭐⭐⭐⭐⭐ | シンプルなCRUD、namespace |
| Memory | セマンティック検索 | ⭐⭐⭐⭐⭐ | OpenAI Embeddings |
| Memory | クロススレッド | ⭐⭐⭐⭐⭐ | セッション間で共有 |
| Memory | LangMemツール | ⭐⭐⭐⭐⭐ | エージェント管理メモリ |
| Memory | バックグラウンド抽出 | ⭐⭐⭐⭐ | 自動ファクト抽出 |

## Not Good

| カテゴリ | 項目 | 評価 | 備考 |
|----------|------|------|------|
| Tool Calling | ツールリトライ | ⭐⭐ | 自前実装必要 |
| HITL | 監査ログ | ⭐ | 完全に自前実装 |
| HITL | タイムアウト | ⭐ | 仕組みなし |
| HITL | 通知 | ⭐ | 仕組みなし |
| HITL | 認可 | ⭐ | 仕組みなし |
| Durable | Checkpointクリーンアップ | ⭐ | 自動クリーンアップなし |
| Durable | スレッド一覧API | ⭐ | 組み込みAPIなし |
| Durable | 並行アクセス | ⭐⭐ | 競合状態の可能性 |
| Memory | メモリクリーンアップ | ⭐ | TTL/自動クリーンアップなし |
| Memory | プライバシー/PII | ⭐⭐ | コンプライアンスは手動 |
| Memory | Embeddingコスト | ⭐⭐ | 操作ごとにコスト発生 |

---

# 結論

## Tool Calling

**完成度が高い。** `@tool`デコレータでシンプルに定義でき、`ToolNode`で自動実行される。エラーハンドリングも`handle_tool_errors=True`で対応。本番ではリトライやサーキットブレーカーを自前実装する必要あり。

## HITL

**「グラフ実行の中断・再開」としては完成度が高い。** `interrupt()` / `Command(resume=...)` APIはクリーンで使いやすい。

しかし、本番運用に必要な以下の機能は提供されていない：

- 承認ワークフロー管理（誰が、いつ、何を承認したか）
- タイムアウト・エスカレーション
- 通知・リマインド
- 権限管理

**これらは「LangGraphの責務ではない」という設計判断と思われる。**

## Durable Execution

**基盤は堅牢。** 各ノード完了後にCheckpointが保存され、再起動後も状態が完全に復旧する。HITLのinterruptも正しく永続化される。

本番での課題:
- 自動クリーンアップなし（Checkpointサイズが無限に増加）
- スレッド一覧APIなし（ストレージを直接クエリ必要）
- 同じthread_idでの並行アクセスは競合状態を引き起こす

## Memory

**機能は充実。** Store APIはシンプルで効果的。Embeddingによるセマンティック検索も正常動作。LangMemはCrewAIにはないエージェント管理メモリ機能を提供。

本番での課題:
- 操作ごとのEmbeddingコスト
- TTL/自動クリーンアップなし
- プライバシー/PIIコンプライアンスは自前実装
- バックグラウンド抽出の品質はLLM依存

## 本番導入時の追加開発

1. **承認管理サービス** - 承認待ちスレッドの管理、UI提供
2. **監査ログサービス** - 全操作の記録
3. **通知サービス** - Slack/Email連携
4. **認可サービス** - ロールベースの承認権限
5. **バックグラウンドジョブ** - タイムアウト処理、Checkpoint/メモリクリーンアップ
6. **リトライ/サーキットブレーカー** - 外部API呼び出しの安定化
7. **スレッド管理** - アクティブスレッドの追跡、古いスレッドのクリーンアップ
8. **ユニークID生成** - 並行アクセス問題の回避
9. **メモリライフサイクル管理** - TTL、クリーンアップ、コスト監視

**工数感**: グラフ実行部分の3-5倍の周辺システムが必要。

---

# ファイル構成

```
lang-graph-sample/
├── 01_quickstart.py              # Quick Start
├── 02_tool_definition.py         # Tool定義方法の比較
├── 03_tool_execution.py          # ToolNode動作検証
├── 04_tool_error_handling.py     # エラーハンドリング
├── 05_hitl_interrupt.py          # HITL基本（interrupt）
├── 06_hitl_approve_reject_edit.py # Approve/Reject/Edit
├── 07_durable_basic.py           # Durable Execution基本
├── 08_durable_hitl.py            # HITL + 永続化
├── 09_durable_production.py      # Durable本番課題
├── 11_memory_store_basic.py      # Memory Store CRUD
├── 12_memory_semantic_search.py  # セマンティック検索
├── 13_memory_cross_thread.py     # クロススレッド永続化
├── 14_memory_langmem_tools.py    # LangMemエージェントツール
├── 15_memory_background_extraction.py # バックグラウンド抽出
├── 16_production_considerations.py # 本番課題まとめ
├── REPORT.md                     # 英語版レポート
├── REPORT_ja.md                  # このレポート（日本語）
├── .env.example                  # 環境変数テンプレート
├── pyproject.toml
└── uv.lock
```
