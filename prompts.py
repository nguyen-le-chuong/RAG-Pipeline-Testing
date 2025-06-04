from langchain.prompts import PromptTemplate

prompt_template_with_options = """
あなたは日本語に堪能なアシスタントです。以下の文脈、関数呼び出しの結果、質問、選択肢に基づいて、Chain-of-Thought（CoT）を使用してステップごとに回答を導き出してください。関数呼び出しの結果がない場合は、文脈のみで推論してください。以下の例を参考に、論理的な推論を行い、選択肢から一つの回答を選んでください。

**例1（関数呼び出しあり）**:
質問: データフローダイアグラム（DFD）の中央ノードは何ですか？
選択肢: A) ノードA B) ノードB C) エッジC D) データストアD
文脈: DFDはノードA、ノードB、データストアDを含む。
関数呼び出しの結果: 画像はDFDで、ノードBが中央に位置し、ノードAとデータストアDに接続。
回答:
1. 質問はDFDの中央ノードを尋ねています。
2. 文脈では、DFDにノードA、ノードB、データストアDが含まれます。
3. 関数呼び出しの結果によると、ノードBが中央ノードです。
4. 選択肢B) ノードBが一致します。
**最終回答**: B) ノードB

**例2（関数呼び出しなし）**:
質問: 日本の首都はどこですか？
選択肢: A) 東京 B) 大阪 C) 京都 D) 福岡
文脈: 日本の首都は東京です。
関数呼び出しの結果: なし
回答:
1. 質問は日本の首都を尋ねています。
2. 文脈によると、首都は東京です。
3. 関数呼び出しの結果がないため、文脈のみで推論します。
4. 選択肢A) 東京が一致します。
**最終回答**: A) 東京

**質問**: {question}
**選択肢**: {options}
**文脈**: {context}
**関数呼び出しの結果**: {function_results}
**回答**:
ステップごとに推論を行い、「最終回答: [選択肢]」の形式で回答してください。
"""
prompt_with_options = PromptTemplate(
    input_variables=["context", "question", "options", "function_results"],
    template=prompt_template_with_options
)
"""
You are an assistant proficient in Japanese. Based on the provided context, function call results, question, and options, use Chain-of-Thought (CoT) to derive the answer step by step. If there are no function call results, reason based solely on the context. Refer to the examples below, perform logical reasoning, and select one answer from the options.

**Example 1 (with function call)**:
Question: What is the central node of a Data Flow Diagram (DFD)?
Options: A) Node A B) Node B C) Edge C D) Data Store D
Context: The DFD includes Node A, Node B, and Data Store D.
Function call result: The image is a DFD, with Node B located centrally, connected to Node A and Data Store D.
Answer:
1. The question asks about the central node of the DFD.
2. The context states that the DFD includes Node A, Node B, and Data Store D.
3. The function call result indicates that Node B is the central node.
4. Option B) Node B matches.
**Final Answer**: B) Node B

**Example 2 (no function call)**:
Question: What is the capital of Japan?
Options: A) Tokyo B) Osaka C) Kyoto D) Fukuoka
Context: The capital of Japan is Tokyo.
Function call result: None
Answer:
1. The question asks about the capital of Japan.
2. The context states that the capital is Tokyo.
3. Since there is no function call result, reasoning is based solely on the context.
4. Option A) Tokyo matches.
**Final Answer**: A) Tokyo

**Question**: {question}
**Options**: {options}
**Context**: {context}
**Function call result**: {function_results}
**Answer**:
Perform reasoning step by step and provide the answer in the format "Final Answer: [option]".
"""