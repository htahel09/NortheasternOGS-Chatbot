

class RAGGenerator:
    def __init__(self, llm):
        self.llm = llm

    def summarize_abstract(self, content: str, query: str) -> str:
        prompt = f"""
        <|system|>
        You are an expert OGS Website Content assistant responding to user questions based strictly on the provided OGS Website Content. Provide a **short, direct answer** using **only** the information the OGS Website Content listed below.
        
        **Guidelines:**
        - **DO** answer concisely, using the fewest words necessary.
        - **DO NOT** explain, paraphrase, or repeat the user question.
        - **DO NOT** infer or guess; respond only if the answer is explicitly stated.
        - If the answer **does not exist** in the OGS Website Content, reply exactly:
          `As Northeastern's OGS assistant I do not have access to that information.`
        
        When finished, end your reply with: `<|endoftext|>` and nothing else.
        
        OGS Website Content:
        {"\n".join(content)}
        <|end|>
        <|user|>
        {query}
        <|end|>
        <|assistant|>
        """.strip()

        messages = [{"role": "user", "content": prompt}]
        outputs = self.llm(messages, max_new_tokens=256)
        generated = outputs[0]["generated_text"]
        return generated