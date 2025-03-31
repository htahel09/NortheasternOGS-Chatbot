

class RAGGenerator:
    def __init__(self, llm):
        self.llm = llm

    #def __call__(self, prompt):
    #    if not isinstance(prompt, list):
    #        prompt = [{"role": "user", "content": str(prompt)}]
    #    outputs = self.llm(prompt, max_new_tokens=256)
    #    generated = outputs[0]["generated_text"]
    #    return generated

    def summarize_abstract(self, content: str, query: str) -> str:
        prompt = (
            "You are Northeastern's university student visa assistant.\n"
            f"Query: {query}\n"
            f"Content: {'\n'.join(content)}\n"
            "If the query can be answered with the context provided in the content, frame an answer. "
            "Otherwise, state: 'As an Northeastern's OGS assistant I do not have an answer to that question.'"
        )
        messages = [{"role": "user", "content": prompt}]
        outputs = self.llm(messages, max_new_tokens=256)
        generated = outputs[0]["generated_text"]
        return generated