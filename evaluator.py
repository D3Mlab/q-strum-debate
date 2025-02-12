import llm
import jinja2


class LLMPairwiseEvaluator():
    def __init__(self, llm: llm.LLM, prompt: str):
        self.llm = llm
        self.prompt = prompt

    def evaluate(self, res_a: dict, res_b: dict, query: str, domain: str) -> dict:
        jinja_prompt = jinja2.Template(self.prompt).render(a=res_a, b=res_b, query=query, domain=domain)
        rating = self.llm.generate(jinja_prompt, max_tokens=4000)

        rating = self.llm.parse_json(rating)

        return rating