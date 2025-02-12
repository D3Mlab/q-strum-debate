import llm
from llm_comparer import LLMComparer
import jinja2
import logging
from tqdm import tqdm

class GeneralComparer(LLMComparer):
    def __init__(self, logger: logging.Logger, llm: llm.LLM, prompts: dict, dataset_name: str = None, mode: str = 'value_merge', debate: bool = False, 
    debate_iter: int = 1, debate_mode: str = 'single'):
        super().__init__(logger, llm, dataset_name)
        self.extract_prompt = prompts['extract']
        self.attr_merge_prompt = prompts['attribute_merge']
        self.filter_prompt = prompts['filter']
        self.value_merge_prompt = prompts['value_merge']
        self.contrast_prompt = prompts['contrast']
        self.usefulness_prompt = prompts['usefulness']
        self.debate_prompt = prompts['debate']
        self.debate_json_prompt = prompts['debate_json']
        if debate_iter > 1:
            self.debate_feedback_prompt = prompts['debate_feedback']
        self.mode = mode
        self.debate = debate
        self.debate_iter = debate_iter
        self.debate_mode = debate_mode
    
    def compare(self, query: str, dest1: str, sentences1: list[str], dest2: str, sentences2: list[str], intermediate_results: dict = {}) -> tuple[dict, dict]:

        if intermediate_results != {}:
            self.logger.info("Using existing intermediate results")

            new_attr_1 = intermediate_results['extracted_atts_1']
            new_attr_2 = intermediate_results['extracted_atts_2']

            if self.mode == 'filter':
                merged_attr_1 = intermediate_results['merged_atts_1']
                merged_attr_2 = intermediate_results['merged_atts_2']

                merged_values_1 = intermediate_results['filter_1']
                merged_values_2 = intermediate_results['filter_2']
            elif self.mode == 'attr_merge':
                merged_attr_1 = intermediate_results['merged_atts_1']
                merged_attr_2 = intermediate_results['merged_atts_2']
                if 'merged_values_1' in intermediate_results and 'merged_values_2' in intermediate_results:
                    del intermediate_results['merged_values_1']
                    del intermediate_results['merged_values_2']
                merged_values_1 = merged_attr_1
                merged_values_2 = merged_attr_2

                new_attr_1 = merged_attr_1
                new_attr_2 = merged_attr_2
            elif self.mode == 'value_merge':
                merged_values_1 = intermediate_results['merged_values_1']
                merged_values_2 = intermediate_results['merged_values_2']
            else:
                if 'merged_values_1' in intermediate_results and 'merged_values_2' in intermediate_results:
                    del intermediate_results['merged_values_1']
                    del intermediate_results['merged_values_2']
                if 'merged_atts_1' in intermediate_results and 'merged_atts_2' in intermediate_results:
                    del intermediate_results['merged_atts_1']
                    del intermediate_results['merged_atts_2']
                merged_values_1 = intermediate_results['extracted_atts_1']
                merged_values_2 = intermediate_results['extracted_atts_2']
    
        else:  

            # Extract attributes from the two entities
            extractor = LLMExtract(self.llm, self.extract_prompt)
            attributes1 = extractor.extract_attributes(query, dest1, sentences1)
            attributes2 = extractor.extract_attributes(query, dest2, sentences2)

            intermediate_results['extracted_atts_1'] = attributes1
            intermediate_results['extracted_atts_2'] = attributes2

            self.logger.debug(f"Extracted attributes for {dest1}: {attributes1}")
            self.logger.debug(f"Extracted attributes for {dest2}: {attributes2}")
            assert type(attributes1) == dict and type(attributes2) == dict

            new_attr_1 = attributes1
            new_attr_2 = attributes2

            if self.mode == 'attr_merge' or self.mode == 'value_merge' or self.mode == 'filter':
                # Merge attributes
                attribute_merger = LLMAttributeMerge(dest1, dest2, attributes1, attributes2, self.llm, self.attr_merge_prompt, query)
                new_attr_1, new_attr_2 = attribute_merger.merge_attributes()

                intermediate_results['merged_atts_1'] = new_attr_1
                intermediate_results['merged_atts_2'] = new_attr_2

                self.logger.debug(f"Merged attributes for {dest1}: {new_attr_1}")
                self.logger.debug(f"Merged attributes for {dest2}: {new_attr_2}")
                assert type(new_attr_1) == dict and type(new_attr_2) == dict

                merged_values_1 = new_attr_1
                merged_values_2 = new_attr_2

            if self.mode == 'value_merge':
                # Merge values
                value_merger = LLMValueMerge(self.llm, self.value_merge_prompt, query)
                merged_values_1 = value_merger.merge_values(dest1, new_attr_1)
                merged_values_2 = value_merger.merge_values(dest2, new_attr_2)

                intermediate_results['merged_values_1'] = merged_values_1
                intermediate_results['merged_values_2'] = merged_values_2

                self.logger.debug(f"Merged values for {dest1}: {merged_values_1}")
                self.logger.debug(f"Merged values for {dest2}: {merged_values_2}")
                assert type(merged_values_1) == dict and type(merged_values_2) == dict
            elif self.mode == 'filter':

                filter = LLMContrast(dest1, dest2, merged_values_1, merged_values_2, self.llm, self.filter_prompt, query)
                filter = filter.contrast()

                assert dest1 in filter and dest2 in filter

                intermediate_results['filter_1'] = filter[dest1]
                intermediate_results['filter_2'] = filter[dest2]

                merged_values_1 = filter[dest1]
                merged_values_2 = filter[dest2]

                self.logger.debug(f"Filtered aspects for {dest1}: {merged_values_1}")
                self.logger.debug(f"Filtered aspects for {dest2}: {merged_values_2}")
            else:
                merged_values_1 = new_attr_1
                merged_values_2 = new_attr_2


        if self.debate:
            self.logger.info("Debating")
            # Debate
            debate = LLMDebate(self.llm, self.debate_prompt, self.debate_json_prompt, dest1, dest2, merged_values_1, merged_values_2, query, self.debate_feedback_prompt if self.debate_iter > 1 else None)
            full_aspects = list(set(merged_values_1.keys()) | set(merged_values_2.keys()))

            debates = {}

            debate_d1_json = {}
            debate_d2_json = {}

            if self.debate_mode == 'single':
                for aspect in tqdm(full_aspects, desc="Debating aspects"):
                    
                    debate_outs = []
                    debate_feedbacks = []
                    feedback = None
                    debate_out = None
                    for _ in range(self.debate_iter):
                        debate_out = debate.debate(aspect, feedback=feedback if self.debate_iter > 1 else None, prev_debate=debate_out if self.debate_iter > 1 else None)
                        debate_outs.append(debate_out)

                        if self.debate_iter > 1:
                            feedback = debate.debate_feedback(debate_out, aspect)
                            debate_feedbacks.append(feedback)


                    debate_json = debate.get_json(debate_out, aspect)
                    self.logger.debug(f"Debate for aspect '{aspect}': {debate_out}")
                    
                    if self.debate_iter > 1:
                        debates[aspect] = [(debate_outs[i], debate_feedbacks[i]) for i in range(len(debate_outs))]
                    else:
                        debates[aspect] = debate_outs

                    debate_d1_json[aspect] = debate_json[dest1]
                    debate_d2_json[aspect] = debate_json[dest2]
            elif self.debate_mode == 'all':
                debate_outs = debate.debate(str(full_aspects), feedback=None, prev_debate=None)
                debates = debate_outs
                for aspect in tqdm(full_aspects, desc="Processing debate results to JSON"):
                    debate_json = debate.get_json(debate_outs, aspect)
                    debate_d1_json[aspect] = debate_json[dest1]
                    debate_d2_json[aspect] = debate_json[dest2]
                

            intermediate_results['debate'] = debates
            
            debate_results = {
                dest1: debate_d1_json,
                dest2: debate_d2_json
            } 

            intermediate_results['debate_results'] = debate_results 

            merged_values_1 = debate_d1_json
            merged_values_2 = debate_d2_json
        else:
            if 'debate' in intermediate_results:
                del intermediate_results['debate'] 
            if 'debate_results' in intermediate_results:
                del intermediate_results['debate_results']           
            
        self.logger.info("Contrasting")
        # Contrast
        contrast = LLMContrast(dest1, dest2, merged_values_1, merged_values_2, self.llm, self.contrast_prompt, query)
        contrast = contrast.contrast()

        # intermediate_results['contrast'] = contrast

        self.logger.debug(f"Contrast between {dest1} and {dest2}: {contrast}")
        assert type(contrast) == dict
        

        # # Usefulness
        # usefulness = LLMUsefulness(self.llm, self.usefulness_prompt)
        # usefulness = usefulness.evaluate(contrast, query)

        # self.logger.debug(f"Usefulness of the comparison: {usefulness}")
        # assert type(usefulness) == dict

        return intermediate_results, contrast

class LLMExtract():
    def __init__(self, llm: llm.LLM, prompt: str):
        self.llm = llm
        self.prompt = prompt

        # prompt must have query, destination and sentences fields, and force output to JSON

    def extract_attributes(self, query: str, destination: str, sentences: list[str]) -> dict:
        # clear all new line characters from the sentences
        sentences = [sentence.replace('\n', ' ') for sentence in sentences]
        # join all the sentences into one string, separated by a new line and with numbering
        sentences = '\n'.join([f'[{i}] {sentence}' for i, sentence in enumerate(sentences, 1)])
        # formatted prompt using jinja2
        prompt = jinja2.Template(self.prompt).render(query=query, destination=destination, sentences=sentences)

        # generate the response
        response = self.llm.generate(prompt, max_tokens=4000)

        response = self.llm.parse_json(response)
        
        return response

class LLMAttributeMerge():
    def __init__(self, dest1: str, dest2: str, attributes1: dict, attributes2: dict, llm: llm.LLM, prompt: str, query: str):
        self.attributes1 = attributes1
        self.attributes2 = attributes2
        self.llm = llm
        self.dest1 = dest1
        self.dest2 = dest2
        self.prompt = prompt
        self.query = query

    def merge_attributes(self) -> dict:
        # Merge attributes from both entities
        attr1_keys, attr2_keys = set(self.attributes1.keys()), set(self.attributes2.keys())
        jinja_prompt = jinja2.Template(self.prompt).render(dest1=self.dest1, dest2=self.dest2, attributes1=str(attr1_keys), attributes2=str(attr2_keys), query=self.query)
        merged_attributes = self.llm.generate(jinja_prompt, max_tokens=4000)

        merged_attributes = self.llm.parse_json(merged_attributes)

        d1_merged = merged_attributes[self.dest1]
        d2_merged = merged_attributes[self.dest2]

        new_attr_1 = {}
        new_attr_2 = {}

        for key in attr1_keys:
            if key in d1_merged:
                if d1_merged[key] in new_attr_1:
                    new_attr_1[d1_merged[key]] += self.attributes1[key]
                else:
                    new_attr_1[d1_merged[key]] = self.attributes1[key]
            else:
                if key in new_attr_1:
                    new_attr_1[key] += self.attributes1[key]
                else:
                    new_attr_1[key] = self.attributes1[key]
        
        for key in attr2_keys:
            if key in d2_merged:
                if d2_merged[key] in new_attr_2:
                    new_attr_2[d2_merged[key]] += self.attributes2[key]
                else:
                    new_attr_2[d2_merged[key]] = self.attributes2[key]
            else:
                if key in new_attr_2:
                    new_attr_2[key] += self.attributes2[key]
                else:
                    new_attr_2[key] = self.attributes2[key]
        return new_attr_1, new_attr_2


class LLMValueMerge():
    def __init__(self, llm: llm.LLM, prompt: str, query: str):
        self.llm = llm
        self.prompt = prompt
        self.query = query

    def merge_values(self, dest: str, attributes: dict) -> str:
        # Merge values from both entities
        jinja_prompt = jinja2.Template(self.prompt).render(dest=dest, attributes=attributes, query=self.query)
        merged_values = self.llm.generate(jinja_prompt, max_tokens=4000)
        
        merged_values = self.llm.parse_json(merged_values)

        return merged_values

class LLMContrast():
    def __init__(self, dest1: str, dest2: str, attributes1: dict, attributes2: dict, llm: llm.LLM, prompt: str, query: str):
        self.attributes1 = attributes1
        self.attributes2 = attributes2
        self.llm = llm
        self.dest1 = dest1
        self.dest2 = dest2
        self.prompt = prompt
        self.query = query

    def contrast(self) -> str:
        # Contrast the two entities
        jinja_prompt = jinja2.Template(self.prompt).render(dest1=self.dest1, dest2=self.dest2, attributes1=self.attributes1, attributes2=self.attributes2, query=self.query)
        contrast = self.llm.generate(jinja_prompt, max_tokens=4000)

        contrast = self.llm.parse_json(contrast)

        return contrast

class LLMUsefulness():
    def __init__(self, llm: llm.LLM, prompt: str):
        self.llm = llm
        self.prompt = prompt

    def evaluate(self, comparison: dict, query: str) -> dict:
        # Evaluate the usefulness of the comparison
        jinja_prompt = jinja2.Template(self.prompt).render(comparison=comparison, query=query)
        usefulness = self.llm.generate(jinja_prompt, max_tokens=4000)

        usefulness = self.llm.parse_json(usefulness)

        return usefulness

class LLMDebate:
    def __init__(self, llm: llm.LLM, debate_prompt: str, json_prompt: str, dest1: str, dest2: str, sents1: dict, sents2: dict, query: str, debate_feedback_prompt: str = None):
        self.llm = llm
        self.debate_prompt = debate_prompt
        self.json_prompt = json_prompt
        self.dest1 = dest1
        self.dest2 = dest2
        self.sents1 = sents1
        self.sents2 = sents2
        self.query = query
        self.debate_feedback_prompt = debate_feedback_prompt

    def debate(self, aspect: str, feedback: str, prev_debate: str) -> str:
        # Debate between two entities
        jinja_prompt = jinja2.Template(self.debate_prompt).render(dest1=self.dest1, dest2=self.dest2, sents1=self.sents1, sents2=self.sents2, aspect=aspect, query=self.query, feedback=feedback, prev_debate=prev_debate)
        debate = self.llm.generate(jinja_prompt, max_tokens=4000, json=False)
        
        return debate
    
    def debate_feedback(self, debate: str, aspect: str) -> dict:
        # Debate between two entities
        jinja_prompt = jinja2.Template(self.debate_feedback_prompt).render(dest1=self.dest1, dest2=self.dest2, sents1=self.sents1, sents2=self.sents2, aspect=aspect, query=self.query, debate=debate)
        feedback = self.llm.generate(jinja_prompt, max_tokens=4000)

        feedback = self.llm.parse_json(feedback)

        return feedback

    def get_json(self, debate: str, aspect: str) -> dict:
        jinja_prompt = jinja2.Template(self.json_prompt).render(dest1=self.dest1, dest2=self.dest2, sents1=self.sents1, sents2=self.sents2, debate=debate, query=self.query, aspect=aspect)
        debate_json = self.llm.generate(jinja_prompt, max_tokens=4000)

        debate_json = self.llm.parse_json(debate_json)

        return debate_json        

