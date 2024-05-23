from functools import cached_property

import numpy as np
import ast
import copy
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from inspect import isfunction

import torch
from asttokens import ASTTokens
from intervaltree import IntervalTree
from loguru import logger
from scipy.stats import ks_2samp
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    BatchEncoding,
)
from textattack.code_transformations.boolean_inversion import BooleanInversionVisitor
from examples import main as generate_examples


@dataclass
class Token:
    idx: int
    value: str
    min: int
    max: int


class PerturbVisitor(ast.NodeTransformer):
    def __init__(self, target_node_id: str):
        self.target_node_id = target_node_id
        self.visitor = BooleanInversionVisitor('')

    def visit_If(self, node):
        if hasattr(node, 'nid') and node.nid == self.target_node_id:
            return self.visitor.transform_node(node)[0]
        return self.generic_visit(node)

    def visit_While(self, node):
        if hasattr(node, 'nid') and node.nid == self.target_node_id:
            return self.visitor.transform_node(node)[0]
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if hasattr(node, 'nid') and node.nid == self.target_node_id:
            return self.visitor.transform_node(node)[0]
        return self.generic_visit(node)


class NodeGradientAnalyzer:
    model_repo = {}

    def __init__(self, *, source_code: str, mutation_targets: list, device: str = None):
        self.source_code = source_code
        self.device = device or self.get_device()
        self.mutation_targets = mutation_targets

        self.tokens: Optional[BatchEncoding] = None
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self.tree_with_locs: Optional[ASTTokens] = None

    @cached_property
    def model(self):
        assert self._model, "No model loaded"
        return self._model

    @cached_property
    def tokenizer(self):
        assert self._tokenizer, "No tokenizer loaded"
        return self._tokenizer

    @staticmethod
    def get_device() -> str:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logger.info("Using device '{}'", device)
        return device

    def load_from_hf(self, *, model_name: str):
        if NodeGradientAnalyzer.model_repo.get(model_name):
            model, tokenizer = self.model_repo[model_name]
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            NodeGradientAnalyzer.model_repo[model_name] = (model, tokenizer)

        self._model = model
        self._tokenizer = tokenizer

    def parse_syntax_tree(self) -> ast.Module:
        # logger.debug("Parsing Syntax Tree for Source Code")
        parse_tree: ast.Module = ast.parse(self.source_code)
        # logger.debug("Syntax Tree:\n{}", ast.dump(parse_tree, indent=4))
        return parse_tree

    def find_line_end(self, *, start_offset: int) -> int:
        target = start_offset
        while target < len(self.source_code) and self.source_code[target] != "\n":
            target += 1
        return target

    def get_tokens_with_positions(self) -> IntervalTree:
        tokens = self.tokenizer(
            self.source_code, return_offsets_mapping=True, return_tensors="pt"
        )
        self.tokens = tokens

        itree = IntervalTree()
        for i, token in enumerate(tokens["input_ids"][0]):
            token_str = self.tokenizer.decode([token.item()])
            start, end = tokens["offset_mapping"][0][i].tolist()

            if start == end:
                continue

            token = Token(idx=i, value=token_str, min=start, max=end)
            itree.addi(begin=start, end=end, data=token)

        tokens["input_ids"] = tokens["input_ids"].to(self.device)
        tokens["attention_mask"] = tokens["attention_mask"].to(self.device)
        return itree

    def is_mutation_target(self, node: ast.AST):
        if node.__class__ in self.mutation_targets:
            return True

        for t in self.mutation_targets:
            if isfunction(t):
                if t(node):
                    return True
        return False

    def build_token_map(self, *, token_tree: IntervalTree) -> Dict[ast.AST, List[Token]]:
        no_loc_index = -1

        tree: ast.Module = self.parse_syntax_tree()
        self.tree_with_locs = ASTTokens(self.source_code, tree=tree)

        token_map: Dict[ast.AST, List[Token]] = defaultdict(list)

        for node in ast.walk(tree):
            loc = f"{getattr(node, 'lineno', no_loc_index)}_{getattr(node, 'col_offset', no_loc_index)}"
            node.nid = hashlib.md5(loc.encode('utf-8')).hexdigest()

            no_loc_index -= 1

            if not self.is_mutation_target(node):
                continue

            start, end = self.tree_with_locs.get_text_range(node)

            if hasattr(node, "body"):
                logger.debug(
                    "Extracting first line from statement block {} on {}",
                    node.__class__,
                    node.lineno,
                )
                end = self.find_line_end(start_offset=start)

            enveloped_tokens = token_tree.envelop(start, end)

            for token in enveloped_tokens:
                token_map[node].append(token.data)

        return token_map

    def compute_gradients(self) -> torch.Tensor:
        embeddings = self.model.transformer.wte(self.tokens["input_ids"])
        embeddings.retain_grad()
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=self.tokens["attention_mask"],
            labels=self.tokens["input_ids"],
        )
        loss = outputs.loss
        self.model.zero_grad()
        loss.backward()
        return embeddings.grad[0]

    def compute_node_gradients(self, strategy: str, percentile: float = 0.9, debug: bool = False) -> Dict[
        ast.AST, float]:
        node_interval_tree: IntervalTree = self.get_tokens_with_positions()
        token_node_map = self.build_token_map(token_tree=node_interval_tree)
        gradients = self.compute_gradients()

        node_gradients = {}
        for node in token_node_map:
            node_tokens = token_node_map[node]
            logger.debug("AST Node {} has {} tokens", node.__class__, len(node_tokens))
            token_gradients = [gradients[token.idx] for token in node_tokens]
            stacked = torch.stack(token_gradients)

            if len(stacked) == 0:
                continue

            match strategy:
                case "norm":
                    agg = torch.norm(stacked, dim=0)
                case "rms":
                    agg = torch.sqrt(torch.mean(stacked ** 2, dim=0))
                case "sum":
                    agg = torch.sum(stacked, dim=0)
                case "max":
                    agg = torch.max(stacked, dim=0)[0]
                case "avg":
                    agg = torch.mean(stacked, dim=0)
                case "median":
                    agg = torch.median(stacked, dim=0)[0]
                case "percentile":
                    agg = torch.quantile(stacked, percentile or 0.9, dim=0)
                case _:
                    raise ValueError(f"Invalid aggregation method: {strategy}")

            node_gradients[node] = agg.norm().log().item()

        if debug:
            self._print_node_gradients(node_gradients)
        return node_gradients

    def _print_node_gradients(self, node_gradients: Dict[ast.AST, float]):
        # logger.info("Source:\n{}", self.source_code)
        sorted_gradients = sorted(
            node_gradients.items(), key=lambda item: item[1], reverse=True
        )
        logger.debug("\nAST Node Aggregated Gradients:")
        logger.debug("-" * 30)
        for node, gradient in sorted_gradients:
            node_type = type(node).__name__
            node_info = ast.dump(node)
            logger.debug(
                f"Node Type: {node_type}\nGradient: {gradient}\nNode Info: {node_info}\n"
                + "-" * 30
            )

    def perturb_node(self, node: ast.AST) -> str:
        perturb_visitor = PerturbVisitor(node.nid)

        # Visit the tree with the visitor that perturbs only the target node
        target_tree = copy.deepcopy(self.tree_with_locs.tree)
        perturbed_tree = perturb_visitor.visit(target_tree)

        perturbed_code = ast.unparse(perturbed_tree)
        return perturbed_code

    def evaluate_impact(self, perturbed_code: str) -> float:
        tokens = self.tokenizer(
            perturbed_code, return_offsets_mapping=True, return_tensors="pt"
        )
        tokens["input_ids"] = tokens["input_ids"].to(self.device)
        tokens["attention_mask"] = tokens["attention_mask"].to(self.device)
        embeddings = self.model.transformer.wte(tokens['input_ids'])
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=tokens["attention_mask"],
            labels=tokens["input_ids"],
        )
        loss = outputs.loss.item()
        return loss

    def test_gradient_impact(self, num_samples: int = 10) -> Tuple[List[float], List[float]]:
        node_gradients = self.compute_node_gradients(strategy="rms", debug=True)
        gradients = [value for _, value in node_gradients.items()]

        if len(node_gradients) == 0:
            return [], []

        try:
            high_threshold = np.percentile(gradients, 90)
            low_threshold = np.percentile(gradients, 10)
        except IndexError:
            return [], []

        high_grad_nodes = [node for node, value in node_gradients.items() if value >= high_threshold]
        low_grad_nodes = [node for node, value in node_gradients.items() if value <= low_threshold]

        # Adjust the number of samples if there are fewer nodes than desired
        high_grad_nodes = high_grad_nodes[:num_samples]
        low_grad_nodes = low_grad_nodes[:num_samples]

        high_impact_losses = []
        low_impact_losses = []

        for node in high_grad_nodes:
            perturbed_code = self.perturb_node(node)
            logger.debug(f"Perturbed Code (High Grad Node):\n{perturbed_code}")
            loss = self.evaluate_impact(perturbed_code)
            high_impact_losses.append(loss)

        for node in low_grad_nodes:
            perturbed_code = self.perturb_node(node)
            logger.debug(f"Perturbed Code (Low Grad Node):\n{perturbed_code}")
            loss = self.evaluate_impact(perturbed_code)
            low_impact_losses.append(loss)

        logger.info("High impact losses: {}", high_impact_losses)
        logger.info("Low impact losses: {}", low_impact_losses)

        return high_impact_losses, low_impact_losses


def main():
    examples = generate_examples(num_samples=100, k=5, depth=2)

    all_high_impact_losses = []
    all_low_impact_losses = []

    for i, example in enumerate(examples):
        source_code = example.strip()

        try:
            analyzer = NodeGradientAnalyzer(
                source_code=source_code,
                mutation_targets=[
                    ast.If,
                    ast.While,
                    lambda node: isinstance(node, ast.Assign) and (
                            (isinstance(node.value, ast.UnaryOp) and isinstance(node.value.op, ast.Not)) or isinstance(
                        node.value, ast.Compare))
                ],
            )

            analyzer.load_from_hf(model_name="gpt2")
            high_impact_losses, low_impact_losses = analyzer.test_gradient_impact(num_samples=5)

            all_high_impact_losses.extend(high_impact_losses)
            all_low_impact_losses.extend(low_impact_losses)
        except:
            logger.exception("Error with i")

    logger.info("All high impact losses: {}", all_high_impact_losses)
    logger.info("All low impact losses: {}", all_low_impact_losses)

    # Ensure there are no identical loss values between high and low impact lists
    if all_high_impact_losses == all_low_impact_losses:
        logger.error("High impact losses and low impact losses are identical. Check perturbation logic.")
    else:
        print(np.mean(all_high_impact_losses))
        print(np.mean(all_low_impact_losses))
        stat, p_value = ks_2samp(all_high_impact_losses, all_low_impact_losses)
        logger.info("KS test statistic: {}", stat)
        logger.info("p-value: {}", p_value)


if __name__ == "__main__":
    main()
