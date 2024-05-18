import ast
import copy
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from asttokens import ASTTokens
from intervaltree import IntervalTree
from loguru import logger
from scipy.stats import mannwhitneyu
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    BatchEncoding,
)

from textattack.shared.utils import remove_comments_and_docstrings


@dataclass
class Token:
    idx: int
    value: str
    min: int
    max: int


class PerturbVisitor(ast.NodeTransformer):
    def __init__(self, target_node_id: str):
        self.target_node_id = target_node_id

    def visit_BinOp(self, node):
        if hasattr(node, 'nid') and node.nid == self.target_node_id:
            if isinstance(node.op, ast.Add):
                new_node = ast.BinOp(left=node.left, op=ast.Sub(), right=ast.UnaryOp(op=ast.USub(), operand=node.right))
            elif isinstance(node.op, ast.Sub):
                new_node = ast.BinOp(left=node.left, op=ast.Add(), right=ast.UnaryOp(op=ast.USub(), operand=node.right))
            elif isinstance(node.op, ast.Mult):
                new_node = ast.BinOp(left=node.left, op=ast.Div(),
                                     right=ast.BinOp(left=ast.Constant(value=1), op=ast.Div(), right=node.right))
            elif isinstance(node.op, ast.Div):
                new_node = ast.BinOp(left=node.left, op=ast.Mult(),
                                     right=ast.BinOp(left=ast.Constant(value=1), op=ast.Div(), right=node.right))
            elif isinstance(node.op, ast.LShift):
                new_node = ast.BinOp(left=node.left, op=ast.RShift(), right=node.right)
            elif isinstance(node.op, ast.RShift):
                new_node = ast.BinOp(left=node.left, op=ast.LShift(), right=node.right)
            else:
                return node
            return ast.copy_location(new_node, node)
        return self.generic_visit(node)


class NodeGradientAnalyzer:
    def __init__(self, *, source_code: str, mutation_targets: list, device: str = None):
        self.source_code = source_code
        self.device = device or self.get_device()
        self.mutation_targets = mutation_targets

        self.tokens: Optional[BatchEncoding] = None
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self.tree_with_locs: Optional[ASTTokens] = None

    @property
    def model(self):
        assert self._model, "No model loaded"
        return self._model

    @property
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
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load(self, *, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def parse_syntax_tree(self) -> ast.Module:
        logger.debug("Parsing Syntax Tree for Source Code")
        parse_tree: ast.Module = ast.parse(self.source_code)
        logger.debug("Syntax Tree:\n{}", ast.dump(parse_tree, indent=4))
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

    def build_token_map(
            self, *, token_tree: IntervalTree
    ) -> Dict[ast.AST, List[Token]]:
        no_loc_index = -1

        tree: ast.Module = self.parse_syntax_tree()
        self.tree_with_locs = ASTTokens(self.source_code, tree=tree)

        token_map: Dict[ast.AST, List[Token]] = defaultdict(list)

        for node in ast.walk(tree):
            loc = f"{getattr(node, 'lineno', no_loc_index)}_{getattr(node, 'col_offset', no_loc_index)}"
            node.nid = hashlib.md5(loc.encode('utf-8')).hexdigest()

            no_loc_index -= 1

            if node.__class__ not in self.mutation_targets:
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
        # if not hasattr(self.model, "model") or not hasattr(
        #         self.model.model, "embed_tokens"
        # ):
        #     raise ValueError("Error: cannot find model embed token function")

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

    def compute_node_gradients(
            self, strategy: str, percentile: float = 0.9, debug: bool = False
    ) -> Dict[ast.AST, float]:
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

            node_gradients[node] = agg.sum().log().item()

        if debug:
            self._print_node_gradients(node_gradients)
        return node_gradients

    def _print_node_gradients(self, node_gradients: Dict[ast.AST, float]):
        logger.info("Source:\n{}", self.source_code)
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
        # print(perturbed_code)
        return perturbed_code

    def evaluate_impact(self, perturbed_code: str) -> float:
        tokens = self.tokenizer(
            perturbed_code, return_offsets_mapping=True, return_tensors="pt"
        )
        tokens["input_ids"] = tokens["input_ids"].to(self.device)
        tokens["attention_mask"] = tokens["attention_mask"].to(self.device)
        # embeddings = self.model.model.embed_tokens(tokens["input_ids"])
        embeddings = self.model.transformer.wte(tokens['input_ids'])
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=tokens["attention_mask"],
            labels=tokens["input_ids"],
        )
        loss = outputs.loss.item()
        return loss

    def test_gradient_impact(self, num_samples: int = 10) -> Tuple[float, float]:
        node_gradients = self.compute_node_gradients(strategy="avg", debug=True)
        sorted_nodes = sorted(
            node_gradients.items(), key=lambda item: item[1], reverse=True
        )
        high_grad_nodes = [node for node, _ in sorted_nodes[:num_samples]]
        low_grad_nodes = [node for node, _ in sorted_nodes[-num_samples:]]

        high_impact_losses = []
        low_impact_losses = []

        for node in high_grad_nodes:
            perturbed_code = self.perturb_node(node)
            loss = self.evaluate_impact(perturbed_code)
            high_impact_losses.append(loss)

        for node in low_grad_nodes:
            perturbed_code = self.perturb_node(node)
            loss = self.evaluate_impact(perturbed_code)
            low_impact_losses.append(loss)

        logger.info("High impact losses: {}", high_impact_losses)
        logger.info("Low impact losses: {}", low_impact_losses)

        stat, p_value = mannwhitneyu(high_impact_losses, low_impact_losses)
        return stat, p_value


def main():
    source_code = """def financial_analysis(prices, quantities, daily_returns, risk_free_rate):
    portfolio_value = sum(p * q for p, q in zip(prices, quantities))
    asset_values = [p * q for p, q in zip(prices, quantities)]
    previous_prices = [p * 0.95 for p in prices]
    returns = [(p - pp) / pp for p, pp in zip(prices, previous_prices)]
    total_value = sum(asset_values)
    weights = [av / total_value for av in asset_values]
    weighted_returns = [w * r for w, r in zip(weights, returns)]
    portfolio_return = sum(weighted_returns)
    portfolio_variance = np.dot(weights, np.dot(np.cov(daily_returns), weights))
    portfolio_volatility = math.sqrt(portfolio_variance)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    historical_prices = [[p * (0.9 + 0.1 * np.random.rand()) for p in prices] for _ in range(10)]
    max_drawdown = max((max(hp) - min(hp)) / max(hp) for hp in zip(*historical_prices))
    cumulative_return = np.prod([1 + r for r in returns]) - 1
    average_return = sum(returns) / len(returns)
    volatility = np.std(daily_returns)
    variance = np.var(daily_returns)
    kurtosis = sum((r - average_return) ** 4 for r in returns) / (len(returns) * volatility ** 4) - 3
    skewness = sum((r - average_return) ** 3 for r in returns) / (len(returns) * volatility ** 3)
    beta = np.cov(daily_returns, returns)[0, 1] / np.var(daily_returns)
    alpha = average_return - beta * risk_free_rate
    """
    source_code = remove_comments_and_docstrings(source_code)
    analyzer = NodeGradientAnalyzer(
        source_code=source_code,
        mutation_targets=[
            ast.BinOp,
        ],
    )

    analyzer.load_from_hf(model_name="gpt2")
    stat, p_value = analyzer.test_gradient_impact(num_samples=20)
    logger.info("Mann Whitney U Test: {}", stat)
    logger.info("p-value: {}", p_value)


if __name__ == "__main__":
    main()
