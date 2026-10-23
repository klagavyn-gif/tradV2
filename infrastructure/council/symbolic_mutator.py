"""
tradV2 — Council Symbolic Alpha Mutator (Genetic Programming Syntax Tree Crucible)
==================================================================================
A zero-token, high-speed Genetic Programming (GP) Syntax Tree engine for quantitative
alpha expression mutation, recombination, and Pareto frontier optimization.

Key Capabilities:
1. Bidirectional AST Parsing: Converts safe Python/Quantum DSL expressions to typed GP Trees and back.
2. Domain-Aware Genetic Operators: Subtree crossover, operator mutation, and Gaussian parameter jitter.
3. Pareto Frontier Filtering: Optimizes jointly for Net Expectancy/Profit Factor vs. Tree Simplicity
   (Occam's Razor complexity penalty to prevent curve-fitting).
4. Pure In-Memory Speed: Mutates and evaluates thousands of formulas in RAM in seconds with ZERO LLM API tokens.
"""

from __future__ import annotations

import ast
import copy
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from .quantum_lab import QuantumLabEngine, QuantumFactorEvaluation, evaluate_quantum_expression

logger = logging.getLogger(__name__)

# Standard Enriched Quantum Features & Domain-Aware Sampling Bounds
DEFAULT_FEATURE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "squeeze_ratio": (0.70, 1.25),
    "squeeze_momentum": (-1.5, 1.5),
    "wick_absorption": (0.5, 3.0),
    "upper_absorption": (0.5, 3.0),
    "vol_skew": (-1.5, 1.5),
    "kinetic_energy": (-2.5, 2.5),
    "efficiency_ratio": (0.2, 0.8),
    "rel_vol": (0.8, 2.5),
    "atr_pct": (0.5, 3.5),
    "pv_correlation": (-0.5, 0.5),
    "vwap_deviation": (-1.5, 1.5),
    "fractal_pinch": (0.5, 2.5),
    "bb_width": (0.01, 0.15),
    "paxg_divergence": (-2.0, 2.0),
    "btc_mom_diff": (-3.0, 3.0),
}

AVAILABLE_FEATURES: List[str] = list(DEFAULT_FEATURE_BOUNDS.keys())
COMPARISON_OPS: List[str] = [">", "<", ">=", "<="]
LOGICAL_OPS: List[str] = ["&", "|"]
TS_FUNCS: List[str] = ["ts_slope", "zscore", "ts_rank", "delta", "ts_mean"]
VALID_PERIODS: List[int] = [5, 10, 14, 16, 20]


# ---------------------------------------------------------------------------
# GP Tree Node Hierarchy
# ---------------------------------------------------------------------------

class GPNode:
    """Base abstract node in the Genetic Programming Expression Tree."""

    def to_dsl(self) -> str:
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def node_count(self) -> int:
        raise NotImplementedError

    def copy(self) -> GPNode:
        return copy.deepcopy(self)

    def get_subtrees(self) -> List[GPNode]:
        """Return all nodes in subtree including self."""
        nodes = [self]
        for child in self.children():
            nodes.extend(child.get_subtrees())
        return nodes

    def children(self) -> List[GPNode]:
        return []

    def replace_child(self, old_child: GPNode, new_child: GPNode) -> bool:
        return False


@dataclass
class ConstantNode(GPNode):
    value: float

    def to_dsl(self) -> str:
        return f"{self.value:.4f}".rstrip("0").rstrip(".")

    def depth(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1


@dataclass
class FeatureNode(GPNode):
    name: str

    def to_dsl(self) -> str:
        return self.name

    def depth(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1


@dataclass
class TsFuncNode(GPNode):
    func_name: str
    child: GPNode
    period: int = 14

    def to_dsl(self) -> str:
        return f"{self.func_name}({self.child.to_dsl()}, {self.period})"

    def depth(self) -> int:
        return 1 + self.child.depth()

    def node_count(self) -> int:
        return 1 + self.child.node_count()

    def children(self) -> List[GPNode]:
        return [self.child]

    def replace_child(self, old_child: GPNode, new_child: GPNode) -> bool:
        if self.child is old_child:
            self.child = new_child
            return True
        return self.child.replace_child(old_child, new_child)


@dataclass
class CompareNode(GPNode):
    left: GPNode
    op: str
    right: GPNode

    def to_dsl(self) -> str:
        return f"({self.left.to_dsl()} {self.op} {self.right.to_dsl()})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def node_count(self) -> int:
        return 1 + self.left.node_count() + self.right.node_count()

    def children(self) -> List[GPNode]:
        return [self.left, self.right]

    def replace_child(self, old_child: GPNode, new_child: GPNode) -> bool:
        if self.left is old_child:
            self.left = new_child
            return True
        if self.right is old_child:
            self.right = new_child
            return True
        return self.left.replace_child(old_child, new_child) or self.right.replace_child(old_child, new_child)


@dataclass
class LogicalNode(GPNode):
    left: GPNode
    op: str
    right: GPNode

    def to_dsl(self) -> str:
        return f"({self.left.to_dsl()} {self.op} {self.right.to_dsl()})"

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def node_count(self) -> int:
        return 1 + self.left.node_count() + self.right.node_count()

    def children(self) -> List[GPNode]:
        return [self.left, self.right]

    def replace_child(self, old_child: GPNode, new_child: GPNode) -> bool:
        if self.left is old_child:
            self.left = new_child
            return True
        if self.right is old_child:
            self.right = new_child
            return True
        return self.left.replace_child(old_child, new_child) or self.right.replace_child(old_child, new_child)


# ---------------------------------------------------------------------------
# Bidirectional DSL <-> AST Parser
# ---------------------------------------------------------------------------

def parse_dsl_to_tree(expression: str) -> GPNode:
    """Safely parse a Python/Quantum DSL expression string into a GPNode tree."""
    clean_expr = expression.strip()
    try:
        parsed_ast = ast.parse(clean_expr, mode="eval")
    except Exception as e:
        raise ValueError(f"Failed to parse DSL expression '{expression}': {e}") from e

    return _ast_to_gp_node(parsed_ast.body)


def _ast_to_gp_node(node: ast.AST) -> GPNode:
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.BitAnd):
            return LogicalNode(
                left=_ast_to_gp_node(node.left),
                op="&",
                right=_ast_to_gp_node(node.right),
            )
        elif isinstance(node.op, ast.BitOr):
            return LogicalNode(
                left=_ast_to_gp_node(node.left),
                op="|",
                right=_ast_to_gp_node(node.right),
            )
        elif isinstance(node.op, ast.Add):
            return CompareNode(
                left=_ast_to_gp_node(node.left),
                op="+",
                right=_ast_to_gp_node(node.right),
            )
        elif isinstance(node.op, ast.Sub):
            return CompareNode(
                left=_ast_to_gp_node(node.left),
                op="-",
                right=_ast_to_gp_node(node.right),
            )
        else:
            raise ValueError(f"Unsupported binary operator in AST: {ast.dump(node.op)}")

    elif isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only single-comparison expressions supported per node")
        op_obj = node.ops[0]
        op_map = {
            ast.Gt: ">",
            ast.Lt: "<",
            ast.GtE: ">=",
            ast.LtE: "<=",
            ast.Eq: "==",
            ast.NotEq: "!=",
        }
        op_str = op_map.get(type(op_obj))
        if not op_str:
            raise ValueError(f"Unsupported comparison operator: {type(op_obj)}")
        return CompareNode(
            left=_ast_to_gp_node(node.left),
            op=op_str,
            right=_ast_to_gp_node(node.comparators[0]),
        )

    elif isinstance(node, ast.Call):
        func_name = getattr(node.func, "id", "")
        if func_name in TS_FUNCS:
            child = _ast_to_gp_node(node.args[0])
            period = 14
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                period = int(node.args[1].value)
            return TsFuncNode(func_name=func_name, child=child, period=period)
        raise ValueError(f"Unsupported function call in GP parser: {func_name}")

    elif isinstance(node, ast.Name):
        return FeatureNode(name=node.id)

    elif isinstance(node, ast.Constant):
        return ConstantNode(value=float(node.value))

    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        if isinstance(node.operand, ast.Constant):
            return ConstantNode(value=-float(node.operand.value))
        raise ValueError("Unary negative operator supported only on constants")

    raise ValueError(f"Unsupported AST node type: {type(node)}")


# ---------------------------------------------------------------------------
# Genetic Tree Operators (Mutation, Crossover, Synthesis)
# ---------------------------------------------------------------------------

class TreeAlphaMutator:
    """
    Applies high-speed Genetic Programming operators on DSL Expression Trees:
    - Point Mutation (perturb constant, toggle operator)
    - Subtree Mutation (replace subtree with fresh terminal/comparison)
    - Subtree Crossover (swap subtrees between two trees)
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.features = features or AVAILABLE_FEATURES
        self.bounds = feature_bounds or DEFAULT_FEATURE_BOUNDS
        self.rng = rng or random.Random(42)

    def sample_constant_for_feature(self, feature_name: str) -> float:
        """Sample a realistic domain-aware constant threshold."""
        min_v, max_v = self.bounds.get(feature_name, (-1.0, 1.0))
        val = self.rng.uniform(min_v, max_v)
        return round(val, 3)

    def generate_random_leaf_comparison(self, max_depth: int = 1) -> CompareNode:
        """Create a single comparison rule: (feature op constant) or (ts_func(feature) op constant)."""
        feature = self.rng.choice(self.features)
        op = self.rng.choice(COMPARISON_OPS)
        const_val = self.sample_constant_for_feature(feature)

        if max_depth > 1 and self.rng.random() < 0.35:
            ts_func = self.rng.choice(TS_FUNCS)
            period = self.rng.choice(VALID_PERIODS)
            left_node = TsFuncNode(func_name=ts_func, child=FeatureNode(feature), period=period)
            if ts_func == "ts_rank":
                const_val = round(self.rng.uniform(0.1, 0.9), 2)
            elif ts_func == "zscore":
                const_val = round(self.rng.uniform(-2.0, 2.0), 2)
        else:
            left_node = FeatureNode(feature)

        return CompareNode(left=left_node, op=op, right=ConstantNode(const_val))

    def generate_random_tree(self, max_depth: int = 3) -> GPNode:
        """Generate a random boolean expression tree with depth <= max_depth."""
        if max_depth <= 1:
            return self.generate_random_leaf_comparison(max_depth=1)

        left = self.generate_random_tree(max_depth - 1)
        right = self.generate_random_tree(max_depth - 1)
        op = self.rng.choice(LOGICAL_OPS)
        return LogicalNode(left=left, op=op, right=right)

    def mutate_constants(self, tree: GPNode, jitter_pct: float = 0.15) -> GPNode:
        """Perturb numeric constants slightly using Gaussian/uniform jitter."""
        new_tree = tree.copy()
        for node in new_tree.get_subtrees():
            if isinstance(node, ConstantNode):
                delta = abs(node.value) * jitter_pct if abs(node.value) > 0.05 else jitter_pct
                shift = self.rng.uniform(-delta, delta)
                node.value = round(node.value + shift, 4)
        return new_tree

    def mutate_operators(self, tree: GPNode) -> GPNode:
        """Mutate a comparison or logical operator in the tree."""
        new_tree = tree.copy()
        candidates = [n for n in new_tree.get_subtrees() if isinstance(n, (CompareNode, LogicalNode))]
        if not candidates:
            return new_tree

        target = self.rng.choice(candidates)
        if isinstance(target, CompareNode):
            comp_swaps = {">": "<", "<": ">", ">=": "<=", "<=": ">="}
            target.op = comp_swaps.get(target.op, ">")
        elif isinstance(target, LogicalNode):
            target.op = "|" if target.op == "&" else "&"

        return new_tree

    def mutate_subtree(self, tree: GPNode, max_depth: int = 2) -> GPNode:
        """Pick a random subtree and replace it with a fresh random comparison rule."""
        new_tree = tree.copy()
        subtrees = new_tree.get_subtrees()
        # Prefer replacing comparison or logical nodes rather than single features
        valid_targets = [n for n in subtrees if isinstance(n, (CompareNode, LogicalNode))]
        if not valid_targets:
            return self.generate_random_leaf_comparison(max_depth=1)

        target = self.rng.choice(valid_targets)
        replacement = self.generate_random_leaf_comparison(max_depth=max_depth)

        if target is new_tree:
            return replacement

        new_tree.replace_child(target, replacement)
        return new_tree

    def crossover(self, parent1: GPNode, parent2: GPNode) -> Tuple[GPNode, GPNode]:
        """Subtree crossover: swap a random subtree of parent1 with a random subtree of parent2."""
        c1 = parent1.copy()
        c2 = parent2.copy()

        subs1 = [n for n in c1.get_subtrees() if isinstance(n, (CompareNode, LogicalNode))]
        subs2 = [n for n in c2.get_subtrees() if isinstance(n, (CompareNode, LogicalNode))]

        if not subs1 or not subs2:
            return c1, c2

        target1 = self.rng.choice(subs1)
        target2 = self.rng.choice(subs2)

        rep1 = target2.copy()
        rep2 = target1.copy()

        if target1 is c1:
            c1 = rep1
        else:
            c1.replace_child(target1, rep1)

        if target2 is c2:
            c2 = rep2
        else:
            c2.replace_child(target2, rep2)

        return c1, c2


# ---------------------------------------------------------------------------
# High-Speed In-Memory Population & Pareto Frontier Evaluator
# ---------------------------------------------------------------------------

@dataclass
class CandidateIndividual:
    factor_name: str
    entry_tree: GPNode
    exit_tree: GPNode
    direction: str = "BUY"
    net_expectancy: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    complexity: int = 0  # Total node count (lower is simpler)
    is_pareto_optimal: bool = False

    @property
    def entry_dsl(self) -> str:
        return self.entry_tree.to_dsl()

    @property
    def exit_dsl(self) -> str:
        return self.exit_tree.to_dsl()


class SymbolicAlphaForge:
    """
    Orchestrates the in-memory evolutionary search for robust quantitative alpha expressions.
    Runs completely in RAM with NumPy & Pandas, applying Pareto frontier selection.
    """

    def __init__(
        self,
        mutator: Optional[TreeAlphaMutator] = None,
        seed: int = 42,
        lab_engine: Optional[QuantumLabEngine] = None,
    ):
        self.mutator = mutator or TreeAlphaMutator(rng=random.Random(seed))
        self.lab_engine = lab_engine or QuantumLabEngine()
        self.rng = random.Random(seed)

    def evaluate_candidate(
        self,
        candidate: CandidateIndividual,
        df: pd.DataFrame,
        symbol: str = "BTCUSDT",
    ) -> CandidateIndividual:
        """Vectorially evaluate candidate expression on DataFrame and compute performance."""
        try:
            entry_expr = candidate.entry_dsl
            exit_expr = candidate.exit_dsl

            eval_res = self.lab_engine.backtest_extreme_factor(
                factor_name=candidate.factor_name,
                hypothesis="GP Tree Evolved Expression",
                entry_expr=entry_expr,
                exit_expr=exit_expr,
                df=df,
                symbol=symbol,
                direction=candidate.direction,
            )

            candidate.total_trades = eval_res.total_trades
            candidate.win_rate = eval_res.win_rate_pct
            candidate.profit_factor = eval_res.profit_factor
            candidate.net_expectancy = eval_res.net_expectancy_pct
            candidate.complexity = candidate.entry_tree.node_count() + candidate.exit_tree.node_count()

        except Exception as e:
            logger.debug("Failed evaluating candidate %s: %s", candidate.factor_name, e)
            candidate.total_trades = 0
            candidate.profit_factor = 0.0
            candidate.net_expectancy = -999.0
            candidate.complexity = candidate.entry_tree.node_count() + candidate.exit_tree.node_count()

        return candidate

    def filter_pareto_frontier(
        self,
        population: List[CandidateIndividual],
        min_trades: int = 8,
    ) -> List[CandidateIndividual]:
        """
        Identify non-dominated solutions on the 2D Pareto plane:
        Objective 1: Maximize Net Expectancy
        Objective 2: Minimize Complexity (Tree Node Count)
        """
        viable = [ind for ind in population if ind.total_trades >= min_trades and ind.profit_factor > 0.0]
        if not viable:
            return population[:5]

        pareto_set: List[CandidateIndividual] = []
        for ind_a in viable:
            dominated = False
            for ind_b in viable:
                if ind_a is ind_b:
                    continue
                # B dominates A if B has >= expectancy and <= complexity (with at least one strictly better)
                if (
                    ind_b.net_expectancy >= ind_a.net_expectancy
                    and ind_b.complexity <= ind_a.complexity
                    and (ind_b.net_expectancy > ind_a.net_expectancy or ind_b.complexity < ind_a.complexity)
                ):
                    dominated = True
                    break
            if not dominated:
                ind_a.is_pareto_optimal = True
                pareto_set.append(ind_a)

        # Sort Pareto set by Net Expectancy descending
        pareto_set.sort(key=lambda x: (x.net_expectancy, x.profit_factor), reverse=True)
        return pareto_set

    def evolve_alpha_pool(
        self,
        df: pd.DataFrame,
        seed_hypotheses: Optional[List[Dict[str, str]]] = None,
        population_size: int = 30,
        generations: int = 3,
        top_k: int = 5,
        symbol: str = "BTCUSDT",
    ) -> List[CandidateIndividual]:
        """
        Run in-memory evolutionary loop over DataFrame:
        1. Initialize population from seeds and random trees.
        2. Evaluate fitness vectorially.
        3. Perform Crossover, Constant Jitter, and Subtree Mutation.
        4. Extract Pareto-optimal Alpha candidates.
        """
        population: List[CandidateIndividual] = []

        # 1. Parse Seeds
        if seed_hypotheses:
            for i, seed in enumerate(seed_hypotheses):
                try:
                    e_tree = parse_dsl_to_tree(seed["entry_expression"])
                    x_tree = parse_dsl_to_tree(seed["exit_expression"])
                    ind = CandidateIndividual(
                        factor_name=seed.get("factor_name", f"Seed_{i+1}"),
                        entry_tree=e_tree,
                        exit_tree=x_tree,
                        direction=seed.get("direction", "BUY"),
                    )
                    population.append(ind)
                except Exception as e:
                    logger.warning("Could not parse seed hypothesis: %s", e)

        # Fill remaining population with random trees
        while len(population) < population_size:
            idx = len(population) + 1
            e_tree = self.mutator.generate_random_tree(max_depth=2)
            x_tree = self.mutator.generate_random_leaf_comparison(max_depth=1)
            population.append(
                CandidateIndividual(
                    factor_name=f"GP_Synth_Factor_{idx:03d}",
                    entry_tree=e_tree,
                    exit_tree=x_tree,
                    direction="BUY",
                )
            )

        # 2. Evolutionary Generations
        for gen in range(generations):
            logger.info("Executing GP Evolution Gen %d/%d (Pop: %d)...", gen + 1, generations, len(population))
            for ind in population:
                self.evaluate_candidate(ind, df, symbol=symbol)

            # Rank by Net Expectancy
            population.sort(key=lambda x: x.net_expectancy, reverse=True)
            elites = population[:max(4, population_size // 5)]

            # Generate offspring
            offspring: List[CandidateIndividual] = [copy.deepcopy(e) for e in elites]
            while len(offspring) < population_size:
                p1 = self.rng.choice(elites)
                p2 = self.rng.choice(elites)

                # 40% Crossover, 30% Constant Jitter, 30% Subtree Mutation
                r = self.rng.random()
                if r < 0.40:
                    new_e1, _ = self.mutator.crossover(p1.entry_tree, p2.entry_tree)
                    child_entry = new_e1
                    child_exit = p1.exit_tree.copy()
                elif r < 0.70:
                    child_entry = self.mutator.mutate_constants(p1.entry_tree, jitter_pct=0.15)
                    child_exit = self.mutator.mutate_constants(p1.exit_tree, jitter_pct=0.15)
                else:
                    child_entry = self.mutator.mutate_subtree(p1.entry_tree, max_depth=2)
                    child_exit = p1.exit_tree.copy()

                offspring.append(
                    CandidateIndividual(
                        factor_name=f"GP_Gen{gen+1}_{len(offspring)+1:03d}",
                        entry_tree=child_entry,
                        exit_tree=child_exit,
                        direction="BUY",
                    )
                )

            population = offspring

        # Final Evaluation of the Last Generation
        for ind in population:
            self.evaluate_candidate(ind, df, symbol=symbol)

        pareto_frontier = self.filter_pareto_frontier(population)
        logger.info("Evolution completed. Pareto Frontier identified %d non-dominated alphas.", len(pareto_frontier))
        return pareto_frontier[:top_k]
