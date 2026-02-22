import torch
from torch import fx, nn
from torch.fx.node import Node
from torch.fx.graph import map_arg

from .rules import get_function_lipfn, get_module_lipfn

class LipUpperBound(fx.Interpreter):
    r"""Lipschitz upper bound computation."""

    def __init__(self, model: nn.Module) -> None:
        r"""Initialize the interpreter with the given Module."""
        self.mod = torch.fx.symbolic_trace(model)
        self.graph = self.mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, inputs, init_lip_const: float = 1.0):
        r"""Propagate the estimation through the graph.

        Args:
            inputs: Input arguments to the model (Only used to infer shapes).
            init_lip_const: Initial Lipschitz constant.
        """
        env : dict[str, Node] = {}

        def load_arg(a: Node)-> tuple:
            r"""Load argument from the environment."""
            return map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    err_msg = f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                    raise RuntimeError(err_msg)
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                outputs = inputs
                lip = init_lip_const
            elif node.op == 'get_attr':
                outputs = fetch_attr(node.target)
            elif node.op == 'call_function':
                lips = [arg.lip for arg in node.args]
                lip = get_function_lipfn(node.target)(lips)
                outputs = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                ## This should be adjudted to support lip computation for methods
                outputs = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                module = self.modules[node.target]
                arg_shape = node.args[0].shape
                arg_lip = node.args[0].lip
                lip = get_module_lipfn(module)(module, arg_shape)*arg_lip
                outputs = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            if isinstance(outputs, torch.Tensor):
                node.shape = outputs.shape
                node.dtype = outputs.dtype
                node.lip = lip

            env[node.name] = outputs

        return load_arg(self.graph.output)
