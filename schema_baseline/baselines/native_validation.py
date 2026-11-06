"""Low-memory native dev loop; upstream score and reduction nodes are unchanged."""
import ast, copy, types
import common as c
import io_utils as io

def epoch_ast(class_name):
    """Preserve every official numerical AST node, including batch sums/rounding."""
    path=c.ROOT_OFFICIAL()/'src/model/model_process.py'
    tree=ast.parse(path.read_text('utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='process_epoch'))
    loops=[n for n in fn.body if isinstance(n,ast.For)]
    io.require(len(loops)==2 and ast.unparse(loops[0].iter)=='self.data_loader','Official validation loop changed')
    loops[0].body.append(ast.Delete(targets=[ast.Name(id=n,ctx=ast.Del()) for n in
        ('head','relation','tail','label','pred','batch_size_range','target_pred','ranks','batch')]))
    fn.body=[ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='no_grad',ctx=ast.Load()),args=[],keywords=[]))],body=fn.body)]
    return ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))


def transformed_epoch(up,processor):
    path=c.ROOT_OFFICIAL()/'src/model/model_process.py'
    module=epoch_ast(type(processor).__name__)
    namespace=dict(processor.process_epoch.__func__.__globals__)
    exec(compile(module,str(path)+'[v1f-no-grad-release]','exec'),namespace)
    return types.MethodType(namespace['process_epoch'],processor)


