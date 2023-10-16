import math
import ast
import inspect
import sys
import numpy as np
import importlib
import amd.rocal.fn as fn

def add_imports(function_code, imports):
    # Join the list of import statements
    import_statements = "\n".join(imports)
    modified_function_code = f"{import_statements}\n\n{function_code}"
    return modified_function_code

def get_imports(sourceFile):
    tree = ast.parse(sourceFile)
    imports = []
    alias_dict = {}
    from_import_dict = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module = []
            for alias in node.names:
                module.append(alias.name)
                if alias.asname:
                    alias_dict[alias.name] = alias.asname
            imports.extend(module)
        elif isinstance(node, ast.ImportFrom):
            module = []
            for alias in node.names:
                module.append(alias.name)
                if alias.asname:
                    from_import_dict[alias.name] = alias.asname
            imports.extend(module)

    return imports, alias_dict, from_import_dict

def external_source(images, filePath = "", pythonScript="", dtype="", size=0, batch = True):
    # Get source code of the current file
    with open(filePath, 'r') as f:
        source = f.read()

    imports, alias_dict, from_import_dict = get_imports(source)

    print("Imported Modules: ", imports)
    print("Alias Imports: ", alias_dict)
    print("From Imports: ", from_import_dict)

    # python_function_code = inspect.getsource(pythonScript)

    importStrings = []

    for module in imports:
        if module in pythonScript:
            importStrings.append("import "+ module)

    for key, value in alias_dict.items():
        if value in pythonScript:
            importStrings.append("import "+ key + " as "+ value)

    modified_code = add_imports(pythonScript, importStrings)

    print(modified_code)
    output = fn.ExternalSource(images, source=modified_code, dtype=dtype, size=size, batch=batch)
    return output