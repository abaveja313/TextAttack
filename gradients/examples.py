import random
import textwrap


def generate_boolean_expression(variables):
    operators = ['and', 'or']
    comparators = ['==', '!=', '<', '>', '<=', '>=', 'is', 'is not']
    variable = random.choice(variables)
    value = random.choice([True, False])
    comparator = random.choice(comparators)
    expression = f"{variable} {comparator} {value}"
    if random.random() > 0.5:
        other_variable = random.choice(variables)
        expression += f" {random.choice(operators)} {other_variable} {random.choice(comparators)} {value}"
    return expression


def generate_if_statement(variables, depth=0):
    condition = generate_boolean_expression(variables)
    body = generate_code(variables, depth + 1)
    orelse = generate_code(variables, depth + 1) if random.random() > 0.5 else ""
    return f"if {condition}:\n{textwrap.indent(body, '    ')}" + (
        f"\nelse:\n{textwrap.indent(orelse, '    ')}" if orelse else "")


def generate_while_loop(variables, depth=0):
    condition = generate_boolean_expression(variables)
    body = generate_code(variables, depth + 1)
    return f"while {condition}:\n{textwrap.indent(body, '    ')}"


def generate_assignment(variables):
    variable = random.choice(variables)
    expression = generate_boolean_expression(variables)
    return f"{variable} = {expression}"


def generate_code(variables, depth=0, k=5):
    code = []
    for _ in range(random.randint(1, k)):
        choice = random.random()
        if choice < 0.3:
            code.append(generate_if_statement(variables, depth))
        elif choice < 0.6:
            code.append(generate_while_loop(variables, depth))
        else:
            code.append(generate_assignment(variables))
        if depth > 2:  # Limit depth of nested structures
            break
    return "\n".join(code)


def main(num_samples=10, k=5, depth=0):
    samples = []
    variables = [f'var{i}' for i in range(1, 6)]
    for _ in range(num_samples):
        code = generate_code(variables, k=k, depth=depth)
        samples.append(code)
    return samples
