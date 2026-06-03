import json, re
from typing import Dict

# MARK: Parsers

def bash_parser(action: str):
    # action = eval(action)
    action = re.sub("\n", " ", action)
    if '```' not in action:
        if action.lower().startswith(f"bash: "):
            action = action[len(f"bash: "):]
            return action, True
        if "command:" in action.lower():
            action = action[action.index("command:") + len("command:"):]
            return action, True

    pattern1 = r'```(?:bash|BASH|sh)?([\S\s]+?)```'
    pattern2 = r'```([\S\s]+?)```'
    matches = re.findall(pattern1, action.lower(), re.DOTALL) + re.findall(pattern2, action.lower(), re.DOTALL)
    if len(matches) == 0:
        return action, True 
    action = " ".join(matches[0].split())
    return action, True

def bash_parser_react(action: str):
    if action == "submit":
         return action, True
    pattern = r'execute\[(.*)\]'
    matches = re.findall(pattern, action, re.DOTALL)
    if len(matches) > 0:
        action = matches[0]
        return action, True
    return action, False

SQL_KEYWORDS = ["SHOW", "SELECT", "DESCRIBE", "DESC"]

def sql_parser(action: str):
    up_to_semicolon = lambda x: x[:x.index(";")] if ";" in x else x
    # additional guradrails for falcon, vicuna
    action = re.sub(r"\\_", "_", action)
    # action = re.sub("\\\*", "*", action)
    if '```' not in action:
        if action.startswith(f"SQL: "):
            action = up_to_semicolon(action[len(f"SQL: ")])
            return action, True
        for x in SQL_KEYWORDS:
            if x in action:
                action = up_to_semicolon(action[action.index(x):])
                return action, True
    
    pattern1 = r'```(?:sql|SQL)?([\S\s]+?)```'
    pattern2 = r'```([\S\s]+?)```'
    matches = re.findall(pattern1, action, re.DOTALL) + re.findall(pattern2, action, re.DOTALL)
    if len(matches) == 0:
        return action, False
    action = " ".join(matches[0].split())
    action = up_to_semicolon(action)
    if not action.split()[0].upper() in SQL_KEYWORDS:
        return action, False
    return action, True

def sql_parser_react(action: str):
    if action == "submit":
        return action, True
    pattern = r'execute\[(.*)\]'
    matches = re.findall(pattern, action, re.DOTALL)
    if len(matches) > 0:
        action = matches[0]
        if ";" in action:
            return action[:action.index(";")], True
        return action, True
    return action, False

def ctf_parser(action: str):
    action = action.strip()
    if action.startswith("Action:"):
        action = action[len("Action:"):].strip()
    return action, True

def _strip_python_block(action: str) -> str:
    print("===== Original Action (claude) =====")
    print(action)
    print("================")
    original = action
    action = action.strip()

    def remove_language_header(text: str) -> str:
        lines = text.splitlines()
        while lines:
            first = lines[0].strip().lower().rstrip(":")
            if not first:
                lines = lines[1:]
                continue
            if first in {"python", "python code", "code"}:
                lines = lines[1:]
                continue
            break
        return "\n".join(lines).strip()

    pattern = r'```(?:python|py)?([\s\S]+?)```'
    matches = re.findall(pattern, action, re.IGNORECASE)
    if matches:
        action = matches[0].strip()
    else:
        lowered = action.lower()
        if lowered.startswith("python "):
            action = action.split(" ", 1)[1].strip()
        else:
            action = remove_language_header(action)
    if action.lower().startswith("```"):
        action = action.strip("`").strip()

    if not action:
        return original

    action = action.replace("\r\n", "\n").strip()
    action = re.sub(r'([^\n])def\s', r'\1\ndef ', action)
    action = re.sub(r':\s*return', ':\n    return', action)
    action = re.sub(r'([\)\]\}])\s+(?=[A-Za-z_])', r'\1\n', action)

    if action.startswith("def"):
        replace_spaces = lambda match: '\n' + ' ' * (len(match.group(0)) - 1)
        action = re.sub(r' {5,}', replace_spaces, action)
    return action or original


def python_parser(action: str):
    cleaned = _strip_python_block(action)
    return cleaned, True


CLAUDE_BREAK_PATTERNS = [
    (r'(?<!\n)(from\b)', r'\n\1'),
    (r'(?<!\n)(import\b)', r'\n\1'),
    (r'(?<!\n)(def\b)', r'\n\1'),
    (r'(?<!\n)(class\b)', r'\n\1'),
    (r'(?<!\n)(return\b)', r'\n\1'),
    (r'(?<!\n)(print\s*\()', r'\n\1'),
    (r'(?<!\n)(for\b|while\b|if\b|elif\b|else\b|try\b|except\b|finally\b|with\b)', r'\n\1'),
]


def _claude_post_process(action: str) -> str:
    if not action:
        return action
    text = action
    for pattern, repl in CLAUDE_BREAK_PATTERNS:
        text = re.sub(pattern, repl, text)
    # Insert newlines before assignment statements that are crammed next to other statements
    text = re.sub(r'(?<!\n)(?<![\w])([A-Za-z_][\w]*\s*=\s*)(?!\=)', r'\n\1', text)
    text = re.sub(r':\s*(?=[A-Za-z_])', ':\n    ', text)
    text = re.sub(r'\n{2,}', '\n', text).strip()

    lines = []
    for raw in text.split("\n"):
        stripped = raw.strip()
        if not stripped:
            continue
        lines.append(stripped)

    formatted = []
    indent_level = 0
    block_stack = []
    for line in lines:
        keyword = line.split(" ", 1)[0]
        lowered = keyword.lower()
        if lowered in {"elif", "else", "except", "finally"} and indent_level > 0:
            indent_level = max(indent_level - 1, 0)
            if block_stack:
                block_stack.pop()
        formatted.append("    " * indent_level + line)
        if line.endswith(":") and lowered in {"def", "class", "for", "while", "if", "elif", "else", "try", "except", "finally", "with"}:
            indent_level += 1
            block_stack.append(lowered)
        elif lowered in {"return", "raise"} and indent_level > 0:
            indent_level = max(indent_level - 1, 0)
            if block_stack:
                block_stack.pop()
    return "\n".join(formatted)


# def python_claude_parser(action: str):
#     cleaned = _strip_python_block(action)
#     return cleaned, True


def python_claude_parser(action: str):
    text = action
    if text.startswith("```"):
        newline_idx = text.find("\n")
        if newline_idx != -1:
            text = text[newline_idx + 1 :]
        else:
            text = ""

        stripped = text.rstrip()
        if stripped.endswith("```"):
            stripped = stripped[:-3]
            trailing_len = len(text) - len(text.rstrip())
            text = stripped + text[len(text) - trailing_len :]

    return text, True


ACTION_PARSER_MAP = {
    "sql": sql_parser,
    "bash": bash_parser,
    "python": python_parser,
    "python_claude": python_claude_parser,
    "ctf": ctf_parser
}
ACTION_PARSER_MAP_REACT = {"sql": sql_parser_react, "bash": bash_parser_react}

# MARK: Handicaps

def handicap_bash(record: Dict) -> str:
    pass

def handicap_sql(record: Dict) -> str:
    # Custom handicap for spider dev dataset
    handicap = "MySQL tables, with their properties\n"
    tables = record["db_tables"]
    for name, columns in tables.items():
        handicap += f'- {name}: {str(columns)}\n'
    return handicap

HANDICAP_MAP = {"bash": handicap_bash, "sql": handicap_sql}

# MARK: Miscellaneous

def gen_react_demos(file_name, num_demos) -> list:
    """
    Generate ReAct style task demonstrations from multiturn evaluation log files
    
    Example Usage: gen_react_demos("path/to/ic_sql_multiturn_10_turns.json", 5)
    """
    trajectories = json.load(open(file_name, "r"))
    good_demos = []
    for _, v in trajectories.items():
        if v['summary']['max_reward'] == 1.0 and len(v['turn_history']['actions']) >= 3:
            good_demos.append({
                "query": v["query"],
                "sequence": v['turn_history']
            })
            if len(good_demos) >= num_demos:
                break
    if len(good_demos) < num_demos:
        print(f"Only got {len(good_demos)} demos (Requested {num_demos})")
    
    react_demo_str = ""
    for demo in good_demos:
        react_demo_str += f"Query: {demo['query']}\n"
        for turn in range(1, len(demo['sequence']['actions']) + 1):
            react_demo_str += f"Thought {turn}: \n"
            react_demo_str += f"Action {turn}: execute[{demo['sequence']['actions'][turn - 1]}]\n"
            obs = demo['sequence']['observations'][turn - 1]
            if "code was found in your last response." in obs:
                react_demo_str += f"Observation {turn}: Error executing query: Your last `execute` action did not contain SQL code\n"
            else:
                react_demo_str += f"Observation {turn}: {obs}\n"
        react_demo_str += f"Thought {turn+1}: \n"
        react_demo_str += f"Action {turn+1}: submit\n"
    return react_demo_str
