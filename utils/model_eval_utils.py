def get_last_boxed(text):
    boxed_start = r"\boxed"
    boxed_start_idx = text.rfind(boxed_start)
    if boxed_start_idx == -1:
        return None
 
    current_idx = boxed_start_idx + len(boxed_start)
 

    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1
 

    if current_idx >= len(text) or text[current_idx] != "{":
        return None
 
    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx
 

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1
 
    
    if brace_depth != 0:
        return None
 
    
    return text[content_start_idx:current_idx-1]

# Regular expression for extracting numeric values from the text
import re

RE_NUMBER = re.compile(
    r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
 
def extract_final_candidate(text, fallback="number_then_full"):
    # Default return value if nothing matches
    result = ""

    # Prefer the last boxed expression if present
    if text:
        boxed = get_last_boxed(text.strip())
        if boxed:
            result = boxed.strip().strip("$ ")
 
        # If no boxed expression, try fallback
        elif fallback in ("number_then_full", "number_only"):
            m = RE_NUMBER.findall(text)
            if m:
                result = m[-1] # Use last number
            elif fallback == "number_then_full":
                
                result = text # Else return full text if no number found
    return result

# LaTeX formatting to be replaced (left: original value, right: new value)
LATEX_FIXES = [
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot", "*"),
    (r"\u00B7|\u00D7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"°", ""),
]

# Strip chat special tokens like "<|assistant|>"
RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")

# Dictionary mapping to convert unicode superscripts to plaintext superscripts
SUPERSCRIPT_MAP = {
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
    "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-", "⁽": "(", "⁾": ")",
}

"""
This function takes an extracted answer string and rewrites it into a standardized format that we can reliably compare against reference solutions. 
It first strips away special tokens and unnecessary LaTeX clutter, such as \left, \right, or degree symbols. 
It then unwraps cases like \text{...}, removes inline math markers, and rewrites common structures into a calculator-style form. 
For example, 
    it turns \sqrt{a} into sqrt(a) and \frac{a}{b} into (a)/(b). 

Finally, it normalizes exponents, mixed numbers, and thousands separators and cleans up braces and casing. 

In short, the function transforms differently formatted LaTeX outputs into a clean, standardized string representation.
"""
def normalize_text(text):
    if not text:
        return ""
    text = RE_SPECIAL.sub("", text).strip()
 

    # Strip leading multiple-choice labels (e.g., like "c. 3" -> 3)
    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)
 
 
    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)

    # Remove angle-degree markers
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("°", "")

    # Unwrap "\text{...}" if the whole string is wrapped
    match = re.match(r"^\\text\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x")

    # Strip inline/display math wrappers: \( \) \[ \]
    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)
 
    # LaTeX canonicalization
    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)
 
 
    def convert_superscripts(s, base=None):
        converted = "".join(
            SUPERSCRIPT_MAP[ch] if ch in SUPERSCRIPT_MAP else ch
            for ch in s
        )
        if base is None:
            return converted
        return f"{base}**{converted}"
 
    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)
 
 
    # Normalize number and root expressions
    text = text.replace("\\%", "%").replace("$", "").replace("%", "")
    text = re.sub(
        r"\\sqrt\s*\{([^}]*)\}",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
    text = re.sub(
        r"\\sqrt\s+([^\\\s{}]+)",
        lambda match: f"sqrt({match.group(1)})",
        text,
    )
 
    # Convert LaTeX fractions into division form
    text = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
    text = re.sub(
        r"\\frac\s+([^\s{}]+)\s+([^\s{}]+)",
        lambda match: f"({match.group(1)})/({match.group(2)})",
        text,
    )
 
    # Handle exponents and mixed numbers
    text = text.replace("^", "**")
    text = re.sub(
        r"(?<=\d)\s+(\d+/\d+)",
        lambda match: "+" + match.group(1),
        text,
    )
 
    # Remove thousands separators in numbers
    text = re.sub(
        r"(?<=\d),(?=\d\d\d(\D|$))",
        "",
        text,
    )
 
    return text.replace("{", "").replace("}", "").strip().lower()


# SymPy parser for mathematical equality check
from sympy.parsing import sympy_parser as spp
from sympy.core.sympify import SympifyError
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError
 
def sympy_parser(expr):
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,

                spp.implicit_multiplication_application,
            ),
 
            evaluate=True,
        )
    except (SympifyError, SyntaxError, TypeError, AttributeError,
            IndexError, TokenError, ValueError, PolynomialError):
        return None

# Equality check function using SymPy
from sympy import simplify
 
def equality_check(expr_gtruth, expr_pred):
    # First, check if the two expressions are exactly the same string
    if expr_gtruth == expr_pred:
        return True
 
    # Parse both expressions into SymPy objects (returns None if parsing fails)
    gtruth, pred = sympy_parser(expr_gtruth), sympy_parser(expr_pred)

    # If both expressions were parsed successfully, try symbolic comparison
    if gtruth is not None and pred is not None:
        try:
            return simplify(gtruth - pred) == 0 # If the difference is 0, they are equivalent
        except (SympifyError, TypeError):
            pass
 
    return False

# Helper function to split tuple-like expressions
def split_into_parts(text):
    result = [text]

    # Check if text looks like a tuple or list, e.g. "(a, b)" or "[a, b]"
    if text:
        if (
            len(text) >= 2
            and text[0] in "([" and text[-1] in ")]"
            and "," in text[1:-1]
        ):
            # Split on commas inside brackets and strip whitespace
            items = [p.strip() for p in text[1:-1].split(",")]
            if all(items):
                result = items
    else:
        result = [] # If text is empty, return an empty list
 
    return result

# Function to grade predicted answers against ground truth
def grade_answer(pred_text, gt_text):
    # Default outcome if checks fail
    result = False
    # Only continue if both inputs are non-empty strings
    if pred_text is not None and gt_text is not None:
        gt_parts = split_into_parts(
            normalize_text(gt_text)
        )
        pred_parts = split_into_parts(
            normalize_text(pred_text)
        )

        # Ensure both sides have same number of valid parts
        if (gt_parts and pred_parts
           and len(gt_parts) == len(pred_parts)):
            # Check each part for mathematical equivalence
            result = all(
                equality_check(gt, pred)
                for gt, pred in zip(gt_parts, pred_parts) 
            )
 
    return result

def render_prompt(prompt):
    template = (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Problem:\n{prompt}\n\nAnswer:"
    )
    return template

