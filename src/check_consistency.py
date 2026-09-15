"""
FAPE, consistency check for the paper and every document that repeats its numbers

Run it after changing a script, rerunning a notebook or editing a document, and
before committing:

    python src/check_consistency.py

It trains nothing and takes a few seconds. The executed notebooks are the record
of what each script printed, so the numbers are checked against their saved output:

 1. docs/results_table.md and docs/cross_domain_results_table.md are exactly what
    make_results_table.py builds now (that script also asserts its own sentences)
 2. Table 2 in the paper matches docs/results_table.md
 3. every value in RESULTS and MEPS_RESULTS is printed, on a line naming the same
    model, in that evaluation's Stage 2 notebook
 4. every three-decimal number and one-decimal percentage in the paper, README,
    outline and cross-domain tables is printed by the notebooks of the evaluation
    it describes or follows from that evaluation's RESULTS
 5. the headline counts in those documents match RESULTS
 6. the paper's reference list matches docs/references.md, every citation in the
    text has an entry and every entry is cited
 7. every figure the paper embeds exists
 8. the body stays within 7,000 words and the abstract within 200
 9. every notebook ran top to bottom without an error, and no notebook that runs
    a script is older than that script's last change
10. the feature fixes for MEPS and Folktables are still in place, and phrases
    corrected in earlier drafts have not come back
11. no script or notebook types a result with two or more decimals into printed or plotted
    text (titles, labels, print statements) instead of computing it; percentages typed
    into the Stage 1 notes are not covered; they were compared with a fresh run of every
    notebook on September 15 2026

Prints one line per problem and exits with status 1 if there is any.
"""

import ast
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Documents that state current results (the phase records in docs/ are dated history).
RESULT_DOCS = ['docs/paper_draft.md', 'README.md', 'docs/paper_outline.md',
               'docs/cross_domain_results_table.md']

STAGE2_NOTEBOOKS = {   # RESULTS key -> (notebook, header that starts the evaluation's block)
    'COMPAS': ('stage2_compas_threshold', None),
    'Folktables': ('stage2_folktables_threshold', None),
    'Law School': ('stage2_lawschool_threshold', None),
    'Lending Club': ('stage2_lendingclub_threshold', None),
    'Agricultural': ('stage2_agricultural_threshold', None),
    'FairGround (Education/law_school_lequy)': ('stage2_fairground_threshold', '--- Dataset: law_school_lequy '),
    'Student (math)': ('stage2_student_threshold', '--- Subject: math '),
    'MEPS': ('stage2_fairground_threshold', '--- Dataset: meps_panel_19_fy2015 '),
}
# Words that name an evaluation -> notebooks that print its numbers, and its RESULTS keys.
# A number has to come from the evaluation it describes: the notebooks together print
# about half of all values between 0.000 and 1.000, so matching against all of them
# would let half of all wrong numbers through.
EVALUATIONS = {
    ('COMPAS', 'recidivism'):
        (['stage2_compas_threshold', 'baseline_compas', 'eda_compas', 'group_size_check'], ['COMPAS']),
    ('Folktables', 'ACS', 'income prediction'):
        (['stage2_folktables_threshold', 'baseline_folktables', 'eda_folktables', 'group_size_check'], ['Folktables']),
    ('Law School', 'bar passage', 'law_school_lequy'):
        (['stage2_lawschool_threshold', 'baseline_lawschool', 'eda_lawschool', 'stage2_fairground_threshold'],
         ['Law School', 'FairGround (Education/law_school_lequy)']),
    ('Lending', 'income quartile'):
        (['stage2_lendingclub_threshold', 'baseline_lendingclub', 'eda_lendingclub'], ['Lending Club']),
    ('Agricultur', 'SBA', 'partnership'):
        (['stage2_agricultural_threshold', 'baseline_agricultural', 'eda_agricultural'], ['Agricultural']),
    ('FairGround', 'creditcard', 'adult'):   # shares law_school_lequy with Law School
        (['stage2_fairground_threshold', 'baseline_fairground', 'eda_fairground', 'stage2_lawschool_threshold'],
         ['FairGround (Education/law_school_lequy)', 'MEPS', 'Law School']),
    ('MEPS', 'healthcare', 'Healthcare'):
        (['stage2_fairground_threshold', 'baseline_fairground', 'threshold_holdout_check'], ['MEPS']),
    ('Student', 'Math', 'Portuguese'):
        (['stage2_student_threshold', 'baseline_student', 'eda_student', 'threshold_holdout_check'], ['Student (math)']),
}
MODEL_NAMES = {'LR': 'LogisticRegression', 'RF': 'RandomForest', 'GB': 'GradientBoosting'}
PRINTED_LABELS = {'acc': ('ACC=', 'Acc='), 'auc': ('AUC=',), 'dpd': ('DPD=', 'DP_diff='),
                  'eod': ('EOD=', 'EO_diff=')}

# Wording from earlier drafts that was wrong. docs/methodology_decisions.md quotes some
# of it on purpose, so it is not searched.
RETIRED_PHRASES = [
    'Fairness-Aware Production ML Pipeline Evaluation',   # names the project had before July
    'Fairness-Aware Predictive Ensemble',
    'compas_group_size_check',                            # replaced by group_size_check.py
    'seven evaluations',                                  # MEPS is the eighth
    'EEOC compliant',                                     # the 0.8 ratio is a research convention here
    'legally actionable',
    'zero fairness tests',                                # Breck et al. (2017) include one
    'non-deterministic in fairlearn',                     # random_state is passed to predict
]

# Fixed reference values that printed and plotted text may state directly: the 0.8 ratio
# convention and its inverse for adverse outcomes, and the DPD reference lines.
CONVENTIONS = {'0.80', '1.25', '0.05', '0.10', '0.20', '0.50', '1.00', '0.00'}

problems = []


def problem(where, message):
    problems.append(f'{where}: {message}')


def read(rel):
    with open(os.path.join(REPO_ROOT, rel), encoding='utf-8') as fh:
        return fh.read()


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, 'src', f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_output(name):
    """Printed text of an executed notebook, images left out."""
    with open(os.path.join(REPO_ROOT, 'notebooks', f'{name}.ipynb'), encoding='utf-8') as fh:
        nb = json.load(fh)
    parts = []
    for cell in nb['cells']:
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'stream':
                parts.append(''.join(out['text']))
            elif out.get('output_type') in ('execute_result', 'display_data'):
                parts.append(''.join(out.get('data', {}).get('text/plain', '')))
    return ''.join(parts)


def block(text, header):
    """The part of a notebook's output that belongs to one evaluation."""
    if header is None:
        return text
    start = text.find(header)
    if start < 0:
        return ''
    end = text.find(header.split(':')[0] + ':', start + len(header))
    return text[start:] if end < 0 else text[start:end]


def section(lines, first, last):
    """Lines from the heading that starts with `first` up to the next line that starts with `last`."""
    start = next((i for i, line in enumerate(lines) if line.startswith(first)), None)
    if start is None:
        return []
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(last)), len(lines))
    return lines[start:end + 1]


def check_generated_tables(make_tables):
    for rel, build in [('docs/results_table.md', make_tables.results_table_text),
                       ('docs/cross_domain_results_table.md', make_tables.cross_table_text)]:
        try:
            text = build()
        except AssertionError as err:
            line = traceback.extract_tb(err.__traceback__)[-1].lineno
            problem(f'src/make_results_table.py:{line}',
                    f'a sentence written for {rel} no longer matches the numbers ({err})')
            continue
        if read(rel) != text:
            problem(rel, 'differs from what src/make_results_table.py builds; run it and review the diff')


def check_table2(paper_lines):
    table = [line for line in read('docs/results_table.md').splitlines() if line.startswith('|')]
    if table[0] not in paper_lines:
        problem('docs/paper_draft.md', 'Table 2 header not found')
        return
    start = paper_lines.index(table[0])
    for offset, (want, have) in enumerate(zip(table, paper_lines[start:start + len(table)])):
        if want != have:
            problem(f'docs/paper_draft.md:{start + offset + 1}', f'Table 2 row differs from docs/results_table.md: {want}')


def check_results_printed(results):
    outputs = {}
    for key, entry in results.items():
        notebook, header = STAGE2_NOTEBOOKS[key]
        if notebook not in outputs:
            outputs[notebook] = notebook_output(notebook)
        text = block(outputs[notebook], header)
        for model, values in entry.items():
            if values is None:
                continue
            lines = [line for line in text.splitlines() if MODEL_NAMES[model] in line]
            for metric, value in values.items():
                if value is None:
                    continue
                labels = PRINTED_LABELS[metric.split('_')[1]]
                pattern = re.compile('|'.join(re.escape(f'{label}{value:.3f}') for label in labels) + r'(?!\d)')
                if not any(pattern.search(line) for line in lines):
                    problem(f'notebooks/{notebook}.ipynb',
                            f'RESULTS[{key!r}][{model!r}][{metric!r}] = {value:.3f} is not printed for {MODEL_NAMES[model]}')


def printed_numbers(notebook):
    """Every number a notebook printed, rounded to three decimals and to one."""
    three, one = set(), set()
    for token in re.findall(r'\d+\.\d+', notebook_output(notebook)):
        if len(token.split('.')[1]) >= 3:
            three.add(f'{float(token):.3f}')
        one.add(f'{float(token):.1f}')
    return three, one


def derived_numbers(entries):
    """Numbers the documents compute from RESULTS: the values, accuracy costs and percentage changes."""
    three, one = set(), set()
    for entry in entries:
        for values in entry.values():
            if values is None:
                continue
            for value in values.values():
                if value is not None:
                    three.add(f'{value:.3f}')
            for before, after in [('baseline_acc', 'dp_acc'), ('baseline_acc', 'eo_acc'),
                                  ('baseline_dpd', 'dp_dpd'), ('baseline_eod', 'eo_eod')]:
                if values.get(before) is None or values.get(after) is None:
                    continue
                three.add(f'{abs(values[before] - values[after]):.3f}')
                if values[before]:
                    one.add(f'{abs(values[before] - values[after]) / values[before] * 100:.1f}')
    return three, one


def check_number_sources(results):
    names = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(REPO_ROOT, 'notebooks', '*.ipynb'))]
    printed = {name: printed_numbers(name) for name in names}
    pools = {}

    def pool(named):
        """Numbers the named evaluations account for, as (three decimals, one decimal)."""
        key = frozenset(named)
        if key not in pools:
            notebooks = {nb for words in named for nb in EVALUATIONS[words][0]} if named else set(names)
            derived = derived_numbers(results[k] for words in (named or EVALUATIONS) for k in EVALUATIONS[words][1])
            pools[key] = ({v for nb in notebooks for v in printed[nb][0]} | derived[0],
                          {v for nb in notebooks for v in printed[nb][1]} | derived[1])
        return pools[key]

    for rel in RESULT_DOCS:
        for lineno, line in enumerate(read(rel).splitlines(), 1):
            numbers = [(m.start(), m.end(), m.group()) for m in re.finditer(r'(?<![\d.])\d\.\d{3}(?![\d.])', line)]
            numbers += [(m.start(), m.end(), m.group()) for m in re.finditer(r'(?<![\d.])\d+\.\d%', line)]
            if not numbers:
                continue
            mentions = sorted((m.start(), m.end(), words) for words in EVALUATIONS
                              for word in words for m in re.finditer(re.escape(word), line))
            breaks = [0] + [m.end() for m in re.finditer(r'(?<=\.)\s+(?=[A-Z])', line)] + [len(line)]
            for start, end, token in numbers:
                # A number belongs to the evaluation named just before it ("MEPS, 0.859") or just
                # after it ("0.920 for Agricultural"), to every evaluation its sentence names when
                # the sentence says "respectively", and to every evaluation on the line otherwise.
                first, last = next((a, b) for a, b in zip(breaks, breaks[1:]) if a <= start < b)
                before = [words for s, e, words in mentions if e <= start][-1:]
                after = [words for s, e, words in mentions if end <= s <= end + 40][:1]
                named = before + after or [words for _, _, words in mentions]
                if 'respectively' in line[first:last]:
                    named = [words for s, _, words in mentions if first <= s < last] or named
                percent = token.endswith('%')
                if token.rstrip('%') not in pool(named)[percent]:
                    source = 'the notebooks for the evaluation it describes' if named else 'any notebook'
                    problem(f'{rel}:{lineno}', f'{token} is not printed by {source} and does not follow from RESULTS')


def check_headline_counts(make_tables, paper_lines):
    everything = [(n, m, e[m]) for n, e in make_tables.ORDER for m in make_tables.MODELS if e[m] is not None]
    high = [r for _, _, r in everything if r['baseline_dpd'] > 0.2]
    low = [r for _, _, r in everything if r['baseline_dpd'] < 0.05]
    improved = sum(r['dp_dpd'] < r['baseline_dpd'] for r in high)
    worsened = sum(r['dp_dpd'] > r['baseline_dpd'] for r in low)
    abstract = '\n'.join(section(paper_lines, '## Abstract', '## 1.'))
    # The README does not state the counts; these three do.
    for where, text in [('docs/paper_draft.md abstract', abstract),
                        ('docs/paper_outline.md', read('docs/paper_outline.md')),
                        ('docs/cross_domain_results_table.md', read('docs/cross_domain_results_table.md'))]:
        for count, total in [(improved, len(high)), (worsened, len(low))]:
            if not re.search(rf'\b{count} of (?:the )?{total}\b', text):
                problem(where, f'does not state "{count} of {total}", the count RESULTS gives')


def check_references(paper):
    body, _, reference_section = paper.partition('\n## References\n')
    entries = [line for line in reference_section.strip().splitlines() if line.strip()]
    listed = [line for line in read('docs/references.md').splitlines()[1:] if line.strip()]
    if entries != listed:
        problem('docs/references.md', "does not match the paper's reference list")
    keys = []
    for entry in entries:
        author = re.match(r'(.+?)(?:,|\.) ', entry)
        year = re.search(r'\((\d{4})[^)]*\)|\((n\.d\.)\)', entry)
        if not author or not year:
            problem('docs/paper_draft.md', f'cannot read author and year from reference: {entry[:60]}')
            continue
        keys.append((author.group(1), year.group(1) or year.group(2)))
    for author, year in keys:
        if not re.search(re.escape(author) + r"[^()]{0,40}\(?" + re.escape(year), body):
            problem('docs/paper_draft.md', f'{author} ({year}) is in the reference list but never cited')
    name = r"[A-Z][A-Za-z\-]+"
    cited = re.findall(rf"({name})(?: et al\.| and {name})?(?:'s)? \((\d{{4}}|n\.d\.)\)", body)
    cited += re.findall(rf"\(([A-Z][A-Za-z\- ]+?)(?: et al\.| and {name})?, (\d{{4}}|n\.d\.)\)", body)
    for author, year in sorted(set(cited)):
        if (author, year) not in keys:
            problem('docs/paper_draft.md', f'cites {author} ({year}), which has no reference entry')


def check_figures(paper):
    for target in re.findall(r'!\[[^\]]*\]\(([^)\s]+)\)', paper):
        if not os.path.exists(os.path.normpath(os.path.join(REPO_ROOT, 'docs', target))):
            problem('docs/paper_draft.md', f'embedded figure {target} does not exist')


def check_limits(paper_lines):
    body = section(paper_lines, '## 1. Introduction', '## Declarations')
    words = len(' '.join(body).split())
    if words > 7000:
        problem('docs/paper_draft.md', f'body is {words} words, over the 7,000 limit')
    abstract = [line for line in section(paper_lines, '## Abstract', '---')
                if not line.startswith(('##', '---', '*'))]
    words = len(' '.join(abstract).split())
    if words > 200:
        problem('docs/paper_draft.md', f'abstract is {words} words, over the 200 limit')


def last_change(rel):
    """Commit time of a file's last change, or its modification time if it has uncommitted edits."""
    status = subprocess.run(['git', 'status', '--porcelain', '--', rel], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout
    if status.strip():
        return os.path.getmtime(os.path.join(REPO_ROOT, rel))
    stamp = subprocess.run(['git', 'log', '-1', '--format=%ct', '--', rel], cwd=REPO_ROOT,
                           capture_output=True, text=True).stdout.strip()
    return int(stamp) if stamp else os.path.getmtime(os.path.join(REPO_ROOT, rel))


def check_notebooks():
    use_git = subprocess.run(['git', 'rev-parse'], cwd=REPO_ROOT, capture_output=True).returncode == 0
    for path in sorted(glob.glob(os.path.join(REPO_ROOT, 'notebooks', '*.ipynb'))):
        rel = os.path.relpath(path, REPO_ROOT)
        with open(path, encoding='utf-8') as fh:
            nb = json.load(fh)
        code = [cell for cell in nb['cells'] if cell['cell_type'] == 'code']
        if any(cell.get('execution_count') is None for cell in code):
            problem(rel, 'has code cells that were never run')
        if any(out.get('output_type') == 'error' for cell in code for out in cell.get('outputs', [])):
            problem(rel, 'has an error in its output')
        name = os.path.splitext(os.path.basename(path))[0]
        script = f'src/{name}.py'
        runs_script = any(f'from {name} import' in ''.join(cell['source']) for cell in code)
        if not (use_git and runs_script and os.path.exists(os.path.join(REPO_ROOT, script))):
            continue
        local = [f'src/{m}.py' for m in re.findall(r'^\s*(?:from|import)\s+(\w+)', read(script), re.M)
                 if os.path.exists(os.path.join(REPO_ROOT, 'src', f'{m}.py'))]
        newest = max([script] + local, key=last_change)
        if last_change(newest) > last_change(rel):
            problem(rel, f'is older than the last change to {newest}; rerun it')


def check_feature_fixes():
    for rel in ['src/stage2_folktables_threshold.py', 'src/baseline_folktables.py']:
        for node in ast.walk(ast.parse(read(rel))):
            if isinstance(node, (ast.List, ast.Tuple, ast.Set)) and any(
                    isinstance(e, ast.Constant) and e.value == 'POVPIP' for e in node.elts):
                problem(f'{rel}:{node.lineno}', 'POVPIP is back in a column list; it is built from family income (Decision 25)')
    for rel in ['src/stage2_fairground_threshold.py', 'src/baseline_fairground.py']:
        if 'documented_feature_columns(' not in read(rel):
            problem(rel, "no longer uses FairGround's documented features; MEPS visit counts define its label (Decision 24)")


def check_typed_numbers():
    """Printed and plotted text has to compute its numbers; a typed one goes stale on the next rerun."""
    calls = {'print', 'set_title', 'suptitle', 'title', 'text', 'annotate', 'set_xlabel', 'set_ylabel',
             'xlabel', 'ylabel'}
    number = re.compile(r'(?<![\w.])[+-]?\d+\.\d{2,}(?![\w.])')
    sources = [(rel, read(rel)) for rel in sorted(glob.glob('src/*.py', root_dir=REPO_ROOT))]
    for path in sorted(glob.glob('notebooks/*.ipynb', root_dir=REPO_ROOT)):
        cells = json.loads(read(path))['cells']
        sources += [(f'{path} cell {i}', ''.join(c['source'])) for i, c in enumerate(cells) if c['cell_type'] == 'code']
    for where, source in sources:
        code = '\n'.join('' if line.lstrip().startswith(('%', '!')) else line for line in source.splitlines())
        for node in ast.walk(ast.parse(code)):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
            texts = node.args + [k.value for k in node.keywords if k.arg == 'label'] if name in calls else \
                [k.value for k in node.keywords if k.arg == 'label']
            for text in texts:
                parts = text.values if isinstance(text, ast.JoinedStr) else [text]
                for part in parts:
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        typed = [n for n in number.findall(part.value) if n.lstrip('+-') not in CONVENTIONS]
                        if typed:
                            problem(f'{where}:{node.lineno}', f'typed-in {", ".join(typed)} in {name or "label"} text; compute it')


def check_retired_phrases():
    files = (['README.md'] + sorted(glob.glob('docs/*.md', root_dir=REPO_ROOT))
             + sorted(glob.glob('src/*.py', root_dir=REPO_ROOT)) + sorted(glob.glob('notebooks/*.ipynb', root_dir=REPO_ROOT)))
    skip = {'docs/methodology_decisions.md', 'src/check_consistency.py'}
    for rel in files:
        if rel in skip:
            continue
        text = read(rel).lower()
        for phrase in RETIRED_PHRASES:
            if phrase.lower() in text:
                problem(rel, f'contains "{phrase}", corrected in an earlier draft')


def main():
    make_tables = load_module('make_results_table')
    results = dict(make_tables.R, MEPS=make_tables.MEPS)
    paper = read('docs/paper_draft.md')
    paper_lines = paper.splitlines()

    checks = [
        ('generated tables', lambda: check_generated_tables(make_tables)),
        ('Table 2 in the paper', lambda: check_table2(paper_lines)),
        ('RESULTS against notebook output', lambda: check_results_printed(results)),
        ('where each number comes from', lambda: check_number_sources(results)),
        ('headline counts', lambda: check_headline_counts(make_tables, paper_lines)),
        ('references and citations', lambda: check_references(paper)),
        ('embedded figures', lambda: check_figures(paper)),
        ('word limits', lambda: check_limits(paper_lines)),
        ('notebooks', check_notebooks),
        ('feature fixes and retired phrases', lambda: (check_feature_fixes(), check_retired_phrases())),
        ('numbers typed into printed or plotted text', check_typed_numbers),
    ]
    for label, run in checks:
        before = len(problems)
        run()
        found = len(problems) - before
        print(f'  {"ok  " if not found else "FAIL"}  {label}' + (f' ({found})' if found else ''))

    if problems:
        print(f'\n{len(problems)} problem(s):')
        for line in problems:
            print(f'  {line}')
        return 1
    print('\nEverything matches.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
