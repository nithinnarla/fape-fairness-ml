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
    model, in that evaluation's intervention notebook (notebooks/stage2_*)
 4. every three-decimal number and every percentage, at one decimal or whole, in the paper, README,
    outline and cross-domain tables is printed by the notebooks of the evaluation
    it describes or follows from that evaluation's RESULTS
 5. the headline counts in those documents match RESULTS, including any sentence or
    figure caption that counts the evaluated model-domain pairs as a whole
 6. the paper's reference list matches docs/references.md, every citation in the
    text has an entry and every entry is cited. Works the phase records cite but the
    paper does not have entries in docs/reading_notes_references.md, so every citation
    in every document resolves without putting an uncited work in the paper's list
 7. every figure the paper embeds exists, every figure and table it captions is
    referred to in the text, and both are numbered in order from 1
 8. the body stays within 7,000 words, the body and declarations together stay within
    7,000 as well since JASIST does not say the declarations are excluded, and the
    abstract stays within 200. Table pipes, bold markers and figure paths are not words
 9. every notebook ran top to bottom without an error, and no notebook that runs
    a script is older than that script's last change
10. the feature fixes for MEPS and Folktables are still in place, and phrases
    corrected in earlier drafts have not come back
11. no script or notebook types a result with two or more decimals into printed or plotted
    text (titles, labels, print statements) instead of computing it. Percentages typed into
    the EDA and baseline notes are not covered; they were compared with a fresh run of
    every notebook on September 15 2026
12. the figure counts the paper, outline and README state match the files in figures/
13. no prose anywhere in the repository uses a typographic dash or quote, or a hyphen
    standing in for a dash. Code and inline code are left alone; only markdown,
    comments, docstrings and string literals are read
14. every figure drawn by a script you have edited has been written since that edit, so a
    label corrected in a script is redrawn before it is committed. Each of the 231 figures
    resolves to the one source that saves it
15. a notebook whose code cells changed carries new run timestamps, so its output and its
    figures come from the code it now holds rather than from an earlier run

Prints one line per problem and exits with status 1 if there is any.
"""

import ast
import functools
import glob
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tokenize
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
    """Numbers the documents compute from RESULTS: the values, accuracy costs and percentage changes.

    Percentage changes are collected at one decimal and as whole numbers, because the prose
    rounds both ways: "96.7%" in one sentence and "by 12% and 30%" in another.
    """
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
                    pct = abs(values[before] - values[after]) / values[before] * 100
                    one.add(f'{pct:.1f}')
                    one.add(f'{pct:.0f}')          # the prose also rounds to whole numbers
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
            # Whole-number percentages written as "by 12% and 30%", the idiom the prose uses for
            # an improvement in a metric. That form hid a wrong MEPS value the one-decimal
            # pattern never saw. Rates the data simply has, "default rises to about 40%", are a
            # different kind of claim and are left alone.
            for m in re.finditer(r'\bby (\d{1,3})%(?:\s+and\s+(\d{1,3})%)?', line):
                for group in (1, 2):
                    if m.group(group):
                        numbers.append((m.start(group), m.end(group), m.group(group) + '%'))
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
    # Anywhere a document counts the evaluated pairs as a whole, in a sentence or a figure
    # caption, the number is the one RESULTS holds; a caption is as easy to get wrong as a
    # sentence. "9 of the 14 pairs" counts a subset, so a number after "of" is left alone.
    total_pairs = re.compile(r'(?<!of )(?<!of the )\b(\d+)\s+(?:evaluated\s+)?(?:model-domain\s+)?pairs\b')
    for rel in RESULT_DOCS:
        for stated in total_pairs.findall(read(rel)):
            if int(stated) != len(everything):
                problem(rel, f'counts {stated} evaluated pairs; RESULTS holds {len(everything)}')


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


def check_exhibits(paper):
    """Every figure and table the paper captions is referred to in the text, and numbered in order."""
    body = paper.split('## 1. Introduction')[1].split('## Declarations')[0]
    captions = re.findall(r'^\*\*(Figure|Table) (\d+)\.\*\*', body, re.M)
    for kind in ('Figure', 'Table'):
        numbers = [int(n) for k, n in captions if k == kind]
        if numbers != list(range(1, len(numbers) + 1)):
            problem('docs/paper_draft.md', f'{kind} captions are numbered {numbers}, not in order from 1')
    for kind, number in captions:
        prose = '\n'.join(line for line in body.splitlines()
                          if not line.startswith('![') and not line.startswith(f'**{kind} {number}.**'))
        if not re.search(rf'\b{kind} {number}\b', prose):
            problem('docs/paper_draft.md', f'{kind} {number} is captioned but never referred to in the text')


def manuscript_words(lines):
    """Words as a journal counts them. The line that embeds a figure is a file path rather
    than manuscript text, and the pipes and asterisks that draw a markdown table or mark bold
    are punctuation, not words; its caption and its cells are counted like any other text."""
    text = ' '.join(line for line in lines if not line.startswith('!['))
    text = re.sub(r'\*{1,2}', '', text.replace('|', ' '))
    return len(re.sub(r'(?<![\w.])[:-]+(?![\w.])', ' ', text).split())


def check_limits(paper_lines):
    body = manuscript_words(section(paper_lines, '## 1. Introduction', '## Declarations'))
    if body > 7000:
        problem('docs/paper_draft.md', f'body is {body} words, over the 7,000 limit')
    # JASIST excludes the abstract, keywords, references and supplemental material from the
    # 7,000 words and says nothing about the declarations, so they are counted here too:
    # the submission has to hold under the strictest reading of the limit, not the kindest.
    declarations = manuscript_words(section(paper_lines, '## Declarations', '## References'))
    if body + declarations > 7000:
        problem('docs/paper_draft.md', f'body and declarations are {body + declarations} words '
                                       f'together ({body} and {declarations}), over the 7,000 limit')
    abstract = manuscript_words([line for line in section(paper_lines, '## Abstract', '---')
                                 if not line.startswith(('##', '---', '*'))])
    if abstract > 200:
        problem('docs/paper_draft.md', f'abstract is {abstract} words, over the 200 limit')


@functools.lru_cache(maxsize=1)
def uncommitted():
    """Every path with an uncommitted edit, read once: this runs over a few hundred figures."""
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout
    return {line[3:].strip().strip('"') for line in status.splitlines()}


@functools.lru_cache(maxsize=None)
def last_change(rel):
    """Commit time of a file's last change, or its modification time if it has uncommitted edits."""
    if rel in uncommitted():
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


FOLDER_VARIABLE = re.compile(r"\w*(?:FIGURES?|OUT)_?DIR\s*=\s*os\.path\.join\([^)]*?'figures',\s*'(\w+)'\s*\)")
FIGURE_NAME = re.compile(r"['\"]((?:[\w./-]*/)?[\w-]+\.png)['\"]")


def figures_drawn(rel):
    """The figures a script or notebook saves, as repo-relative paths.

    Scripts join a bare file name onto a folder held in a FIGURES_DIR or OUT_DIR variable;
    notebooks write the path out, relative to notebooks/. Both are read here, so two figures
    that share a file name in different folders stay apart."""
    if rel.endswith('.ipynb'):
        source = '\n'.join(''.join(cell['source']) for cell in json.loads(read(rel))['cells']
                           if cell['cell_type'] == 'code')
    else:
        source = read(rel)
    folders = set(FOLDER_VARIABLE.findall(source))
    drawn = set()
    for name in FIGURE_NAME.findall(source):
        if '/' in name:
            drawn.add(os.path.normpath(os.path.join('figures', name.split('figures/')[-1])))
        elif len(folders) == 1:
            drawn.add(f'figures/{next(iter(folders))}/{name}')
    return {path for path in drawn if os.path.exists(os.path.join(REPO_ROOT, path))}


CITATION = re.compile(r"\b([A-Z][A-Za-zÀ-ſ'-]+)(?:\s+(?:and|&)\s+[A-Z][A-Za-z'-]+|\s+et al\.?)?,?\s*\((\d{4})[a-z]?\)")
# The phase records, which cite more widely than the paper does.
THEORY_DOCS = ['docs/literature_review.md', 'docs/literature_analysis.md',
               'docs/research_design_rationale.md', 'docs/methodology_decisions.md']
# docs/references.md is the paper's list and stays exactly as the paper prints it. Works the
# phase records cite but the paper does not go in the second file.
REFERENCE_FILES = ['docs/references.md', 'docs/reading_notes_references.md']
NOT_AUTHORS = {'Decision', 'Section', 'Figure', 'Table', 'Stage', 'Phase', 'Protocol', 'Gap',
               'Panel', 'Appendix', 'Equation', 'Step'}


def check_doc_citations():
    """Every work the phase records cite has an entry in one of the two reference lists."""
    entries = []
    for rel in REFERENCE_FILES:
        if os.path.exists(os.path.join(REPO_ROOT, rel)):
            entries += [line for line in read(rel).splitlines() if line.strip()]
    for rel in THEORY_DOCS:
        for surname, year in sorted(set(CITATION.findall(read(rel)))):
            if surname in NOT_AUTHORS:
                continue
            if not any(re.search(rf'\b{re.escape(surname)}\b', line) and year in line for line in entries):
                problem(rel, f'cites {surname} ({year}), which has no entry in '
                             f'{" or ".join(os.path.basename(f) for f in REFERENCE_FILES)}')


def last_run(notebook):
    """When the notebook's stored run finished, from the timestamps nbclient records."""
    stamps = [cell['metadata']['execution'].get('shell.execute_reply')
              for cell in notebook['cells']
              if cell.get('metadata', {}).get('execution', {}).get('shell.execute_reply')]
    return max(stamps) if stamps else None


def check_figures_current():
    """Every figure a script draws has been written since the script was last edited.

    Only scripts with uncommitted edits are checked, which is the case that matters: a label
    corrected in the working tree and committed without drawing the figure again. Modification
    times decide it rather than the file's contents, since redrawing a figure whose appearance
    did not change leaves the same bytes and git would see no edit at all. Figures a notebook
    draws are covered by check_notebook_rerun instead: nbconvert saves the notebook after the
    figures it wrote, so their times cannot be compared this way."""
    if subprocess.run(['git', 'rev-parse'], cwd=REPO_ROOT, capture_output=True).returncode:
        return
    dirty = uncommitted()

    def written(rel):
        return os.path.getmtime(os.path.join(REPO_ROOT, rel))

    for rel in sorted(glob.glob('src/*.py', root_dir=REPO_ROOT)):
        drawn = figures_drawn(rel)
        if not drawn:
            continue
        source = read(rel)
        local = [f'src/{m}.py' for m in re.findall(r'^\s*(?:from|import)\s+(\w+)', source, re.M)
                 if os.path.exists(os.path.join(REPO_ROOT, 'src', f'{m}.py'))]
        sources = [rel] + local
        if not any(path in dirty for path in sources):
            continue
        newest = max(sources, key=written)
        for figure in sorted(drawn):
            if written(newest) > written(figure):
                problem(figure, f'has not been drawn since {newest} was edited; run {rel} again')


def check_notebook_rerun():
    """A notebook whose code changed has been run again, so its output and figures come from
    the code it now holds. Comparing the run timestamps with the committed copy says whether
    the edit was followed by a run, which a modification time cannot."""
    if subprocess.run(['git', 'rev-parse'], cwd=REPO_ROOT, capture_output=True).returncode:
        return
    for rel in sorted(glob.glob('notebooks/*.ipynb', root_dir=REPO_ROOT)):
        if rel not in uncommitted():
            continue
        committed = subprocess.run(['git', 'show', f'HEAD:{rel}'], cwd=REPO_ROOT,
                                   capture_output=True, text=True)
        if committed.returncode:      # a notebook added in this commit has nothing to compare
            continue
        before, after = json.loads(committed.stdout), json.loads(read(rel))
        code = lambda nb: [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
        if code(before) != code(after) and last_run(before) == last_run(after):
            problem(rel, 'has edited code cells but the same run timestamps as the committed '
                         'copy; run it again so its output and figures match the code')


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


def check_figure_counts():
    """The figure counts the paper, outline and README state match the files in figures/."""
    def count(folder, prefix=''):
        return sum(1 for f in os.listdir(os.path.join(REPO_ROOT, 'figures', folder))
                   if f.endswith('.png') and f.startswith(prefix))
    eda, baseline, stage = count('eda'), count('baseline'), count('stage2')
    aggregation, cross, drift = count('stage2', 'aggregation_'), count('stage2', 'cross_domain_'), count('stage2', 'drift_')
    domain = stage - aggregation - cross - drift
    statements = [
        ('docs/paper_draft.md', r'(\d+) EDA, (\d+) baseline and (\d+) intervention and monitoring figures',
         (eda, baseline, stage)),
        ('docs/paper_outline.md', r'(\d+) EDA, (\d+) baseline and (\d+) intervention and monitoring figures', (eda, baseline, stage)),
        ('README.md', r'with (\d+) EDA figures and (\d+) baseline figures', (eda, baseline)),
        ('README.md', r'with (\d+) figures: (\d+) from the seven domain scripts, (\d+) aggregation and (\d+) cross-domain',
         (domain + aggregation + cross, domain, aggregation, cross)),
        ('README.md', r'regress under the shift[^\n]*?; (\d+) figures committed', (drift,)),
    ]
    for rel, pattern, want in statements:
        found = re.search(pattern, read(rel))
        if not found:
            problem(rel, f'no figure count statement matching {pattern!r}')
        elif tuple(int(g) for g in found.groups()) != want:
            problem(rel, f'states figure counts {tuple(int(g) for g in found.groups())}, figures/ holds {want}')


# Named by code point so this file holds none of the characters it looks for.
TYPOGRAPHIC = {chr(code): name for code, name in [
    (0x2013, 'en dash'), (0x2014, 'em dash'), (0x2015, 'horizontal bar'), (0x2212, 'minus sign'),
    (0x2018, 'curly quote'), (0x2019, 'curly quote'), (0x201c, 'curly quote'), (0x201d, 'curly quote'),
    (0x2026, 'ellipsis character')]}
# A hyphen with a space on each side, standing in for a dash between two words.
SPACED_HYPHEN = re.compile(r"(?<=[\w)\]%.!?'\"])[ \t]-[ \t](?=[\w(\[+$'\"])")


def literal_text(token):
    """The literal part of a string token: an f-string's replacement fields hold code, not prose."""
    prefix = re.match(r"([rRbBuUfF]*)", token).group(1)
    if 'f' not in prefix.lower():
        return token
    out, buf, depth, i = [], [], 0, 0
    while i < len(token):
        char = token[i]
        if depth == 0 and token.startswith('{{', i):
            buf.append('{{'); i += 2; continue
        if depth == 0 and char == '{':
            out.append(''.join(buf)); buf = []; depth = 1; i += 1; continue
        if depth:
            depth += (char == '{') - (char == '}')
            i += 1; continue
        buf.append(char); i += 1
    out.append(''.join(buf))
    return ' '.join(out)


def prose_of_python(source):
    """Every comment, docstring and string literal, as (line number, text)."""
    masked = '\n'.join('#' + line if line.lstrip().startswith(('%', '!')) else line
                       for line in source.split('\n'))
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(masked).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return
    for token in tokens:
        if token.type == tokenize.COMMENT:
            yield token.start[0], token.string
        elif token.type == tokenize.STRING:
            yield token.start[0], literal_text(token.string)


def prose_pieces(rel):
    """Prose only: markdown, and the comments, docstrings and string literals of code."""
    if rel.endswith('.ipynb'):
        for index, cell in enumerate(json.loads(read(rel))['cells']):
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'markdown':
                yield f'cell {index}', source
            else:
                for line, text in prose_of_python(source):
                    yield f'cell {index} line {line}', text
    elif rel.endswith('.py'):
        for line, text in prose_of_python(read(rel)):
            yield f'line {line}', text
    else:
        yield '', read(rel)


def check_writing_marks():
    """No typographic dashes or quotes, and no hyphen standing in for a dash, in any prose."""
    files = (['README.md'] + sorted(glob.glob('docs/*.md', root_dir=REPO_ROOT))
             + sorted(glob.glob('src/*.py', root_dir=REPO_ROOT))
             + sorted(glob.glob('notebooks/*.ipynb', root_dir=REPO_ROOT)))
    for rel in files:
        for where, text in prose_pieces(rel):
            place = f'{rel}, {where}' if where else rel
            fences = re.sub(r'```.*?```|`[^`\n]*`', ' ', text, flags=re.S) if rel.endswith(('.md', '.ipynb')) else text
            for char, name in TYPOGRAPHIC.items():
                if char in fences:
                    problem(place, f'{name} ({char!r}) in prose; write it out or use plain ASCII')
            found = SPACED_HYPHEN.search(fences)
            if found:
                around = fences[max(0, found.start() - 40):found.end() + 40].replace('\n', ' ')
                problem(place, f'hyphen used as a dash in "{around.strip()}"')


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
        ('references and citations', lambda: (check_references(paper), check_doc_citations())),
        ('embedded figures', lambda: (check_figures(paper), check_exhibits(paper))),
        ('word limits', lambda: check_limits(paper_lines)),
        ('notebooks', check_notebooks),
        ('figures newer than the scripts that draw them', check_figures_current),
        ('notebooks rerun after a code change', check_notebook_rerun),
        ('feature fixes and retired phrases', lambda: (check_feature_fixes(), check_retired_phrases())),
        ('numbers typed into printed or plotted text', check_typed_numbers),
        ('figure counts', check_figure_counts),
        ('dashes and quotes in prose', check_writing_marks),
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
