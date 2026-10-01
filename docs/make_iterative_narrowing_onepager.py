"""Render a one-page, evidence-bound explanation. Records only; no model inference."""
from pathlib import Path
from statistics import NormalDist
import hashlib
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'applications/design-comparison/results/development_001'
OUT = ROOT / 'output/pdf'
sys.path.insert(0, str(ROOT / 'src'))
from causal_decidability.compatible_set import compatible_set


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(name):
    return [json.loads(line) for line in (RUN / name).read_text().splitlines()]


def collect():
    generation = json.loads((RUN / 'generation.json').read_text())
    for name, expected in generation['artifacts'].items():
        assert digest(RUN / name) == expected, name
    cfg = json.loads((RUN / 'manifest.json').read_text())['config']
    cases = rows('public_cases.jsonl')
    labels = {r['case_id']: r for r in rows('private_labels.jsonl')}
    # Explicitly retrospective, explanatory selection; not an inferential sample.
    case = next(c for c in cases if c['family'] == 'linear'
                and labels[c['case_id']]['truth_name'] == 'context_switch')
    assert case['case_id'] == 'fc04b0fee691e4362c18'
    raw = next(r for r in rows('measurements.jsonl') if r['case_id'] == case['case_id'])
    n = cfg['samples']
    z = NormalDist().inv_cdf(1 - cfg['alpha_per_case_policy'] / (2 * len(case['menu'])))
    radii = [z * sigma / math.sqrt(n) + case['numerical_bound'] for sigma in case['known_sigmas']]
    means = [math.fsum(samples[:n]) / n for samples in raw['samples']]
    stages = []
    for cells in ([0, 1, 2, 3], [0, 1, 2, 3, 5], [0, 1, 2, 3, 5, 11]):
        result = compatible_set({k: [v[i] for i in cells] for k, v in case['predictions'].items()},
            [means[i] for i in cells], [radii[i] for i in cells],
            equivalence_groups=case['equivalence_groups'])
        display_cell = 1 if len(cells) == 4 else cells[-1]
        stages.append({'cumulative_cells': [case['menu'][i] for i in cells],
                       'displayed_cell': case['menu'][display_cell],
                       'mean': means[display_cell], 'comparison_radius': radii[display_cell],
                       'retained_groups': result['retained_groups'], 'outcome': result['outcome']})
    assert [len(s['retained_groups']) for s in stages] == [6, 2, 1]
    assert stages[-1]['retained_groups'] == [['context_switch']]
    data = {'status': 'retrospective illustrative replay; not an adaptive experiment',
        'case_selection': 'First linear context_switch case in archived generation order; chosen after outcomes for explanation.',
        'case_id': case['case_id'], 'truth': labels[case['case_id']]['truth_name'],
        'samples_per_cell': n, 'menu_cells': len(case['menu']), 'alpha': cfg['alpha_per_case_policy'],
        'noise_model': cfg['measurement_model'], 'numerical_bound': case['numerical_bound'],
        'stages': stages, 'full_context_1': {'mean': means[3], 'radius': radii[3]},
        'input_sha256': {str((RUN / name).relative_to(ROOT)): digest(RUN / name)
                        for name in ('manifest.json', 'public_cases.jsonl', 'measurements.jsonl', 'private_labels.jsonl')},
        'builder_sha256': digest(Path(__file__)),
        'classifier_sha256': digest(ROOT / 'src/causal_decidability/compatible_set.py')}
    return data


def render(data):
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from pypdf import PdfReader

    fonts = Path('/System/Library/Fonts/Supplemental')
    for name, filename in [('Note', 'Arial.ttf'), ('Note-Bold', 'Arial Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily('Note', normal='Note', bold='Note-Bold')
    W, H = A4
    margin, width = 36, W - 72
    navy, teal, muted = '#163048', '#08786F', '#526571'
    path = OUT / 'ITERATIVE_CAUSAL_NARROWING.pdf'
    OUT.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle('How interventions narrow causal explanations')
    c.setAuthor('Felix Borck')
    c.setSubject('Retrospective replay of a measured synthetic circuit; real-model evidence distinguished')

    def para(text, x, top, w, size=10, leading=None, color=navy, bold=False, limit=None):
        style = ParagraphStyle('p', fontName='Note-Bold' if bold else 'Note', fontSize=size,
                               leading=leading or size * 1.3, textColor=HexColor(color))
        p = Paragraph(text, style)
        _, h = p.wrap(w, H)
        if limit is not None:
            assert h <= limit, (text, h, limit)
        assert top + h < H - 15, (text, top + h)
        p.drawOn(c, x, H - top - h)
        return h

    def box(x, top, w, h, color, radius=8):
        c.setFillColor(HexColor(color))
        c.roundRect(x, H - top - h, w, h, radius, stroke=0, fill=1)

    para('CAUSAL DECIDABILITY  /  WORKED EXAMPLE', margin, 31, width, 8.5, color=teal, bold=True)
    para('How interventions narrow<br/>causal explanations', margin, 51, width, 26, 29, bold=True)
    para('Felix Borck  |  A concrete path from one patch effect to fewer compatible explanations',
         margin, 116, width, 9.3, color=muted)
    box(margin, 142, width, 29, '#FFF1D9', 5)
    para('EXISTING DATA  •  Retrospective replay of one measured synthetic circuit. No new inference.',
         margin + 10, 150, width - 20, 8.5, color='#765321', bold=True)

    para('The question: which internal computation produced the effect?', margin, 188, width, 12.3, bold=True)
    para('A and B are two internal channels. All six candidate classes predict the same full-patch '
         'output, about 0.199. Selective patches and a context change make their predictions disagree.',
         margin, 211, width, 10.1, limit=32)

    gap = 14
    cw = (width - 2 * gap) / 3
    top, ch = 254, 214
    titles = ['SHARED EFFECT', 'SELECT THE SOURCE', 'CHANGE THE CONTEXT']
    counts = [6, 2, 1]
    conditions = ['Transfer A + B<br/>Context 0', 'Transfer A only<br/>Context 0', 'Transfer A only<br/>Context 1']
    survivors = [
        'A alone; B alone; average; both required; context switch; reverse switch.',
        '<b>A always controls output</b><br/>or<br/><b>the context selects A/B.</b>',
        '<b>Context switch:</b><br/>A in context 0;<br/>B in context 1.',
    ]
    for i, stage in enumerate(data['stages']):
        x = margin + i * (cw + gap)
        box(x, top, cw, ch, '#EDF6F4' if i == 2 else '#EFF3F7')
        para(f'0{i+1}  {titles[i]}', x + 11, top + 12, cw - 22, 8, bold=True, color=teal)
        para(str(counts[i]), x + 11, top + 32, cw - 22, 35, 38, bold=True, color=teal if i == 2 else navy)
        para('compatible class' if i == 2 else 'compatible classes', x + 12, top + 73, cw - 24, 9.2)
        para(conditions[i], x + 12, top + 94, cw - 24, 10, 12, bold=True)
        number = f"{stage['mean']:.3f} ± {stage['comparison_radius']:.3f}"
        para(number, x + 12, top + 125, cw - 24, 15, 17, bold=True)
        para(survivors[i], x + 12, top + 155, cw - 24, 9.2, 12, limit=52)
        if i < 2:
            xx = x + cw + 3
            yy = H - top - 54
            c.setStrokeColor(HexColor(teal))
            c.setLineWidth(1.5)
            c.line(xx, yy, xx + 8, yy)
            c.line(xx + 5, yy + 3, xx + 8, yy)
            c.line(xx + 5, yy - 3, xx + 8, yy)

    para('Each step retains the earlier measurements. In step 3, A-only predicts about 0.199; '
         'the context switch predicts 0. The measured output separates those remaining rivals.',
         margin, 480, width, 10, limit=31)

    box(margin, 526, width, 74, '#F5F7F8', 5)
    para('WHAT THE NARROWING ESTABLISHES', margin + 12, 537, width - 24, 8.5, bold=True, color=teal)
    para('One class remains <b>within the declared candidate set and intervention menu</b>. '
         'Equivalent implementations are treated as one class. This is an illustrative replay, not a '
         'validated adaptive selection policy or a uniquely identified mechanism in an LLM.',
         margin + 12, 555, width - 24, 9.3, 12.5, limit=40)

    para('What already exists in GPT-2', margin, 617, width, 12, bold=True)
    para('<b>Source comparison:</b> the null-read candidate predicts better in 64/64 fresh pairs '
         '(Makelov-derived application). No absolute adequacy threshold was tested. '
         '<b>Later rounds:</b> simple route and invariance profiles were excluded. These are separate '
         'questions, not a single nested identification chain.', margin, 640, width, 9.5, 12.5, limit=53)

    para('The next prospective step', margin, 704, width, 12, bold=True)
    para('Freeze a next-intervention rule and its error budget; log the surviving set after each '
         'measurement; then test the remaining explanation on untouched cases. New rivals or failed '
         'controls can reopen the explanation.', margin, 726, width, 9.4, 12, limit=38)

    c.setStrokeColor(HexColor('#CED8DF')); c.setLineWidth(0.6)
    c.line(margin, H - 775, W - margin, H - 775)
    para('Measurement: 8 simulated observations/cell; known independent Gaussian errors. ± is a '
         'menu-wide simultaneous comparison radius (α=0.005, 16 cells), plus a numerical allowance; '
         'generic scores, not nats. Case fc04b0fee691e4362c18; development_001. Full anchors in both contexts are retained.',
         margin, 783, width, 7.2, 9.0, color=muted, limit=29)
    para('<link href="https://arxiv.org/abs/2301.04709v4" color="#08786F">Geiger et al.: causal abstraction</link>'
         '  |  <link href="https://arxiv.org/abs/2311.17030" color="#08786F">Makelov et al.: source comparison context</link>'
         '  |  Data and replay: ITERATIVE_CAUSAL_NARROWING.md', margin, 816, width, 7.2, 8, color=muted)
    c.save()
    reader = PdfReader(path)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    for phrase in ('Retrospective replay', '64/64', '0.200', '-0.002', '0.040'):
        assert phrase in text, phrase
    return path


def write_note(data):
    text = ['# How interventions narrow causal explanations', '',
        '**Retrospective replay of one measured synthetic development circuit.** No new model '
        'inference and no prospective adaptive selection claim.', '',
        'The first linear `context_switch` case in the archived generation order was selected '
        'after the run for illustration. This is not a representative sample or a new confirmation.', '',
        '| Cumulative evidence | Displayed score ± comparison radius | Compatible classes |',
        '|---|---:|---|']
    descriptions = ['Native/full anchors in both contexts', 'Add A-only in context 0', 'Add A-only in context 1']
    for description, stage in zip(descriptions, data['stages']):
        text.append(f"| {description} | {stage['mean']:.6f} ± {stage['comparison_radius']:.6f} | "
                    + '; '.join(' / '.join(g) for g in stage['retained_groups']) + ' |')
    text += ['', 'Seven implementation labels form six full-menu classes: `channel_a_alias` remains '
        'equivalent to `channel_a` under every allowed intervention. The final class agrees with '
        'the known executed graph. This does not assert unique implementation or natural LLM usage.', '',
        'The classifier consumes all previous cells at each stage; candidate equivalence is never '
        'redefined using only the selected cells. Both native/full context anchors are included, '
        'although the figure displays the context-0 full patch.', '',
        '## Measurement and uncertainty', '',
        'Each score is the mean of the first eight stored artificial noisy measurements of a '
        'deterministic float32 circuit. The known Gaussian model gives Bonferroni radii over the '
        'complete 16-cell menu at alpha=0.005; the calibrated numerical allowance is added. These '
        'are generic scores, not nats and not posterior probabilities of mechanism truth. '
        'The retrospective selection of an illustrative case is not a fresh population test.', '',
        'The displayed sequence was constructed after the run. The implemented selector chooses '
        'a batch of interventions before outcomes, not the next cell after each retained set. '
        'Other B-only cells are already in the archive: they cannot be described as unseen validation.', '',
        '## Real-model evidence is separate', '',
        '[Q1](CONFIRMED_CASE.md): null-read predicts better in 64/64 fresh GPT-2 pairs. No adequacy '
        'tolerance was declared. [Round 2](ROUND2_RESULT.md) and [Round 3A](ROUND3A_CONFIRMATION.md) '
        'exclude different narrow profiles; they are not successive subsets of this six-class space. '
        '[Round 3B](ROUND3B_STAGE_A_RESULT.md) stopped before patching.', '',
        '## Reproduce and inspect', '',
        '`python3 docs/make_iterative_narrowing_onepager.py` requires reportlab and pypdf. '
        'It reads and hash-checks existing data; it loads no model.', '',
        '[PDF](../output/pdf/ITERATIVE_CAUSAL_NARROWING.pdf) | '
        '[Exact values, cells and input hashes](ITERATIVE_CAUSAL_NARROWING.json)', '',
        '**Connection to Geiger:** causal abstraction relates a high-level explanation to an '
        'underlying computation through corresponding interventions. This demonstration concerns '
        'which measured interventions distinguish competing candidate computations; the present '
        'six-class example is not itself a validated semantic abstraction of an LLM. '
        '[Geiger et al.](https://arxiv.org/abs/2301.04709v4)', '']
    (ROOT / 'docs/ITERATIVE_CAUSAL_NARROWING.md').write_text('\n'.join(text))
    (ROOT / 'docs/ITERATIVE_CAUSAL_NARROWING.json').write_text(json.dumps(data, indent=2) + '\n')


def render_external_plan():
    """A second, separate one-page proposal. Symbolic checks are not Tracr outcomes."""
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.pagesizes import A4
    from pypdf import PdfReader
    W, H = A4
    margin, width = 36, W - 72
    navy, teal, muted = '#163048', '#08786F', '#526571'
    path = OUT / 'EXTERNAL_REPO_ITERATIVE_TEST.pdf'
    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle('An external demonstration of iterative causal narrowing')
    c.setAuthor('Felix Borck')

    def p(text, top, size=10, bold=False, color=navy, x=margin, w=width, leading=None, maxheight=100):
        style = ParagraphStyle('p', fontName='Note-Bold' if bold else 'Note', fontSize=size,
                               leading=leading or size * 1.3, textColor=HexColor(color))
        para = Paragraph(text, style)
        _, height = para.wrap(w, H)
        assert height <= maxheight and top + height < H-15, (text, height)
        para.drawOn(c, x, H-top-height)
        return height

    def box(top, height, color):
        c.setFillColor(HexColor(color))
        c.roundRect(margin, H-top-height, width, height, 6, stroke=0, fill=1)

    p('CAUSAL DECIDABILITY  /  NEXT EXTERNAL DEMONSTRATION', 31, 8.5, True, teal)
    p('From our benchmark<br/>to an external Transformer', 51, 26, True, leading=29)
    p('Felix Borck  |  Proposed test in DeepMind\'s Tracr repository', 116, 9.3, color=muted)
    box(142, 32, '#FFF1D9')
    p('PLANNED, NOT RUN  •  Upstream interfaces checked; intervention adapter still needed.',
      152, 8.5, True, '#765321', x=margin+10, w=width-20)

    p('Why Tracr?', 192, 12.3, True)
    p('Tracr compiles programs into Transformer weights. Its reversal program provides a known '
      'computation for independent checking. This is a third-party compiled Transformer, '
      'not a pretrained language model.', 214, 10, maxheight=41)

    p('The rivals: an address, or a copied answer?', 265, 12.3, True)
    p('<b>Address:</b> retrieve the recipient token at the transferred position. '
      '<b>Answer copy:</b> reproduce the donor\'s answer. Transfer donor query 1\'s intermediate '
      'address 2 into recipient query 0; count positions from zero, excluding BOS.',
      288, 9.6, maxheight=40)

    examples = [
        ('1  Shared prediction', ['A','B','C','D'], ['W','X','C','Z']),
        ('2  Separating condition', ['A','B','C','D'], ['W','X','Y','Z']),
        ('3  Prediction check', ['A','B','U','D'], ['W','X','Y','Z']),
    ]
    ts = ParagraphStyle('table', fontName='Note', fontSize=9, leading=12, textColor=HexColor(navy))
    hs = ParagraphStyle('header', parent=ts, fontName='Note-Bold', textColor=white)
    rows = [[Paragraph(t, hs) for t in ('Stage', 'Recipient', 'Donor', 'Address<br/>predicts', 'Copy<br/>predicts')]]
    expected = [('C','C'),('C','Y'),('U','Y')]
    for (stage, recipient, donor), pair in zip(examples, expected):
        address = len(donor) - 1 - 1
        assert (recipient[address], donor[::-1][1]) == pair
        assert recipient[::-1][0] == 'D'
        rows.append([Paragraph(value, ts) for value in
            (stage, '['+','.join(recipient)+']', '['+','.join(donor)+']', '<b>'+pair[0]+'</b>', '<b>'+pair[1]+'</b>')])
    table = Table(rows, colWidths=[153,106,106,79,width-444], rowHeights=[36,34,34,34])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),HexColor(navy)),
        ('BACKGROUND',(0,1),(-1,1),HexColor('#EFF3F7')),
        ('BACKGROUND',(0,2),(-1,2),HexColor('#EDF6F4')),
        ('BACKGROUND',(0,3),(-1,3),HexColor('#F5F7F8')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),
        ('RIGHTPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,-1),0.6,white)]))
    table.wrapOn(c, width, H); table.drawOn(c, margin, H-343-138)
    p('These are predictions, not observations. The unpatched recipient answers D in every row. '
      'The decisive change preserves the donor address while changing its answer.',
      491, 9.3, color=muted, maxheight=28)

    p('Build only what this test needs', 537, 12.3, True)
    steps = [
        ('1. Qualify the task and patch.', 'Compile reversal once; reproduce native outputs. Add a '
         'forward adapter, check the actual copied coordinates, BOS offset, identities and numerical precision.'),
        ('2. Freeze the narrowing rule.', 'Declare rivals, allowed cells, budget, stopping rules and '
         'sequential uncertainty. Keep compiler answer labels out of the decision process.'),
        ('3. Log the trajectory.', 'Show both candidates surviving the shared condition; use their '
         'disagreement to select the next condition. Preserve both, neither and invalid outcomes.'),
        ('4. Test fresh predictions.', 'Use untouched sequence families. A later test transfers to '
         'another program and compares selection policies at equal budgets.'),
    ]
    for i, (title, body) in enumerate(steps):
        p(f'<b>{title}</b> {body}', 560 + i*39, 9.4, leading=12, maxheight=36)

    box(727, 54, '#F5F7F8')
    p('<b>What a successful demonstration would establish:</b> an explicit intervention '
      'distinguished two declared interpretations, and the survivor predicted new outcomes. '
      'It would not by itself prove adaptive superiority or identify a native LLM mechanism.',
      738, 9.1, x=margin+11, w=width-22, leading=12, maxheight=37)
    p('<link href="https://github.com/google-deepmind/tracr" color="#08786F">google-deepmind/tracr</link>'
      '  |  pinned revision 9ce2b8c82b6b  |  compiler/lib.py: make_reverse',
      797, 7.5, color=muted)
    p('Verified interfaces: compiler/assemble.py and transformer/model.py. No built-in patch callback. '
      'Full plan, controls and data requirements: EXTERNAL_ITERATIVE_DEMONSTRATION.md',
      808, 7.2, leading=9, color=muted, maxheight=19)
    c.save()
    reader = PdfReader(path)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    for expected_text in ('PLANNED, NOT RUN', '[A,B,U,D]', 'prediction', 'Tracr'):
        assert expected_text in text
    return path


if __name__ == '__main__':
    data = collect()
    write_note(data)
    print(render(data))
    print(render_external_plan())
