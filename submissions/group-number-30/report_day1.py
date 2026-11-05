"""Generate Q1/Q2 and figures from actual completed experiment records."""
import argparse
import json
from pathlib import Path
import shutil

NAMES={'word_lr':'Word TF-IDF + logistic regression','char_lr':'Character TF-IDF + logistic regression',
       'word_nb':'Word TF-IDF + ComplementNB','word_svm':'Word TF-IDF + calibrated linear SVM',
       'style_lr':'Style features + logistic regression','pos_lr':'POS n-grams + logistic regression',
       'surprisal_lr':'Class-conditional POS surprise + logistic regression','word_style_lr':'Word TF-IDF + style + logistic regression'}


def read(root,name): return json.loads((root/name).read_text())
def settings(r): return ', '.join(f'{k}={v}' for k,v in r['config'].items() if k!='kind')
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def esc(s):
    mapping={'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#','{':r'\{','}':r'\}','$':r'\$'}
    return ''.join(mapping.get(c,c) for c in str(s))
def table(headers,rows,colspec):
    return '\n'.join([r'\begin{center}\small\begin{tabular}{'+colspec+'}',r'\toprule',
        ' & '.join(map(esc,headers))+r' \\',r'\midrule',*[' & '.join(map(esc,row))+r' \\' for row in rows],
        r'\bottomrule\end{tabular}\end{center}'])


def generate(root):
    root=Path(root).resolve(); runs=read(root,'comparison.json'); curves=read(root,'learning_curve.json')
    selection=read(root,'selection.json'); best=selection.get('selected',selection)
    if 'config' not in best: best=max(runs,key=lambda r:r['validation_auroc'])
    curves=sorted(curves,key=lambda r:r['train_rows'])
    if len(runs)!=10 or {r['config']['kind'] for r in runs}!={'word_lr','char_lr','word_nb','word_svm'}:
        raise ValueError('Expected all ten predeclared core comparisons')
    if [r['train_rows'] for r in curves]!=[500,1000,2000,5000,8000]: raise ValueError('Incomplete learning curve')
    if any(r['validation_rows']!=2000 for r in runs+curves): raise ValueError('Holdout changed')
    if any(r['config']!=best['config'] for r in curves): raise ValueError('Curve configuration changed')
    if any(r['validation_ids_sha256']!=runs[0]['validation_ids_sha256'] for r in runs+curves): raise ValueError('Validation IDs changed')
    if best!=max(runs,key=lambda r:r['validation_auroc']): raise ValueError('Selection does not follow declared metric')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.3,4.3),layout='constrained')
    for key,label,color in [('train_auroc','Training','#3556a4'),('validation_auroc','Validation','#12806c')]:
        ax.plot([r['train_rows'] for r in curves],[r[key] for r in curves],marker='o',label=label,color=color)
    ax.set(xlabel='Labeled training rows',ylabel='AUROC',title=f'Fixed configuration: {best["config"]["kind"]}, {settings(best)}',ylim=(.45,1.015))
    ax.set_xticks([500,1000,2000,5000,8000],['500','1,000','2,000','5,000','8,000']);ax.legend();ax.grid(alpha=.2)
    for ext in ['png','pdf']: fig.savefig(root/f'learning_curve.{ext}',dpi=180)
    plt.close(fig)
    protocol=('The supplied training file contains 10,000 passages with 5,000 labels per class and no blank or exact duplicated passages. '
        'We saved one stratified 8,000/2,000 split with seed 20261023, including ordered IDs and input hashes. '
        'Both partitions are balanced. The starter file is a subset and was not added. All fitted preprocessing uses only the training partition. '
        'The holdout is reused for model selection, so the winning AUROC is a selection estimate, not an unbiased independent test score. '
        'The full-corpus audit found no word-trigram Jaccard pairs at or above 0.70, but a few shared openings/moderate-overlap pairs cross the split. '
        'We retain the prescribed holdout and report a secondary score excluding eight validation rows that share the normalized first 80 characters or word-trigram Jaccard similarity of at least 0.50 with a training passage. This leaves 1,992 rows (998 human, 994 AI); it does not certify source independence.')
    preprocessing=('Word features use lowercase word unigrams/bigrams with the default scikit-learn token rule (at least two word characters), '
        'min_df=2 and a 60,000-feature cap. Character features use lowercase character 3-5 grams, min_df=3 and an 80,000-feature cap. '
        'Both use sublinear term frequency, smoothed IDF and L2 normalization. We retain stop words, punctuation in character features, and original formatting '
        'for structure experiments; we do not stem or infer semantic quality. The actual corpus mixes recipes, news, abstracts and informal posts, '
        'motivating both lexical and subword/formatting representations. Logistic regression uses liblinear and max_iter=2000. '
        'The SVM uses three seeded training-only folds for sigmoid calibration; vocabulary/IDF are shared across those internal folds but never fit on the outer holdout. '
        'Fit times include preprocessing and use two numerical-library threads.')
    family_best={kind:max([r for r in runs if r['config']['kind']==kind],key=lambda r:r['validation_auroc']) for kind in ['word_lr','char_lr','word_nb','word_svm']}
    tuning=('Before inspecting any model validation score, we fixed C in {0.3,1,3} for word and character logistic regression and alpha in {0.1,1,3} '
        'for ComplementNB. The calibrated SVM is an additional fixed C=1 comparison. Maximum validation AUROC selects each family winner and the overall '
        'Day 1 configuration, with the first listed candidate breaking exact ties. Log loss and source-overlap sensitivity are diagnostics and do not select '
        'the model. We did not consult Kaggle scores or hidden labels. All ten candidates are shown, including weaker settings. '
        + 'Within-family choices: '+ '; '.join(f'{kind}: {settings(r)}' for kind,r in family_best.items())+'.')
    decision=(f'The Day 1 core recommendation is {NAMES[best["config"]["kind"]]} ({settings(best)}): training AUROC '
        f'{best["train_auroc"]:.4f}, validation AUROC {best["validation_auroc"]:.4f}, fit time {best["fit_seconds"]:.2f} seconds. '
        f'The validation log loss is {best["validation_log_loss"]:.4f}. Excluding the eight overlap-flagged passages leaves '
        f'{best["sensitivity_rows"]} validation rows and sensitivity AUROC {best["sensitivity_auroc"]:.4f}; this does not establish complete source independence. '
        'The saved selected_config.json preserves this configuration. '
        'This is a provisional recommendation for the final submission, based on the declared selection metric and measured laptop cost. '
        'The gap to other candidates does not establish stability across domains or generators. The handwritten-hypothesis ablations are separate controlled comparisons; '
        'later final-model changes must be recorded explicitly. No private or public Kaggle score is claimed.')
    extension=None
    if (root/'extension_comparison.json').exists():
        extension=read(root,'extension_comparison.json')
        e=extension['run'];effect=extension['effect_vs_core'];lo,hi=effect['ci_95_percentile']
        extension_text=(f'After the core comparison, we tested one equal-weight average of character C=3 and word-plus-style C=1, without searching weights. '
            f'Its validation AUROC is {e["validation_auroc"]:.4f} (change {effect["delta_auroc"]:+.4f}; exploratory paired 95% interval [{lo:+.4f}, {hi:+.4f}]). '
            f'The overlap-excluded AUROC is {e["sensitivity_auroc"]:.4f} on {e["sensitivity_rows"]} rows. '
            + ('This blend is the Day 1 handoff recommendation; the character model remains the simpler fallback. ' if extension['promoted_for_day1_handoff'] else 'We retain the character model as the Day 1 handoff recommendation. ')
            + 'This adaptive extension reused the validation set after earlier selections; its interval is not confirmatory. The required learning curve remains for the fixed character model.')
        decision += ' '+extension_text
    delta=curves[-1]['validation_auroc']-curves[0]['validation_auroc'];tail=curves[-1]['validation_auroc']-curves[-2]['validation_auroc']
    gap=curves[-1]['train_auroc']-curves[-1]['validation_auroc']
    interpretation=(f'Validation AUROC changes from {curves[0]["validation_auroc"]:.4f} at 500 rows to {curves[-1]["validation_auroc"]:.4f} '
        f'at 8,000 rows ({delta:+.4f}); the 5,000-to-8,000 change is {tail:+.4f}. The final training-minus-validation gap is {gap:+.4f}. '
        + ('The positive gap is consistent with overfitting, although sampling and source variation can contribute. ' if gap>0 else 'The final gap does not show a training advantage on this split. ')
        + ('The overall validation increase supports using the available labeled examples for this configuration. ' if delta>0 else 'This curve does not establish a benefit from the larger sample for this configuration. ')
        + 'Each size uses one nested stratified sample and one seed, with preprocessing refit on that sample. We held hyperparameters and the 2,000 validation IDs fixed. '
        'The curve does not quantify sampling variability or justify extrapolating beyond the supplied data. High training performance with lower validation performance '
        'suggests a generalization limitation rather than evidence that a larger-capacity model alone is needed.')
    comp_rows=[[r['name'],settings(r),f'{r["train_auroc"]:.4f}',f'{r["validation_auroc"]:.4f}',f'{r["fit_seconds"]:.2f}'] for r in runs]
    curve_rows=[[r['train_rows'],f'{r["train_auroc"]:.4f}',f'{r["validation_auroc"]:.4f}',f'{r["fit_seconds"]:.2f}'] for r in curves]
    md=['# Day 1 measured report — Q1 and Q2',protocol,'## Q1.1 — Preprocessing and representations',preprocessing,
        '## Q1.2 — Model comparison','We compared word logistic regression, character logistic regression, word Complement Naive Bayes and a calibrated linear SVM.',mdtable(['Run','Setting','Train AUROC','Validation AUROC','Fit seconds'],comp_rows),
        '### Related-text sensitivity (diagnostic only)',mdtable(['Run','Retained validation rows','Sensitivity AUROC'],[[r['name'],r['sensitivity_rows'],f'{r["sensitivity_auroc"]:.4f}'] for r in runs]),
        '## Q1.3 — Tuning and selection',tuning,'## Q1.4 — Day 1 recommendation',decision,
        '## Q2.1 — Fixed-configuration learning curve',f'Configuration: `{best["config"]}`. Subset IDs are saved; every size refits vocabulary and IDF.',
        mdtable(['Training rows','Train AUROC','Validation AUROC','Fit seconds'],curve_rows),
        '## Q2.2 — Curve','![Learning curve](learning_curve.png)','## Q2.3 — Interpretation',interpretation]
    bonus_file=root/'bonus_comparison.json'
    if bonus_file.exists():
        bonus=read(root,'bonus_comparison.json');b_runs=bonus.get('runs',[])
        md+=['## Additional notebook hypotheses — exploratory controlled tests',
            'The actual training audit motivated tests of repetition, formatting and sentence structure. Positivity/adjective/personalization differences were weak or opposite to the suggested direction. '
            'POS annotations are predictions of a disclosed pretrained NLTK English tagger, not grammar truth. All ablations use C=1; word/style features have fixed weights 1 and 0.1. '
            'The class-conditional POS model is a smoothed trigram model with three-fold cross-fitted training features. It is not a neural network or an error detector. '
            'Reported resubstitution training AUROC for this model uses final all-training tag models and is optimistic. Cache reuse makes bonus timing order dependent.',
            mdtable(['Approach','Train AUROC','Validation AUROC','Fit seconds'],[[r['name'],f'{r["train_auroc"]:.4f}',f'{r["validation_auroc"]:.4f}',f'{r["fit_seconds"]:.2f}'] for r in b_runs])]
        failures=bonus.get('failures',[])
        md += [f'Completed {len(b_runs)} of {len(b_runs)+len(failures)} attempted ablations.']
        if failures:
            md += ['Failed ablations (not negative measured results):', '\n'.join(f'- {f["config"]}: {f["type"]}: {f["error"]}' for f in failures)]
        effects=bonus.get('effects_vs_word_C1',{})
        if effects:
            md+=[mdtable(['Versus word LR C=1','AUROC change','Paired 95% interval'],[[k,f'{v["delta_auroc"]:+.4f}',f'[{v["ci_95_percentile"][0]:+.4f}, {v["ci_95_percentile"][1]:+.4f}]'] for k,v in effects.items()]),
                'Intervals are exploratory paired stratified bootstrap intervals conditional on these fits; they do not correct for multiple comparisons, model selection, training variability or correlated sources.']
    if extension:
        md += ['## One adaptive equal-weight blend',extension_text, f'Full configuration: `{extension["recommended_config"]}`. The declared plan is extension_plan.json; recommended_config.json is the runnable handoff selection.']
    warnings=[(r['name'],w) for r in runs+curves for w in r.get('warnings',[])]
    if bonus_file.exists(): warnings += [(r['name'],w) for r in b_runs for w in r.get('warnings',[])]
    warning_text='No fit/prediction warnings were recorded.' if not warnings else '\n'.join(f'- {name}: {w}' for name,w in warnings)
    md += ['## Recorded warnings',warning_text]
    md+=['## Sources','[scikit-learn TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html), '
        '[ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html), '
        '[calibration](https://scikit-learn.org/stable/modules/calibration.html), '
        '[NLTK POS tagger](https://www.nltk.org/api/nltk.tag.perceptron.html).',
        '## Verification and disclosure','Codex implemented, ran and checked these experiments. This statement does not claim that teammates have independently verified them; their actual review belongs in Q6.3.']
    (root/'Q1_Q2.md').write_text('\n\n'.join(md)+'\n')
    report=root.parent/'report';(report/'images').mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'learning_curve.pdf',report/'images'/'learning_curve.pdf')
    tex=[r'\section{Model comparison and selection (15 points)}',esc(protocol),r'\subsection*{Q1.1 --- Preprocessing}',esc(preprocessing),
        r'\subsection*{Q1.2 --- Measured comparisons}',table(['Run','Setting','Train','Validation','Fit (s)'],comp_rows,'p{3.1cm}p{2.2cm}rrr'),
        r'\subsection*{Related-text sensitivity (diagnostic)}',table(['Run','Retained rows','AUROC'],[[r['name'],r['sensitivity_rows'],f'{r["sensitivity_auroc"]:.4f}'] for r in runs],'p{5cm}rr'),
        r'\subsection*{Q1.3 --- Tuning}',esc(tuning),r'\subsection*{Q1.4 --- Day 1 recommendation}',esc(decision),
        r'\section{Effect of training data size (10 points)}',r'\subsection*{Q2.1 --- Fixed-configuration results}',esc(f'Configuration: {best["config"]}. Nested samples; all preprocessing refit within each size.'),
        table(['Training rows','Train AUROC','Validation AUROC','Fit (s)'],curve_rows,'rrrr'),r'\subsection*{Q2.2 --- Learning curve}',
        r'\providecommand{\DayOneFigurePath}{images/learning_curve.pdf}',r'\includegraphics[width=\linewidth]{\DayOneFigurePath}',
        r'\subsection*{Q2.3 --- Interpretation}',esc(interpretation),r'\paragraph{Recorded warnings}',esc(warning_text),r'\paragraph{Sources and assistance.}',
        r'Method references: \href{https://scikit-learn.org/stable/modules/feature_extraction.html}{scikit-learn text features}, \href{https://scikit-learn.org/stable/modules/naive_bayes.html}{Naive Bayes}, and \href{https://scikit-learn.org/stable/modules/calibration.html}{calibration}. Codex implemented, ran and checked these experiments. Teammate verification is not claimed and must be documented separately in Q6.3.']
    if bonus_file.exists():
        tex += [r'\section*{Exploratory extensions for Day 2 review}',
            esc('These are fixed C=1 comparisons against word logistic regression C=1. They test statistical structure proxies, not grammatical quality. POS tagging uses the disclosed pretrained NLTK English tagger. Class-conditional POS training features are cross-fitted; its resubstitution training score remains optimistic. Per-process cache reuse makes bonus timing order dependent.'),
            table(['Approach','Train','Validation','Fit (s)'],[[r['name'],f'{r["train_auroc"]:.4f}',f'{r["validation_auroc"]:.4f}',f'{r["fit_seconds"]:.2f}'] for r in b_runs],'p{5cm}rrr'),
            esc('Negative results are retained. Detailed definitions, paired intervals and confounds are in HYPOTHESES.md and Q1_Q2.md. These comparisons provide evidence for Day 2 Q5 review; they do not complete the unlabeled-data experiment or final generalization analysis.') ]
        if failures: tex += [esc('Failed ablations: '+str(failures))]
    if extension: tex += [r'\paragraph{One adaptive blend.}',esc(extension_text)]
    (report/'day1_sections.tex').write_text('\n\n'.join(tex)+'\n')
    (report/'main.tex').write_text(r'''\documentclass[11pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,booktabs,graphicx,hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\begin{document}
\title{Human or AI? --- Group Number 30\\Day 1 measured report}
\date{November 5, 2026}
\maketitle
\input{day1_sections.tex}
\end{document}
''')
    return root/'Q1_Q2.md'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,default=Path('results'));args=parser.parse_args()
    print(generate(args.results))
