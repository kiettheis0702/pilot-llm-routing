"""Research report and poster from complete, independently replayed local results."""
import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader

ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent
SOURCE=BASE/'aggregate';FIGURES=BASE/'figures';DEST=BASE/'deliverables'
if DEST.exists():raise FileExistsError(DEST)
manifest=json.loads((SOURCE/'manifest.json').read_text())
assert manifest['complete'] and manifest['independently_replayed_groups']==108
for n,h in manifest['files'].items():assert hashlib.sha256((SOURCE/n).read_bytes()).hexdigest()==h
figure_manifest=json.loads((FIGURES/'figure-data.json').read_text())
assert figure_manifest['aggregate_manifest_sha256']==hashlib.sha256((SOURCE/'manifest.json').read_bytes()).hexdigest()
for n,h in figure_manifest['files'].items():assert hashlib.sha256((FIGURES/n).read_bytes()).hexdigest()==h
with (SOURCE/'mean-min-max.csv').open(encoding='utf-8',newline='') as f:table=list(csv.DictReader(f))
with (SOURCE/'per-seed.csv').open(encoding='utf-8',newline='') as f:individual=list(csv.DictReader(f))
with (SOURCE/'paired-deltas.csv').open(encoding='utf-8',newline='') as f:paired=list(csv.DictReader(f))
labels={'pilot':'PILOT','linucb':'LinUCB','epoch-greedy':'EpochGreedy','explore-only':'ExploreOnly','random':'Random',
    'fixed-00':'Fixed WizardLM 13B','fixed-01':'Fixed Claude Instant','fixed-02':'Fixed Claude v1','fixed-03':'Fixed GPT-3.5',
    'fixed-04':'Fixed GPT-4','fixed-05':'Fixed Code Llama 34B','fixed-06':'Fixed Llama 2 70B','fixed-07':'Fixed Yi 34B'}
def selected(domain,method,budget=.5):
    return next(r for r in table if r['domain']==domain and r['method']==method and r['variant']=='main'
                and float(r['learning_fraction'])==1 and float(r['budget_fraction'])==budget)
def score(r):return float(r['service_utility_mean'])
def domain_finding(domain):
    values=[selected(domain,m) for m in labels];best=max(values,key=score);pilot=selected(domain,'pilot')
    return (f"At 0.5 of the reference budget, PILOT delivered {score(pilot):.3f} reward per requested query "
        f"with {float(pilot['coverage_mean']):.1%} coverage. The highest mean at this budget was "
        f"{labels[best['method']]} ({score(best):.3f}). These are descriptive results on fixed streams.")
overruns=sum(float(r['actual_budget_excess'])>0 for r in individual)
DEST.mkdir()
NAVY=colors.HexColor('#102e42');TEAL=colors.HexColor('#087f8c');GRAY=colors.HexColor('#52616b');PALE=colors.HexColor('#edf4f5')
styles={size:ParagraphStyle('body'+str(size),fontName='Helvetica',fontSize=size,leading=size*1.38,textColor=NAVY) for size in (10,11,12,16,22,24,26,30)}
def paragraph(c,text,x,y,width,size=11):
    p=Paragraph(text,styles[size]);w,h=p.wrap(width,10000);p.drawOn(c,x,y-h);return y-h-10
def fitted_image(c,path,x,y,width,height):
    img=ImageReader(str(path));iw,ih=img.getSize();scale=min(width/iw,height/ih)
    w,h=iw*scale,ih*scale;c.drawImage(img,x+(width-w)/2,y+(height-h)/2,width=w,height=h,mask='auto')
W,H=landscape(A4);c=canvas.Canvas(str(DEST/'research-report.pdf'),pagesize=(W,H))
c.setTitle('PILOT: complete eight-arm local offline experiment');c.setAuthor('Team 8 - HCMUS course project')
page=0
def header(title,subtitle='Complete declared local experiment | not an eleven-arm paper reproduction'):
    global page
    page+=1;c.setFillColor(NAVY);c.rect(0,H-72,W,72,fill=1,stroke=0)
    c.setFillColor(colors.white);c.setFont('Helvetica-Bold',22);c.drawString(34,H-34,title)
    c.setFont('Helvetica',10);c.drawString(34,H-54,subtitle)
    c.setFillColor(GRAY);c.setFont('Helvetica',8)
    c.drawString(34,20,'PILOT local study - 29 October 2026 - all model feedback is prerecorded; no paid generation calls')
    c.drawRightString(W-34,20,str(page))
header('PILOT: an offline routing study')
y=H-95
y=paragraph(c,'<b>Complete local scope:</b> 8 evidence-admitted model arms, 3 replicate seeds, 13 primary methods, '
    '8 budget fractions, 4 learning prefixes and 5 controlled ablations. All 1,098 deployment conditions and '
    '108 method/variant groups passed independent numerical replay. This validates the declared local experiment, '
    'not equality with the original paper.',34,y,W-68,12)
y=paragraph(c,'<b>RouterBench.</b> '+escape(domain_finding('routerbench')),34,y,W-68,11)
y=paragraph(c,'<b>MMLU subset.</b> '+escape(domain_finding('mmlu')),34,y,W-68,11)
y=paragraph(c,f'<b>Accounting:</b> {overruns:,} of 1,098 conditions recorded actual benchmark spend above the estimated budget. '
    'The selector enforces estimated affordability; actual recorded costs are reported separately. These costs are '
    'historical benchmark units, not charges incurred by this experiment.',34,y,W-68,11)
data=[['Method','RouterBench utility','RB coverage','MMLU utility','MMLU coverage']]
for method in labels:
    a,b=selected('routerbench',method),selected('mmlu',method)
    data.append([labels[method],f"{score(a):.4f}",f"{float(a['coverage_mean']):.1%}",f"{score(b):.4f}",f"{float(b['coverage_mean']):.1%}"])
t=Table(data,colWidths=[190,145,105,145,105],rowHeights=17)
t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),colors.white),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
    ('ALIGN',(1,0),(-1,-1),'RIGHT'),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
_,th=t.wrap(W-68,H);t.drawOn(c,34,y-th)
paragraph(c,'Table: mean across three seeds, full learning, budget fraction 0.5. Full seed/budget tables accompany this report.',34,y-th-8,W-68,10)
c.showPage()

header('Design, controls and interpretation')
y=H-98
blocks=[
('Research basis','Panda et al. formulate adaptive LLM routing as a contextual bandit, initialize it using human preferences, '
 'and combine routing with an online budget policy. Our implementation follows this structure with explicitly documented local substitutions. '
 'Reference: Adaptive LLM Routing under Budget Constraints, Findings of EMNLP 2025, DOI 10.18653/v1/2025.findings-emnlp.1301.'),
('Data and representation','Pinned RouterBench 0-shot data: 1,000 tuning / 32,270 learning / 3,227 deployment queries. '
 'MMLU retains the same frozen phase order after task-prefix filtering: 398 / 12,369 / 1,275. MMLU is nested in RouterBench. '
 'Arena preparation supplies 1,245 training and 313 holdout preference pairs. Local BGE-M3 encodes all inputs at a 2,048-token cap; '
 'no routing input is truncated, and 13 preference inputs are truncated.'),
('Training and selection','Each seed (7, 42, 2026) retrains a 64D projection/prior for 30/40 epochs, learning rate 0.03 and margin 0.2. '
 'Independent holdout checks determine finite initialization coefficients for every arm. Tuning selects PILOT alpha=1, LinUCB alpha=5 '
 'and EpochGreedy window=10 in both domains. Tuning states never initialize routing learning. Only chosen-arm feedback updates learning; '
 'deployment uses a frozen checkpoint.'),
('Common comparison','Every method uses identical query order, model pool, tuning-only length estimates and absolute budget per condition. '
 'Budget fractions are 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1 and 1.25 times the domain reference budget. '
 'Reference budgets are 10.23165528 for RouterBench and 1.7940659547738693 for MMLU. Bin size is 100. '
 'Linear predictions are unclipped; random/fixed scores are declared utility conventions, not calibrated probabilities.'),
('What the metrics mean','Service utility is total observed reward divided by all requested queries; unserved service utility is zero. '
 'Coverage and served-query mean are separate. Learning regret uses the best stored reward among the eight admitted arms as a retrospective '
 'oracle; the policy never receives that oracle. Seed mean/min/max describes these three runs, not a confidence interval or significance test.')]
for title,text in blocks:y=paragraph(c,'<b>'+title+'.</b> '+escape(text),34,y,W-68,11)
c.showPage()

captions=[
('service-utility','Quality delivered under each budget','All 13 methods and all eight budget fractions. Lines show the mean over the three fixed seeds. '
 'Unserved queries contribute zero service utility. Full per-seed results remain in the CSV tables.'),
('coverage','Budget exhaustion and served coverage','Coverage is essential to interpreting reward on served queries. The common policy terminates a stream '
 'when neither its threshold rule nor its affordability fallback admits a permitted model. Fixed policies may use only their named arm.'),
('learning-regret','Learning feedback and empirical regret','Snapshots at 10%, 25%, 50% and 100% are nested prefixes of one uninterrupted stream. '
 'The oracle is retrospective and unconstrained by deployment budgets. Different stream lengths make the two domains separate descriptive panels.'),
('learning-size','Sensitivity to routing-learning data','All four prefix sizes use the same deployment stream and budget fraction 0.5. '
 'Preference prior, seed, cost estimates and policy parameters stay fixed within a replicate; no tuning occurs on deployment outcomes.'),
('ablations','Controlled changes relative to full PILOT','All five declared variants are reported, including unfavorable or identical outcomes. '
 'Bands are observed min/max paired seed deltas, not confidence intervals. The 8D variant retrains the same 30/40-epoch recipe; '
 'it is not an embedding-provider comparison. Zero prior directions retain the learned projection and coefficients.'),
('recorded-cost','Recorded cost versus declared budgets','These are recorded benchmark costs, separately accumulated after routing decisions. '
 'Output-length estimates come only from tuning. An estimated spending bound does not guarantee a bound on recorded actual spending; '
 'per-run excess is included in the tables.')]
for name,title,caption in captions:
    header(title);fitted_image(c,FIGURES/(name+'.png'),20,95,W-40,H-175)
    paragraph(c,escape(caption),34,82,W-68,11);c.showPage()

header('Limits, evidence and reproduction')
y=H-98
limits=[
 'The model pool has eight arms. Claude-v2, Mistral-7B-chat and Mixtral-8x7B-chat remain excluded because their provider checkpoint bridges are unresolved. Canonical name identity alone does not establish serving equivalence.',
 'The original paper release/shot selection, 86 local task labels versus 64 reported tasks, and exact Arena preparation remain unresolved. This study uses a declared 313-pair holdout rather than claiming the paper\'s 500-example evaluation.',
 'BGE-M3, the 2,048-token cap, 64D/8D projections, initialization/training recipe and baseline deployment score conventions are explicit local adaptations. An 8D sensitivity result is not evidence that BGE-M3 matches the paper\'s embedding provider.',
 'Historical price inputs use a documented factor-of-ten dataset/source calibration and a shared token-count approximation. Triplet cost ordering assumes a 1:1 input/output reference mix. Actual cost overruns and score-scale differences must remain visible.',
 'Three seeds and nested MMLU do not establish population uncertainty or independent dataset replication. The complete grid is reported; individual favorable configurations are not treated as general superiority.']
for text in limits:y=paragraph(c,'- '+escape(text),34,y,W-68,11)
y=paragraph(c,'<b>Evidence:</b> aggregate/manifest.json pins every completed run and all four CSV tables. '
    'verification/ contains independent action, state, cost and metric replay for every condition group. '
    'figures/figure-data.json records the exact numeric series behind every figure. Raw prompts and generated vectors stay out of Git.',34,y,W-68,11)
y=paragraph(c,'<b>Reproduction:</b> docs/full-experiments.md lists the command sequence and immutable resume behavior. '
    'Use configs/frozen_experiment_protocol.json and configs/research_prior_index.json without changing pins. '
    'Run one job/worker at a time with at least 8 GiB available RAM. No provider generation API is called.',34,y,W-68,11)
y=paragraph(c,'<b>Reference:</b> <link href="https://aclanthology.org/2025.findings-emnlp.1301/">'
    'Panda et al. (2025), Adaptive LLM Routing under Budget Constraints</link>. '
    'Dataset and cost-source identities are recorded in the frozen protocol and docs/source-cost-evidence-2026-10-29.md.',34,y,W-68,10)
c.showPage();c.save()

# A1 landscape poster: one narrative column, two wide result panels.
PW,PH=2383.94,1683.78;p=canvas.Canvas(str(DEST/'research-poster.pdf'),pagesize=(PW,PH))
p.setTitle('PILOT: an offline routing study - local eight-arm experiment')
p.setFillColor(NAVY);p.rect(0,PH-240,PW,240,fill=1,stroke=0)
p.setFillColor(colors.white);p.setFont('Helvetica-Bold',72);p.drawString(70,PH-105,'PILOT: an offline routing study')
p.setFont('Helvetica',30);p.drawString(74,PH-160,'Team 8 | HCMUS course project | Complete declared local experiments')
p.setFillColor(colors.HexColor('#87e2d1'));p.setFont('Helvetica-Bold',27)
p.drawString(74,PH-208,'8 model arms   /   3 seeds   /   1,098 budget runs   /   No paid generation calls')
y=PH-295;left=74;width=650
for title,text in [
 ('The question','How much recorded reward can a router deliver under a limited estimated budget, and how does preference initialization change its behavior?'),
 ('Offline workflow','Human preferences → local BGE-M3 embeddings → seeded projection and arm priors → chosen-arm learning feedback → frozen budget evaluation.'),
 ('Data and controls','RouterBench: 32,270 learning and 3,227 deployment queries. MMLU subset: 12,369 and 1,275. Tuning is separate. All methods share each domain\'s pool, order, budgets and estimated-cost rules.'),
 ('RouterBench result',domain_finding('routerbench')),
 ('MMLU result',domain_finding('mmlu')),
 ('Evidence and limits',f'All 108 method/variant groups independently replayed. {overruns} conditions exceeded the budget in recorded cost despite estimated affordability. Three seeds provide descriptive ranges only. BGE-M3, model pool and source details differ from the paper; full eleven-arm equivalence remains unresolved.')]:
    p.setFillColor(TEAL);p.setFont('Helvetica-Bold',30);p.drawString(left,y,title);y-=18
    y=paragraph(p,escape(text.replace('→',' > ')),left,y,width,24)-26
right=790;rw=PW-right-65
p.setFillColor(NAVY);p.setFont('Helvetica-Bold',32);p.drawString(right,PH-295,'Budgeted service utility - every primary method')
fitted_image(p,FIGURES/'service-utility.png',right,PH-930,rw,590)
p.setFillColor(NAVY);p.setFont('Helvetica-Bold',32);p.drawString(right,PH-970,'Five paired ablations - including unfavorable results')
fitted_image(p,FIGURES/'ablations.png',right,100,rw,565)
p.setFillColor(PALE);p.rect(0,0,PW,72,fill=1,stroke=0);p.setFillColor(GRAY);p.setFont('Helvetica',20)
p.drawString(74,32,'Reference: Panda et al., Findings of EMNLP 2025 | DOI 10.18653/v1/2025.findings-emnlp.1301 | Local experiment, not paper-equivalent')
p.showPage();p.save()
report=dict(version=1,kind='local-research-deliverables',aggregate_manifest_sha256=hashlib.sha256((SOURCE/'manifest.json').read_bytes()).hexdigest(),
    figure_manifest_sha256=hashlib.sha256((FIGURES/'figure-data.json').read_bytes()).hexdigest(),report_pages=page,
    poster_pages=1,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),synthetic=False,paper_equivalent=False,
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir()})
with (DEST/'manifest.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False,allow_nan=False)
print(json.dumps(report))
