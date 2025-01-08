from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

def compute_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

def compute_bleu(reference, generated):
    smoothing_function = SmoothingFunction().method1
    reference_tokens = reference.split()
    generated_tokens = generated.split()
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing_function)
    return bleu_score

reference_summaries = ["In the year 1878 I took my degree of Doctor of Medicine of the University of London, and proceeded to Netley to go through the course prescribed for surgeons in the army. Having completed my studies there, I was duly attached to the Fifth Northumberland Fusiliers as Assistant Surgeon. The regiment was stationed in India at the time, and before I could join it, the second Afghan war had broken out. On landing at Bombay, I learned that my corps had advanced through the passes, and was already deep in the enemy’s country. I followed, however, with many other officers who were in the same situation as myself, and succeeded in reaching Candahar in safety, where I found my regiment, and at once entered upon my new duties."] 
rag_summaries = ["In 1878 I took my degree of Doctor of Medicine of the University of London. I was attached to the Fifth Northumberland Fusiliers as Assistant Surgeon. The regiment was stationed in India at the time, and before I could join it the second Afghan war had broken out."]  
baseline_summaries = ["in the year 1878, I took my degree of doctor of medicine from the university of london. he went through the course prescribed for surgeons in the army. the regiment was stationed in India at the time, and before joining it, the second Afghan war had broken out."]  


rouge_scores_rag = []
bleu_scores_rag = []
rouge_scores_llm = []
bleu_scores_llm = []


for ref, gen in zip(reference_summaries, rag_summaries):
    rouge_scores_rag.append(compute_rouge(ref, gen))
    bleu_scores_rag.append(compute_bleu(ref, gen))


for ref, gen in zip(reference_summaries, baseline_summaries):
    rouge_scores_llm.append(compute_rouge(ref, gen))
    bleu_scores_llm.append(compute_bleu(ref, gen))


avg_rouge_rag = {key: np.mean([score[key].fmeasure for score in rouge_scores_rag]) for key in rouge_scores_rag[0]}
avg_bleu_rag = np.mean(bleu_scores_rag)
avg_rouge_llm = {key: np.mean([score[key].fmeasure for score in rouge_scores_llm]) for key in rouge_scores_llm[0]}
avg_bleu_llm = np.mean(bleu_scores_llm)

print(f"RAG ROUGE Scores: {avg_rouge_rag}")
print(f"RAG BLEU Score: {avg_bleu_rag}")
print(f"Baseline ROUGE Scores: {avg_rouge_llm}")
print(f"Baseline BLEU Score: {avg_bleu_llm}")
