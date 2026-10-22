# Glossary

run file :: TOML that describes one run: corpus, questions, parameters -> hermes_report.config.RunConfig
questions file :: JSON of questions with expected answers, keywords, evidence -> hermes_report.questions.QuestionSet
corpus :: folder of Markdown files, one document each -> hermes_report.corpus.Corpus
source :: path of a document relative to the corpus folder; payload["source"] -> CorpusDocument.source
run record :: everything a run saw and produced, written as run.json -> hermes_report.record.RunRecord
raw point :: a point as stored in Qdrant: id, payload, vector summary -> hermes_report.record.PointRecord
rule verdict :: grade from keywords and evidence phrases -> hermes_report.evaluate.rule_verdict
judge :: LLM that grades an answer against the expected one -> hermes_report.evaluate.judge
evidence :: phrases of the corpus that must reach the context for an answer to be reachable -> Question.evidence
touched document :: document with a chunk retrieved, added or expected for a question -> report
metric family :: a named group of scores computed together, selected by name -> hermes_report.metrics.Metric
score :: one number of a metric family for one question, in [0, 1] or a count -> MetricResult.scores
metric input :: the neutral view of one answered question that metrics read -> hermes_report.metrics.MetricInput
judge session :: memoised judge operations (claims, verification, relevance) for one question -> hermes_report.metrics.JudgeSession
claim :: one statement extracted from an answer or the expected answer -> hermes_report.metrics.judge.Claim
budget loss :: evidence recall after retrieval minus evidence recall in the final context -> evidence_retrieval
experiment history :: one row per notebook run and collection, in history.csv / history.jsonl -> notebooks/rag_evaluation.ipynb
