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
