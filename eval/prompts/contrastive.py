"""
Contrastive prompt template: the model is shown the mention, the query image,
and the candidate entities as numbered options, and must pick one.

TODO: implement build_prompt(instance, kb, cfg) -> str, honouring the
`prompt:` section of the eval config (answer_none, shuffle_candidates).
"""
