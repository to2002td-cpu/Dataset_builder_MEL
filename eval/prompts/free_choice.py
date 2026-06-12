"""
Free-choice prompt template: the model is shown the mention and the query
image only (no candidate list) and must name the entity directly.

TODO: implement build_prompt(instance, kb, cfg) -> str, honouring the
`prompt:` section of the eval config.
"""
"free": "What entity does the label '{mention}' represent in this image?"
