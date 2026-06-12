"""
Contrastive prompt template: the model is shown the mention, the query image,
and the candidate entities as numbered options, and must pick one.

TODO: implement build_prompt(instance, kb, cfg) -> str, honouring the
`prompt:` section of the eval config (answer_none, shuffle_candidates).
"""
  "contrast": [
    "What entity does the label '{mention}' represent in this image?",
    {
      "possibilities-nom": "Here are the names of the possibilities:\n{candidates}",
      "possibilities-nom-description": "Here are the names and a brief description of each possibility:\n{candidates}",
      "possibilities-nom-description-image": "Here are the names, a brief description, and an image of each possibility:\n{candidates}"
    }
  ],
"response format": "You must respond with a single valid JSON object only. No preamble, no explanation outside the JSON, no markdown backticks.\n\nFormat:\n{\n  \"number\": <integer position of the candidate in the list>,\n  \"answer\": \"<name of the selected candidate>\",\n  \"explanation\": \"<brief explanation of your choice>\"\n}"
