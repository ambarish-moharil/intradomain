# intradomain_toolkit
The toolkit is a first experimental release based on the paper "Identification of Intra-Domain Ambiguity Using Transformer Based Learning". The mentioned package can be installed and used for the contextual disambiguation of a natural language corpus (English).

# INTRADOMAIN AMBIGUITY

DISAMBIGUATE INTRADOMAIN CONTEXTUAL AMBIGUITY

## Instructions

1. Install:

```
pip install intradomain_toolkit
```

2. Disambiguate a certain corpus:

```python

from intradomain_toolkit import disambiguation

# initialize context object (to disambiguate a corpus)
contextual = disambiguation.disambiguate()
