"""OpenAI cache provenance and exact-input checks, without an API client."""
import json

from scripts.pretraining import _digest, _hash

MODEL, DIMENSION = 'text-embedding-3-small', 1536


def provenance(synthetic=False):
    return dict(model=MODEL,
        version='synthetic-schema-fixture' if synthetic else MODEL+'; provider snapshot not supplied',
        source=json.dumps(dict(model=MODEL, dimensions=DIMENSION, snapshot='not exposed',
            normalization='provider unit vector, preserved as float32', synthetic=synthetic), sort_keys=True),
        normalization='provider unit vector, preserved as float32', synthetic=synthetic)


def validate_inputs(cache, inputs):
    """Identity validation, not independent attestation of a provider response."""
    if cache.dimension != DIMENSION or cache.provenance != provenance(False):
        raise ValueError('Unsupported OpenAI cache model, dimension or provenance')
    metadata = cache.metadata
    if metadata.get('input_rows_sha256') != _digest(inputs):
        raise ValueError('Verbatim OpenAI input rows differ from preparation')
    for key in ('source_sha256', 'plan_identity'):
        _hash(metadata.get(key))
    ids = metadata.get('provider_request_ids')
    if (metadata.get('synthetic') is not False or not isinstance(ids, list) or not ids
            or any(not isinstance(v, str) or not v.strip() for v in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('OpenAI cache requires explicit response provenance and request IDs')
