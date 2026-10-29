"""Explicit evidence levels; an alias never becomes a literal exact ID."""
from __future__ import annotations

import re

POLICIES = ('exact_id', 'verified_alias', 'identity_match')


def resolve_mappings(mappings: list[dict], policy: str = 'exact_id') -> dict[str, str]:
    if policy not in POLICIES:
        raise ValueError('Unknown mapping policy')
    allowed = set(POLICIES[:POLICIES.index(policy) + 1])
    sources, targets, accepted = set(), set(), {}
    for entry in mappings:
        if not isinstance(entry, dict) or not {'arena', 'routerbench', 'status'} <= entry.keys():
            raise ValueError('Incomplete mapping')
        source, target = entry['arena'], entry['routerbench']
        if (not all(isinstance(x, str) and x.strip() for x in (source, target))
                or source in sources or target in targets):
            raise ValueError('Ambiguous or invalid model mapping')
        sources.add(source)
        targets.add(target)
        status = entry['status']
        if status == 'exact_id' and source != target:
            raise ValueError('Exact mappings must have literal identical model IDs')
        if status not in allowed:
            continue
        if status != 'exact_id':
            evidence = entry.get('evidence')
            if not isinstance(evidence, list) or not evidence:
                raise ValueError('Nonliteral mapping requires pinned evidence')
            for item in evidence:
                if (not isinstance(item, dict)
                        or any(not isinstance(item.get(k), str) or not item[k].strip()
                               for k in ('url', 'revision', 'relationship', 'limitation'))
                        or not re.fullmatch(r'[0-9a-f]{40}', item['revision'])
                        or not item['url'].startswith('https://')
                        or item['revision'] not in item['url']):
                    raise ValueError('Invalid or unpinned mapping evidence')
        accepted[source] = target
    if not targets:
        raise ValueError('Empty model mapping')
    return accepted
