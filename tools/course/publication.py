"""Read separately verified public links without treating SDK receipts as public proof."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


def publication_state(records, file=None):
    """Join explicitly verified publication evidence by deterministic run ID.

    This validates supplied evidence; it does not perform an anonymous access
    check itself. Create the input only after the signed-out check has occurred.
    """
    known={r['id'] for r in records}
    urls={};projects=set()
    if file is not None:
        document=json.loads(Path(file).read_text(encoding='utf-8'))
        if set(document)!={'version','runs'} or document['version']!=1 or not isinstance(document['runs'],list):
            raise ValueError('Invalid publication evidence schema')
        for row in document['runs']:
            if set(row)!={'id','url','public_verified','verified_at','verification_method'}:
                raise ValueError('Invalid public link fields')
            if row['id'] not in known or row['id'] in urls:
                raise ValueError('Unknown or duplicate public run ID')
            if row['public_verified'] is not True or row['verification_method']!='signed-out-browser':
                raise ValueError('An SDK receipt is not verified public access')
            checked=datetime.fromisoformat(row['verified_at'])
            if checked.tzinfo is None or checked.utcoffset() is None:
                raise ValueError('Public verification timestamp requires timezone')
            match=re.fullmatch(r'https://wandb\.ai/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)/runs/([A-Za-z0-9_-]+)',row['url'])
            if not match or match.group(3)!=row['id']:
                raise ValueError('Public URL does not identify the recorded W&B run')
            urls[row['id']]=row['url']
            projects.add('https://wandb.ai/'+match.group(1)+'/'+match.group(2))
        if len(projects)>1:
            raise ValueError('Course publication must identify one reviewed project')
    return dict(urls=urls,verified_count=len(urls),total_count=len(known),
                complete=bool(known) and len(urls)==len(known),
                project_url=next(iter(projects),None))
