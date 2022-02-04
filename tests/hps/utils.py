import collections
from typing import Union, Any, Dict, List

from embeddings.pipeline.pipelines_metadata import LightningPipelineMetadata


def _flatten(
    d: Union[collections.MutableMapping[Any, Any], LightningPipelineMetadata]
) -> Dict[Any, Any]:
    items: List[tuple[Any, Any]] = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten(v).items())
        else:
            items.append((new_key, v))
    return dict(items)
