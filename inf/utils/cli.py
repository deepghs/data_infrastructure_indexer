import argparse
import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union, get_args, get_origin

from .duration import parse_duration_to_seconds

_MISSING = object()
_DURATION_PARAMETER_NAMES = {
    'access_interval',
    'deploy_span',
    'max_time_limit',
    'no_recent',
    'tag_refresh_time',
    'upload_time_span',
}
_UNLIMITED_DURATION_PARAMETER_NAMES = {
    'access_interval',
    'max_time_limit',
}
_NONE_TOKENS = {'', 'none', 'null', 'nil'}
_ARGUMENT_USAGE_HINTS = {
    'repository': 'Target Hugging Face dataset repository to read from and write to',
    'public_repository': 'Public Hugging Face dataset repository to process alongside the private repo',
    'public_repository_4m': '4M public Hugging Face dataset repository to process alongside the private repo',
    'previous_repository': 'Historical Hugging Face dataset repository to process',
    'index_repository': 'Index Hugging Face dataset repository to process',
    'exist_repo': 'Existing Hugging Face dataset repository used as the source index',
    'upload_time_span': 'Minimum interval between upload batches',
    'deploy_span': 'Minimum interval between deploy or upload commits',
    'max_time_limit': 'Stop the sync after this total runtime',
    'sync_mode': 'Continue incremental sync behavior instead of a fresh rebuild',
    'site_username': 'Site username used for authenticated upstream requests',
    'site_apikey': 'Site API key used for authenticated upstream requests',
    'site_golden': 'Enable the Danbooru golden-account request mode',
    'max_part_rows': 'Maximum rows to keep in one parquet or table shard before rotating',
    'sync_from_archives': 'Seed the sync from archived data before live updates',
    'no_recent': 'Skip records newer than this recency threshold',
    'user_id': 'Site user ID used for authenticated upstream requests',
    'api_key': 'Site API key used for authenticated upstream requests',
    'proxy_pool': 'Proxy endpoint or pool URL to attach to upstream requests',
    'access_interval': 'Minimum interval between site API requests',
    'tag_refresh_time': 'Refresh cached tag metadata when older than this threshold',
    'try_failed_ids_first': 'Retry previously failed record IDs before scanning new ones',
    'start_from_id': 'Start scanning from this explicit record ID instead of the stored pointer',
    'min_commits': 'Only squash repositories when commit history reaches at least this count',
    'commit_message': 'Commit message to use for squash operations',
    'commit_message_template': 'Template used for squash commit messages; supports {repo_id}',
}


@dataclass(frozen=True)
class EnvDefault:
    env_name: str
    default: Any = _MISSING

    def resolve(self) -> Any:
        if self.env_name in os.environ:
            return os.environ[self.env_name]
        if self.default is not _MISSING:
            return self.default
        raise KeyError(self.env_name)


def env_default(env_name: str, default: Any = _MISSING) -> EnvDefault:
    return EnvDefault(env_name=env_name, default=default)


def parse_cli_bool(value: Union[str, bool, None], allow_none: bool = False) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        if allow_none:
            return None
        raise ValueError('Boolean value should not be empty.')

    text = str(value).strip().lower()
    if allow_none and text in _NONE_TOKENS:
        return None
    if text in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Invalid boolean value - {value!r}.')


def _is_optional_annotation(annotation) -> bool:
    return get_origin(annotation) is Union and type(None) in get_args(annotation)


def _unwrap_optional(annotation):
    if _is_optional_annotation(annotation):
        return next(arg for arg in get_args(annotation) if arg is not type(None))
    return annotation


def _is_duration_parameter(name: str) -> bool:
    return name in _DURATION_PARAMETER_NAMES or name.endswith('_time_span') or name.endswith('_time_limit')


def _source_default_is_none(source: Any) -> bool:
    if isinstance(source, EnvDefault):
        return source.default is None
    return source is None


def _resolve_default(name: str, parameter: inspect.Parameter, defaults: Dict[str, Any]) -> Any:
    if name in defaults:
        source = defaults[name]
        if isinstance(source, EnvDefault):
            return source.resolve()
        return source
    if parameter.default is not inspect.Signature.empty:
        return parameter.default
    raise KeyError(name)


def _describe_default(name: str, parameter: inspect.Parameter, defaults: Dict[str, Any]) -> str:
    if name in defaults:
        source = defaults[name]
        if isinstance(source, EnvDefault):
            if source.default is _MISSING:
                return f'default: env {source.env_name}'
            return f'default: env {source.env_name} or {source.default!r}'
        return f'default: {source!r}'

    if parameter.default is not inspect.Signature.empty:
        return f'default: {parameter.default!r}'

    return 'required'


def _describe_parameter(name: str, parameter: inspect.Parameter, defaults: Dict[str, Any],
                        argument_help: Optional[Dict[str, str]] = None) -> str:
    annotation = parameter.annotation
    optional = _is_optional_annotation(annotation)
    base_annotation = _unwrap_optional(annotation)
    default_source = defaults[name] if name in defaults else _MISSING
    allow_none = optional or parameter.default is None or _source_default_is_none(default_source)
    if _is_duration_parameter(name) and name in _UNLIMITED_DURATION_PARAMETER_NAMES:
        allow_none = True

    parts = []
    help_text = (argument_help or {}).get(name) or _ARGUMENT_USAGE_HINTS.get(name)
    if help_text:
        parts.append(help_text)

    if _is_duration_parameter(name):
        parts.append('duration in seconds or text like 5h, 48min, 2d')
        if allow_none:
            if name in _UNLIMITED_DURATION_PARAMETER_NAMES:
                parts.append('use none or unlimited to disable the limit')
            else:
                parts.append('use none to clear the value')
    elif base_annotation is bool or isinstance(parameter.default, bool):
        parts.append('boolean: true/false/yes/no/on/off/1/0')
        if allow_none:
            parts.append('use none to clear the value')
    elif base_annotation is int:
        parts.append('integer')
        if allow_none:
            parts.append('use none to clear the value')
    elif base_annotation is float:
        parts.append('number')
        if allow_none:
            parts.append('use none to clear the value')
    elif base_annotation is str:
        parts.append('string')
        if allow_none:
            parts.append('use none to clear the value')
    else:
        parts.append('value')
        if allow_none:
            parts.append('use none to clear the value')

    parts.append(_describe_default(name=name, parameter=parameter, defaults=defaults))
    return '; '.join(parts)


def _build_converter(name: str, parameter: inspect.Parameter, defaults: Dict[str, Any]) -> Callable[[str], Any]:
    annotation = parameter.annotation
    optional = _is_optional_annotation(annotation)
    base_annotation = _unwrap_optional(annotation)
    default_source = defaults[name] if name in defaults else _MISSING
    allow_none = optional or parameter.default is None or _source_default_is_none(default_source)
    if _is_duration_parameter(name) and name in _UNLIMITED_DURATION_PARAMETER_NAMES:
        allow_none = True

    if _is_duration_parameter(name):
        def _convert_duration(raw_value: str):
            value = parse_duration_to_seconds(raw_value, allow_none=allow_none)
            if value is None:
                return None
            if base_annotation is int:
                return int(value)
            return value

        return _convert_duration

    if base_annotation is bool or isinstance(parameter.default, bool):
        return lambda raw_value: parse_cli_bool(raw_value, allow_none=allow_none)

    if base_annotation is int:
        def _convert_int(raw_value: str):
            if allow_none and str(raw_value).strip().lower() in _NONE_TOKENS:
                return None
            return int(raw_value)

        return _convert_int

    if base_annotation is float:
        def _convert_float(raw_value: str):
            if allow_none and str(raw_value).strip().lower() in _NONE_TOKENS:
                return None
            return float(raw_value)

        return _convert_float

    if base_annotation is str:
        def _convert_str(raw_value: str):
            if allow_none and str(raw_value).strip().lower() in _NONE_TOKENS:
                return None
            return str(raw_value)

        return _convert_str

    def _convert_fallback(raw_value: str):
        if allow_none and str(raw_value).strip().lower() in _NONE_TOKENS:
            return None
        return raw_value

    return _convert_fallback


def build_cli_parser(func: Callable, defaults: Optional[Dict[str, Any]] = None,
                     description: Optional[str] = None,
                     argument_help: Optional[Dict[str, str]] = None) -> argparse.ArgumentParser:
    defaults = dict(defaults or {})
    parser = argparse.ArgumentParser(description=description or inspect.getdoc(func))
    signature = inspect.signature(func)

    for name, parameter in signature.parameters.items():
        if parameter.kind not in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}:
            continue

        parser.add_argument(
            f'--{name.replace("_", "-")}',
            dest=name,
            default=argparse.SUPPRESS,
            type=_build_converter(name=name, parameter=parameter, defaults=defaults),
            help=_describe_parameter(
                name=name,
                parameter=parameter,
                defaults=defaults,
                argument_help=argument_help,
            ),
        )

    return parser


def resolve_cli_arguments(func: Callable, defaults: Optional[Dict[str, Any]] = None,
                          argv: Optional[list] = None, description: Optional[str] = None,
                          argument_help: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    defaults = dict(defaults or {})
    parser = build_cli_parser(
        func=func,
        defaults=defaults,
        description=description,
        argument_help=argument_help,
    )
    signature = inspect.signature(func)
    namespace = parser.parse_args(argv)
    parsed_values = vars(namespace)

    values = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind not in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}:
            continue

        if name in parsed_values:
            values[name] = parsed_values[name]
        else:
            values[name] = _resolve_default(name=name, parameter=parameter, defaults=defaults)

    return values


def run_callable_from_cli(func: Callable, defaults: Optional[Dict[str, Any]] = None,
                          argv: Optional[list] = None, description: Optional[str] = None,
                          argument_help: Optional[Dict[str, str]] = None):
    values = resolve_cli_arguments(
        func=func,
        defaults=defaults,
        argv=argv,
        description=description,
        argument_help=argument_help,
    )
    return func(**values)


__all__ = [
    'EnvDefault',
    'build_cli_parser',
    'env_default',
    'parse_cli_bool',
    'resolve_cli_arguments',
    'run_callable_from_cli',
]
