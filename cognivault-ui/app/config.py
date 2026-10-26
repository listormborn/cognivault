"""Application paths, defaults, and config load/save.

This module is intentionally free of any FastAPI / third-party imports so it can
be imported and exercised standalone (bootstrap, tests, quick checks).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


class AppPaths:
    """Filesystem layout under ``~/.cognivault-ui``.

    Directories are created lazily (see :meth:`ensure_dirs`); simply importing
    or instantiating this class touches nothing on disk.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root: Path = root or Path(os.path.expanduser("~/.cognivault-ui"))

    @property
    def config_file(self) -> Path:
        return self.root / "config.json"

    @property
    def certs_dir(self) -> Path:
        return self.root / "certs"

    @property
    def history_dir(self) -> Path:
        return self.root / "history"

    @property
    def venv_dir(self) -> Path:
        return self.root / "venv"

    @property
    def tmp_dir(self) -> Path:
        return self.root / "tmp"

    @property
    def confluence_dir(self) -> Path:
        return self.root / "confluence"

    def ensure_dirs(self) -> None:
        """Create the root plus the writable sub-directories if missing."""
        for d in (
            self.root,
            self.certs_dir,
            self.history_dir,
            self.tmp_dir,
            self.confluence_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# Module-level singleton used across the app.
PATHS = AppPaths()


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "cognivault": {"base_url": "http://localhost:3000", "token": ""},
    # One section, two transports. The certificate, temperature and token budget
    # are shared — only `provider` and the `kitai_*` keys decide where the request
    # goes, so switching back is one env var and no re-entered credentials.
    "gigachat": {
        # "gigachat" — прямой mTLS-клиент, настоящий стриминг токенов;
        # "kitai"    — платформа KitAI (POST → polling → commit), стриминга нет.
        "provider": "gigachat",
        "base_url": "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1",
        "model": "GigaChat-3-Ultra-preview",
        "cert_path": "~/.cognivault-ui/certs/client_crt.crt",
        "key_path": "~/.cognivault-ui/certs/client_key.key",
        "key_passphrase": "",
        "verify_ssl": False,
        "temperature": 0.2,
        "max_tokens": 4096,
        "model_context_tokens": 32768,
        # ── KitAI ──
        # `kitai_host` — БЕЗ /v1: пути версионируются самим API (/api/v1/query/...).
        "kitai_host": "https://hcscr-ift.delta.sbrf.ru",
        # Пустое значение = взять `model`. Держим отдельным ключом, чтобы
        # переключение провайдера не переписывало имя модели другого транспорта.
        # glm-5.1, а не 5.2: на IFT-контуре 5.2 принимается платформой и падает у
        # неё же на апстриме (`response_code=503`, «upstream connect error»),
        # то есть роутинга под этим именем там нет. Каталог моделей закрыт для
        # нашего сертификата (403), поэтому имя выяснено перебором.
        "kitai_model": "glm-5.1",
        # Идентификация вызывающей системы: уезжает в заголовках
        # x-identification-system / x-identification-module.
        "kitai_system_name": "csp_lab",
        "kitai_module_name": "csp_lab_antifraud_edge",
        "kitai_profanity_check": False,
        # Дополнительные поля запроса к платформе (`UniversalModelQueryPDto`),
        # объектом. Нужно, чтобы оператор мог передать параметр платформы —
        # например, выключить режим рассуждений модели — без пересборки образа.
        # Только ДОБАВЛЯЕТ ключи: `query_id`/`model_name`/`messages` и прочие
        # наши поля перекрыть нельзя. В ENV — строка JSON (`KITAI_EXTRA_BODY`).
        "kitai_extra_body": {},
        # Пусто = «тот же сертификат, что у GigaChat». Заполнять, только если у
        # KitAI своя пара: это другой контур, и один сертификат на оба — частный
        # случай, а не правило.
        "kitai_cert_path": "",
        "kitai_key_path": "",
        "kitai_key_passphrase": "",
        # Бюджет ожидания ответа. Это НЕ таймаут сокета: запрос ставится в
        # очередь, и всё это время мы опрашиваем результат.
        "kitai_poll_timeout": 240.0,
        "kitai_poll_initial_delay": 2.0,
        "kitai_poll_delay": 2.0,
    },
    "rag": {
        "default_on": False,
        # Auto smart-expansion is the default. When mode == "auto" the RAG
        # pipeline uses its OWN internals (hybrid search, k=10, section/whole-file
        # expansion, computed char budget) and ignores any stale stored
        # source/limit, so installs whose config predates these keys get the new
        # behaviour automatically (deep-merge just fills in mode="auto").
        "mode": "auto",
        "source": "hybrid",
        "limit": 10,
        "min_score": None,
        "token_budget": 3000,
        # Волна 0 (фидбек 2026-08-26): 24000 → 48000 и 4000 → 12000.
        #
        # Замер на снапшоте корпуса (127 страниц, 920 чанков, 260 секций): 44
        # секции длиннее старого окна 4000 и держат 70.6% всего текста базы.
        # Проверенный пример — вопрос «для каких моделей финэффект НЕ
        # рассчитывается»: все 7 моделей лежат в ОДНОЙ секции (24 379 символов),
        # в 7 чанках, разнесённых на 16 762 символа. В окно 4000 помещается
        # максимум 3 из 7 — агент и называл ровно 3. Это арифметика окна, а не
        # качество ранжирования: hit@1 гибрида 0.82, hit@5 0.97.
        #
        # `max_context_chars` поднят вместе с окном, иначе расширенные секции
        # просто не влезут в бюджет. Потолок сверху всё равно считает
        # `_compute_budget` от контекста модели: при 32768 токенов и max_tokens
        # 4096 вычисленный лимит ~70k символов, то есть 48000 достижимы.
        #
        # Цена: контекст на ход примерно вдвое дороже, и при 12000 на блок в
        # бюджет влезает ~4 источника вместо 5. Это осознанный размен полноты
        # на разнообразие, и он ОБРАТИМ из UI — обе ручки в USER_EDITABLE_KEYS.
        "max_context_chars": 48000,
        "file_full_chars": 6000,
        "section_max_chars": 12000,
        "max_expanded_files": 2,
        # Волна 2: два скрытых вызова GigaChat в конвейере чата.
        # `condense_enabled` — интент-маршрутизация + переписывание вопроса;
        # `grader_enabled` — батч-оценка релевантности (она же реранкер);
        # `rerank_candidates` — ширина ретрива, которую видит грейдер.
        #
        # Волна 3: 20 → 40. Recall на этапе поиска — потолок для всего, что дальше,
        # а грейдер как раз и делает широкую сеть безопасной. Дорожает только этап
        # оценки: 40 кандидатов бьются на батчи по 12 (`rag_pipeline._BATCH_SIZE`),
        # то есть ЧЕТЫРЕ вызова грейдера вместо двух — примерно вдвое дороже, но
        # по-прежнему одна параллельная волна по латентности. Ручка правится из UI,
        # если нужно вернуть 20.
        "condense_enabled": True,
        # Шаг 2б: запускать ли condense на ПЕРВОЙ реплике, где истории ещё нет.
        # Выключено по умолчанию, и это осознанная цена: включение добавляет
        # ровно один вызов GigaChat к КАЖДОМУ первому сообщению чата (~300
        # входных токенов и полсекунды-полторы до первого токена), а покупает
        # только поле `scope` — то есть возможность показать оговорку про охват
        # уже на первом вопросе. Переписывание вопроса и классификация интента
        # на первом ходу игнорируются в любом случае (`rag_pipeline.condense`):
        # без истории разрешать нечего, а ошибочный `smalltalk` стоил бы
        # пользователю поиска. Метавопрос первым сообщением ловится без вызова
        # модели (`corpus_scope.match_meta`), поэтому флаг именно опция, а не
        # необходимость.
        "condense_first_turn": False,
        # Шаг 3: подмешивать ли в ход дерево разделов из каталога
        # (`GET /api/vault/catalog`) вместо короткого блока «состав базы».
        # Выключено по умолчанию, и это не осторожность ради осторожности:
        # дерево отвечает на «какие витрины ClickHouse описаны в базе» только
        # потому, что иерархия Confluence И ЕСТЬ дерево продуктов. На вольте
        # вида `2024/Q1/заметки` тот же блок предъявит модели календарь и
        # пригласит отвечать про продукты названиями месяцев. Цена включения —
        # ~8 500 символов (~3 400 токенов) на боевом корпусе из 127 документов в
        # КАЖДОМ ходе, потолок 9 500 символов; платит за неё история диалога —
        # `trim_history` защищает последнее сообщение и режет старые реплики;
        # вызовов модели не добавляется ни одного, дерево берётся из каталога и
        # кэшируется на 5 минут. Блок ДОБАВЛЯЕТСЯ к источникам, не заменяет их:
        # отбор фрагментов, грейдер и его отказ работают ровно как раньше.
        "corpus_tree_enabled": False,
        # Поводки скрытых вызовов. Прежние 10/20 подбирались под стриминговый
        # GigaChat и оказались короче реальности: в прогоне `baseline` стадия
        # `grade` упёрлась ровно в 20 с на 41 ходе из 46, грейдер вернул оценки
        # на 2 парах из 44, и контекст собирался сырым порядком поиска. Ошибки
        # при этом нет нигде — отбор при неоценённых фрагментах пропускает их
        # дальше, — так что деградация была полностью бесшумной. Успевшие
        # вызовы шли 11–19 с, то есть 20 с резали по медиане.
        #
        # Числа совпадают с `deploy/dropapp/03-configmap-ui.yaml` не случайно:
        # ENV оттуда побеждает этот дефолт, и разойтись им нельзя (за этим
        # следит `test_prod_configmap_matches_the_code_defaults_for_tuned_keys`).
        # Тот прогон и случился на дефолтах кода — ConfigMap до пода не доехал.
        "condense_timeout": 45.0,
        "grader_timeout": 90.0,
        # Бюджет ВЫВОДА скрытых вызовов. Раньше был зашит в код: 1024 у грейдера,
        # 512 у condense — под GigaChat, который отдаёт JSON и ничего кроме.
        # Рассуждающие модели (qwen-семейство в режиме thinking на KitAI)
        # тратят из того же `max_tokens` сначала на рассуждения, и на сам JSON
        # его не остаётся: на проде это «GigaChat вернул пустой ответ» с
        # `finish_reason=length` и мёртвый реранкер на каждом ходе. Токены
        # вывода на входной контекст не давят (грейдер и так видит только
        # превью), так что запас дешёвый. Ручки правятся из UI.
        "grader_max_tokens": 4096,
        "condense_max_tokens": 2048,
        "grader_enabled": True,
        "grader_threshold": 4,
        # Волна 0: 2 → 0. `grader_keep_top` отдавал два верхних по СЫРОМУ рангу
        # поиска в контекст безусловно — даже с оценкой 1 («не связан с
        # вопросом»), и слоты резервировались до кэпа `_MAX_SELECTED`. То есть
        # 40% контекста каждого хода не фильтровалось вообще. Именно так в ответ
        # про мониторинг нормализованного скора попал пункт про Min/Max PSI из
        # карточки модели (жалоба пользователя «3 пункт про psi лишний»).
        #
        # 0 → 1. Ноль оказался перебором: разбор реального диалога показал, что
        # страница, стоявшая на поиске ПЕРВОЙ, вылетала из ответа из-за одной
        # ошибки судьи — страховать стало нечем. Но и старая двойка была плоха
        # тем, что тащила фрагменты с оценкой 1 («не связан с вопросом»).
        # Компромисс — одна позиция И только если судья не забраковал её прямо
        # (`_INSURANCE_MIN_GRADE`), см. `rag_pipeline.select`.
        "grader_keep_top": 1,
        "rerank_candidates": 40,
    },
    # Prompt overrides. ``None`` (NOT the prompt text) is the default on purpose:
    # it means "use the built-in prompt from the code". Storing the full text as
    # the default would freeze every install that ever saved settings on the
    # edition of the prompt that happened to be current that day — later prompt
    # improvements would never reach them. Only a deliberate user edit persists
    # text here; clearing the field writes ``None`` again and re-joins the
    # built-in default.
    "prompts": {"system": None, "context_reminder": None},
    "env": {
        "pip_index_url": "https://sberosc.sigma.sbrf.ru/repo/pypi/simple",
        "pip_token": "",
    },
    "ui": {"theme": "auto"},
}


# --------------------------------------------------------------------------- #
# pip.conf helper (stdlib only — importable by the app AND the bootstrap)
# --------------------------------------------------------------------------- #


def migrate_pip_index_url(url: str) -> str:
    """Self-heal a stale SberOSC pip index URL missing the ``/repo/`` prefix.

    Early test runs persisted ``https://sberosc.<host>/pypi/simple`` (the wrong
    path — the mirror actually lives under ``/repo/pypi/simple``). Because stored
    config overrides the built-in default, that stale value keeps breaking pip.

    This is deliberately conservative: only when the host is a SberOSC host
    (hostname contains ``sberosc.``) AND the path is *exactly* ``/pypi/simple``
    do we rewrite the path to ``/repo/pypi/simple``. Scheme, ``user:token@``
    userinfo, host, port, query and fragment are preserved. Everything else —
    already-correct ``/repo/pypi/simple``, ``nexus-ci`` URLs, any non-sberosc
    index — is returned unchanged.
    """
    try:
        parts = urlsplit(url)
    except (ValueError, TypeError):
        return url
    host = parts.hostname or ""
    if "sberosc." in host and parts.path == "/pypi/simple":
        return urlunsplit(parts._replace(path="/repo/pypi/simple"))
    return url


def build_pip_conf(index_url: str, token: str) -> tuple[str, str, str]:
    """Build a pip ``[global]`` config for the SberOSC mirror.

    Returns ``(pip_conf_text, redacted_index_url, trusted_host)`` where:

    * ``pip_conf_text`` is a ready-to-write ``pip.conf`` body pinning
      ``index-url``/``trusted-host``/``default-timeout`` (the SberOSC-recommended
      method — inline ``--index-url``/``--trusted-host`` flags break transitive
      dependency resolution).
    * ``redacted_index_url`` is the effective index URL with the token masked as
      ``***`` — safe for logging / SSE.
    * ``trusted_host`` is the hostname parsed from ``index_url``.

    When ``token`` is non-empty and ``index_url`` carries no ``@`` userinfo yet,
    ``token:<token>@`` is injected right after the scheme, yielding
    ``https://token:<token>@<rest>``. Empty token (or a URL that already has
    credentials) leaves the URL unchanged.
    """
    parts = urlsplit(index_url)
    trusted_host = parts.hostname or ""
    has_userinfo = "@" in parts.netloc

    if token and not has_userinfo:
        prefix = f"{parts.scheme}://" if parts.scheme else ""
        rest = index_url[len(prefix):] if prefix and index_url.startswith(prefix) else index_url
        auth_url = f"{prefix}token:{token}@{rest}"
        redacted_index_url = f"{prefix}token:***@{rest}"
    else:
        auth_url = index_url
        redacted_index_url = index_url

    pip_conf_text = (
        "[global]\n"
        f"index-url={auth_url}\n"
        f"trusted-host={trusted_host}\n"
        "default-timeout=120\n"
    )
    return pip_conf_text, redacted_index_url, trusted_host

# Keys whose *string* values are filesystem paths and should be expanduser'd.
_PATH_KEYS: dict[str, tuple[str, ...]] = {
    "gigachat": ("cert_path", "key_path"),
}


# --------------------------------------------------------------------------- #
# Merge / expand helpers
# --------------------------------------------------------------------------- #


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` and return it.

    Dicts are merged key-by-key; any non-dict value in ``override`` (including
    ``None``) replaces the corresponding value in ``base``.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_paths(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with known path fields run through ``expanduser``."""
    result = copy.deepcopy(config)
    for section, keys in _PATH_KEYS.items():
        sect = result.get(section)
        if not isinstance(sect, dict):
            continue
        for key in keys:
            val = sect.get(key)
            if isinstance(val, str) and val:
                sect[key] = os.path.expanduser(val)
    return result


# --------------------------------------------------------------------------- #
# Load / save
# --------------------------------------------------------------------------- #


def load_config(paths: AppPaths | None = None) -> dict[str, Any]:
    """Read config fresh from disk, deep-merged over the defaults.

    A missing or unreadable/invalid file yields the defaults (with expanded
    paths). Path fields are always expanded so callers get absolute paths.
    """
    paths = paths or PATHS
    stored: dict[str, Any] = {}
    try:
        raw = paths.config_file.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            stored = parsed
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        stored = {}
    merged = deep_merge(DEFAULT_CONFIG, stored)
    env = merged.get("env")
    if isinstance(env, dict) and isinstance(env.get("pip_index_url"), str):
        env["pip_index_url"] = migrate_pip_index_url(env["pip_index_url"])
    return _expand_paths(merged)


def _read_raw_config(paths: AppPaths) -> dict[str, Any]:
    """Read the on-disk config verbatim (no defaults, no expansion)."""
    try:
        parsed = json.loads(paths.config_file.read_text(encoding="utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


class ConfigError(ValueError):
    """Raised when a partial config update has an invalid shape."""


def _validate_partial(partial: dict[str, Any]) -> None:
    """Loosely validate a partial config object before persisting.

    We only guard against obviously wrong shapes: top level must be an object,
    and any key that is a dict in DEFAULT_CONFIG must be a dict here too.
    """
    if not isinstance(partial, dict):
        raise ConfigError("config must be a JSON object")
    for key, value in partial.items():
        default_val = DEFAULT_CONFIG.get(key)
        if isinstance(default_val, dict) and not isinstance(value, dict):
            raise ConfigError(f"section '{key}' must be an object")


def save_config(
    partial: dict[str, Any], paths: AppPaths | None = None
) -> dict[str, Any]:
    """Deep-merge ``partial`` over the CURRENT file contents and persist.

    Writes atomically (temp file + ``os.replace``). Returns the freshly loaded
    (defaults-merged, path-expanded) config. Raises :class:`ConfigError` on a
    bad shape.
    """
    paths = paths or PATHS
    _validate_partial(partial)
    paths.ensure_dirs()

    current = _read_raw_config(paths)
    updated = deep_merge(current, partial)

    tmp = paths.config_file.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(tmp, paths.config_file)

    return load_config(paths)
