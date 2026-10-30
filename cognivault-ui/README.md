# CogniVault UI

Chat UI for CogniVault + GigaChat. A small FastAPI server serves the SPA from
`static/`, proxies to the CogniVault backend and to GigaChat over mTLS, and runs
the RAG pipeline (retrieval → grading → context assembly) itself.

## Two modes

`COGNIVAULT_UI_MODE` (default `local`) selects how the service is configured and
who it serves. `app/settings.py` is authoritative.

| | `local` | `server` |
| --- | --- | --- |
| Who it serves | one person, on their own machine | many users behind one deployment |
| Bind | `127.0.0.1:8787` (`run.sh`) | `0.0.0.0:8787` (`UI_HOST`/`UI_PORT`) |
| Auth | none — the CogniVault token sits in the config | `Authorization: Bearer <cv-token>` on every `/api` call; the token *is* the tenant id, validated against the backend |
| Config source | `~/.cognivault-ui/config.json`, deep-merged over `DEFAULT_CONFIG` | built from env by `settings.server_config()`, with per-user overrides merged on top |
| Data | `~/.cognivault-ui/{certs,history,tmp}` | `$UI_DATA_DIR/users/<sha256(token)[:16]>/{config.json,history,certs,tmp,confluence,rag_log.jsonl}` |
| `/api/env/*` | available | not registered (local-only concern) |

Server mode is what the cluster runs (`Dockerfile` → `COGNIVAULT_UI_MODE=server`,
`UI_DATA_DIR=/data`, `uvicorn app.main:app --host 0.0.0.0 --port 8787`); the
manifests and the full deployment procedure live in
[`deploy/dropapp/README.md`](../deploy/dropapp/README.md). The rest of this file
describes the local mode.

### Token roles in server mode

Token-based RBAC is opt-in and keeps existing deployments backward-compatible.
With `UI_RBAC_ENABLED=false` every valid token has the historical `admin` role.
To enable it, set `UI_ADMIN_TOKEN_HASHES` to one or more comma-separated
SHA-256 hashes of administrator CogniVault tokens, then set
`UI_RBAC_ENABLED=true`. Never put the tokens themselves in this variable.

Administrators can open settings, configure Confluence, upload/evaluate data and
maintain the index. Regular users can chat, keep history, follow source links and
leave feedback. The UI hides the settings entry, and the protected APIs enforce
the same boundary with HTTP 403.

## First run (bootstrap, local mode)

The bootstrap script uses only the Python standard library, so it runs before
any dependency is installed:

```bash
python3 bootstrap.py
```

It will:

1. create `~/.cognivault-ui/{certs,history,tmp}`,
2. write a default `~/.cognivault-ui/config.json` (only if absent),
3. build a virtualenv at `~/.cognivault-ui/venv`,
4. install `fastapi`, `uvicorn`, `httpx`, `python-multipart` from the SberOSC
   mirror.

Then drop your GigaChat client certificate and key into
`~/.cognivault-ui/certs/` (default names `client_crt.crt` / `client_key.key`)
and set `cognivault.token` in the config (or via the UI).

> Caveat: `bootstrap.py` carries its own copy of `DEFAULT_CONFIG` (it must stay
> stdlib-only, so it cannot import `app/config.py`), and that copy has drifted —
> it writes `rag.source: "semantic"`, `rag.limit: 5`, `rag.max_context_chars:
> 12000` and no `prompts` block. Since a written value wins over the built-in
> default on deep-merge, a fresh install starts with a lower context budget than
> `app/config.py` intends. `mode` is not written at all, so the pipeline still
> runs in `auto` (where `source`/`limit` are ignored). Fix by editing
> `~/.cognivault-ui/config.json` or the corresponding fields in the UI.

## SberOSC PyPI mirror

Dependencies install from the SberOSC proxy. Two config values (in
`~/.cognivault-ui/config.json` under `env`, or via **Настройки → Окружение** in
the UI) drive it:

- `env.pip_index_url` — default
  `https://sberosc.sigma.sbrf.ru/repo/pypi/simple` (note the `/repo/` prefix).
- `env.pip_token` — your personal SberOSC token, copied from the profile page
  <https://sberosc.sigma.sbrf.ru/dashboard/profile/>.

Rather than passing `--index-url`/`--trusted-host` inline (which can break
resolution of transitive dependencies), bootstrap and the in-app setup write a
`pip.conf` into the venv and run plain `pip install` with `PIP_CONFIG_FILE`
pointing at it:

```ini
[global]
index-url=https://token:<TOKEN>@sberosc.sigma.sbrf.ru/repo/pypi/simple
trusted-host=sberosc.sigma.sbrf.ru
default-timeout=120
```

The literal username is `token`; the password is your SberOSC token. The token
is never printed to logs — only a redacted `https://token:***@…` form is shown.

Notes:

- Brand-new package versions may be held under a ~3-day SberOSC quarantine
  before they appear in the proxy.
- For libraries that were explicitly ordered / marked «Загружен в Nexus», the
  proxy path may 404; in that case set `env.pip_index_url` to the Nexus mirror
  instead: `https://login:token@nexus-ci.sigma.sbrf.ru/repository/pypi-lib-ext/simple/`.

## Run (local mode)

```bash
./run.sh
```

which is equivalent to:

```bash
~/.cognivault-ui/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8787
```

Then open **http://localhost:8787**.

> Open the UI only through `http://localhost:8787`. Opening the built SPA via
> `file://` will not work — the browser must talk to this server (same-origin)
> so the `/api` calls and SSE streams resolve.

## Layout

- `app/` — FastAPI app, routers, and clients (CogniVault, GigaChat, RAG).
  `app/rag.py` assembles the context, `app/rag_pipeline.py` holds the two hidden
  GigaChat calls, `app/settings.py` owns mode + config policy.
- `static/` — the built SPA (served at `/`).
- `~/.cognivault-ui/` — runtime data in local mode: `config.json`, `certs/`,
  `history/`, `venv/`, `tmp/`. In server mode the same tree lives per user under
  `$UI_DATA_DIR/users/<hash>/`.

## Конвейер чата

Тумблер RAG в чате — простой вкл/выкл (`rag.default_on` задаёт начальное
положение). При включённом RAG на каждый вопрос происходит **два скрытых вызова
GigaChat** сверх самой генерации ответа (`app/rag_pipeline.py`):

1. **Интент + переписывание вопроса** (`condense`, temperature 0, таймаут 10 с,
   история — последние 6 реплик). Модель классифицирует реплику как
   `smalltalk` / `clarify` / `kb_question` и, для последнего, переписывает её в
   самодостаточный поисковый запрос (подставляет вместо местоимений конкретные
   названия из истории). `smalltalk` и `clarify` поиск вообще не запускают.
   Тем же ответом приходит **охват** вопроса — `scope: "document" | "corpus"`
   (ответ лежит в одном документе или вопрос про базу целиком); отсутствующее
   или незнакомое значение читается как `document`, то есть как поведение до
   появления поля. Первая реплика без истории вызов пропускает — если только не
   включён `condense_first_turn`, и тогда с первого хода берётся **только**
   `scope`: разрешать без истории нечего, а ошибочный `smalltalk` на первой
   реплике стоил бы пользователю поиска.
2. **Батч-грейдер релевантности** (`grade`, таймаут 20 с) — он же реранкер.
   Все кандидаты оцениваются по шкале 1–5 одним вызовом; при >15 кандидатах
   они режутся на батчи по 12 и оцениваются параллельно (40 кандидатов → 4 вызова).

Между ними — сам поиск: `POST /api/vault/search/hybrid` c
`group_by_section: true` и `section_max_chars` (откат на `search/semantic`),
ширина выборки — `rag.rerank_candidates`.

Отбор после грейдинга (`select`, чистая функция): проходят фрагменты с оценкой
≥ `grader_threshold`; если таких нет — порог опускается до 3; топ-`grader_keep_top`
по рангу поиска добавляются всегда; в контекст попадает не больше 5 фрагментов.
Если не прошло ничего — генерация **не вызывается**, пользователь получает
готовый ответ «в доступных мне документах ответа на этот вопрос не нашлось»
(`finish_reason: "no_context"`).

Дальше отобранное расширяется до читаемых блоков: для верхних
`max_expanded_files` файлов подтягивается полный текст (`/api/vault/content`) и
выбирается глубина — **весь файл** (короткий документ или ≥3 попаданий, если
влезает в бюджет), **раздел** (`section_text` из ответа бэкенда) или
**фрагмент** (исходный чанк). Всё собирается в один блок под символьный бюджет,
вычисляемый из окна модели (`gigachat.model_context_tokens`) с запасом на
русский текст. Ответ обязан ссылаться на источники как `[Источник N]` —
это часть системного промпта, номера вне списка источников в UI не линкуются.

Перед блоком «Источники» в то же сообщение подставляется **состав базы**
(`app/corpus_map.py`): сколько всего документов в вольте, сколько фрагментов
модели показали и как корпус делится по верхним разделам со счётчиками. Без
него модель не знает, что держит 5 документов из 127, и уверенно отвечает по
одной случайно подошедшей странице-списку. Блок строится из кэшированного
`GET /api/vault/files` (TTL 5 минут на вольт, таймаут 5 с) плюс одного поля
каталога, не стоит ни одного вызова модели, не растёт с корпусом (свёртка на
глубине 1–2, потолок 700 символов ≈ 150 токенов) и **молча исчезает**, если
листинг недоступен. Текст блока принадлежит коду, а не редактируемым промптам:
пометка «это не источник, ссылаться нельзя» обязана доехать и до пользователя,
который уже сохранил свой `prompts.system`. В `context_chars`, в лог запросов и
в оценку eval он не входит — там по-прежнему ровно блок источников.

Что считать документом, модуль **не решает сам**: набор расширений приезжает
полем `document_extensions` из `GET /api/vault/catalog`, где выводится из той же
константы, по которой поллер сканирует вольт. Прежний локальный список врал —
`txt` и `markdown` индексатор не сканирует вовсе, и каждый `.txt` в вольте
считался документом, про который можно спросить и который поиск никогда не
вернёт. Каталог недоступен — блока нет; встроенного списка-запасника не осталось.

**Дерево разделов** (`app/corpus_tree.py`, флаг `rag.corpus_tree_enabled`,
по умолчанию выключен) — тот же слот в голове сообщения, но вместо свёртки со
счётчиками туда встаёт полный перечень разделов и страниц из
`GET /api/vault/catalog`: каждый проиндексированный документ по имени, с
аннотациями индексатора для верхних двух уровней. Он существует потому, что
некоторые вопросы («какие витрины ClickHouse описаны в базе», «что лежит в
разделе Архив») не отвечаются ни одним документом: страница-список `Продукты` в
корпусе есть и она **пустая**, перечень продуктов существует только как форма
дерева.

Три вещи помечаются явно, иначе список заголовков звучит как инвентаризация и
врёт авторитетно: `[архив]` — устаревшая ветка, `[раздел: N]` — страница-контейнер
с N документами внутри, `[пусто]` — у страницы нет своего текста (на боевом
корпусе таких 17 из 127 — поиск по ним не вернёт ничего). Перечень строится по
индексу, а не по страницам-оглавлениям: те перечисляют не всё («База знаний»
называет 2 из 3 своих детей, «Fincert» — 5 из 6, «General» ссылается за пределы
своего поддерева).

Дерево **добавляется** к источникам, а не заменяет их: блок «Источники»
отбирается, грейдится и режется по своему бюджету ДО того, как дерево вообще
запрашивается, поэтому оно не может ни вытеснить фрагмент, ни укоротить его, и в
`context_chars` не входит. Ход, который грейдер отверг целиком, до дерева не
доходит вовсе — отказ по-прежнему выносит только грейдер. Цена: ~8 500 символов
(≈3 400 токенов) в каждом ходе на боевом корпусе, потолок 9 500 символов с
лестницей деградации (сначала аннотации по уровню, потом глубина). Платит за неё
история диалога: `trim_history` защищает последнее сообщение и выбрасывает
старые реплики. Флаг выключен по умолчанию, потому что дерево informative ровно
настолько, насколько иерархия папок совпадает с деревом продуктов; на вольте
вида `2024/Q1/заметки` тот же блок предъявит модели календарь.

**Вопрос про саму базу** («что ты знаешь?», «о каких проектах знаешь?») ловится
до всего этого — узким набором якорных формулировок в `app/corpus_scope.py`, без
единого вызова модели, поэтому он работает и на первой реплике, где condense
пропускается. Ответ собирается из той же структуры
(`corpus_tree.render_overview` при включённом флаге, иначе
`corpus_map.render_overview`): реальные названия и счётчики, свой системный
промпт (`rag.META_SYSTEM_PROMPT` — `NO_RAG_SYSTEM_PROMPT` здесь не годится, он
запрещает утверждать то, чего нет в истории диалога, то есть запрещает и рассказ
об охвате). Совпадение обязано быть точным: «Что ты знаешь про PSI?» уходит в
обычный поиск. Непопадание и недоступный листинг — тоже обычный поиск, то есть
сегодняшнее поведение, включая отказ грейдера.

Матчер не видит истории, поэтому каждая формулировка обязана быть безопасной как
голая реплика второго хода — а часть из них не была: у «какие разделы?», «что ты
ещё знаешь?», «о каких продуктах есть информация?» опущено дополнение, и после
«Расскажи про продукт Fincert» они означают разделы ПРОДУКТА. Такие формулировки
предлагаются только на ходу без истории (`match_meta(..., has_history=...)`, тот
же признак, по которому condense считает ход первым): начиная со второго хода их
разбирает condense, который историю видит и переписывает вопрос в поиск. Якорные
формулировки («…в базе знаний», «кто ты?», «что ты умеешь?») работают на любом
ходу — поток «привет» → «что ты умеешь?» не ломается.

**Оговорка по концентрации доказательств** дописывается ПОСЛЕ ответа, когда
охват вопроса `corpus`, а все выбранные фрагменты пришли из одного документа
(«ответ описывает только его»). Чистый Python, ноль токенов, ответ она уточняет,
а не заменяет; отказ она не подменяет никогда — отказ по-прежнему выносит только
грейдер. В `rag_log.jsonl` пишутся поля `scope` и `hedge`, чтобы долю оговорок
можно было мерить на живом стенде.

Каждый источник в UI получает бейдж глубины (`файл` / `раздел` / `фрагмент`) и
неброскую строку вида «контекст: 9 800 симв. · 3 источника». SSE-события идут в
порядке `meta` → (`notice` | `sources`) → `token`… → `done`; `sources`
эмитится **до** первого токена и несёт `n`, `title`, `path`, `section_path`,
`score`, `depth`, `grade` (оценка грейдера) и `url` для страниц, приехавших из
Confluence.

Отказоустойчивость: любая ошибка скрытого вызова (кривой JSON, таймаут,
транспорт) логируется предупреждением и не ломает ответ — condense
деградирует к сырому вопросу, грейдер пропускает кандидаты как есть.

## Настройки в UI

Панель настроек редактирует конфиг через `PUT /api/config` тремя независимыми
блоками. В server-режиме сохраняется **только** список
`settings.USER_EDITABLE_KEYS` и **только в конфиг вызывающего пользователя**
(`$UI_DATA_DIR/users/<hash>/config.json`); всё остальное из тела запроса
отбрасывается и возвращается в поле `ignored`. В local-режиме пишется
`~/.cognivault-ui/config.json` целиком.

**Модель и генерация** (`gigachat.*`): `model`, `temperature` (0–2),
`max_tokens` (1…`model_context_tokens`). Поле `model_context_tokens` показано,
но заблокировано — его задаёт администратор.

**Поиск и контекст** (`rag.*`):

| Ключ | По умолчанию | Назначение |
| --- | --- | --- |
| `default_on` | `false` | начальное положение тумблера RAG |
| `limit` | `10` | ширина выборки (только legacy-режим `mode != "auto"`) |
| `rerank_candidates` | `40` | сколько кандидатов запрашивать у бэкенда и отдавать грейдеру |
| `grader_enabled` | `true` | включить батч-грейдер |
| `grader_threshold` | `4` | минимальная оценка (1–5), с которой фрагмент проходит |
| `grader_keep_top` | `2` | сколько верхних по рангу поиска пропускать всегда |
| `condense_enabled` | `true` | включить интент + переписывание вопроса |
| `condense_first_turn` | `false` | разбирать и первую реплику: +1 вызов на каждое первое сообщение, покупает только `scope` (и через него — оговорку про охват) |
| `corpus_tree_enabled` | `false` | подмешивать полный перечень разделов и страниц из каталога вместо короткого блока «состав базы»: ноль вызовов модели, ~3 400 токенов на ход, осмысленно только когда папки вольта = разделы базы |
| `max_context_chars` | `24000` | потолок символьного бюджета |
| `file_full_chars` | `6000` | порог «взять файл целиком» |
| `section_max_chars` | `4000` | лимит на текст раздела |
| `max_expanded_files` | `2` | сколько верхних файлов расширять |
| `min_score` | `null` | нижний порог по score |

**Промпты** (`prompts.*`) — ровно два редактируемых: `system` (правила ответа,
включая формат `[Источник N]`) и `context_reminder` (напоминание между блоком
источников и вопросом). Пустое поле или кнопка «Сбросить к стандартному»
возвращают встроенный текст (`null` в конфиге, а не копия строки), рядом
показывается бейдж «изменён». Промпты самого конвейера (condense и грейдер)
показаны в свёрнутом блоке **только для чтения**: их ответ разбирается как JSON,
поэтому править их из UI нельзя.

Ключи, которые UI показывает заблокированными (`ADMIN_LOCKED_KEYS`):
`cognivault.base_url`/`token`, все `gigachat.*` про адрес и сертификаты,
`gigachat.model_context_tokens`, `rag.mode`, `rag.source`, `rag.token_budget`,
`env.*`. В server-режиме они приходят из переменных окружения пода.

Границы валидации (`settings.validate_user_overrides`): `limit` и
`rerank_candidates` 1–100, `grader_threshold` 1–5, `grader_keep_top` 0–10,
`max_expanded_files` 0–10, `max_context_chars` 500–200000, `file_full_chars` и
`section_max_chars` 100–100000, `min_score` 0–1 или `null`, промпты ≤20000
символов.

Поскольку ручки грейдера и ширина ретрива живут в конфиге пользователя, A/B двух
наборов настроек делается без пересборки образа — см. `tools/eval/README.md`.

## Индекс и поиск (переиндексация и пересоздание коллекции)

У оператора нет доступа ни к шеллу, ни к векторной базе, поэтому обе операции
обслуживания живут в панели настроек, блок **«Индекс и поиск»**:

- **«Переиндексировать вольт»** — заново разбирает и векторизует документы
  пользователя. Файлы не трогает. Показывает `обработано N из M файлов` и в
  конце — сводку с пофайловыми ошибками.
- **«Пересоздать коллекцию»** — разрушающая операция на весь кластер: удаляет
  векторы **всех** пользователей и собирает коллекцию заново. Единственная
  защита — подтверждение: предупреждение прямым текстом (что удаляются векторы
  всех, что до конца перестройки поиск ничего не находит) плюс поле, куда надо
  **вручную ввести название коллекции** (берётся из `GET /api/admin/collection`,
  в поле не подставляется). Показывает этап, `пользователей X из Y` и ошибки.

Рядом одной фразой сказано состояние схемы индекса: при
`schemeVersion != expectedSchemeVersion` — «поиск по словам работает хуже
обычного, пока коллекция не пересоздана» (номера версий не печатаются).

Обе операции — **задачи на сервере**, а не запросы: `POST` возвращает `jobId`,
прогресс читается опросом статуса раз в 2.5 с. Поэтому задача переживает
закрытие вкладки, а вернувшийся оператор при открытии панели снова цепляется к
тому, что ещё идёт. Двойной клик безопасен: `409` от бэкенда не показывается
ошибкой, а подключает к уже идущей задаче. `jobId` живёт в памяти процесса
(`admin_routes._JOBS`), не в пользовательском `config.json`. На старом бэкенде
без `/api/admin/collection*` прокси отвечает `501 COLLECTION_API_UNAVAILABLE`,
UI прячет пересоздание и объясняет почему — переиндексация продолжает работать.

## API surface

| Method | Path                        | Purpose                                       |
| ------ | --------------------------- | --------------------------------------------- |
| GET    | `/healthz`                  | liveness/readiness probe                       |
| GET    | `/api/whoami`               | identity; in server mode validates the token   |
| GET    | `/api/config`               | effective config + defaults, locked keys, read-only prompts |
| PUT    | `/api/config`               | deep-merge + persist a partial config          |
| GET    | `/api/status`               | health/cert/env/history status                 |
| POST   | `/api/chat`                 | streaming chat (SSE), optional RAG             |
| POST   | `/api/feedback`             | 👍/👎 on an answer → `rag_log.jsonl`           |
| GET    | `/api/history`              | list newest chats                              |
| GET    | `/api/history/{id}`         | load a chat                                    |
| DELETE | `/api/history/{id}`         | delete a chat                                  |
| POST   | `/api/upload`               | forward a vault zip to CogniVault              |
| POST   | `/api/admin/reindex`        | start a vault reindex → `{jobId, attached}`    |
| GET    | `/api/admin/reindex/status` | reindex progress (`jobId` optional → remembered job, else idle) |
| GET    | `/api/admin/collection`     | collection name/alias, scheme version, points  |
| POST   | `/api/admin/collection/rebuild` | rebuild the collection (`confirm` = collection name) |
| GET    | `/api/admin/collection/rebuild/status` | rebuild phase / users done / errors |
| GET    | `/api/confluence/config`    | current Confluence source config (no secrets)  |
| PUT    | `/api/confluence/config`    | save Confluence credentials + root link        |
| POST   | `/api/confluence/validate`  | check credentials / resolve the root page      |
| POST   | `/api/confluence/sync`      | run a sync, streaming progress (SSE)           |
| GET    | `/api/confluence/status`    | last sync result and manifest counters         |
| POST   | `/api/env/setup`            | provision venv + deps (SSE) — local only       |
| GET    | `/api/env/export`           | download an env backup zip — local only        |
| POST   | `/api/env/import`           | restore an env backup zip — local only         |

`/api/confluence/*` is registered only when the `CONFLUENCE_ENABLED` admin flag
is on (default), `/api/env/*` only in local mode.
