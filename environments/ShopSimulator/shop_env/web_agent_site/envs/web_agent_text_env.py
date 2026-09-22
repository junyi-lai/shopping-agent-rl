"""ShopSimulator Environment v2.1: the pure simulator served by ``pack_api``.

The environment exposes one session per lease.  It renders the shop pages the
agent observes, keeps the transactional session state (search results, opened
product, selected variant) and reports raw terminal facts.  It never scores a
trajectory: termination is liveness-only (max steps, exact repeats, no progress)
and the reward lives in the project's ``shopping_agent.reward`` package.
"""
import json
import math
import os

from bs4 import BeautifulSoup
from bs4.element import Comment
from collections import defaultdict
from flask import Flask
from web_agent_site.engine.engine import (
    load_products,
    init_search_engine,
    get_top_n_product_from_keywords,
    map_action_to_html,
    parse_action,
    get_product_per_page,
    ACTION_TO_TEMPLATE,
    PRODUCT_WINDOW,
    SEARCH_RETURN_N,
    END_BUTTON, NEXT_PAGE, PREV_PAGE, BACK_TO_SEARCH,
)
from web_agent_site.engine.search import normalize_query
from web_agent_site.engine.goal import get_goals
from web_agent_site.engine.observation import (
    build_observation_state,
    page_type_from_name,
)
from web_agent_site.engine.config import (
    ENVIRONMENT_VERSION,
    load_config,
)
from web_agent_site.utils import BASE_DIR, DEFAULT_FILE_PATH

# Sessions are addressed by a synthetic URL; the browser and the page renderer
# parse its path, never the host, and no HTTP request is ever made.
SESSION_URL_ROOT = "https://shopsimulator.invalid"

# ---------------------------------------------------------------------------
# Minimal environment machinery: price resolution (for observation display) and
# liveness termination (max steps / exact repeats / no-progress). All scoring
# lives in the project's ``shopping_agent.reward`` package.
# ---------------------------------------------------------------------------

_AXIS_ALIASES = {
    "color": {"颜色", "颜色分类"},
    "size": {"尺码", "鞋码"},
    "dimensions": {"尺寸", "大小"},
    "net_content": {"净含量", "总净含量"},
    "flavor": {"口味", "食品口味"},
    "specification": {"规格", "规格描述", "规格类型"},
    "bundle": {"套餐", "套餐类型", "组合套餐"},
    "capacity": {"容量", "规格容量"},
}


def _norm_option_text(value):
    import re as _re
    import unicodedata
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("/", "|")
    return _re.sub(r"\s+", "", text)


def _canonical_axis(value):
    normalized = _norm_option_text(value)
    for canonical, aliases in _AXIS_ALIASES.items():
        if normalized in {_norm_option_text(alias) for alias in aliases}:
            return canonical
    return normalized


def _finite_price(value):
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price >= 0 else None


def _resolve_selected_price(product, options):
    """Minimal deterministic variant-price resolution.

    Mirrors ``shopping_agent.reward.variant_price.resolve_variant_price``; used
    here only to display the selected variant price in observations.
    """
    raw = product.get("customization_options") or {}
    axes = {}
    for raw_axis, entries in raw.items():
        canonical = _canonical_axis(raw_axis)
        values = {}
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            normalized = _norm_option_text(entry.get("value"))
            if normalized:
                values[normalized] = _finite_price(entry.get("price"))
        if canonical in axes:
            return {"status": "unverifiable", "price": None}
        axes[canonical] = values

    selected = {}
    for raw_axis, raw_value in (options or {}).items():
        canonical = _canonical_axis(raw_axis)
        if canonical in selected:
            return {"status": "unverifiable", "price": None}
        selected[canonical] = _norm_option_text(raw_value)

    for variant in product.get("variant_combinations") or []:
        if isinstance(variant, dict) and isinstance(variant.get("options"), dict):
            normalized_variant = {
                _canonical_axis(key): _norm_option_text(value)
                for key, value in variant["options"].items()
            }
            if normalized_variant == selected:
                price = _finite_price(variant.get("price"))
                return {
                    "status": "pass" if price is not None else "unverifiable",
                    "price": price,
                }

    effective_axes = []
    all_prices = []
    for canonical, values in axes.items():
        prices = {price for price in values.values() if price is not None}
        all_prices.extend(prices)
        if len(prices) > 1:
            effective_axes.append(canonical)
    if len(effective_axes) > 1:
        return {"status": "unverifiable", "price": None}
    if len(effective_axes) == 1:
        axis = effective_axes[0]
        value = selected.get(axis)
        price = axes[axis].get(value) if value else None
        return {
            "status": "pass" if price is not None else "unverifiable",
            "price": price,
        }
    unique_prices = {price for price in all_prices}
    if len(unique_prices) == 1:
        return {"status": "pass", "price": next(iter(unique_prices))}
    pricing = [
        price
        for price in (_finite_price(value) for value in product.get("pricing") or [])
        if price is not None
    ]
    if not axes and len(set(pricing)) == 1:
        return {"status": "pass", "price": pricing[0]}
    return {"status": "unverifiable", "price": None}


class MinimalTerminationTracker:
    """Liveness only: exact repeats and max steps.

    上游 ShopSimulator 没有"无进展即终止"这条规则。本仓曾按 no_progress_limit=4 提前终止,
    与已验收的训练数据不兼容(1192 条 gold 轨迹里约 93% 含连续 4 步以上无进展),故该终止
    判据已移除。no_progress_steps 仍作为诊断量统计并回报,但不再决定终止。
    """

    def __init__(self, max_steps=35, exact_repeat_limit=2):
        self.max_steps = int(max_steps)
        self.exact_repeat_limit = int(exact_repeat_limit)
        self.steps = 0
        self.consecutive_repeats = 0
        self.no_progress_steps = 0
        self.last_signature = None
        self.seen_asins = set()
        self.opened_asins = set()
        self.distinct_normalized_queries = set()

    def record(self, action_name, action_arg, visible_asins=()):
        self.steps += 1
        signature = (
            str(action_name or "").casefold(),
            str(action_arg or "").casefold(),
        )
        if signature == self.last_signature:
            self.consecutive_repeats += 1
        else:
            self.consecutive_repeats = 0
        self.last_signature = signature

        visible = set(str(asin) for asin in visible_asins)
        new_asins = visible - self.seen_asins
        self.seen_asins.update(visible)
        name = str(action_name or "").casefold()
        if name == "search":
            self.distinct_normalized_queries.add(normalize_query(str(action_arg or "")))
        if name == "click" and str(action_arg or "").isdigit():
            self.opened_asins.add(str(action_arg))

        if new_asins or name == "search":
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        reason = None
        if self.consecutive_repeats >= self.exact_repeat_limit:
            reason = "repeat_loop"
        elif self.steps >= self.max_steps:
            reason = "max_steps"
        return {
            "termination_version": "shopping-termination-v4",
            "termination_reason": reason,
            "step_count": self.steps,
            "consecutive_repeats": self.consecutive_repeats,
            "no_progress_steps": self.no_progress_steps,
            "new_asin_count": len(new_asins),
            "seen_asin_count": len(self.seen_asins),
            "opened_candidate_count": len(self.opened_asins),
            "distinct_normalized_queries": len(self.distinct_normalized_queries),
        }

app = Flask(__name__)


class WebAgentTextEnv:
    """One leased ShopSimulator session: search, browse, select, buy."""

    def __init__(
            self,
            observation_mode='html',
            file_path=DEFAULT_FILE_PATH,
            server=None,
            **kwargs
        ):
        """Create the session state of one leased environment slot.

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        server -- a shared SimServer so every slot reuses one corpus copy
        session / session_prefix -- explicit session id (one per slot)
        show_attrs -- render the Attributes block on the item page
        """
        self.observation_mode = observation_mode
        self.kwargs = kwargs

        self.file_path = file_path

        self.base_url = SESSION_URL_ROOT
        self.server = SimServer(
            self.base_url,
            self.file_path,
            self.kwargs.get('show_attrs', False),
        ) if server is None else server
        self.browser = SimBrowser(self.server)
        self.session = self.kwargs.get('session')
        self.session_prefix = self.kwargs.get('session_prefix')

    def step(self, action):
        """Apply one action string and return (observation, status, info).

        Arguments:
        action (`str`) -- ``search[keywords]``, ``click[value]`` or
                          ``finish``; anything else is a no-op.
        """
        info = None
        self.get_available_actions()

        # Determine action type (click, search) and argument
        try:
            action_name, action_arg = parse_action(action)
        except (ValueError, TypeError, AttributeError):
            # 更具体的异常处理，避免捕获所有异常
            action_name, action_arg = "", ""
        if action_arg is not None:
            action_arg = action_arg.lower()
        if action_name == 'finish':
            status = self.server.finish_without_purchase(self.session)
        elif (action_name == 'search' and
            action_arg is not None and
            action_arg != ''):
            # 执行搜索
            status = self.browser.search(action_arg)
        elif (action_name == 'click' and
              action_arg in self.text_to_clickable.keys() and
              action_arg != 'search'):
            status = self.browser.click(action_arg, self.text_to_clickable)
        else:
            status = dict(reward=0, done=False)

        # Update observation, state with the new action
        ob = self.observation
        if not status.get("done"):
            progress = self.server.record_progress(
                self.session,
                action_name,
                action_arg,
                self.server.visible_asins(self.session, self.browser.current_url),
                current_url=self.browser.current_url,
            )
            status["progress"] = progress
            if progress["termination_reason"]:
                status = self.server.terminate_session(
                    self.session,
                    progress["termination_reason"],
                    progress=progress,
                )
        return ob, status, info

    def get_available_actions(self):
        """Returns list of available actions at the current step"""
        html_obj = self._parse_html()

        # Collect search bar, buttons, links, and options as clickables
        search_bar = html_obj.find(id='search_input')
        has_search_bar = True if search_bar is not None else False
        buttons = html_obj.find_all(class_='btn')
        product_links  = html_obj.find_all(class_='product-link')
        buying_options = html_obj.select('input[type="radio"]')

        self.text_to_clickable = {
            f'{b.get_text()}'.lower(): b
            for b in buttons + product_links
        }
        for opt in buying_options:
            opt_value = opt.get('value')
            self.text_to_clickable[f'{opt_value}'] = opt
        return dict(
            has_search_bar=has_search_bar,
            clickables=list(self.text_to_clickable.keys()),
        )

    def get_instruction_text(self):
        """Read the user requirement rendered into the current page."""
        html_obj = self._parse_html(self.browser.page_source)
        instruction_text = html_obj.find(id='instruction-text').h4.text
        return instruction_text

    def _parse_html(self, html=None):
        """Return the current page (or ``html``) as a BeautifulSoup object."""
        if html is None:
            html = self.state['html']
        html_obj = BeautifulSoup(html, 'html.parser')
        return html_obj

    @property
    def observation(self):
        """Compiles state into either the `html` or `text` observation mode"""
        html = self.state['html']
        if self.observation_mode == 'html':
            return html
        elif self.observation_mode == 'text':
            return self.convert_html_to_text(html, simple=True)
        elif self.observation_mode == 'text_rich':
            return self.convert_html_to_text(html, simple=False)
        elif self.observation_mode == 'url':
            return self.state['url']
        else:
            raise ValueError(
                f'Observation mode {self.observation_mode} not supported.'
            )

    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        return dict(
            url=self.browser.current_url,
            html=self.browser.page_source,
            instruction_text=self.instruction_text,
        )

    def convert_html_to_text(self, html, simple=False):
        """Strip HTML of tags and add separators to convert observation into simple mode"""
        texts = self._parse_html(html).findAll(text=True)
        visible_texts = filter(tag_visible, texts)
        if simple:
            # For `simple` mode, return just [SEP] separators
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ''
            for t in visible_texts:
                if t == '\n':
                    continue
                if t.parent.name == 'button':  # button
                    processed_t = f'[button] {t} [button_]'
                elif t.parent.name == 'label':  # options
                    if f'"{t}"' in self.state['url']:
                        processed_t = f'  [clicked button] {t} [clicked button_]'
                        observation = f'You have clicked {t}.\n' + observation
                    else:
                        processed_t = f'  [button] {t} [button_]'
                elif t.parent.get('class') == ["product-link"]: # product asins
                    if f'{t}' in self.server.user_sessions[self.session]['asins']:
                        processed_t = f'\n[clicked button] {t} [clicked button_]'
                    else:
                        processed_t = f'\n[button] {t} [button_]'
                else: # regular, unclickable text
                    processed_t =  str(t)
                observation += processed_t + '\n'
            return observation

    def reset(self,idx=None,session=None, instruction_text=None):
        """Create a new session and reset environment variables"""
        session_int = None

        if session is not None:
            self.session = str(session)
            session_int = idx
        elif self.session_prefix is not None:
            self.session = f"{self.session_prefix}-{idx}"
            session_int = idx
        else:
            self.session = idx

        init_url = f'{self.base_url}/{self.session}'
        self.browser.get(init_url, session_id=self.session, session_int=session_int)

        self.text_to_clickable = None
        self.instruction_text = self.get_instruction_text() if instruction_text is None else instruction_text
        obs = self.observation
        return obs, None

    def structured_observation(self):
        available_actions = self.get_available_actions()
        page_name = self.server.get_page_name(self.browser.current_url)
        return build_observation_state(
            page_type=page_type_from_name(page_name),
            session=self.server.user_sessions[self.session],
            product_item_dict=self.server.product_item_dict,
            available_actions=available_actions,
        )

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


class SimServer:
    """Deterministic page renderer for one leased session."""

    def __init__(self, base_url, file_path, show_attrs=False):
        """Load the frozen corpus, goals and search index once per process.

        Arguments:
        base_url -- synthetic session URL root (no HTTP is ever issued)
        file_path -- frozen product corpus
        show_attrs -- render the Attributes block on the item page
        """
        self.environment_version = ENVIRONMENT_VERSION
        config_path = os.environ.get(
            "SHOP_ENV_CONFIG",
            os.path.join(BASE_DIR, "../configs/environment.json"),
        )
        self.environment_config = load_config(config_path)

        # Load all products, goals, and search engine
        self.base_url = base_url
        self.all_products, self.product_item_dict, self.product_prices = \
            load_products(filepath=file_path)
        self.search_engine = init_search_engine(product_filepath=file_path)
        search_config = self.environment_config["search"]
        if int(search_config["top_k"]) != SEARCH_RETURN_N:
            raise ValueError("search top_k differs from the engine runtime")
        if int(search_config["page_size"]) != PRODUCT_WINDOW:
            raise ValueError("search page_size differs from the engine runtime")
        if (
            self.search_engine.manifest.get("field_weights")
            != search_config["field_weights"]
        ):
            raise ValueError("search index weights differ from the config")
        self.goals = get_goals(self.all_products)
        self.show_attrs = show_attrs
        print(f'Loaded {len(self.goals)} goals.')
        self.user_sessions = dict()
        configured_max_steps = int(
            self.environment_config["termination"]["max_steps"]
        )
        self.max_steps = int(os.environ.get("SHOP_MAX_STEPS", configured_max_steps))
        if self.max_steps != configured_max_steps:
            raise ValueError(
                "SHOP_MAX_STEPS differs from the frozen Environment v2 config"
            )

    @app.route('/', methods=['GET', 'POST'])
    def index(self, session_id, **kwargs):
        """Redirect to the search page with the given session ID"""
        html = map_action_to_html(
            'start',
            session_id=session_id,
            instruction_text=kwargs['instruction_text'],
        )
        url = f'{self.base_url}/{session_id}'
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def search_results(self, session_id, **kwargs):
        """Initialize session and return the search results page"""
        session = self.user_sessions[session_id]
        keywords = kwargs['keywords']
        assert isinstance(keywords, list)
        requested_page = kwargs.get("page")
        page = 1 if requested_page is None else requested_page
        if page < 1:
            raise ValueError("search result page must be positive")
        normalized_query = normalize_query(' '.join(keywords))
        is_new_query = (
            requested_page is None
            or session.get("normalized_query") != normalized_query
            or "search_result_asins" not in session
        )
        session["page"] = page
        session["keywords"] = keywords
        session["asin"] = None
        session["options"] = {}

        if is_new_query:
            session["actions"]["search"] += 1
            session["normalized_query"] = normalized_query
            session.setdefault("distinct_normalized_queries", set()).add(normalized_query)
            top_n_products = get_top_n_product_from_keywords(
                keywords,
                self.search_engine,
                self.product_item_dict,
            )
            session["search_result_asins"] = [product["asin"] for product in top_n_products]
        else:
            top_n_products = [
                self.product_item_dict[asin]
                for asin in session["search_result_asins"]
                if asin in self.product_item_dict
            ]

        total_pages = max(1, (len(top_n_products) + PRODUCT_WINDOW - 1) // PRODUCT_WINDOW)
        if page > total_pages:
            raise ValueError(
                f"search result page {page} is beyond the final page {total_pages}"
            )

        # Get product list from search result asins and get list of corresponding URLs
        products = get_product_per_page(top_n_products, page)
        session["current_page_asins"] = [product["asin"] for product in products]
        session["total_results"] = len(top_n_products)
        session["total_pages"] = total_pages

        keywords_url_string = '+'.join(keywords)
        url = (
            f'{self.base_url}/search_results/{session_id}/'
            f'{keywords_url_string}/{page}'
        )

        # Render HTML search page
        html = map_action_to_html(
            'search',
            session_id=session_id,
            products=products,
            keywords=session["keywords"],
            page=page,
            total=len(top_n_products),
            total_pages=total_pages,
            normalized_query=normalized_query,
            instruction_text=session["goal"]["instruction_text"],
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def item_page(self, session_id, **kwargs):
        """Render and return the HTML for a product item page"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        text_to_clickable = kwargs['text_to_clickable']
        clickable = text_to_clickable[clickable_name]

        # Update session logs with information of last product asin selected
        if (clickable.get('class') is not None and
            clickable.get('class')[0] == 'product-link'):
            session["asin"] = clickable_name.upper()
            session["actions"]["asin"] += 1
            session["asins"].add(session["asin"])
        elif clickable.get('name') is not None:
            clickable_key = clickable['name'].lower()
            session["options"][clickable_key] = clickable_name
            session["actions"]["options"] += 1

        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        keywords_url_string = '+'.join(session["keywords"])
        option_string = json.dumps(session['options'])

        # 获取当前选中的 option 和价格（纯展示，不涉及奖励）
        price_resolution = _resolve_selected_price(
            product_info,
            session["options"],
        )
        selected_price = (
            price_resolution["price"]
            if price_resolution["status"] == "pass"
            else None
        )
        session["price_resolution"] = price_resolution
        session["selected_price"] = selected_price
        session["subpage"] = None

        url = (
            f'{self.base_url}/item_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/'
            f'{session["page"]}/{option_string}'
        )

        html = map_action_to_html(
            'click',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
            show_attrs=self.show_attrs,
            selected_price=selected_price,
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def item_sub_page(self, session_id, **kwargs):
        """Render and return the HTML for a product's sub page (i.e. description, features)"""
        session = self.user_sessions[session_id]
        clickable_name = kwargs['clickable_name']
        for k in ACTION_TO_TEMPLATE:
            if clickable_name.lower() == k.lower():
                clickable_name = k
                break

        # Set fields + url of page, then render page's HTML
        product_info = self.product_item_dict[session["asin"]]
        session["subpage"] = clickable_name
        session["actions"][clickable_name] += 1
        keywords_url_string = '+'.join(session["keywords"])
        url = (
            f'{self.base_url}/item_sub_page/{session_id}/'
            f'{session["asin"]}/{keywords_url_string}/{session["page"]}/'
            f'{clickable_name}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{clickable_name}]',
            session_id=session_id,
            product_info=product_info,
            keywords=session["keywords"],
            page=session["page"],
            asin=session["asin"],
            options=session["options"],
            instruction_text=session["goal"]["instruction_text"],
        )
        return html, url

    @app.route('/', methods=['GET', 'POST'])
    def done(self, session_id, **kwargs):
        """Render the done page and return the raw purchase facts."""
        session = self.user_sessions[session_id]
        goal = self.user_sessions[session_id]['goal']
        purchased_product = self.product_item_dict[session["asin"]]
        session["actions"]["purchase"] += 1
        price = session.get("selected_price")
        if price is None:
            price = self.product_prices.get(session["asin"])

        self.user_sessions[session_id]['done'] = True
        self.user_sessions[session_id]['termination_reason'] = "purchase"

        url = (
            f'{self.base_url}/done/{session_id}/'
            f'{session["asin"]}/{session["options"]}'
        )
        html = map_action_to_html(
            f'click[{END_BUTTON}]',
            session_id=session_id,
            asin=session["asin"],
            options=session["options"],
        )
        return (
            html, url, price, purchased_product, goal, session["options"],
            self.product_item_dict[goal["asin"]],
        )

    def visible_asins(self, session_id, current_url):
        session = self.user_sessions[session_id]
        page_name = self.get_page_name(current_url)
        if page_name == "search_results":
            return tuple(session.get("current_page_asins") or ())
        if page_name in {"item_page", "item_sub_page"} and session.get("asin"):
            return (session["asin"],)
        return ()

    def record_progress(
        self,
        session_id,
        action_name,
        action_arg,
        visible_asins,
        *,
        current_url=None,
    ):
        session = self.user_sessions[session_id]
        tracker = session["progress_tracker"]
        return tracker.record(
            action_name,
            action_arg,
            visible_asins,
        )

    def finish_without_purchase(self, session_id):
        session = self.user_sessions[session_id]
        tracker = session["progress_tracker"]
        session["done"] = True
        session["termination_reason"] = "abstain"
        return {
            "done": True,
            "purchase": {},
            "goal": session["goal"],
            "target_product": self.product_item_dict[session["goal"]["asin"]],
            "termination_reason": "abstain",
            "abstain_facts": {
                "distinct_normalized_queries": sorted(tracker.distinct_normalized_queries),
                "opened_asins": sorted(tracker.opened_asins),
                "step_count": tracker.steps,
            },
        }

    def terminate_session(self, session_id, reason, *, progress=None):
        session = self.user_sessions[session_id]
        session["done"] = True
        session["termination_reason"] = reason
        status = {
            "done": True,
            "purchase": {},
            "goal": session["goal"],
            "target_product": self.product_item_dict[session["goal"]["asin"]],
            "termination_reason": reason,
        }
        if progress is not None:
            status["progress"] = progress
        return status

    def receive(self, session_id, current_url, session_int=None, **kwargs):
        """Map action to the corresponding page"""
        status = dict(reward=0.0, done=False)
        with app.app_context(), app.test_request_context():
            # Create/determine goal, instruction_text from current session
            if session_id not in self.user_sessions:
                idx = int(session_int) if session_int is not None else int(session_id)
                goal = self.goals[idx]
                instruction_text = goal['instruction_text']
                self.user_sessions[session_id] = {'goal': goal, 'done': False}
            else:
                instruction_text = \
                    self.user_sessions[session_id]['goal']['instruction_text']
            session = self.user_sessions[session_id]

            if not kwargs:
                # If no action, reset the session variables
                kwargs['instruction_text'] = instruction_text
                html, url = self.index(session_id, **kwargs)
                self.user_sessions[session_id].update(
                    {
                        'keywords': None,
                        'page': None,
                        'asin': None,
                        'asins': set(),
                        'distinct_normalized_queries': set(),
                        'normalized_query': None,
                        'search_result_asins': [],
                        'current_page_asins': [],
                        'total_results': 0,
                        'total_pages': 0,
                        'options': dict(),
                        'selected_price': None,
                        'subpage': None,
                        'price_resolution': None,
                        'progress_tracker': MinimalTerminationTracker(
                            max_steps=self.max_steps,
                            exact_repeat_limit=int(
                                self.environment_config["termination"][
                                    "exact_repeat_limit"
                                ]
                            ),
                        ),
                        'actions': defaultdict(int)
                    }
                )
            elif 'keywords' in kwargs:
                # If search keywords are available, run a search
                html, url = self.search_results(session_id, **kwargs)
            elif 'clickable_name' in kwargs:
                clickable_name = kwargs['clickable_name'].lower()
                if clickable_name == END_BUTTON.lower():
                    # 购买:只返回原始事实(商品/规格/价格/goal),奖励由项目层计算
                    html, url, price, purchased_product, goal, options, target_product = self.done(session_id, **kwargs)
                    status['done'] = True
                    # 价格由 done() 单独返回,必须先注入再构造 purchase_light,
                    # 否则 get_purchase_info 读 purchase["price"] 必抛 KeyError。
                    purchased_product = dict(purchased_product)
                    purchased_product['price'] = price
                    status['purchase'] = self.get_purchase_info(purchased_product, options)
                    status['purchased_product'] = dict(purchased_product)
                    status['goal'] = goal
                    status['target_product'] = dict(target_product)
                    status['options'] = options
                    status['price'] = price
                    status['price_resolution'] = session.get("price_resolution")
                    status['termination_reason'] = "purchase"
                elif clickable_name == BACK_TO_SEARCH.lower():
                    # Return to the search form without erasing trajectory-level
                    # progress, explored queries, or opened products.
                    html, url = self.index(
                        session_id,
                        instruction_text=instruction_text,
                    )
                    session.update(
                        {
                            "page": None,
                            "asin": None,
                            "current_page_asins": [],
                            "options": {},
                            "selected_price": None,
                            "price_resolution": None,
                            "subpage": None,
                        }
                    )
                elif (clickable_name == NEXT_PAGE.lower() and
                      self.get_page_name(current_url) == 'search_results'):
                    # If "next page" clicked from search results, re-render with `page` enumerated
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] + 1,
                    )
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'search_results'):
                    # If "prev page" clicked from search results, re-render with `page` denumerated
                    html, url, status = self.receive(
                        session_id,
                        current_url,
                        keywords=session["keywords"],
                        page=session["page"] - 1,
                    )
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'item_sub_page'):
                    # If "prev page" clicked from sub page, return to corresponding item page
                    html, url = self.item_page(session_id, **kwargs)
                elif (clickable_name == PREV_PAGE.lower() and
                      self.get_page_name(current_url) == 'item_page'):
                    # If "prev page" clicked from item page, return to search results page
                    html, url = self.search_results(
                        session_id,
                        keywords=session["keywords"],
                        page=session["page"],
                        **kwargs
                    )
                elif clickable_name in [k.lower() for k in ACTION_TO_TEMPLATE]:
                    # Render item_sub_page if clickable is description, features, or reviews
                    html, url = self.item_sub_page(session_id, **kwargs)
                else:
                    # Otherwise, render current item page
                    html, url = self.item_page(session_id, **kwargs)
            return html, url, status

    def get_purchase_info(self, purchase, option):
        purchase_light = {"asin": purchase["asin"],
                          "category": purchase["category"],
                          "query": purchase["query"],
                          "name": purchase["title"],
                          "product_category": purchase["product_category"],
                          "instruction_text": purchase["instruction_text"],
                          "attributes": purchase["Attributes"],
                           "price": purchase["price"],
                           "options":option}
        return purchase_light


    def get_page_name(self, url):
        """Determine which page (i.e. item_page, search_results) the given URL is pointing at"""
        if url is None:
            return None
        page_names = [
            'search_results',
            'item_page',
            'item_sub_page',
            'done'
        ]
        for page_name in page_names:
            if page_name in url:
                return page_name
        return ''  # index page


class SimBrowser:
    """Session browser: renders pages through SimServer without any HTTP call."""

    def __init__(self, server):
        self.server = server
        self.current_url = None
        self.page_source = None
        self.session_id = None

    def get(self, url, session_id=None, session_int=None):
        """Set browser variables to corresponding link, page HTML for URL"""
        self.session_id = url.split('/')[-1] if session_id is None else session_id
        self.page_source, _, _ = \
            self.server.receive(self.session_id, self.current_url, session_int=session_int)
        self.current_url = url

    def click(self, clickable_name, text_to_clickable):
        """Wrapper for `receive` handler for performing click action on current page"""
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                clickable_name=clickable_name,
                text_to_clickable=text_to_clickable,
            )
        return status

    def search(self, keywords):
        """Wrapper for `receive` handler for performing search action on current page"""
        if isinstance(keywords, str):
            keywords = keywords.split(' ')
        self.page_source, self.current_url, status = \
            self.server.receive(
                self.session_id,
                current_url=self.current_url,
                keywords=keywords,
        )
        return status