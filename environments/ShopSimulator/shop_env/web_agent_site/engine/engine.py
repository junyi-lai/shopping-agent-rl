"""Page rendering and product loading for the ShopSimulator Environment v2.1.

The environment is a pure simulator: this module renders the HTML pages that the
agent observes, resolves products from the frozen corpus and exposes the
multi-field BM25 index. No scoring happens here.
"""
import json
import os
import re
from functools import lru_cache
from ast import literal_eval
from tqdm import tqdm
from flask import render_template_string
from rich import print
from web_agent_site.utils import BASE_DIR
from web_agent_site.engine.search import (
    MultiFieldBM25Searcher,
    sha256_file,
)

TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

SEARCH_RETURN_N = 150
PRODUCT_WINDOW = 20
TOP_K_ATTR = 10

END_BUTTON = 'Buy Now'
NEXT_PAGE = 'Next >'
PREV_PAGE = '< Prev'
BACK_TO_SEARCH = 'Back to Search'

ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def map_action_to_html(action, **kwargs):
    action_name, action_arg = parse_action(action)
    if action_name == 'start':
        path = os.path.join(TEMPLATE_DIR, 'search_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'search':
        path = os.path.join(TEMPLATE_DIR, 'results_page.html')
        html = render_template_string(
            read_html_template(path=path),
            session_id=kwargs['session_id'],
            products=kwargs['products'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            total=kwargs['total'],
            total_pages=kwargs.get('total_pages'),
            normalized_query=kwargs.get('normalized_query'),
            instruction_text=kwargs['instruction_text'],
        )
    elif action_name == 'click' and action_arg == END_BUTTON:
        path = os.path.join(TEMPLATE_DIR, 'done_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            asin=kwargs['asin'],
            options=kwargs['options'],
        )
    elif action_name == 'click' and action_arg in ACTION_TO_TEMPLATE:
        path = os.path.join(TEMPLATE_DIR, ACTION_TO_TEMPLATE[action_arg])
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text')
        )
    elif action_name == 'click':
        path = os.path.join(TEMPLATE_DIR, 'item_page.html')
        html = render_template_string(
            read_html_template(path),
            session_id=kwargs['session_id'],
            product_info=kwargs['product_info'],
            keywords=kwargs['keywords'],
            page=kwargs['page'],
            asin=kwargs['asin'],
            options=kwargs['options'],
            instruction_text=kwargs.get('instruction_text'),
            show_attrs=kwargs['show_attrs'],
            selected_option=kwargs.get('selected_option'),
            selected_price=kwargs.get('selected_price'),
        )
    else:
        raise ValueError('Action name not recognized.')
    return html


def read_html_template(path):
    with open(path, encoding="utf-8") as f:
        template = f.read()
    return template


def parse_action(action):
    """
    Parse action string to action name and its arguments.
    """
    pattern = re.compile(r'(.+)\[(.+)\]')
    m = re.match(pattern, action)
    if m is None:
        action_name = action
        action_arg = None
    else:
        action_name, action_arg = m.groups()
    return action_name, action_arg


def convert_web_app_string_to_var(name, string):
    if name == 'keywords':
        keywords = string
        if keywords.startswith('['):
            keywords = literal_eval(keywords)
        else:
            keywords = [keywords]
        var = keywords
    elif name == 'page':
        page = string
        page = int(page)
        var = page
    else:
        raise ValueError('Name of variable not recognized.')
    return var


def get_top_n_product_from_keywords(
        keywords,
        search_engine,
        product_item_dict,
    ):
    """Resolve one query to the frozen corpus through the BM25 index."""
    query = ' '.join(keywords)
    hits = search_engine.search(query, k=SEARCH_RETURN_N)
    return [
        product_item_dict[hit.asin]
        for hit in hits
        if hit.asin in product_item_dict
    ]


def get_product_per_page(top_n_products, page):
    if not isinstance(page, int) or page < 1:
        raise ValueError("page must be a positive integer")
    return top_n_products[(page - 1) * PRODUCT_WINDOW:page * PRODUCT_WINDOW]


def generate_product_prices(all_products):
    product_prices = dict()
    for product in all_products:
        asin = product['asin']
        pricing = product['pricing']
        if not pricing:
            price = 100.0
        elif len(pricing) >= 1:
            price = pricing[0]
        product_prices[asin] = price
    return product_prices


@lru_cache(maxsize=4)
def _product_file_sha256(filepath):
    return sha256_file(filepath)


def init_search_engine(product_filepath=None):
    """Open the frozen index and check it was built from the current products."""
    index_path = os.environ.get(
        "SHOP_SEARCH_INDEX",
        os.path.join(BASE_DIR, "../search_engine/products.sqlite3"),
    )
    expected_sha = (
        _product_file_sha256(os.path.abspath(product_filepath))
        if product_filepath
        else None
    )
    return MultiFieldBM25Searcher(
        index_path,
        expected_product_sha256=expected_sha,
    )


def clean_product_keys(products):
    for product in products:
        product.pop('product_information', None)
        product.pop('brand', None)
        product.pop('brand_url', None)
        product.pop('list_price', None)
        product.pop('availability_quantity', None)
        product.pop('availability_status', None)
        product.pop('total_reviews', None)
        product.pop('total_answered_questions', None)
        product.pop('seller_id', None)
        product.pop('seller_name', None)
        product.pop('fulfilled_by_amazon', None)
        product.pop('fast_track_message', None)
        product.pop('aplus_present', None)
        product.pop('small_description_old', None)
    return products


def load_products(filepath):
    """Load the frozen product corpus and index it by ASIN."""
    with open(filepath, encoding="utf-8") as f:
        products = json.load(f)

    print('Products loaded.')
    products = clean_product_keys(products)

    asins = set()
    all_products = []
    for i, p in tqdm(enumerate(products), total=len(products)):
        asin = p['asin']
        if asin == 'nan' or len(asin) > 20:
            continue
        if asin in asins:
            continue
        asins.add(asin)

        products[i]['shop_name'] = p['shop_name']
        products[i]['category'] = p['category']
        products[i]['query'] = p.get('query', '')
        products[i]['product_category'] = p.get('product_category', '')
        products[i]['Title'] = p['title']
        products[i]['Description'] = p.get('full_description', '')
        # The frozen corpus carries no review data; the Reviews sub-page has to
        # render an empty list instead of an undefined template variable.
        products[i]['Reviews'] = []
        products[i]['Rating'] = 'N.A.'
        products[i]['BulletPoints'] = p.get('small_description', '') \
            if isinstance(p.get('small_description', ''), list) else [p.get('small_description', '')]

        pricing = p.get('pricing')
        if pricing is None or not pricing:
            pricing = [100.0]
            price_tag = '100.0'
        else:
            if len(pricing) == 1:
                price_tag = f"{pricing[0]}"
            else:
                price_tag = f"{pricing[0]} to {pricing[1]}"
                pricing = pricing[:2]
        products[i]['pricing'] = pricing
        products[i]['Price'] = price_tag

        options = dict()
        customization_options = p.get('customization_options', '')
        option_to_image = dict()
        option_to_price = dict()
        if customization_options:
            for option_name, option_contents in customization_options.items():
                if option_contents is None:
                    continue
                option_name = option_name.lower()

                option_values = []
                for option_content in option_contents:
                    option_value = option_content['value'].strip().replace('/', ' | ').lower()
                    option_image = option_content.get('image', None)

                    option_values.append(option_value)
                    option_to_image[option_value] = option_image
                    option_to_price[option_value] = option_content.get('price', None)
                options[option_name] = option_values
        products[i]['options'] = options
        products[i]['option_to_image'] = option_to_image
        products[i]['option_to_price'] = option_to_price

        products[i]['Attributes'] = products[i]['attribute']
        products[i]['instruction_text'] = p['instructions'][0]['instruction']

        products[i]['MainImage'] = p['images'][0]
        products[i]['query'] = p['query'].lower().strip()

        all_products.append(products[i])

    product_item_dict = {p['asin']: p for p in all_products}
    product_prices = generate_product_prices(all_products)
    return all_products, product_item_dict, product_prices
