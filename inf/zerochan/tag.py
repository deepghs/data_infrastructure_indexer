import re
from urllib.parse import quote_plus, urljoin, unquote_plus

import httpx
import requests.exceptions
from ditk import logging
from hbutils.system import urlsplit
from markdownify import MarkdownConverter, chomp
from pyquery import PyQuery as pq
from waifuc.utils import srequest

from .base import _ROOT, get_session


class ImageAndLinkBlockConverter(MarkdownConverter):
    def __init__(self, current_page_url: str = None, **options):
        MarkdownConverter.__init__(self, **options)
        self.current_page_url = current_page_url

    def convert_img(self, el, text, convert_as_inline):
        alt = el.attrs.get('alt', None) or ''
        src = el.attrs.get('src', None) or ''
        if self.current_page_url:
            src = urljoin(str(self.current_page_url), src)
        title = el.attrs.get('title', None) or ''
        title_part = ' "%s"' % title.replace('"', r'\"') if title else ''
        if (convert_as_inline
                and el.parent.name not in self.options['keep_inline_images_in']):
            return alt

        return '![%s](%s%s)' % (alt, src, title_part)

    def convert_a(self, el, text, convert_as_inline):
        prefix, suffix, text = chomp(text)
        if not text:
            return ''
        href = el.get('href')
        if self.current_page_url:
            href = urljoin(str(self.current_page_url), href)
        title = el.get('title')
        # For the replacement see #29: text nodes underscores are escaped
        if (self.options['autolinks']
                and text.replace(r'\_', '_') == href
                and not title
                and not self.options['default_title']):
            # Shortcut syntax
            return '<%s>' % href
        if self.options['default_title'] and not title:
            title = href
        title_part = ' "%s"' % title.replace('"', r'\"') if title else ''
        return '%s[%s](%s%s)%s' % (prefix, text, href, title_part, suffix) if href else text


def _get_tag_info(tag, session=None):
    session = session or get_session()
    try:
        resp = srequest(session, 'GET', f'{_ROOT}/{quote_plus(tag)}')
    except (httpx.HTTPError, requests.exceptions.RequestException) as err:
        if err.response.status_code in {410, 404, 500}:
            return None
        raise
    if not resp.text or not resp.text.strip():
        return None
    page = pq(resp.text)

    if page('#content h1 #copy-link'):
        name = page('#content h1 #copy-link').attr('data-value')
    else:
        name = page('#content h1 span').text().strip()

    info_wrapper = page('section#tag-info-wrapper')
    if not info_wrapper.html() or not info_wrapper.html().strip():
        return None

    raw_cls = info_wrapper('h2').attr('class').strip()
    categories = list(filter(bool, re.split(r'\s+', raw_cls)))
    if len(categories) == 1 and ',' not in categories[0]:
        category = categories[0]
    else:
        logging.warning(f'Unknown category type - {raw_cls!r}.')
        category = 'unknown'
        # raise ValueError(f'Unable to determine category of tag {tag!r}, h2_categories: {categories!r}.')

    tag_cover_a = page('#tag-cover-wrapper a')
    if tag_cover_a:
        cover_image_id = int(urlsplit(urljoin(str(resp.url), tag_cover_a.attr('href'))).path_segments[-1])
    else:
        cover_image_id = None

    tag_cover_img = page('#tag-cover')
    if tag_cover_img:
        cover_image_url = urljoin(str(resp.url), tag_cover_img.attr('src'))
    else:
        cover_image_url = None

    socials = []
    for social_item in page('#socials > a').items():
        try:
            social_url = urljoin(str(resp.url), social_item.attr('href'))
        except ValueError as err:
            logging.warning(f'Social url failed - {err!r}, it will be ignored.')
            continue
        if social_item('s').attr('class'):
            s_classes = list(filter(bool, re.split(r'\s+', social_item('s').attr('class'))))
            social_type = s_classes[-1]
            socials.append({
                'type': social_type,
                'url': social_url,
            })

    md_conv = ImageAndLinkBlockConverter(current_page_url=str(resp.url))
    description_md = md_conv.convert(page('#description').html() or '').strip()

    langs = {}
    aliases = []
    for alias_item in page('#aliases ul li').items():
        alias_type = alias_item('i').text().strip()
        if not alias_item.attr('class'):
            logging.info(f'Unknown alias class - {alias_item.attr("class")!r}.')
            continue
        is_language = 'language' in alias_item.attr('class').split(' ')
        alias_name = alias_item('span').attr('data-value')
        aliases.append({
            'name': alias_name,
            'type': alias_type,
            'is_language': is_language,
        })
        if is_language:
            if alias_type not in langs:
                langs[alias_type] = alias_name

    tags = []
    for tag_item in page('#tags li').items():
        if 'more' in (tag_item.attr('class') or '').split(' '):
            continue
        tags.append(unquote_plus(urlsplit(urljoin(str(resp.url), tag_item('a').attr('href'))).path_segments[-1]))

    outfits = []
    for outfit_item in page('#outfits li').items():
        if 'more' in (outfit_item.attr('class') or '').split(' '):
            continue
        outfits.append(unquote_plus(urlsplit(urljoin(str(resp.url), outfit_item('a').attr('href'))).path_segments[-1]))

    # page_no = 1
    # children_all = True
    # children_tags = []
    # while True:
    #     try:
    #         resp = srequest(
    #             session, 'GET', f'{_ROOT}/{quote_plus(tag)}',
    #             params={'children': '', 'p': str(page_no)},
    #         )
    #     except (httpx.HTTPStatusError, requests.exceptions.RequestException) as err:
    #         if err.response.status_code == 403:
    #             logging.warning(f'Max page limit for {tag!r}, skipped')
    #             children_all = False
    #             break
    #         else:
    #             raise
    #     resp.raise_for_status()
    #     page = pq(resp.text)
    #     has_new_add = False
    #     for child_item in page('ul#children-grid li').items():
    #         tag_name = unquote_plus(
    #             urlsplit(urljoin(str(resp.url), child_item('p b a').attr('href'))).path_segments[-1])
    #         p_tags = set(filter(bool, child_item('p').attr('class').split(' ')))
    #         ps_tags = set(filter(bool, child_item('p s').attr('class').split(' ')))
    #         p_ps_tags = list(p_tags & ps_tags)
    #         if len(p_ps_tags) == 1:
    #             children_tags.append({
    #                 'name': tag_name,
    #                 'category': p_ps_tags[0]
    #             })
    #             has_new_add = True
    #         else:
    #             raise ValueError(f'Unable to determine child tag category, '
    #                              f'p_tags: {p_tags!r}, ps_tags: {ps_tags!r}.')
    #
    #     if not has_new_add:
    #         break
    #     page_no += 1

    return {
        'name': name,
        'category': category,
        'raw_category': raw_cls,
        'cover': {
            'image_id': cover_image_id,
            'image_url': cover_image_url,
        },
        'socials': socials,
        'description_md': description_md,
        'aliases': aliases,
        'langs': langs,
        'related_tags': tags,
        'outfit_tags': outfits,
        # 'children_tags': children_tags,
        # 'children_tags_all': children_all,
    }


if __name__ == '__main__':
    logging.try_init_root(logging.INFO)
    print(_get_tag_info('Kafei'))
