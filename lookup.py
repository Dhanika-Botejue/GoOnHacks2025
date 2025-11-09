import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin


def _requests_session(retries=3, backoff=0.6, timeout=15):
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; lookup-bot/1.0)"})
    s.request_timeout = timeout
    return s


def lookup(username, clan_name):
    """Search royaleapi for a player by username, find the profile that matches clan_name,
    then parse the player's decks and return the 8 card names from the most used deck.

    Notes/assumptions:
    - This scrapes HTML from https://royaleapi.com. Site structure may change and break parsing.
    - We try several fallbacks: extracting card names from image alt/title attributes or image filenames.
    - Returns list of 8 strings (card names) or raises RuntimeError if not found.
    """
    base = "https://royaleapi.com"
    session = _requests_session()

    search_url = f"{base}/player/search/results?q={requests.utils.requote_uri(username)}&exact_match=on"
    resp = session.get(search_url, timeout=session.request_timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Search page request failed: {resp.status_code}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Find candidate profile links. Look for hrefs that contain '/player/'
    candidates = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        if '/player/' in href:
            # try to read clan text nearby
            text = (a.get_text(separator=' ', strip=True) or '').strip()
            # try to find clan name in ancestor
            clan = None
            parent = a.parent
            for _ in range(3):
                if parent is None:
                    break
                # look for an element that contains 'clan' or similar
                txt = parent.get_text(separator=' ', strip=True)
                if txt and clan_name.lower() in txt.lower():
                    clan = clan_name
                    break
                parent = parent.parent
            candidates.append((href, text, clan))

    # Prefer candidates that matched clan in nearby text
    profile_url = None
    for href, text, clan in candidates:
        if clan:
            profile_url = urljoin(base, href)
            break

    # Fallback: pick first candidate whose visible text contains the clan name
    if profile_url is None:
        for href, text, clan in candidates:
            if clan_name.lower() in (text or '').lower():
                profile_url = urljoin(base, href)
                break

    # If still not found, try to load each candidate page and inspect clan there
    if profile_url is None:
        for href, text, clan in candidates:
            url = urljoin(base, href)
            try:
                r = session.get(url, timeout=session.request_timeout)
                if r.status_code != 200:
                    continue
                psoup = BeautifulSoup(r.text, 'html.parser')
                page_text = psoup.get_text(separator=' ', strip=True)
                if clan_name.lower() in page_text.lower():
                    profile_url = url
                    break
            except Exception:
                continue
            time.sleep(0.1)

    if profile_url is None:
        raise RuntimeError("Could not find a player profile matching the clan")

    # Load profile page and find decks
    r = session.get(profile_url, timeout=session.request_timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Failed loading profile page: {r.status_code}")
    psoup = BeautifulSoup(r.text, 'html.parser')

    # Search for deck containers: try common patterns
    deck_containers = []
    # common: elements with class containing 'deck' and containing several images
    for div in psoup.find_all(True, class_=lambda c: c and 'deck' in c.lower()):
        imgs = div.find_all('img')
        if len(imgs) >= 8:
            deck_containers.append((div, imgs))

    # fallback: find any group with 8 images
    if not deck_containers:
        all_divs = psoup.find_all(True)
        for div in all_divs:
            imgs = div.find_all('img')
            if len(imgs) >= 8:
                deck_containers.append((div, imgs))
                if len(deck_containers) >= 6:
                    break

    if not deck_containers:
        raise RuntimeError('No deck containers found on profile page')

    # For each deck container, try to find usage metric nearby
    best_deck = None
    best_score = -1.0
    for div, imgs in deck_containers:
        # Look for a usage percent in text
        text = div.get_text(separator=' ', strip=True)
        score = 0.0
        import re
        m = re.search(r"(\d{1,3}(?:\.\d+)?%)", text)
        if m:
            try:
                score = float(m.group(1).strip('%'))
            except Exception:
                score = 0.0
        # prefer the one with highest score or first if none
        if score > best_score:
            best_score = score
            best_deck = imgs[:8]

    # If no usage metrics found, pick first deck's first 8 images
    if best_deck is None:
        best_deck = deck_containers[0][1][:8]

    # Extract card names from images (alt/title or filename)
    cards = []
    for img in best_deck:
        name = None
        if img.has_attr('alt') and img['alt'].strip():
            name = img['alt'].strip()
        elif img.has_attr('title') and img['title'].strip():
            name = img['title'].strip()
        else:
            src = img.get('src', '')
            # try basename
            from os.path import basename
            name = basename(src).split('.')[0].replace('-', ' ').replace('_', ' ').strip()
        cards.append(name)

    # Ensure we return exactly 8 items
    if len(cards) >= 8:
        return cards[:8]
    else:
        raise RuntimeError('Could not extract 8 card names from deck')

# usage example
#cards = lookup('Dhanika', 'Dababy Lets Go')
#rint(cards)