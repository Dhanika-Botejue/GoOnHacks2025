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


def lookup(username, clan_name, debug=False):
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

    # Prefer structured search-result entries when present (more reliable)
    candidates = []
    for entry in soup.find_all(True, class_=lambda c: c and 'player_search_results__result_container' in c):
        # each entry contains an <a class="header" href="/player/...">display name</a>
        a = entry.find('a', class_='header')
        if not a or not a.has_attr('href'):
            continue
        href = a['href']
        display = a.get_text(separator=' ', strip=True)
        # look for a clan anchor within the entry
        clan_anchor = entry.find('a', href=lambda h: h and h.startswith('/clan/'))
        clan_text = None
        if clan_anchor:
            # clan text often contains name + tag, e.g. "L-O-A-D-I-N-G  #PJRC9CGR"
            clan_text = clan_anchor.get_text(separator=' ', strip=True)
        candidates.append((href, display, clan_text))

    # If we didn't find structured entries, fallback to previous broad scan
    if not candidates:
        for a in soup.find_all('a', href=True):
            href = a['href']
            if '/player/' in href:
                text = (a.get_text(separator=' ', strip=True) or '').strip()
                # try to find clan name in ancestor
                clan = None
                parent = a.parent
                for _ in range(3):
                    if parent is None:
                        break
                    txt = parent.get_text(separator=' ', strip=True)
                    if txt and clan_name.lower() in txt.lower():
                        clan = clan_name
                        break
                    parent = parent.parent
                candidates.append((href, text, clan))

    if debug:
        print(f"[debug] found {len(candidates)} candidate links")
        for i, (href, text, clan) in enumerate(candidates[:20]):
            print(f"[debug] candidate {i}: href={href} clanNearby={bool(clan)} text_snippet={repr(text[:120])}")

    # Prefer candidates that include a clan anchor in the search result and match exactly
    def _normalize(s):
        import re
        if not s:
            return ''
        return re.sub(r'[^a-z0-9]', '', s.lower())

    profile_url = None
    for href, display, clan_text in candidates:
        if clan_text:
            # clan_text often like "L-O-A-D-I-N-G  #PJRC9CGR" -> take part before tag
            clan_part = clan_text.split('#')[0].strip()
            if _normalize(clan_part) == _normalize(clan_name) or _normalize(clan_name) in _normalize(clan_part):
                profile_url = urljoin(base, href)
                if debug:
                    print(f"[debug] selected by clan_text match: {clan_text} -> {profile_url}")
                break

    # Fallback: pick first candidate whose visible text contains the clan name (less strict)
    if profile_url is None:
        for href, display, clan_text in candidates:
            if clan_name.lower() in (display or '').lower():
                profile_url = urljoin(base, href)
                if debug:
                    print(f"[debug] selected by display match: {display} -> {profile_url}")
                break

    # If still not found, load candidate profile pages and inspect the profile header for a clan link
    if profile_url is None:
        for href, display, clan_text in candidates:
            url = urljoin(base, href)
            try:
                r = session.get(url, timeout=session.request_timeout)
                if r.status_code != 200:
                    continue
                psoup = BeautifulSoup(r.text, 'html.parser')
                # look for a clan link in the profile header
                clan_anchor = psoup.find('a', href=lambda h: h and h.startswith('/clan/'))
                if clan_anchor:
                    page_clan = clan_anchor.get_text(separator=' ', strip=True)
                    page_clan_part = page_clan.split('#')[0].strip()
                    if _normalize(page_clan_part) == _normalize(clan_name) or _normalize(clan_name) in _normalize(page_clan_part):
                        profile_url = url
                        if debug:
                            print(f"[debug] matched clan on profile page: {url} -> {page_clan}")
                        break
                # as a looser fallback, inspect page text
                page_text = psoup.get_text(separator=' ', strip=True)
                if clan_name.lower() in page_text.lower():
                    profile_url = url
                    if debug:
                        print(f"[debug] matched clan in page text: {url}")
                    break
            except Exception:
                continue
            time.sleep(0.1)

    if profile_url is None:
        raise RuntimeError("Could not find a player profile matching the clan")

    if debug:
        print(f"[debug] selected profile_url: {profile_url}")

    # Load profile page and try to reach the "Decks" view.
    # Many player pages expose decks under a separate subpath (/player/<tag>/decks).
    # Try to follow an explicit "decks" link from the profile page first, otherwise
    # fall back to appending '/decks' to the profile URL.
    r = session.get(profile_url, timeout=session.request_timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Failed loading profile page: {r.status_code}")
    psoup = BeautifulSoup(r.text, 'html.parser')

    # look for an explicit decks link on the profile (safer than guessing URL)
    decks_url = None
    decks_link = psoup.find('a', href=lambda h: h and h.endswith('/decks') and '/player/' in h)
    if decks_link and decks_link.has_attr('href'):
        decks_url = urljoin(base, decks_link['href'])
        if debug:
            print(f"[debug] found decks link on profile page: {decks_url}")
    else:
        # fallback: append '/decks' to profile url (handle trailing slash)
        decks_url = profile_url.rstrip('/') + '/decks'
        if debug:
            print(f"[debug] falling back to decks_url: {decks_url}")

    # Fetch the decks page (if different from profile_url)
    try:
        r2 = session.get(decks_url, timeout=session.request_timeout)
        if r2.status_code == 200:
            psoup = BeautifulSoup(r2.text, 'html.parser')
            if debug:
                print(f"[debug] loaded decks page: {decks_url}")
        else:
            if debug:
                print(f"[debug] decks page request returned {r2.status_code}, using profile page HTML")
    except Exception:
        if debug:
            print("[debug] exception fetching decks page, using profile page HTML")
        # keep original psoup

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

    if debug:
        print(f"[debug] found {len(deck_containers)} deck candidate containers")
        for i, (div, imgs) in enumerate(deck_containers[:6]):
            sample_srcs = [img.get('alt') or img.get('title') or img.get('src') for img in imgs[:8]]
            print(f"[debug] container {i} has {len(imgs)} images; sample: {sample_srcs}")

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

    if debug:
        print(f"[debug] best_score={best_score}")
        if best_deck is not None:
            print("[debug] best_deck sample srcs:", [img.get('alt') or img.get('title') or img.get('src') for img in best_deck])

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
    if debug:
        print(f"[debug] extracted card names: {cards}")
    if len(cards) >= 8:
        return cards[:8]
    else:
        raise RuntimeError('Could not extract 8 card names from deck')

# usage example
cards = lookup('-Viper-', 'L-O-A-D-I-N-G', debug=False)
print(cards)

card1 = lookup('Dhanika', 'Dababy Lets Go', debug=False)
print(card1)

card2 = lookup('moose', 'L-O-A-D-I-N-G', debug=False)
print(card2)