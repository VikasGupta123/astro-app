"""
astro_engine.py
----------------
Calculates a real Vedic/Jyotish birth chart (Kundli) using the Swiss
Ephemeris library (pyswisseph): sidereal Rashi (signs) with the Lahiri
ayanamsa (the standard used by AstroSage, drikpanchang, and Indian
government calendars), Nakshatra, Rahu/Ketu, whole-sign houses (Lagna),
each planet's house (Bhava) placement, a computed Vimshottari Mahadasha/
Antardasha timeline, today's real planetary transits (Gochar) including
Sade Sati detection, a South Indian-style visual chart grid, and Ashtakoot
Kundli Milan (marriage compatibility matching) between two charts, including
Mangal Dosha detection.

You generally do not need to edit this file. app.py calls into it.
"""

import datetime
import swisseph as swe
from timezonefinder import TimezoneFinder
import pytz

# Lahiri ayanamsa - the official Indian government standard and the default
# used by mainstream Vedic astrology software/apps.
swe.set_sid_mode(swe.SIDM_LAHIRI)

RASHI_NAMES = [
    "Mesha", "Vrishabha", "Mithuna", "Karka", "Simha", "Kanya",
    "Tula", "Vrishchik", "Dhanu", "Makar", "Kumbha", "Meena",
]
RASHI_ENGLISH = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "Purva Ashadha", "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "Purva Bhadrapada", "Uttara Bhadrapada", "Revati",
]

HOUSE_MEANINGS = {
    1: "Self, body, personality, how you come across (Tanu Bhava)",
    2: "Wealth, family, speech, food/values (Dhana Bhava)",
    3: "Courage, siblings, communication, short journeys (Sahaja Bhava)",
    4: "Home, mother, emotional roots, comfort (Sukha Bhava)",
    5: "Children, intelligence, romance, creativity (Putra Bhava)",
    6: "Health, obstacles, debts, daily work/service (Ripu/Roga Bhava)",
    7: "Marriage, partnerships, business relationships (Kalatra Bhava)",
    8: "Transformation, longevity, shared resources, the unknown (Ayu Bhava)",
    9: "Fortune, father, dharma, higher learning, luck (Bhagya Bhava)",
    10: "Career, status, public reputation, life direction (Karma Bhava)",
    11: "Gains, income, friendships, hopes/goals (Labha Bhava)",
    12: "Losses, expenses, foreign lands, spirituality, letting go (Vyaya Bhava)",
}

DASHA_ORDER = ["Ketu", "Venus", "Sun", "Moon", "Mars", "Rahu", "Jupiter", "Saturn", "Mercury"]
DASHA_YEARS = {
    "Ketu": 7, "Venus": 20, "Sun": 6, "Moon": 10, "Mars": 7,
    "Rahu": 18, "Jupiter": 16, "Saturn": 19, "Mercury": 17,
}
DASHA_YEAR_DAYS = 365.25

PLANETS = [
    ("Sun", swe.SUN),
    ("Moon", swe.MOON),
    ("Mercury", swe.MERCURY),
    ("Venus", swe.VENUS),
    ("Mars", swe.MARS),
    ("Jupiter", swe.JUPITER),
    ("Saturn", swe.SATURN),
]

SANSKRIT_PLANET_NAMES = {
    "Sun": "Surya", "Moon": "Chandra", "Mercury": "Budh", "Venus": "Shukra",
    "Mars": "Mangal", "Jupiter": "Guru (Brihaspati)", "Saturn": "Shani",
    "Rahu": "Rahu", "Ketu": "Ketu",
}

PLANET_ABBR = {
    "Sun": "Su", "Moon": "Mo", "Mercury": "Me", "Venus": "Ve", "Mars": "Ma",
    "Jupiter": "Ju", "Saturn": "Sa", "Rahu": "Ra", "Ketu": "Ke",
}

PLANET_SYMBOLS = {
    "Sun": "☉", "Moon": "☽", "Mercury": "☿", "Venus": "♀", "Mars": "♂",
    "Jupiter": "♃", "Saturn": "♄", "Rahu": "☊", "Ketu": "☋",
}

SOUTH_INDIAN_LAYOUT = [
    [11, 0, 1, 2],
    [10, None, None, 3],
    [9, None, None, 4],
    [8, 7, 6, 5],
]

_tf = TimezoneFinder()
_NAK_SPAN = 360.0 / 27.0


def sign_index(longitude: float) -> int:
    return int(longitude // 30) % 12


def degree_in_sign(longitude: float) -> float:
    return longitude % 30


def nakshatra_index(longitude: float) -> int:
    return int(longitude // _NAK_SPAN) % 27


def nakshatra_pada(longitude: float) -> int:
    pos_in_nak = longitude % _NAK_SPAN
    pada_span = _NAK_SPAN / 4
    return int(pos_in_nak // pada_span) + 1


def house_from_index(sign_idx: int, reference_idx: int) -> int:
    return ((sign_idx - reference_idx) % 12) + 1


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def find_timezone(lat: float, lon: float) -> str:
    tz_name = _tf.timezone_at(lat=lat, lng=lon)
    if not tz_name:
        raise ValueError(
            "Could not determine timezone for that location. "
            "Try a nearby larger city, or enter latitude/longitude manually."
        )
    return tz_name


def _geocode_open_meteo(place_name: str):
    import requests

    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place_name, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    results = data.get("results")
    if not results:
        return None

    r = results[0]
    label_parts = [r.get("name"), r.get("admin1"), r.get("country")]
    label = ", ".join(p for p in label_parts if p)
    return r["latitude"], r["longitude"], label


def _geocode_nominatim(place_name: str):
    try:
        from geopy.geocoders import Nominatim

        geolocator = Nominatim(user_agent="ai_astrology_chat_app_prototype", timeout=10)
        location = geolocator.geocode(place_name)
    except Exception:
        return None

    if location is None:
        return None
    return location.latitude, location.longitude, location.address


def geocode_place(place_name: str):
    for geocoder in (_geocode_open_meteo, _geocode_nominatim):
        result = geocoder(place_name)
        if result is not None:
            return result

    raise ValueError(
        f"Could not find a location matching '{place_name}' using either geocoding "
        "service. Try adding the country (e.g. 'Kushinagar, India'), or check "
        "'use manual coordinates' below and enter latitude/longitude directly."
    )


def search_places(query: str):
    query = (query or "").strip()
    if len(query) < 2:
        return []

    results = []
    try:
        import requests

        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 8, "language": "en", "format": "json"},
            timeout=8,
        )
        resp.raise_for_status()
        for r in resp.json().get("results") or []:
            label_parts = [r.get("name"), r.get("admin2"), r.get("admin1"), r.get("country")]
            label = ", ".join(p for p in label_parts if p)
            results.append((label, (r["latitude"], r["longitude"], label)))
    except Exception:
        pass

    if not results:
        try:
            from geopy.geocoders import Nominatim

            geolocator = Nominatim(user_agent="ai_astrology_chat_app_prototype", timeout=8)
            for loc in geolocator.geocode(query, exactly_one=False, limit=8) or []:
                results.append((loc.address, (loc.latitude, loc.longitude, loc.address)))
        except Exception:
            pass

    return results


def _rashi_info(longitude: float) -> dict:
    idx = sign_index(longitude)
    nidx = nakshatra_index(longitude)
    return {
        "rashi": RASHI_NAMES[idx],
        "rashi_english": RASHI_ENGLISH[idx],
        "degree": round(degree_in_sign(longitude), 2),
        "nakshatra": NAKSHATRAS[nidx],
        "pada": nakshatra_pada(longitude),
        "longitude": longitude,
    }


def calculate_chart(birth_date: datetime.date, birth_time: datetime.time,
                     lat: float, lon: float, tz_name: str) -> dict:
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    tz = pytz.timezone(tz_name)
    local_dt = tz.localize(datetime.datetime.combine(birth_date, birth_time))
    utc_dt = local_dt.astimezone(pytz.utc)

    jd_ut = swe.julday(
        utc_dt.year, utc_dt.month, utc_dt.day,
        utc_dt.hour + utc_dt.minute / 60 + utc_dt.second / 3600,
    )

    sidereal_flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

    result = {
        "planets": {},
        "timezone": tz_name,
        "utc_datetime": utc_dt.strftime("%Y-%m-%d %H:%M UTC"),
        "local_datetime": local_dt.strftime("%Y-%m-%d %H:%M %Z"),
        "ayanamsa": round(swe.get_ayanamsa_ut(jd_ut), 4),
    }

    cusps, ascmc = swe.houses_ex(jd_ut, lat, lon, b"W", sidereal_flag)
    asc_long = ascmc[0]
    asc_idx = sign_index(asc_long)
    result["ascendant"] = _rashi_info(asc_long)

    for name, body_id in PLANETS:
        pos, _flag = swe.calc_ut(jd_ut, body_id, sidereal_flag)
        longitude = pos[0]
        speed = pos[3]
        info = _rashi_info(longitude)
        info["retrograde"] = speed < 0
        info["house"] = house_from_index(sign_index(longitude), asc_idx)
        result["planets"][name] = info

    rahu_pos, _flag = swe.calc_ut(jd_ut, swe.MEAN_NODE, sidereal_flag)
    rahu_long = rahu_pos[0]
    ketu_long = (rahu_long + 180) % 360
    for pname, plong in (("Rahu", rahu_long), ("Ketu", ketu_long)):
        info = _rashi_info(plong)
        info["retrograde"] = True
        info["house"] = house_from_index(sign_index(plong), asc_idx)
        result["planets"][pname] = info

    result["houses"] = {
        h: RASHI_NAMES[(asc_idx + h - 1) % 12] for h in range(1, 13)
    }

    moon_longitude = result["planets"]["Moon"]["longitude"]
    result["dasha"] = compute_vimshottari_dasha(moon_longitude, local_dt)

    return result


def build_south_indian_grid(chart: dict) -> list:
    rashi_to_house = {rashi: house for house, rashi in chart["houses"].items()}
    planets_by_rashi = {}
    for pname, pdata in chart["planets"].items():
        planets_by_rashi.setdefault(pdata["rashi"], []).append({
            "name": pname,
            "abbr": PLANET_ABBR.get(pname, pname[:2]),
            "retrograde": pdata["retrograde"],
        })
    lagna_rashi = chart["ascendant"]["rashi"]

    grid = []
    for row in SOUTH_INDIAN_LAYOUT:
        grid_row = []
        for idx in row:
            if idx is None:
                grid_row.append(None)
                continue
            rashi = RASHI_NAMES[idx]
            grid_row.append({
                "rashi": rashi,
                "rashi_english": RASHI_ENGLISH[idx],
                "house": rashi_to_house.get(rashi),
                "is_lagna": rashi == lagna_rashi,
                "planets": planets_by_rashi.get(rashi, []),
            })
        grid.append(grid_row)
    return grid


def compute_vimshottari_dasha(moon_longitude: float, birth_dt: datetime.datetime) -> list:
    nidx = nakshatra_index(moon_longitude)
    pos_in_nak = moon_longitude % _NAK_SPAN
    elapsed_fraction = pos_in_nak / _NAK_SPAN

    order_start = nidx % 9
    lords = [DASHA_ORDER[(order_start + i) % 9] for i in range(9)]

    years_list = [DASHA_YEARS[lords[0]] * (1 - elapsed_fraction)] + [
        DASHA_YEARS[lord] for lord in lords[1:]
    ]

    mahadashas = []
    cursor = birth_dt
    for lord, years in zip(lords, years_list):
        duration = datetime.timedelta(days=years * DASHA_YEAR_DAYS)
        start, end = cursor, cursor + duration
        mahadashas.append({
            "lord": lord,
            "start": start,
            "end": end,
            "antardashas": _compute_antardashas(lord, start, years),
        })
        cursor = end

    return mahadashas


def _compute_antardashas(maha_lord: str, maha_start: datetime.datetime, maha_years: float) -> list:
    start_idx = DASHA_ORDER.index(maha_lord)
    cursor = maha_start
    antardashas = []
    for i in range(9):
        sub_lord = DASHA_ORDER[(start_idx + i) % 9]
        sub_years = maha_years * DASHA_YEARS[sub_lord] / 120
        duration = datetime.timedelta(days=sub_years * DASHA_YEAR_DAYS)
        start, end = cursor, cursor + duration
        antardashas.append({"lord": sub_lord, "start": start, "end": end})
        cursor = end
    return antardashas


def find_current_dasha(dasha_list: list, as_of: datetime.datetime = None):
    if not dasha_list:
        return None, None
    if as_of is None:
        as_of = datetime.datetime.now(dasha_list[0]["start"].tzinfo)
    for maha in dasha_list:
        if maha["start"] <= as_of < maha["end"]:
            for antar in maha["antardashas"]:
                if antar["start"] <= as_of < antar["end"]:
                    return maha, antar
            return maha, None
    return None, None


def compute_transits(as_of: datetime.datetime = None) -> tuple:
    swe.set_sid_mode(swe.SIDM_LAHIRI)
    if as_of is None:
        as_of = datetime.datetime.now(pytz.utc)

    jd_ut = swe.julday(
        as_of.year, as_of.month, as_of.day,
        as_of.hour + as_of.minute / 60 + as_of.second / 3600,
    )
    sidereal_flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

    transits = {}
    for name, body_id in PLANETS:
        pos, _flag = swe.calc_ut(jd_ut, body_id, sidereal_flag)
        longitude = pos[0]
        speed = pos[3]
        info = _rashi_info(longitude)
        info["retrograde"] = speed < 0
        transits[name] = info

    rahu_pos, _flag = swe.calc_ut(jd_ut, swe.MEAN_NODE, sidereal_flag)
    rahu_long = rahu_pos[0]
    ketu_long = (rahu_long + 180) % 360
    for pname, plong in (("Rahu", rahu_long), ("Ketu", ketu_long)):
        info = _rashi_info(plong)
        info["retrograde"] = True
        transits[pname] = info

    return transits, as_of


def compute_gochar(natal_moon_longitude: float, transits: dict) -> dict:
    natal_moon_idx = sign_index(natal_moon_longitude)
    gochar = {}
    for planet, data in transits.items():
        t_idx = sign_index(data["longitude"])
        gochar[planet] = {**data, "house_from_moon": house_from_index(t_idx, natal_moon_idx)}
    return gochar


def check_sade_sati(gochar: dict) -> dict:
    house = gochar["Saturn"]["house_from_moon"]
    if house == 12:
        return {"active": True, "phase": "Rising phase (Arohi) - the first ~2.5 years"}
    if house == 1:
        return {"active": True, "phase": "Peak phase (Madhya) - typically the most intense ~2.5 years"}
    if house == 2:
        return {"active": True, "phase": "Setting phase (Uttarardh) - the final ~2.5 years"}
    return {"active": False, "phase": None}


def chart_to_prompt_text(chart: dict, name: str = "the user") -> str:
    lines = [
        f"Vedic birth chart (Kundli) for {name}, sidereal/Lahiri ayanamsa "
        f"({chart['ayanamsa']}°):",
        f"- Lagna (Ascendant): {chart['ascendant']['rashi']} "
        f"({chart['ascendant']['rashi_english']}), "
        f"{chart['ascendant']['degree']}°, "
        f"Nakshatra {chart['ascendant']['nakshatra']} pada {chart['ascendant']['pada']}",
    ]
    for planet, data in chart["planets"].items():
        sanskrit = SANSKRIT_PLANET_NAMES.get(planet, planet)
        retro = " (Vakri/retrograde)" if data["retrograde"] else ""
        house_meaning = HOUSE_MEANINGS.get(data["house"], "")
        lines.append(
            f"- {sanskrit} ({planet}): {data['rashi']} ({data['rashi_english']}), "
            f"{data['degree']}°{retro}, Nakshatra {data['nakshatra']} pada {data['pada']}, "
            f"{ordinal(data['house'])} house ({house_meaning})"
        )

    lines.append("Houses (whole-sign, from Lagna):")
    for h, rashi in chart["houses"].items():
        lines.append(f"  {ordinal(h)} house [{HOUSE_MEANINGS[h]}]: {rashi}")

    dasha_list = chart.get("dasha", [])
    if dasha_list:
        maha, antar = find_current_dasha(dasha_list)
        if maha:
            lines.append(
                f"Current Mahadasha: {maha['lord']} "
                f"({maha['start'].strftime('%Y-%m-%d')} to {maha['end'].strftime('%Y-%m-%d')})"
            )
            if antar:
                lines.append(
                    f"Current Antardasha: {antar['lord']} within {maha['lord']} Mahadasha "
                    f"({antar['start'].strftime('%Y-%m-%d')} to {antar['end'].strftime('%Y-%m-%d')})"
                )
        lines.append("Full Mahadasha timeline from birth:")
        for m in dasha_list:
            lines.append(
                f"  {m['lord']}: {m['start'].strftime('%Y-%m-%d')} to {m['end'].strftime('%Y-%m-%d')}"
            )

    lines.append(f"Birth time (local): {chart['local_datetime']}")
    return "\n".join(lines)


def gochar_to_prompt_text(chart: dict) -> str:
    transits, as_of = compute_transits()
    natal_moon_longitude = chart["planets"]["Moon"]["longitude"]
    gochar = compute_gochar(natal_moon_longitude, transits)
    sade_sati = check_sade_sati(gochar)

    lines = [
        f"Today's Gochar (live planetary transits) as of {as_of.strftime('%Y-%m-%d')}, "
        f"counted from natal Chandra Rashi/Moon sign "
        f"({chart['planets']['Moon']['rashi']}):",
    ]
    for planet, data in gochar.items():
        sanskrit = SANSKRIT_PLANET_NAMES.get(planet, planet)
        retro = " (Vakri/retrograde)" if data["retrograde"] else ""
        lines.append(
            f"- {sanskrit} ({planet}) transiting {data['rashi']} "
            f"({data['rashi_english']}){retro}, {ordinal(data['house_from_moon'])} house from Moon"
        )
    if sade_sati["active"]:
        lines.append(f"Sade Sati is CURRENTLY ACTIVE - {sade_sati['phase']}.")
    else:
        lines.append("Sade Sati is not currently active for this person.")

    return "\n".join(lines)


# =============================================================================
# Ashtakoot Kundli Milan (marriage compatibility matching) + Mangal Dosha
# =============================================================================

RASHI_LORDS = [
    "Mars", "Venus", "Mercury", "Moon", "Sun", "Mercury",
    "Venus", "Mars", "Jupiter", "Saturn", "Saturn", "Jupiter",
]

PLANET_FRIENDSHIP = {
    "Sun":     {"Moon": "friend", "Mars": "friend", "Jupiter": "friend", "Mercury": "neutral", "Venus": "enemy", "Saturn": "enemy"},
    "Moon":    {"Sun": "friend", "Mercury": "friend", "Mars": "neutral", "Jupiter": "neutral", "Venus": "neutral", "Saturn": "neutral"},
    "Mars":    {"Sun": "friend", "Moon": "friend", "Jupiter": "friend", "Venus": "neutral", "Saturn": "neutral", "Mercury": "enemy"},
    "Mercury": {"Sun": "friend", "Venus": "friend", "Mars": "neutral", "Jupiter": "neutral", "Saturn": "neutral", "Moon": "enemy"},
    "Jupiter": {"Sun": "friend", "Moon": "friend", "Mars": "friend", "Saturn": "neutral", "Mercury": "enemy", "Venus": "enemy"},
    "Venus":   {"Mercury": "friend", "Saturn": "friend", "Mars": "neutral", "Jupiter": "neutral", "Sun": "enemy", "Moon": "enemy"},
    "Saturn":  {"Mercury": "friend", "Venus": "friend", "Jupiter": "neutral", "Sun": "enemy", "Moon": "enemy", "Mars": "enemy"},
}

RASHI_VARNA = {
    "Karka": "Brahmin", "Vrishchik": "Brahmin", "Meena": "Brahmin",
    "Mesha": "Kshatriya", "Simha": "Kshatriya", "Dhanu": "Kshatriya",
    "Vrishabha": "Vaishya", "Kanya": "Vaishya", "Makar": "Vaishya",
    "Mithuna": "Shudra", "Tula": "Shudra", "Kumbha": "Shudra",
}
VARNA_RANK = {"Brahmin": 4, "Kshatriya": 3, "Vaishya": 2, "Shudra": 1}

VASHYA_GROUP_WHOLE = {
    "Mesha": "Chatushpada", "Vrishabha": "Chatushpada", "Simha": "Chatushpada",
    "Mithuna": "Manav", "Kanya": "Manav", "Tula": "Manav", "Kumbha": "Manav",
    "Karka": "Jalachar", "Meena": "Jalachar",
    "Vrishchik": "Keeta",
}
VASHYA_COMPATIBILITY = {
    frozenset(["Chatushpada"]): 2, frozenset(["Manav"]): 2,
    frozenset(["Jalachar"]): 2, frozenset(["Keeta"]): 2,
    frozenset(["Manav", "Chatushpada"]): 1,
    frozenset(["Manav", "Jalachar"]): 1,
    frozenset(["Chatushpada", "Jalachar"]): 1,
    frozenset(["Manav", "Keeta"]): 0,
    frozenset(["Chatushpada", "Keeta"]): 0,
    frozenset(["Jalachar", "Keeta"]): 0,
}

TARA_GOOD = {1, 2, 4, 6, 8, 9}
TARA_BAD = {3, 5, 7}

NAKSHATRA_YONI = {
    "Ashwini": "Horse", "Shatabhisha": "Horse",
    "Bharani": "Elephant", "Revati": "Elephant",
    "Krittika": "Sheep", "Pushya": "Sheep",
    "Rohini": "Serpent", "Mrigashira": "Serpent",
    "Ardra": "Dog", "Mula": "Dog",
    "Punarvasu": "Cat", "Ashlesha": "Cat",
    "Magha": "Rat", "Purva Phalguni": "Rat",
    "Uttara Phalguni": "Cow", "Uttara Bhadrapada": "Cow",
    "Hasta": "Buffalo", "Swati": "Buffalo",
    "Chitra": "Tiger", "Vishakha": "Tiger",
    "Anuradha": "Deer", "Jyeshtha": "Deer",
    "Purva Ashadha": "Monkey", "Shravana": "Monkey",
    "Uttara Ashadha": "Mongoose",
    "Dhanishta": "Lion", "Purva Bhadrapada": "Lion",
}
YONI_ENEMY_PAIRS = {
    frozenset(["Horse", "Buffalo"]),
    frozenset(["Elephant", "Lion"]),
    frozenset(["Sheep", "Monkey"]),
    frozenset(["Serpent", "Mongoose"]),
    frozenset(["Dog", "Deer"]),
    frozenset(["Cat", "Rat"]),
    frozenset(["Cow", "Tiger"]),
}

NAKSHATRA_GANA = {
    "Ashwini": "Deva", "Mrigashira": "Deva", "Punarvasu": "Deva", "Pushya": "Deva",
    "Hasta": "Deva", "Swati": "Deva", "Anuradha": "Deva", "Shravana": "Deva", "Revati": "Deva",
    "Bharani": "Manushya", "Rohini": "Manushya", "Ardra": "Manushya",
    "Purva Phalguni": "Manushya", "Uttara Phalguni": "Manushya",
    "Purva Ashadha": "Manushya", "Uttara Ashadha": "Manushya",
    "Purva Bhadrapada": "Manushya", "Uttara Bhadrapada": "Manushya",
    "Krittika": "Rakshasa", "Ashlesha": "Rakshasa", "Magha": "Rakshasa", "Chitra": "Rakshasa",
    "Vishakha": "Rakshasa", "Jyeshtha": "Rakshasa", "Mula": "Rakshasa",
    "Dhanishta": "Rakshasa", "Shatabhisha": "Rakshasa",
}

NAKSHATRA_NADI = {
    "Ashwini": "Aadi", "Ardra": "Aadi", "Punarvasu": "Aadi", "Uttara Phalguni": "Aadi",
    "Hasta": "Aadi", "Jyeshtha": "Aadi", "Mula": "Aadi", "Shatabhisha": "Aadi", "Purva Bhadrapada": "Aadi",
    "Bharani": "Madhya", "Mrigashira": "Madhya", "Pushya": "Madhya", "Purva Phalguni": "Madhya",
    "Chitra": "Madhya", "Anuradha": "Madhya", "Purva Ashadha": "Madhya", "Dhanishta": "Madhya",
    "Uttara Bhadrapada": "Madhya",
    "Krittika": "Antya", "Rohini": "Antya", "Ashlesha": "Antya", "Magha": "Antya",
    "Swati": "Antya", "Vishakha": "Antya", "Uttara Ashadha": "Antya", "Shravana": "Antya", "Revati": "Antya",
}


def _vashya_group(moon_longitude: float) -> str:
    rashi = RASHI_NAMES[sign_index(moon_longitude)]
    if rashi in ("Dhanu", "Makar"):
        deg = degree_in_sign(moon_longitude)
        if rashi == "Dhanu":
            return "Chatushpada" if deg < 15 else "Manav"
        else:
            return "Chatushpada" if deg < 15 else "Jalachar"
    return VASHYA_GROUP_WHOLE[rashi]


def _koot_varna(rashi_a: str, rashi_b: str) -> dict:
    va, vb = RASHI_VARNA[rashi_a], RASHI_VARNA[rashi_b]
    points = 1 if VARNA_RANK[va] >= VARNA_RANK[vb] else 0
    return {"name": "Varna", "max": 1, "points": points, "detail": f"{va} & {vb}"}


def _koot_vashya(moon_a: float, moon_b: float) -> dict:
    ga, gb = _vashya_group(moon_a), _vashya_group(moon_b)
    points = VASHYA_COMPATIBILITY.get(frozenset([ga, gb]), 1)
    return {"name": "Vashya", "max": 2, "points": points, "detail": f"{ga} & {gb}"}


def _koot_tara(nak_a: int, nak_b: int) -> dict:
    count_ab = ((nak_b - nak_a) % 9) + 1
    count_ba = ((nak_a - nak_b) % 9) + 1
    good_ab = count_ab in TARA_GOOD
    good_ba = count_ba in TARA_GOOD
    if good_ab and good_ba:
        points = 3
    elif good_ab or good_ba:
        points = 1.5
    else:
        points = 0
    return {"name": "Tara", "max": 3, "points": points, "detail": f"counts {count_ab}/{count_ba} of 9"}


def _koot_yoni(nak_a: str, nak_b: str) -> dict:
    ya, yb = NAKSHATRA_YONI[nak_a], NAKSHATRA_YONI[nak_b]
    if ya == yb:
        points = 4
    elif frozenset([ya, yb]) in YONI_ENEMY_PAIRS:
        points = 0
    else:
        points = 2
    return {"name": "Yoni", "max": 4, "points": points, "detail": f"{ya} & {yb}"}


def _koot_graha_maitri(rashi_a: str, rashi_b: str) -> dict:
    lord_a = RASHI_LORDS[RASHI_NAMES.index(rashi_a)]
    lord_b = RASHI_LORDS[RASHI_NAMES.index(rashi_b)]
    if lord_a == lord_b:
        points = 5
    else:
        rel_ab = PLANET_FRIENDSHIP[lord_a].get(lord_b, "neutral")
        rel_ba = PLANET_FRIENDSHIP[lord_b].get(lord_a, "neutral")
        rels = {rel_ab, rel_ba}
        if rels == {"friend"}:
            points = 5
        elif rels == {"friend", "neutral"}:
            points = 4
        elif rels == {"neutral"}:
            points = 3
        elif rels == {"friend", "enemy"}:
            points = 1
        elif rels == {"neutral", "enemy"}:
            points = 0.5
        else:
            points = 0
    return {"name": "Graha Maitri", "max": 5, "points": points, "detail": f"lords {lord_a} & {lord_b}"}


def _koot_gana(nak_a: str, nak_b: str) -> dict:
    ga, gb = NAKSHATRA_GANA[nak_a], NAKSHATRA_GANA[nak_b]
    if ga == gb:
        points = 6
    elif {ga, gb} == {"Deva", "Manushya"}:
        points = 5
    elif {ga, gb} == {"Manushya", "Rakshasa"}:
        points = 3
    else:
        points = 0
    return {"name": "Gana", "max": 6, "points": points, "detail": f"{ga} & {gb}"}


def _koot_bhakoot(rashi_a: str, rashi_b: str) -> dict:
    idx_a, idx_b = RASHI_NAMES.index(rashi_a), RASHI_NAMES.index(rashi_b)
    diff = ((idx_b - idx_a) % 12) + 1
    dosha = diff in (2, 12, 5, 9, 6, 8)
    points = 0 if dosha else 7
    return {"name": "Bhakoot", "max": 7, "points": points, "detail": "Bhakoot Dosha present" if dosha else "no dosha"}


def _koot_nadi(nak_a: str, nak_b: str) -> dict:
    na, nb = NAKSHATRA_NADI[nak_a], NAKSHATRA_NADI[nak_b]
    dosha = na == nb
    points = 0 if dosha else 8
    return {"name": "Nadi", "max": 8, "points": points, "detail": "Nadi Dosha present (same Nadi)" if dosha else f"{na} & {nb}"}


def compute_kundli_milan(chart_a: dict, chart_b: dict) -> dict:
    moon_a, moon_b = chart_a["planets"]["Moon"], chart_b["planets"]["Moon"]

    koots = [
        _koot_varna(moon_a["rashi"], moon_b["rashi"]),
        _koot_vashya(moon_a["longitude"], moon_b["longitude"]),
        _koot_tara(nakshatra_index(moon_a["longitude"]), nakshatra_index(moon_b["longitude"])),
        _koot_yoni(moon_a["nakshatra"], moon_b["nakshatra"]),
        _koot_graha_maitri(moon_a["rashi"], moon_b["rashi"]),
        _koot_gana(moon_a["nakshatra"], moon_b["nakshatra"]),
        _koot_bhakoot(moon_a["rashi"], moon_b["rashi"]),
        _koot_nadi(moon_a["nakshatra"], moon_b["nakshatra"]),
    ]
    total = sum(k["points"] for k in koots)

    if total >= 32:
        verdict = "Excellent match"
    elif total >= 24:
        verdict = "Good match"
    elif total >= 18:
        verdict = "Average match (acceptable in most traditions)"
    else:
        verdict = "Below the traditional minimum threshold"

    return {
        "koots": koots,
        "total": total,
        "max_total": 36,
        "verdict": verdict,
        "nadi_dosha": koots[7]["points"] == 0,
        "bhakoot_dosha": koots[6]["points"] == 0,
    }


def check_mangal_dosha(chart: dict) -> dict:
    mars_house = chart["planets"]["Mars"]["house"]
    dosha_houses = {1, 2, 4, 7, 8, 12}
    active = mars_house in dosha_houses
    return {
        "active": active,
        "mars_house": mars_house,
        "detail": (
            f"Mars is in the {ordinal(mars_house)} house from the Lagna - "
            f"{'a Mangal Dosha (Manglik) house' if active else 'not a Mangal Dosha house'}."
        ),
    }


def milan_to_prompt_text(name_a: str, chart_a: dict, name_b: str, chart_b: dict,
                          milan: dict, mangal_a: dict, mangal_b: dict) -> str:
    lines = [
        f"Kundli Milan (Ashtakoot compatibility matching) between {name_a} and {name_b}:",
        f"- {name_a} Chandra Rashi/Nakshatra: {chart_a['planets']['Moon']['rashi']} "
        f"({chart_a['planets']['Moon']['rashi_english']}) / {chart_a['planets']['Moon']['nakshatra']}",
        f"- {name_b} Chandra Rashi/Nakshatra: {chart_b['planets']['Moon']['rashi']} "
        f"({chart_b['planets']['Moon']['rashi_english']}) / {chart_b['planets']['Moon']['nakshatra']}",
        "",
        f"Total score: {milan['total']}/{milan['max_total']} - {milan['verdict']}",
    ]
    for k in milan["koots"]:
        lines.append(f"- {k['name']}: {k['points']}/{k['max']} ({k['detail']})")
    if milan["nadi_dosha"]:
        lines.append("NOTE: Nadi Dosha is present (traditionally considered significant).")
    if milan["bhakoot_dosha"]:
        lines.append("NOTE: Bhakoot Dosha is present.")
    lines.append(f"{name_a} Mangal Dosha (Manglik): {'YES' if mangal_a['active'] else 'No'} - {mangal_a['detail']}")
    lines.append(f"{name_b} Mangal Dosha (Manglik): {'YES' if mangal_b['active'] else 'No'} - {mangal_b['detail']}")
    return "\n".join(lines)
