"""
app.py
------
AI Chat Astrologer (Vedic/Jyotish) - a Streamlit prototype.

Run it with:
    streamlit run app.py

What it does:
1. Takes the user's birth date, time, and place (with live search-as-you-type
   suggestions for the place, backed by astro_engine.search_places).
2. Calculates their real Vedic birth chart / Kundli (astro_engine.py, using
   Swiss Ephemeris with the Lahiri ayanamsa): Rashi, Nakshatra, Lagna,
   Rahu/Ketu, whole-sign houses (with each planet's house/Bhava placement),
   and a computed Vimshottari Mahadasha/Antardasha timeline.
3. Shows a visual South Indian-style chart diagram plus quick highlight
   cards (Lagna, Moon sign, Sun sign, current Dasha) right away - the full
   planet-by-planet breakdown and dasha timeline are tucked into an
   expander so the important stuff isn't buried.
4. Computes today's real planetary transits (Gochar) against your natal
   Moon sign, including Sade Sati detection - recalculated fresh every time
   you open the app, so it naturally updates day to day.
5. Gives a one-time Stella-written interpretation of the full Kundli, then
   lets them chat with an AI astrologer persona (styled as left/right chat
   bubbles, Stella on the left) whose answers are grounded in all of the
   above, using the Claude API. Stella is tuned to be direct and realistic
   rather than diplomatically vague - and to actually explain her reasoning
   rather than just handing down a short verdict - see the system prompt
   below.
6. Kundli Milan: Ashtakoot compatibility matching between two people's
   charts, plus individual Mangal Dosha (Manglik) checks, a grounded
   follow-up chat about the match, and a shareable text summary.

API key resolution order: ANTHROPIC_API_KEY environment variable, then
Streamlit secrets (st.secrets - used when deployed with a shared key on
Streamlit Community Cloud), then a key the visitor pastes in themselves. When
the key comes from the app itself (env var or secrets) rather than being
pasted by the visitor, a per-session message cap applies (MAX_MESSAGES_PER_SESSION
below) so a shared key can't be run up by one visitor.

Note on the birth details form: it lives in the MAIN content area (not the
sidebar). Streamlit's sidebar is collapsed by default on mobile browsers, so
anything sidebar-only is effectively invisible to mobile visitors unless
they know to tap the ">>" icon. Keeping it in the main area means it's
visible on both desktop and mobile without extra taps.

Note on birth-place input: it uses a live search-as-you-type box
(streamlit_searchbox) instead of a plain text field, so the user picks from
real matching places rather than typing a guess and hoping the geocoder
resolves it correctly. This is deliberately NOT wrapped in an st.form,
because Streamlit forms only rerun on submit - a live-search box needs to
rerun on every keystroke to fetch new suggestions.

Note on styling: colors/fonts come from .streamlit/config.toml (a soft
light lavender theme). Extra CSS is injected below for the chart diagram,
highlight cards, chat bubble left/right layout, and trimming Streamlit's
large default top padding - scoped to Streamlit's documented `data-testid`
attributes rather than internal class names, which is the more stable way
to target Streamlit's built-in components, though these testids can still
change on a major Streamlit version bump.

See README.md for full setup instructions, and DEPLOYMENT.md for how to put
this online for others to try.
"""

import datetime
import os

import streamlit as st
from anthropic import Anthropic
from streamlit_searchbox import st_searchbox

from astro_engine import (
    HOUSE_MEANINGS,
    PLANET_SYMBOLS,
    build_south_indian_grid,
    calculate_chart,
    chart_to_prompt_text,
    check_mangal_dosha,
    check_sade_sati,
    compute_gochar,
    compute_kundli_milan,
    compute_transits,
    find_current_dasha,
    find_timezone,
    gochar_to_prompt_text,
    milan_to_prompt_text,
    ordinal,
    search_places,
)

st.set_page_config(page_title="AI Astrology Chat", page_icon="✨", layout="centered")

MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096

# Stella's avatar in the chat. Unicode doesn't have a literal "Buddha statue"
# emoji, so this is the closest fit (a meditating figure) - if you have an
# actual Buddha image you'd like used instead, save it as e.g.
# "stella_avatar.png" in this same folder and change STELLA_AVATAR below to
# that filename; st.chat_message's avatar parameter accepts a local image
# path directly.
STELLA_AVATAR = "🧘"

# Cost/context control: only the most recent N messages (user + assistant combined)
# are sent to the API as conversation history. Older messages stay visible on
# screen but are dropped from what gets sent, so input-token cost per message
# stops growing once a conversation passes this length.
MAX_HISTORY_MESSAGES = 20

# Only enforced when using the app's own (shared) API key - see
# is_using_shared_key() below. Counts every AI call across both the Chat tab
# (including the one-time Kundli overview) and Kundli Milan (interpretation +
# follow-ups) combined, per browser session.
MAX_MESSAGES_PER_SESSION = 15

QUICK_PROMPTS = [
    "Today's Rashifal",
    "My career",
    "Love & relationships",
    "Health & energy",
]

# Explicit defaults for every birth-form widget, keyed to session_state.
# Using a stable `key=` (rather than just `value=`) is the robust Streamlit
# pattern - it's what stops widgets from silently resetting to their default
# on reruns (which happen after every chart generation and every chat
# message in this app).
FORM_DEFAULTS = {
    "birth_name_input": "",
    "birth_date_input": datetime.date(2000, 1, 1),
    "birth_time_input": datetime.time(12, 0),
    "manual_coords_checkbox": False,
    "lat_input": 0.0,
    "lon_input": 0.0,
}

SYSTEM_PROMPT_TEMPLATE = """You are "Stella", a direct, realistic AI Vedic astrologer (Jyotishi) chatting inside an app.

You are given the user's real, calculated Vedic birth chart (Kundli) AND
today's real planetary transits (Gochar) below - sidereal Rashi (signs,
using the Lahiri ayanamsa), Nakshatra, Lagna (Ascendant), Rahu/Ketu,
whole-sign houses with each planet's house placement, their current
Mahadasha/Antardasha (Vimshottari dasha system), and today's transiting
planets relative to their natal Moon (including Sade Sati status). This is
ALL real, precisely computed data - use it directly rather than estimating
or recalculating any of it yourself. When asked "today's rashifal" or
similar, use the Gochar section below, not just the natal chart.

Style:
- Speak primarily in Vedic/Jyotish terms (Rashi, Lagna, Nakshatra, Graha,
  Bhava/house, Mahadasha/Antardasha, Gochar) using the Sanskrit names given,
  with the English/Western equivalent in parentheses the first time you
  mention each one so the user isn't lost (e.g. "your Chandra Rashi (Moon
  sign) is Kanya (Virgo)").
- When discussing a planet's effect, connect its house placement to what
  that house governs (use the house meaning given to you) rather than just
  naming the sign.
- Be direct and realistic, not diplomatic. If a placement or transit points
  to real friction, delay, weakness, or difficulty, say so plainly instead
  of softening it into vague positivity - a reading that sounds equally
  rosy no matter what the chart actually shows isn't useful to anyone.
  Skip hedge-everything phrasing ("this could perhaps in some ways...") -
  say what the chart indicates, clearly.
- Being direct is not the same as being needlessly harsh - deliver a hard
  read with respect, not dismissiveness. The goal is honest and useful,
  not blunt for its own sake or unkind about things the user can't control.
- Don't just hand down a verdict - explain the reasoning behind it. Name
  which planet, house, dasha, or transit is driving the read, and briefly
  say why that combination means what it means, so the user actually
  understands their own chart rather than getting a one-line conclusion.
  A flat "this will be hard" with no explanation is just as unhelpful as
  sugarcoating - directness should come WITH substance and depth, not
  instead of it. Being engaging and thorough is part of being useful here.
- Favor being complete over being short. Don't artificially compress a
  reply just to hit a word count - if more than one factor is relevant,
  walk through all of them. Depth and clarity matter more than brevity.
- Format for easy reading on a phone screen: short paragraphs (2-4
  sentences each), and use a markdown bullet list whenever you're covering
  several distinct things (multiple planets, transits, or factors) rather
  than cramming them into one dense paragraph.
- Bold the single most important takeaway or phrase in each response (e.g.
  "**a good week to have that money conversation**") so it's easy to spot
  at a glance - don't over-bold, just the key point(s).
- There's no fixed length cap - answer as fully as the question deserves.
  A quick factual question can get a short answer, but anything about a
  reading, prediction, or interpretation should be explained in enough
  depth that the user walks away understanding the "why" behind it, not
  just the verdict.
- It's fine to ask the user follow-up questions about their life to tailor the reading.
- Don't claim false certainty about the future - frame things as strong
  tendencies and real probabilities based on the chart, not guarantees. This
  is about not fabricating certainty, not about hedging everything into mush.
- Note: you may only see the most recent part of a long conversation (older
  messages are trimmed to keep things fast and affordable) - if the user
  references something you don't have context for, just ask them to remind you.

Important guardrails:
- This is for entertainment and self-reflection. If the user asks something that
  sounds like a real medical, legal, financial, or safety emergency, gently say
  astrology isn't the right tool for that and suggest they talk to an
  appropriate professional or trusted person.
- Don't fabricate specific predictions presented as guaranteed facts (e.g. exact
  dates of death, diagnoses, exact lottery numbers) - this holds regardless of
  how direct/blunt the rest of your tone is.

{chart_text}

{gochar_text}
"""

KUNDLI_OVERVIEW_INSTRUCTION = """

The user is seeing their very first reading of this Kundli - this is a
one-time full overview, not a regular chat reply. Cover their Lagna
(Ascendant) and what it says about how they come across, their Chandra
Rashi (Moon sign) and what it says about their inner/emotional nature,
their Surya Rashi (Sun sign), their current Mahadasha/Antardasha and what
that period tends to emphasize (be specific about what it's good and bad
for), and today's Gochar (transits) and how it's currently affecting them.
For each point, briefly explain the reasoning (which planet/house/sign is
driving it and why that matters) rather than just stating a conclusion -
this should genuinely teach the user about their own chart, not just hand
them a summary. There's no fixed word count here - go deep enough that each
section actually makes sense, even to someone new to Vedic astrology. Use
short paragraphs and bullet lists (e.g. for the Mahadasha/Gochar section) so
it stays readable rather than becoming a wall of text. Be direct about
anything genuinely challenging in the chart rather than glossing over it -
explain what makes it challenging, not just that it is. Bold the most
important takeaways. End by inviting them to ask about anything specific -
career, love, health, or timing."""

MILAN_SYSTEM_PROMPT = """You are "Stella", a direct, realistic AI Vedic astrologer (Jyotishi).

You are given a real, precisely computed Ashtakoot Kundli Milan (marriage
compatibility) result between two people below - all 8 koot scores, the
total out of 36, Nadi/Bhakoot Dosha flags, and Mangal Dosha (Manglik) status
for each person. This is real computed data - use it directly, don't
recalculate or estimate anything. You may also be shown your own earlier
written interpretation of this same match, if the user is now asking a
follow-up question about it - stay consistent with what you said before
unless the user points out something you should reconsider.

When first asked to interpret: write a direct, thorough interpretation
covering the overall verdict and what the total score suggests, each koot
that stands out (best and weakest) and what it concretely means for the
couple (not just its name and number), whether any dosha (Nadi/Bhakoot/
Mangal) is present and what that traditionally implies, and a grounded
closing note. Explain the reasoning behind each point - why a given koot
score comes out the way it does - rather than just stating the score, so
the couple actually understands the match rather than getting a bare
verdict. There's no fixed word count - go deep enough that the reasoning is
clear rather than compressing it into a short summary. Use Sanskrit terms
with English in parentheses on first mention. Format for easy reading:
short paragraphs, a bullet list for the standout koots rather than a dense
paragraph, and bold the most important takeaways. Be honest, not
diplomatic - if the score is weak or a serious dosha is present, say so
plainly and explain why rather than softening it to spare feelings; don't
oversell a weak match or undersell a strong one. End by noting this is a
traditional first-pass screening tool for entertainment/reference, and real
marriage decisions should also weigh compatibility of values,
communication, and life goals - not just Kundli matching - and that a
qualified astrologer can assess dosha cancellations this simplified tool
doesn't check for.

For follow-up questions: answer directly and thoroughly, explain your
reasoning rather than just giving a verdict, and use bullets/bold where it
aids readability. There's no fixed length cap - answer as fully as the
question deserves rather than artificially cutting it short.

{milan_text}
"""


def inject_custom_css():
    """A small amount of custom styling on top of the .streamlit/config.toml
    theme: the South Indian chart diagram grid, the quick-highlight stat
    cards, rounder buttons, left/right chat bubbles (Stella on the left,
    you on the right), and a trimmed top page margin. The chat bubble and
    top-margin rules key off Streamlit's `data-testid` attributes."""
    st.markdown(
        """
        <style>
        div.stButton > button {
            border-radius: 10px;
        }

        /* Trim Streamlit's large default top padding so the title/caption
           header doesn't eat so much vertical space before the tabs. */
        div[data-testid="stMainBlockContainer"] {
            padding-top: 1.8rem;
        }
        h1 {
            margin-bottom: 0.2rem !important;
        }

        .highlight-cards {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 10px 0 18px 0;
        }
        .highlight-card {
            flex: 1 1 150px;
            background: linear-gradient(160deg, #ffffff, #f3ecfd);
            border: 1px solid #ddd0f5;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 1px 3px rgba(90, 60, 160, 0.08);
        }
        .highlight-card .hc-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #8a7aa8;
            margin-bottom: 4px;
        }
        .highlight-card .hc-value {
            font-size: 1.05rem;
            font-weight: 700;
            color: #35284f;
        }
        .highlight-card .hc-sub {
            font-size: 0.78rem;
            color: #6f6089;
            margin-top: 2px;
        }

        .kundli-chart-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            grid-template-rows: repeat(4, 1fr);
            gap: 4px;
            aspect-ratio: 1 / 1;
            max-width: 620px;
            margin: 6px auto 18px auto;
            background: #e6d9f7;
            border: 2px solid #8a63f2;
            border-radius: 16px;
            padding: 8px;
        }
        .kundli-cell {
            background: #ffffff;
            border: 1px solid #e2d6f7;
            border-radius: 10px;
            padding: 6px 4px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            min-width: 0;
            overflow: hidden;
        }
        .kundli-lagna {
            border: 2px solid #d99a2b;
            background: #fff7e6;
        }
        .kundli-house-num {
            font-size: 0.7rem;
            color: #9382b8;
            font-weight: 600;
        }
        .kundli-rashi {
            font-size: 0.9rem;
            color: #33284d;
            font-weight: 700;
            margin: 2px 0 4px 0;
            line-height: 1.1;
        }
        .kundli-planets {
            display: flex;
            flex-direction: column;
            gap: 2px;
            align-items: center;
            width: 100%;
        }
        .kundli-planet {
            background: #8a63f2;
            color: white;
            font-size: 0.68rem;
            padding: 1px 6px;
            border-radius: 6px;
            font-weight: 600;
            white-space: nowrap;
        }
        .kundli-planet.retro {
            background: #d9695f;
        }
        .kundli-center {
            grid-row: 2 / span 2;
            grid-column: 2 / span 2;
            display: flex;
            align-items: center;
            justify-content: center;
            background: transparent;
            border: none;
        }
        .kundli-center-label {
            color: #8a63f2;
            font-weight: 700;
            font-size: 1rem;
            opacity: 0.45;
        }

        /* Chat bubbles: Stella (custom avatar) on the left, you (default
           user avatar) on the right. Streamlit renders each chat message
           as a flex row of [avatar, content]; reversing the row for user
           messages puts the avatar on the right and pushes the bubble
           along with it. */
        div[data-testid="stChatMessageContent"] {
            max-width: 80%;
            border-radius: 14px;
            padding: 10px 14px;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarCustom"]) {
            justify-content: flex-start;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarCustom"]) div[data-testid="stChatMessageContent"] {
            background: #ffffff;
            border: 1px solid #e2d6f7;
            border-radius: 4px 14px 14px 14px;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
            flex-direction: row-reverse;
            justify-content: flex-start;
        }
        div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
            background: #efe3fb;
            border-radius: 14px 4px 14px 14px;
            margin-left: auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_south_indian_chart_html(grid: list) -> str:
    """Turns astro_engine.build_south_indian_grid()'s output into an HTML/CSS
    grid - a South Indian-style Kundli chart diagram, rendered via
    st.markdown(unsafe_allow_html=True). Uses full planet names (the chart
    is sized generously enough to fit them) rather than 2-letter codes."""
    cells_html = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell is None:
                if r == 1 and c == 1:
                    cells_html.append(
                        '<div class="kundli-cell kundli-center">'
                        '<span class="kundli-center-label">Kundli</span></div>'
                    )
                continue
            planets_html = "".join(
                f'<span class="kundli-planet{" retro" if p["retrograde"] else ""}">'
                f'{p["name"]}{" ℞" if p["retrograde"] else ""}</span>'
                for p in cell["planets"]
            )
            lagna_class = " kundli-lagna" if cell["is_lagna"] else ""
            cells_html.append(
                f'<div class="kundli-cell{lagna_class}" style="grid-row:{r + 1}; grid-column:{c + 1};">'
                f'<div class="kundli-house-num">House {cell["house"]}</div>'
                f'<div class="kundli-rashi">{cell["rashi"]}</div>'
                f'<div class="kundli-planets">{planets_html}</div>'
                f'</div>'
            )
    return '<div class="kundli-chart-grid">' + "".join(cells_html) + "</div>"


def _get_secret_key():
    """Reads ANTHROPIC_API_KEY from Streamlit's secrets manager (used when
    deployed on Streamlit Community Cloud with a shared key). Safe to call
    even when no secrets are configured at all (e.g. running locally)."""
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def is_using_shared_key() -> bool:
    """True when the API key came from the app itself (env var or Streamlit
    secrets) rather than the visitor pasting their own - this is when the
    per-session usage cap applies."""
    return bool(os.environ.get("ANTHROPIC_API_KEY")) or bool(_get_secret_key())


def get_api_key() -> str:
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key
    secret_key = _get_secret_key()
    if secret_key:
        return secret_key
    return st.session_state.get("api_key", "")


def sidebar_api_key():
    """API key input, shown at the very top of the sidebar so it's visible
    no matter which tab (Chat or Kundli Milan) you're using - both need it.
    (Left in the sidebar deliberately: beta testers using the shared key
    never need to touch this at all, so it being less prominent on mobile
    isn't an issue the way the birth form was.)"""
    st.sidebar.header("Anthropic API key")

    if is_using_shared_key():
        st.sidebar.success("Using this app's built-in API key - nothing to enter.")
        used = st.session_state.get("message_count", 0)
        remaining = max(0, MAX_MESSAGES_PER_SESSION - used)
        st.sidebar.caption(
            f"This is a shared demo key, capped at {MAX_MESSAGES_PER_SESSION} AI "
            f"messages per browser session to keep costs manageable - "
            f"{remaining} left this session."
        )
        st.sidebar.divider()
        return

    st.sidebar.text_input(
        "API key",
        type="password",
        key="api_key",
        help="Get one at console.anthropic.com (add a little credit to use it). "
             "Only kept for this session, never saved to disk.",
    )
    if not st.session_state.get("api_key"):
        st.sidebar.caption(
            "Needed to chat with Stella or get a Kundli Milan interpretation. "
            "Generating your Kundli, Gochar, and Milan scores works without it."
        )
    st.sidebar.divider()


def _init_form_defaults():
    for key, default in FORM_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def birth_details_form():
    """Birth-details input form, rendered in the MAIN content area (not the
    sidebar) so it's visible immediately on both desktop and mobile.

    Deliberately NOT wrapped in st.form - the place search box needs to
    rerun on every keystroke to fetch live suggestions, which st.form
    (submit-only reruns) would block.
    """
    _init_form_defaults()

    name = st.text_input("Name (optional)", key="birth_name_input")
    birth_date = st.date_input(
        "Birth date",
        min_value=datetime.date(1920, 1, 1),
        max_value=datetime.date.today(),
        key="birth_date_input",
    )
    birth_time = st.time_input("Birth time (as accurate as possible)", key="birth_time_input")

    st.caption("Birth place - start typing and pick from the suggestions that appear")
    selected_place = st_searchbox(
        search_places,
        key="birth_place_searchbox",
        placeholder="e.g. Jaipur, Uttar Pradesh, India",
    )
    st.caption(
        "Can't find your exact village/town in the suggestions? Try just the "
        "district or nearest bigger town, or use exact coordinates below "
        "instead - very small villages aren't always in the map databases "
        "these suggestions come from."
    )

    with st.expander("Optional: enter latitude/longitude manually instead"):
        st.caption(
            "For the most accurate chart: open Google Maps, right-click your "
            "exact birth location (e.g. the hospital or town center), and click "
            "the coordinates that pop up to copy them - then paste the two "
            "numbers in below."
        )
        manual = st.checkbox("Use manual coordinates instead of place name", key="manual_coords_checkbox")
        lat_manual = st.number_input("Latitude", format="%.6f", key="lat_input")
        lon_manual = st.number_input("Longitude", format="%.6f", key="lon_input")

    if st.button("Generate my Kundli", type="primary"):
        try:
            if manual:
                lat, lon = lat_manual, lon_manual
                place_label = "Manually entered coordinates"
            else:
                if not selected_place:
                    st.error("Please select a birth place from the suggestions above, or check 'use manual coordinates'.")
                    return
                lat, lon, place_label = selected_place

            tz_name = find_timezone(lat, lon)
            chart = calculate_chart(birth_date, birth_time, lat, lon, tz_name)

            st.session_state["chart"] = chart
            st.session_state["user_name"] = name or "the user"
            st.session_state["place_label"] = place_label
            st.session_state["place_coords"] = (lat, lon)
            st.session_state["messages"] = []  # reset chat when a new chart is generated
            st.session_state["kundli_interpretation"] = None  # reset the one-time overview too
            st.success(f"Using **{place_label}** ({lat:.4f}, {lon:.4f}) for your Kundli.")
            st.rerun()
        except Exception as e:
            st.error(str(e))


def render_kundli_highlights(chart: dict):
    """The important stuff, visible immediately (no clicks needed): quick
    stat cards for Lagna, Moon sign, Sun sign, and current Dasha (Mahadasha
    and Antardasha shown together), plus the South Indian-style visual
    chart diagram."""
    asc = chart["ascendant"]
    moon = chart["planets"]["Moon"]
    sun = chart["planets"]["Sun"]
    dasha_list = chart.get("dasha", [])
    maha, antar = find_current_dasha(dasha_list) if dasha_list else (None, None)

    if maha and antar:
        dasha_value = f"{maha['lord']} → {antar['lord']}"
        dasha_sub = "Mahadasha → Antardasha"
    elif maha:
        dasha_value = maha["lord"]
        dasha_sub = "Mahadasha"
    else:
        dasha_value = "—"
        dasha_sub = ""

    place_label = st.session_state.get("place_label")
    user_name = st.session_state.get("user_name", "the user")
    st.caption(
        f"For **{user_name}** · born {chart['local_datetime']}"
        + (f" · {place_label}" if place_label else "")
    )

    cards_html = f"""
    <div class="highlight-cards">
      <div class="highlight-card">
        <div class="hc-label">Lagna (Ascendant)</div>
        <div class="hc-value">{asc['rashi']}</div>
        <div class="hc-sub">{asc['rashi_english']} · {asc['degree']}°</div>
      </div>
      <div class="highlight-card">
        <div class="hc-label">Chandra Rashi (Moon)</div>
        <div class="hc-value">{moon['rashi']}</div>
        <div class="hc-sub">{moon['rashi_english']} · {moon['nakshatra']}</div>
      </div>
      <div class="highlight-card">
        <div class="hc-label">Surya Rashi (Sun)</div>
        <div class="hc-value">{sun['rashi']}</div>
        <div class="hc-sub">{sun['rashi_english']} · {ordinal(sun['house'])} house</div>
      </div>
      <div class="highlight-card">
        <div class="hc-label">Current Dasha</div>
        <div class="hc-value">{dasha_value}</div>
        <div class="hc-sub">{dasha_sub}</div>
      </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

    grid = build_south_indian_grid(chart)
    st.markdown(render_south_indian_chart_html(grid), unsafe_allow_html=True)
    st.caption(
        "South Indian style chart - each Rashi (sign) always sits in the same "
        "box; the gold-highlighted box is your Lagna. Each box shows the "
        "house number and every graha (planet) placed there."
    )


def render_chart_full_details(chart: dict):
    """The full planet-by-planet breakdown and dasha timeline - tucked into
    an expander since the highlights + chart diagram above already surface
    the most useful information at a glance."""
    with st.expander("Full chart details (every planet, house, and the full Dasha timeline)"):
        st.caption(f"Ayanamsa (Lahiri): {chart['ayanamsa']}°")

        st.markdown("**Grahas (planets) - sign, house & meaning:**")
        for planet, data in chart["planets"].items():
            symbol = PLANET_SYMBOLS.get(planet, "")
            retro = " ℞" if data["retrograde"] else ""
            house_meaning = HOUSE_MEANINGS[data["house"]]
            st.write(
                f"- {symbol} **{planet}**: {data['rashi']} ({data['rashi_english']}) "
                f"{data['degree']}°{retro} · {ordinal(data['house'])} house — {house_meaning}"
            )

        dasha_list = chart.get("dasha", [])
        if dasha_list:
            st.markdown("**Full Mahadasha timeline:**")
            for m in dasha_list:
                st.write(
                    f"{m['lord']}: {m['start'].strftime('%d %b %Y')} - "
                    f"{m['end'].strftime('%d %b %Y')}"
                )


def render_gochar_summary():
    """Today's transits vs. your natal Moon - recomputed fresh every time the
    app runs, so this section naturally reflects 'today' without any extra
    scheduling logic."""
    chart = st.session_state.get("chart")
    if not chart:
        return

    transits, as_of = compute_transits()
    natal_moon_longitude = chart["planets"]["Moon"]["longitude"]
    gochar = compute_gochar(natal_moon_longitude, transits)
    sade_sati = check_sade_sati(gochar)

    with st.expander(f"Today's Gochar (transits) - {as_of.strftime('%d %b %Y')}", expanded=False):
        st.caption(
            f"Counted from your natal Chandra Rashi (Moon sign): "
            f"{chart['planets']['Moon']['rashi']} ({chart['planets']['Moon']['rashi_english']})"
        )
        if sade_sati["active"]:
            st.warning(f"⚠️ Sade Sati is currently active - {sade_sati['phase']}.")
        for planet, data in gochar.items():
            retro = " ℞" if data["retrograde"] else ""
            st.write(
                f"- **{planet}**: transiting {data['rashi']} ({data['rashi_english']}){retro} "
                f"· {ordinal(data['house_from_moon'])} house from Moon"
            )


def render_kundli_interpretation(chart: dict, api_key: str):
    """One-time Stella-written interpretation of the full Kundli, shown right
    in the Kundli section - same idea as the Kundli Milan interpretation,
    just for a solo chart. Shown once per generated chart; the open-ended
    chat below is where follow-up questions go."""
    interpretation = st.session_state.get("kundli_interpretation")

    if interpretation:
        st.markdown("**Stella's interpretation of your Kundli:**")
        st.markdown(interpretation)
        st.divider()
        return

    if not api_key:
        st.info("Add your Anthropic API key in the sidebar to get Stella's full interpretation of your Kundli.")
        return

    if st.button("✨ Get Stella's interpretation of your Kundli", type="primary"):
        chart_text = chart_to_prompt_text(chart, st.session_state.get("user_name", "the user"))
        gochar_text = gochar_to_prompt_text(chart)
        system_prompt = (
            SYSTEM_PROMPT_TEMPLATE.format(chart_text=chart_text, gochar_text=gochar_text)
            + KUNDLI_OVERVIEW_INSTRUCTION
        )
        placeholder = st.empty()
        interpretation = call_stella(
            api_key, system_prompt,
            [{"role": "user", "content": "Please give me a full reading of my Kundli."}],
            placeholder=placeholder,
        )
        st.session_state["kundli_interpretation"] = interpretation
        st.rerun()


def chat_tab():
    chart = st.session_state.get("chart")
    if not chart:
        st.info("Enter your birth details below and tap **Generate my Kundli** to start chatting.")
        birth_details_form()
        return

    with st.expander("✏️ Edit Birth Details"):
        birth_details_form()

    render_kundli_highlights(chart)
    render_chart_full_details(chart)
    render_gochar_summary()

    api_key = get_api_key()

    render_kundli_interpretation(chart, api_key)

    if not api_key:
        st.warning("Add your Anthropic API key in the sidebar to start chatting with Stella.")
        return

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    total_messages = len(st.session_state["messages"])
    if total_messages > MAX_HISTORY_MESSAGES:
        st.caption(
            f"💬 {total_messages} messages so far - Stella is only seeing the most "
            f"recent {MAX_HISTORY_MESSAGES} to keep responses fast and costs flat."
        )

    for msg in st.session_state["messages"]:
        avatar = STELLA_AVATAR if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Quick-tap suggested prompts - clicking one sends it exactly as if typed.
    clicked_prompt = None
    cols = st.columns(len(QUICK_PROMPTS))
    for col, prompt_text in zip(cols, QUICK_PROMPTS):
        if col.button(prompt_text, use_container_width=True):
            clicked_prompt = prompt_text

    typed_input = st.chat_input(
        "Ask Stella anything - love, career, timing, today's energy...",
        key="main_chat_input",
    )
    user_input = typed_input or clicked_prompt

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        chart_text = chart_to_prompt_text(chart, st.session_state.get("user_name", "the user"))
        gochar_text = gochar_to_prompt_text(chart)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(chart_text=chart_text, gochar_text=gochar_text)

        history_to_send = st.session_state["messages"][-MAX_HISTORY_MESSAGES:]
        api_messages = [{"role": m["role"], "content": m["content"]} for m in history_to_send]

        with st.chat_message("assistant", avatar=STELLA_AVATAR):
            placeholder = st.empty()
            response_text = call_stella(api_key, system_prompt, api_messages, placeholder=placeholder)
        st.session_state["messages"].append({"role": "assistant", "content": response_text})
        st.rerun()  # clears the quick-prompt buttons' one-shot click state cleanly


def call_stella(api_key: str, system_prompt: str, api_messages: list,
                 placeholder=None) -> str:
    """Shared streaming call to Claude, with a silent one-shot retry on an
    empty response, rich debug info if it still comes back empty, and a
    per-session usage cap when a shared/deployed key is in use."""
    own_placeholder = placeholder is None
    if own_placeholder:
        placeholder = st.empty()

    if is_using_shared_key():
        used = st.session_state.get("message_count", 0)
        if used >= MAX_MESSAGES_PER_SESSION:
            capped_msg = (
                f"_(This demo shares one API key across all visitors, capped at "
                f"{MAX_MESSAGES_PER_SESSION} AI messages per browser session to keep "
                f"costs manageable - you've reached that limit for now. Thanks for "
                f"trying it out! Refresh in a new browser tab for a fresh session, or "
                f"come back later.)_"
            )
            placeholder.warning(capped_msg)
            return capped_msg
        st.session_state["message_count"] = used + 1

    placeholder.markdown("_thinking..._")
    client = Anthropic(api_key=api_key)

    full_response = ""
    stop_reason = None
    final_message = None
    try:
        for attempt in range(2):
            full_response = ""
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=api_messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")
                final_message = stream.get_final_message()
                stop_reason = final_message.stop_reason

            if full_response:
                break
            if attempt == 0:
                placeholder.markdown("_thinking again..._")

        if not full_response:
            block_types = [b.type for b in final_message.content] if final_message and final_message.content else []
            output_tokens = final_message.usage.output_tokens if final_message and final_message.usage else "?"
            full_response = (
                "_(The AI returned no visible text, twice in a row.)_\n\n"
                f"Debug info - stop_reason=`{stop_reason}`, content_block_types=`{block_types}`, "
                f"output_tokens=`{output_tokens}`. Try again or rephrase."
            )
        elif stop_reason == "max_tokens":
            full_response += (
                "\n\n_(Note: this answer was cut off because it hit the length "
                "limit - ask 'continue' if you want the rest.)_"
            )
        placeholder.markdown(full_response)
    except Exception as e:
        error_type = type(e).__name__
        full_response = f"**Error calling the AI ({error_type}):**\n\n```\n{e}\n```"
        placeholder.error(full_response)
        st.info(
            "Common causes: the API key is invalid/expired, the Anthropic account "
            "has no credit balance yet (add credit at console.anthropic.com under "
            "'Billing'), or a network/firewall issue."
        )

    return full_response


# ---------------------------------------------------------------------------
# Kundli Milan (compatibility matching) tab
# ---------------------------------------------------------------------------

MILAN_FORM_DEFAULTS = {
    "groom_name": "", "groom_date": datetime.date(1995, 1, 1), "groom_time": datetime.time(12, 0),
    "groom_manual": False, "groom_lat": 0.0, "groom_lon": 0.0,
    "bride_name": "", "bride_date": datetime.date(1997, 1, 1), "bride_time": datetime.time(12, 0),
    "bride_manual": False, "bride_lat": 0.0, "bride_lon": 0.0,
}


def _init_milan_defaults():
    for key, default in MILAN_FORM_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _person_form(prefix: str, label: str):
    """Renders one person's birth-detail inputs (used twice: groom & bride)."""
    st.markdown(f"**{label}**")
    st.text_input("Name", key=f"{prefix}_name")
    st.date_input(
        "Birth date", min_value=datetime.date(1920, 1, 1),
        max_value=datetime.date.today(), key=f"{prefix}_date",
    )
    st.time_input("Birth time", key=f"{prefix}_time")
    st.caption("Birth place - start typing and pick from the suggestions")
    st_searchbox(
        search_places,
        key=f"{prefix}_place_searchbox",
        placeholder="e.g. Jaipur, Uttar Pradesh, India",
    )
    with st.expander("Enter coordinates manually instead"):
        st.checkbox("Use manual coordinates", key=f"{prefix}_manual")
        st.number_input("Latitude", format="%.6f", key=f"{prefix}_lat")
        st.number_input("Longitude", format="%.6f", key=f"{prefix}_lon")


def _build_chart_from_person_form(prefix: str):
    """Returns (name, chart, place_label, (lat, lon)) - the place/coords are
    kept so the results view can show exactly what was used, for the user to
    verify (same idea as the birth-details recap on the solo Kundli tab)."""
    name = st.session_state[f"{prefix}_name"] or prefix.capitalize()
    if st.session_state[f"{prefix}_manual"]:
        lat, lon = st.session_state[f"{prefix}_lat"], st.session_state[f"{prefix}_lon"]
        place_label = "Manually entered coordinates"
    else:
        selected = st.session_state.get(f"{prefix}_place_searchbox")
        if not selected:
            raise ValueError(f"Please select a birth place for {name} from the suggestions, or use manual coordinates.")
        lat, lon, place_label = selected
    tz_name = find_timezone(lat, lon)
    chart = calculate_chart(
        st.session_state[f"{prefix}_date"], st.session_state[f"{prefix}_time"], lat, lon, tz_name
    )
    return name, chart, place_label, (lat, lon)


def build_milan_share_text(result: dict) -> str:
    """Formats the match into a plain-text summary suitable for pasting into
    WhatsApp/Instagram/etc. Includes Stella's interpretation if it's been
    generated yet."""
    milan = result["milan"]
    gc, bc = result["groom_chart"], result["bride_chart"]
    mg, mb = result["mangal_groom"], result["mangal_bride"]

    lines = [
        f"✨ Kundli Milan: {result['groom_name']} ✕ {result['bride_name']}",
        "",
        f"{result['groom_name']}: born {gc['local_datetime']}, {result['groom_place']}",
        f"  Chandra Rashi {gc['planets']['Moon']['rashi']} ({gc['planets']['Moon']['rashi_english']}), "
        f"Nakshatra {gc['planets']['Moon']['nakshatra']}",
        f"{result['bride_name']}: born {bc['local_datetime']}, {result['bride_place']}",
        f"  Chandra Rashi {bc['planets']['Moon']['rashi']} ({bc['planets']['Moon']['rashi_english']}), "
        f"Nakshatra {bc['planets']['Moon']['nakshatra']}",
        "",
        f"Total Guna Score: {milan['total']}/{milan['max_total']} - {milan['verdict']}",
    ]
    for k in milan["koots"]:
        lines.append(f"  {k['name']}: {k['points']}/{k['max']} ({k['detail']})")
    if milan["nadi_dosha"]:
        lines.append("  ⚠ Nadi Dosha present")
    if milan["bhakoot_dosha"]:
        lines.append("  ⚠ Bhakoot Dosha present")
    lines.append(f"{result['groom_name']} Manglik: {'Yes' if mg['active'] else 'No'} ({mg['detail']})")
    lines.append(f"{result['bride_name']} Manglik: {'Yes' if mb['active'] else 'No'} ({mb['detail']})")

    if result.get("interpretation"):
        lines += ["", "— Stella's interpretation —", result["interpretation"]]

    lines += [
        "",
        "(Generated by an AI Vedic astrology prototype - for entertainment/reference. "
        "Consult a qualified astrologer for a full assessment.)",
    ]
    return "\n".join(lines)


def render_milan_share_block(result: dict):
    st.divider()
    st.markdown("**Share this match**")
    share_text = build_milan_share_text(result)
    if not result.get("interpretation"):
        st.caption("Tip: get Stella's interpretation above first to include it in the shared summary.")
    st.code(share_text, language=None)  # has a built-in one-click copy button
    st.download_button(
        "Download as text file",
        data=share_text,
        file_name=f"kundli_milan_{result['groom_name']}_{result['bride_name']}.txt".replace(" ", "_"),
        mime="text/plain",
    )


def milan_tab():
    _init_milan_defaults()
    st.caption(
        "Ashtakoot Guna Milan: the classical 8-factor, 36-point Vedic compatibility "
        "check, based on each person's Moon Rashi (sign) and Nakshatra, plus an "
        "individual Mangal Dosha (Manglik) check for each person. For entertainment "
        "and reference - a qualified astrologer can assess dosha cancellations and "
        "nuances this simplified tool doesn't cover."
    )

    col1, col2 = st.columns(2)
    with col1:
        _person_form("groom", "Groom's birth details")
    with col2:
        _person_form("bride", "Bride's birth details")

    if st.button("Check Compatibility", type="primary"):
        try:
            groom_name, groom_chart, groom_place, groom_coords = _build_chart_from_person_form("groom")
            bride_name, bride_chart, bride_place, bride_coords = _build_chart_from_person_form("bride")
            milan = compute_kundli_milan(groom_chart, bride_chart)
            mangal_groom = check_mangal_dosha(groom_chart)
            mangal_bride = check_mangal_dosha(bride_chart)
            st.session_state["milan_result"] = {
                "groom_name": groom_name, "groom_chart": groom_chart,
                "groom_place": groom_place, "groom_coords": groom_coords,
                "bride_name": bride_name, "bride_chart": bride_chart,
                "bride_place": bride_place, "bride_coords": bride_coords,
                "milan": milan, "mangal_groom": mangal_groom, "mangal_bride": mangal_bride,
                "interpretation": None, "milan_text": None,
            }
            st.session_state["milan_messages"] = []  # reset follow-up chat for the new match
        except Exception as e:
            st.error(str(e))

    result = st.session_state.get("milan_result")
    if not result:
        return

    milan = result["milan"]
    st.divider()
    st.subheader(f"{result['groom_name']} ✕ {result['bride_name']}")

    # Birth details recap for both people, side by side, so it's easy to
    # verify exactly whose charts produced this result.
    st.markdown("**Birth details used for this match:**")
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        gc = result["groom_chart"]
        st.write(f"**{result['groom_name']}** (Groom)")
        st.write(f"- Born: {gc['local_datetime']} (tz {gc['timezone']})")
        st.write(f"- Place: {result['groom_place']} ({result['groom_coords'][0]:.4f}, {result['groom_coords'][1]:.4f})")
        st.caption(f"Chandra Rashi: {gc['planets']['Moon']['rashi']} · Nakshatra: {gc['planets']['Moon']['nakshatra']}")
    with bcol2:
        bc = result["bride_chart"]
        st.write(f"**{result['bride_name']}** (Bride)")
        st.write(f"- Born: {bc['local_datetime']} (tz {bc['timezone']})")
        st.write(f"- Place: {result['bride_place']} ({result['bride_coords'][0]:.4f}, {result['bride_coords'][1]:.4f})")
        st.caption(f"Chandra Rashi: {bc['planets']['Moon']['rashi']} · Nakshatra: {bc['planets']['Moon']['nakshatra']}")

    st.metric("Total Guna Score", f"{milan['total']} / {milan['max_total']}", milan["verdict"])

    for k in milan["koots"]:
        st.write(f"**{k['name']}**: {k['points']}/{k['max']} — {k['detail']}")

    if milan["nadi_dosha"]:
        st.warning("⚠️ Nadi Dosha present (same Nadi) - traditionally considered significant.")
    if milan["bhakoot_dosha"]:
        st.warning("⚠️ Bhakoot Dosha present.")

    mg, mb = result["mangal_groom"], result["mangal_bride"]
    st.write(f"**{result['groom_name']} Mangal Dosha (Manglik):** {'Yes' if mg['active'] else 'No'} — {mg['detail']}")
    st.write(f"**{result['bride_name']} Mangal Dosha (Manglik):** {'Yes' if mb['active'] else 'No'} — {mb['detail']}")

    api_key = get_api_key()
    if not api_key:
        st.info("Add your Anthropic API key in the sidebar to get Stella's written interpretation of this match.")
        render_milan_share_block(result)
        return

    if result["interpretation"] is None:
        if st.button("Get Stella's interpretation"):
            milan_text = milan_to_prompt_text(
                result["groom_name"], result["groom_chart"],
                result["bride_name"], result["bride_chart"],
                milan, mg, mb,
            )
            system_prompt = MILAN_SYSTEM_PROMPT.format(milan_text=milan_text)
            placeholder = st.empty()
            interpretation = call_stella(
                api_key, system_prompt,
                [{"role": "user", "content": "Please interpret this Kundli Milan result."}],
                placeholder=placeholder,
            )
            st.session_state["milan_result"]["interpretation"] = interpretation
            st.session_state["milan_result"]["milan_text"] = milan_text
            st.rerun()
        render_milan_share_block(result)
        return

    st.markdown("**Stella's interpretation:**")
    st.markdown(result["interpretation"])

    # Follow-up chat, grounded in this specific match (separate thread from
    # the main Chat tab, so questions here stay focused on the match).
    st.divider()
    st.markdown("**Ask Stella about this match:**")
    if "milan_messages" not in st.session_state:
        st.session_state["milan_messages"] = []

    for msg in st.session_state["milan_messages"]:
        avatar = STELLA_AVATAR if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    followup = st.chat_input(
        "Ask about timing, doshas, what this means for you two...",
        key="milan_chat_input",
    )
    if followup:
        st.session_state["milan_messages"].append({"role": "user", "content": followup})
        with st.chat_message("user"):
            st.markdown(followup)

        system_prompt = MILAN_SYSTEM_PROMPT.format(milan_text=result["milan_text"])
        system_prompt += f"\n\nYour earlier interpretation of this match:\n{result['interpretation']}"

        history = st.session_state["milan_messages"][-MAX_HISTORY_MESSAGES:]
        api_messages = [{"role": m["role"], "content": m["content"]} for m in history]

        with st.chat_message("assistant", avatar=STELLA_AVATAR):
            placeholder = st.empty()
            response = call_stella(api_key, system_prompt, api_messages, placeholder=placeholder)
        st.session_state["milan_messages"].append({"role": "assistant", "content": response})
        st.rerun()

    render_milan_share_block(result)


def main():
    inject_custom_css()
    st.title("✨ AI Astrology Chat")
    st.caption(
        "A prototype AI Vedic astrologer (Jyotishi), grounded in your real Kundli. "
        "_For entertainment and self-reflection - not medical, legal, or financial advice._"
    )

    sidebar_api_key()

    tab_chat, tab_milan = st.tabs(["💬 Chat with Stella", "💑 Kundli Milan (Matching)"])
    with tab_chat:
        chat_tab()
    with tab_milan:
        milan_tab()


if __name__ == "__main__":
    main()
