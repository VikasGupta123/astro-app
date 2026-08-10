"""
app.py
------
AI Chat Astrologer (Vedic/Jyotish) - a Streamlit prototype.

Run it with:
    streamlit run app.py

What it does:
1. Takes the user's birth date, time, and place.
2. Calculates their real Vedic birth chart / Kundli (astro_engine.py, using
   Swiss Ephemeris with the Lahiri ayanamsa): Rashi, Nakshatra, Lagna,
   Rahu/Ketu, whole-sign houses (with each planet's house/Bhava placement),
   and a computed Vimshottari Mahadasha/Antardasha timeline.
3. Computes today's real planetary transits (Gochar) against your natal
   Moon sign, including Sade Sati detection - recalculated fresh every time
   you open the app, so it naturally updates day to day.
4. Lets them chat with an AI astrologer persona whose answers are grounded
   in all of the above, using the Claude API.
5. Kundli Milan: Ashtakoot compatibility matching between two people's
   charts, plus individual Mangal Dosha (Manglik) checks, a grounded
   follow-up chat about the match, and a shareable text summary.

API key resolution order: ANTHROPIC_API_KEY environment variable, then
Streamlit secrets (st.secrets - used when deployed with a shared key on
Streamlit Community Cloud), then a key the visitor pastes in themselves. When
the key comes from the app itself (env var or secrets) rather than being
pasted by the visitor, a per-session message cap applies (MAX_MESSAGES_PER_SESSION
below) so a shared key can't be run up by one visitor.

See README.md for full setup instructions, and DEPLOYMENT.md for how to put
this online for others to try.
"""

import datetime
import os

import streamlit as st
from anthropic import Anthropic

from astro_engine import (
    HOUSE_MEANINGS,
    calculate_chart,
    chart_to_prompt_text,
    check_mangal_dosha,
    check_sade_sati,
    compute_gochar,
    compute_kundli_milan,
    compute_transits,
    find_current_dasha,
    find_timezone,
    geocode_place,
    gochar_to_prompt_text,
    milan_to_prompt_text,
    ordinal,
)

st.set_page_config(page_title="AI Astrology Chat", page_icon="✨", layout="centered")

MODEL = "claude-sonnet-5"
MAX_TOKENS = 4096

# Cost/context control: only the most recent N messages (user + assistant combined)
# are sent to the API as conversation history. Older messages stay visible on
# screen but are dropped from what gets sent, so input-token cost per message
# stops growing once a conversation passes this length.
MAX_HISTORY_MESSAGES = 20

# Only enforced when using the app's own (shared) API key - see
# is_using_shared_key() below. Counts every AI call across both the Chat tab
# and Kundli Milan (interpretation + follow-ups) combined, per browser session.
MAX_MESSAGES_PER_SESSION = 15

QUICK_PROMPTS = [
    "Today's Rashifal",
    "My career",
    "Love & relationships",
    "Health & energy",
]

# Explicit defaults for every sidebar form widget, keyed to session_state.
# Using a stable `key=` (rather than just `value=`) is the robust Streamlit
# pattern - it's what stops widgets from silently resetting to their default
# on reruns (which happen after every chart generation and every chat
# message in this app).
FORM_DEFAULTS = {
    "birth_name_input": "",
    "birth_date_input": datetime.date(2000, 1, 1),
    "birth_time_input": datetime.time(12, 0),
    "birth_place_input": "",
    "manual_coords_checkbox": False,
    "lat_input": 0.0,
    "lon_input": 0.0,
}

SYSTEM_PROMPT_TEMPLATE = """You are "Stella", a warm, insightful AI Vedic astrologer (Jyotishi) chatting inside an app.

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
- Warm, conversational, a little mystical, but never vague filler.
- Keep answers focused and readable in a chat bubble (short paragraphs, no giant essays unless asked).
- Keep every reply under roughly 250 words unless the user explicitly asks for
  exhaustive detail (e.g. a full dasha timeline or full chart breakdown) - give
  a concise, useful summary first and offer to go deeper rather than dumping
  everything at once.
- It's fine to ask the user follow-up questions about their life to tailor the reading.
- Never claim certainty about the future - frame things as tendencies, energies, and possibilities.
- Note: you may only see the most recent part of a long conversation (older
  messages are trimmed to keep things fast and affordable) - if the user
  references something you don't have context for, just ask them to remind you.

Important guardrails:
- This is for entertainment and self-reflection. If the user asks something that
  sounds like a real medical, legal, financial, or safety emergency, gently say
  astrology isn't the right tool for that and suggest they talk to an
  appropriate professional or trusted person.
- Don't fabricate specific predictions presented as guaranteed facts (e.g. exact
  dates of death, diagnoses, exact lottery numbers).

{chart_text}

{gochar_text}
"""

MILAN_SYSTEM_PROMPT = """You are "Stella", a warm, insightful AI Vedic astrologer (Jyotishi).

You are given a real, precisely computed Ashtakoot Kundli Milan (marriage
compatibility) result between two people below - all 8 koot scores, the
total out of 36, Nadi/Bhakoot Dosha flags, and Mangal Dosha (Manglik) status
for each person. This is real computed data - use it directly, don't
recalculate or estimate anything. You may also be shown your own earlier
written interpretation of this same match, if the user is now asking a
follow-up question about it - stay consistent with what you said before
unless the user points out something you should reconsider.

When first asked to interpret: write a warm, readable interpretation
(300-400 words) covering the overall verdict and what the total score
suggests, the 2-3 koots that stand out (best and weakest) and what they mean
practically, whether any dosha (Nadi/Bhakoot/Mangal) is present and what
that traditionally implies, and a grounded closing note. Use Sanskrit terms
with English in parentheses on first mention. Be encouraging but honest -
don't oversell a weak match or undersell a strong one. End by noting this is
a traditional first-pass screening tool for entertainment/reference, and
real marriage decisions should also weigh compatibility of values,
communication, and life goals - not just Kundli matching - and that a
qualified astrologer can assess dosha cancellations this simplified tool
doesn't check for.

For follow-up questions: answer conversationally and keep it under ~200
words unless more detail is genuinely asked for.

{milan_text}
"""


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
    no matter which tab (Chat or Kundli Milan) you're using - both need it."""
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


def sidebar_birth_form():
    _init_form_defaults()
    st.sidebar.header("Your birth details")

    with st.sidebar.form("birth_form"):
        name = st.text_input("Name (optional)", key="birth_name_input")
        birth_date = st.date_input(
            "Birth date",
            min_value=datetime.date(1920, 1, 1),
            max_value=datetime.date.today(),
            key="birth_date_input",
        )
        birth_time = st.time_input("Birth time (as accurate as possible)", key="birth_time_input")

        st.caption("Birth place")
        place = st.text_input("City, Country", placeholder="e.g. Jaipur, India", key="birth_place_input")
        st.caption(
            "Free place lookup can be off by a few to ~15km (it resolves to a "
            "representative point for the town, not a specific address). That's "
            "fine for most readings. For maximum precision - especially the "
            "Lagna (Ascendant) - use exact coordinates below instead."
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

        submitted = st.form_submit_button("Generate my Kundli")

    if submitted:
        try:
            if manual:
                lat, lon = lat_manual, lon_manual
                place_label = "Manually entered coordinates"
            else:
                if not place.strip():
                    st.sidebar.error("Please enter a birth place, or check 'use manual coordinates'.")
                    return
                lat, lon, place_label = geocode_place(place)

            tz_name = find_timezone(lat, lon)
            chart = calculate_chart(birth_date, birth_time, lat, lon, tz_name)

            st.session_state["chart"] = chart
            st.session_state["user_name"] = name or "the user"
            st.session_state["place_label"] = place_label
            st.session_state["place_coords"] = (lat, lon)
            st.session_state["messages"] = []  # reset chat when a new chart is generated
            st.sidebar.success(
                f"Found: **{place_label}**\n\nCoordinates: {lat:.4f}, {lon:.4f}\n\n"
                "Double-check this matches where you were born - if it's off, try "
                "adding more detail (e.g. state/country), or switch to manual "
                "coordinates above for exact precision."
            )
        except Exception as e:
            st.sidebar.error(str(e))


def render_chart_summary():
    chart = st.session_state.get("chart")
    if not chart:
        return
    with st.expander("Your Kundli (Vedic chart)", expanded=False):
        # Birth details recap, right alongside the location - so it's always
        # clear exactly which chart you're looking at.
        user_name = st.session_state.get("user_name", "the user")
        place_label = st.session_state.get("place_label")
        coords = st.session_state.get("place_coords")
        st.markdown("**Birth details used for this Kundli:**")
        st.write(f"- Name: {user_name}")
        st.write(f"- Born: {chart['local_datetime']} (timezone {chart['timezone']})")
        if place_label and coords:
            st.write(f"- Place: {place_label} ({coords[0]:.4f}, {coords[1]:.4f})")
        st.caption(f"Ayanamsa (Lahiri): {chart['ayanamsa']}°")

        asc = chart["ascendant"]
        st.write(
            f"**Lagna (Ascendant):** {asc['rashi']} ({asc['rashi_english']}) "
            f"{asc['degree']}° · Nakshatra {asc['nakshatra']} pada {asc['pada']}"
        )

        st.markdown("**Grahas (planets) - sign, house & meaning:**")
        for planet, data in chart["planets"].items():
            retro = " ℞" if data["retrograde"] else ""
            house_meaning = HOUSE_MEANINGS[data["house"]]
            st.write(
                f"- **{planet}**: {data['rashi']} ({data['rashi_english']}) "
                f"{data['degree']}°{retro} · {ordinal(data['house'])} house — {house_meaning}"
            )

        dasha_list = chart.get("dasha", [])
        if dasha_list:
            maha, antar = find_current_dasha(dasha_list)
            st.markdown("**Current Dasha:**")
            if maha:
                st.write(
                    f"Mahadasha: **{maha['lord']}** "
                    f"({maha['start'].strftime('%d %b %Y')} - {maha['end'].strftime('%d %b %Y')})"
                )
                if antar:
                    st.write(
                        f"Antardasha: **{antar['lord']}** "
                        f"({antar['start'].strftime('%d %b %Y')} - {antar['end'].strftime('%d %b %Y')})"
                    )
            with st.expander("Full Mahadasha timeline"):
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


def chat_tab():
    chart = st.session_state.get("chart")
    if not chart:
        st.info("Enter your birth details in the sidebar and click **Generate my Kundli** to start chatting.")
        return

    render_chart_summary()
    render_gochar_summary()

    api_key = get_api_key()
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
        with st.chat_message(msg["role"]):
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

        response_text = call_stella(api_key, system_prompt, api_messages)
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
    "groom_place": "", "groom_manual": False, "groom_lat": 0.0, "groom_lon": 0.0,
    "bride_name": "", "bride_date": datetime.date(1997, 1, 1), "bride_time": datetime.time(12, 0),
    "bride_place": "", "bride_manual": False, "bride_lat": 0.0, "bride_lon": 0.0,
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
    st.text_input("City, Country", placeholder="e.g. Jaipur, India", key=f"{prefix}_place")
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
        place = st.session_state[f"{prefix}_place"]
        if not place.strip():
            raise ValueError(f"Please enter a birth place for {name}, or use manual coordinates.")
        lat, lon, place_label = geocode_place(place)
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
        with st.chat_message(msg["role"]):
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

        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = call_stella(api_key, system_prompt, api_messages, placeholder=placeholder)
        st.session_state["milan_messages"].append({"role": "assistant", "content": response})
        st.rerun()

    render_milan_share_block(result)


def main():
    st.title("✨ AI Astrology Chat")
    st.caption("A prototype AI Vedic astrologer (Jyotishi), grounded in your real Kundli.")
    st.caption("_For entertainment and self-reflection - not medical, legal, or financial advice._")

    sidebar_api_key()
    sidebar_birth_form()

    tab_chat, tab_milan = st.tabs(["💬 Chat with Stella", "💑 Kundli Milan (Matching)"])
    with tab_chat:
        chat_tab()
    with tab_milan:
        milan_tab()


if __name__ == "__main__":
    main()
