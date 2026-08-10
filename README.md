# AI Astrology Chat — Prototype

A working prototype of an AI chat astrologer: users enter their birth date, time,
and place, the app calculates their real birth chart, and they chat with an AI
astrologer persona ("Stella") whose answers are grounded in that chart.

No coding experience needed to run this — just follow the steps below.

## 1. Install Python (if you don't have it)

Download and install Python from https://www.python.org/downloads/ (version 3.10
or newer). During install on Windows, check the box "Add Python to PATH".

## 2. Get an Anthropic API key

1. Go to https://console.anthropic.com and create an account.
2. Go to **API Keys** and create a new key.
3. Add a small amount of credit to the account (a few dollars is plenty to test with).
4. Copy the key somewhere safe — you'll paste it into the app.

## 3. Install the app's dependencies

Open a terminal (Command Prompt / Terminal app) and navigate into this folder
(type `cd ` then drag the `astro_app` folder into the terminal window to
auto-fill the path, then press Enter).

**On macOS**, modern Macs block installing packages directly (you may see an
"externally-managed-environment" error) — use a virtual environment instead:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Your terminal prompt should now show `(venv)` at the start. Every time you
close and reopen Terminal, `cd` back into this folder and run
`source venv/bin/activate` again before running the app.

**On Windows**, this usually works directly:

```
pip install -r requirements.txt
```

If that fails, try `pip3 install -r requirements.txt`, or the same
virtual-environment approach as above (`python -m venv venv` then
`venv\Scripts\activate`).

## 4. Run the app

Still in this folder (with `(venv)` active if you used one), run:

```
streamlit run app.py
```

A browser tab should open automatically at `http://localhost:8501`. If not, open
that address manually.

## 5. Using it

1. In the left sidebar, enter a name (optional), birth date, birth time, and
   birth place (e.g. "Jaipur, India"), then click **Generate my chart**.
   - This needs an internet connection to look up the place and calculate
     the chart accurately.
   - If it can't find your city, tick "Use manual coordinates" and enter
     latitude/longitude instead (you can look these up by searching
     "[city name] latitude longitude").
2. Paste your Anthropic API key when prompted (only kept for your current
   session — never saved to a file).
3. Start chatting — ask about love, career, timing, "what's today's energy for
   me", etc.

## How your API key is billed

Every message you send costs a small amount (a fraction of a cent to a few
cents depending on length) via your Anthropic account. Keep an eye on usage at
console.anthropic.com if you share this with other people.

## What's next (see the business plan doc)

This is a local prototype for you to try and tweak. To turn it into a real
product other people can use, the next steps are: deploy it online (free on
Streamlit Community Cloud), add user accounts and payments, and pick a niche
angle to stand out. All covered in `Astrology_App_Business_Plan.docx`.

## Customizing the AI's personality

Open `app.py` and edit the `SYSTEM_PROMPT_TEMPLATE` variable near the top —
that's the instructions that shape how "Stella" talks. Change the name,
tone, or add rules (e.g. "always mention today's date's transits") and
save the file; Streamlit will reload automatically.

## Files in this folder

- `app.py` — the Streamlit app (the UI and chat logic).
- `astro_engine.py` — calculates the real birth chart (Swiss Ephemeris astronomy).
- `requirements.txt` — the Python packages needed.
