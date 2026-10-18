# Submission checklist — Nebius x NVIDIA Global AI Hackathon

Source: official rules + overview on nebiusglobalaihackathon.devpost.com (read 2026-10-18).
**Deadline: Friday Oct 30, 2026, 10:00 am PT** — aim to submit Oct 29.

## Track
**Best Apps and Agents** — "any app or agent someone would actually use … a productivity tool or copilot". Margin is a
study copilot. (Personal AI asks for NemoClaw/OpenShell/Hermes Agent-style tooling we don't use.)
Stage One is pass/fail on genuinely fitting the track's goal.

Their tip for this track: Nemotron 3 Ultra for serious reasoning, Nano/Super for fast calls; Nebius Serverless
Endpoints/Jobs encouraged (not required).

## Prizes we can realistically target
- Overall ($20k / $10k / $6k) — open to all tracks. A project wins ONE overall award, OR one track award + one bonus.
- Track: Best Apps and Agents → NVIDIA Jetson Orin Nano.
- Bonus: **Best Use of Tavily ($3,000)** — needs a functional runtime call to the Tavily API.
- Bonus: Most Valuable Feedback ($100 + swag, 10 winners) — complete, specific feedback.
- NOT eligible: City Winner Awards (only for entrants who attended a Builders & Brews event).

## Required
- [ ] Registered on Devpost ("Join Hackathon") — confirm
- [ ] Eligible: age of majority where you live (18+ in CT) — confirm
- [x] Runs on Nebius Token Factory (runtime calls) and uses NVIDIA open models (Nemotron 3.5 Super-VL, 3.5 Lightning, 3 Super; Parakeet)
- [x] Newly created during the Submission Period (Aug 26 – Oct 30; started Oct 16)
- [x] Public repo: https://github.com/hansonq888/margin (keep it complete as we build)
- [x] MIT LICENSE, detected by GitHub (shows in About)
- [ ] README: setup + how to run; **highlight NVIDIA models used, where Token Factory helped, other Nebius tools**
- [ ] Working demo URL: hosted demo and/or test build (GitHub release + install steps); free, unrestricted access through Dec 15
- [ ] Demo video < 3 min, public on YouTube, shows it working on the real device; no copyrighted music; avoid third-party trademarks/logos
- [ ] Text description: features and functionality
- [ ] Track selected on the form
- [ ] Feedback on Token Factory / AI Cloud / NVIDIA models & tools
- [ ] Must install and run consistently and work as shown in the video

## Be ready for
- Winner verification includes "the role of the winner in the creation of the submission" — be able to explain every
  component. The ideas and product decisions are yours; know how the code works.
- Third-party licenses: Parakeet (NVIDIA open model), KaTeX (MIT), Python libs — fine. Check FERMAT's license before using it.
- Don't commit personal data: `.env`, `frames/`, `experiments/notes/` are git-ignored. Keep it that way.

## Feedback notes to include (collect as we go)
- Nemotron 3 Nano Omni was not available on Token Factory (third-party listings suggested it might be); switched to Nemotron 3.5 Super-VL.
- The hackathon promo credit form gave no confirmation and the $25 didn't appear right away; only the $1 trial credit showed.
- Token Factory accepted `chat_template_kwargs.enable_thinking=false` and `response_format=json_object` on all Nemotron models we used — worked well.
- Data retention: inference logs kept unless Zero Data Retention is enabled; training/"speculative decoding" terms unclear from public docs.
