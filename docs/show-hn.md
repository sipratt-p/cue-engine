# Show HN draft

**Title:** Show HN: Cue – skip songs inside DJ mixes and mixtapes

**Body:**

DJ mixes are one long file. If a song you hate comes on you scrub blind; if a song you love comes on you can't find it again. Mixcloud shows tracklists but won't let you skip, because licensing treats the mix as one unit.

Cue is an open-source engine plus a small iPhone player (the app isn't public yet; the engine is) that finds where each song starts inside a mix, so Next on your car stereo jumps to the next song, and double-Next jumps to the next mixtape.

How it finds the starts (all measured against DJ-published timestamps, boundaries within 30 s):

- Fingerprinting says which song is playing and roughly when it was first heard. Any provider works; the interface is pluggable.
- A small transition detector trained on the public DJ Mix Dataset places the exact cut just before that. Fingerprint offsets alone put 62 of 190 boundaries within 10 s; fingerprint plus detector puts 99.
- Free 30-second preview clips of the page's tracklist, aligned into the mix with tempo-tolerant chroma DTW, anchor songs without any fingerprint service. On a dancehall mix that alone matches Shazam.
- Published tracklists name remixes correctly and set the song count when nothing fingerprints.

Cut-style mixes land at the level of human annotators (about 9 s disagreement). Long beat-matched house blends are still "right neighbourhood", around 25 s median, and honestly ambiguous even for people.

Output is a plain m4a with MP4 chapters, so any chapter-aware player can use it. The engine is Python, CPU only, about a minute per hour of audio. The app is SwiftUI and keeps everything on the phone.

Repo: https://github.com/sipratt-p/cue-engine. 30-second demo of the app: https://github.com/sipratt-p/cue-engine/blob/main/docs/media/demo.mp4. Happy to talk about what didn't work: LLMs listening to audio placed boundaries no better than a novelty curve and invented titles, and beat-grid snapping did nothing against DJ-typed timestamps.
