# Reddit drafts

## r/DJs, r/Beatmatch, r/mixes

**Title:** I built a mixtape player that lets you skip songs inside a DJ mix (open source engine)

Every DJ mix is one long file, so you can't skip the song you hate or find the one you love. I got tired of scrubbing blind in the car and built Cue.

Drop in a mix (SoundCloud download, Mixcloud purchase, whatever you own). It finds where each song starts, automatically, no marking. Next on the steering wheel skips to the next song, double-Next skips to the next mixtape, and the lock screen shows the song and the mixtape name.

Under the hood: fingerprinting to tell which song is playing, a transition model trained on 170 real mixes to place the exact cut, free preview clips aligned into the mix for songs the fingerprinter misses, and the DJ's published tracklist for naming remixes. On cut-style mixes it's as accurate as a human annotator; on long house blends it's within about half a minute.

The engine is open source and CPU-only, and it writes normal m4a files with chapters, so it works with any chapter-aware player too. iPhone app is in TestFlight.

What would make this useful to you? Which mixes should I test it on?

## r/JPOD-adjacent communities (Shambhala, Bass Coast groups)

**Title:** Tracklists that actually skip: every JPOD BlissCoast mix, songs marked

Same body, with: "I ran all 17 JPOD mixes through it. Red Red Wine in Dubs & Daydreams starts at 3:53, if you ever wanted to jump straight there."
