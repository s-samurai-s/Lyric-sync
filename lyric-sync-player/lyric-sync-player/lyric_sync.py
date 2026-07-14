#!/usr/bin/env python3
import sys
import os
import re
import time
import shutil
import argparse
import threading
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pygame

try:
    import pygame  # type: ignore
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


LRC_TIME_RE = re.compile(r'\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]')
LRC_META_RE = re.compile(r'^\[([a-zA-Z]+):(.*)\]$')


def parse_lrc(path):
    offset_ms = 0
    lines = []

    with open(path, 'r', encoding='utf-8-sig') as f:
        raw_lines = f.readlines()

    for raw in raw_lines:
        raw = raw.rstrip('\r\n')
        stripped = raw.strip()
        if not stripped:
            continue

        meta = LRC_META_RE.match(stripped)
        timestamps = LRC_TIME_RE.findall(raw)

        if meta and not timestamps:
            if meta.group(1).lower() == 'offset':
                try:
                    offset_ms = int(meta.group(2).strip())
                except ValueError:
                    pass
            continue

        if not timestamps:
            continue

        text = LRC_TIME_RE.sub('', raw).strip()

        for m, s, cs in timestamps:
            minutes, seconds = int(m), int(s)
            centi = (cs or '0').ljust(3, '0')[:3]
            total = minutes * 60 + seconds + int(centi) / 1000.0
            total += offset_ms / 1000.0
            lines.append((total, text))

    lines.sort(key=lambda x: x[0])
    return lines


def visual_width(text):
    width = 0
    for ch in text:
        width += 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
    return width


def center_text(text, term_width):
    pad = max(0, (term_width - visual_width(text)) // 2)
    return ' ' * pad + text


def clear_line():
    cols = shutil.get_terminal_size().columns
    sys.stdout.write('\r' + ' ' * cols + '\r')


def typewriter_reveal(text, duration, term_width, stop_event):
    n = len(text)
    if n == 0 or stop_event.is_set():
        return
    
    target_fps = 120
    frame_interval = 1.0 / target_fps
    
    start = time.time()
    prev_displayed = -1
    
    while True:
        if stop_event.is_set():
            break
        
        frame_start = time.time()
        elapsed = frame_start - start
        
        if duration > 0:
            progress = min(elapsed / duration, 1.0)
            char_pos = progress * n
        else:
            char_pos = n
        
        displayed_chars = int(char_pos)
        
        if displayed_chars != prev_displayed:
            clear_line()
            sys.stdout.write(center_text(text[:displayed_chars], term_width))
            sys.stdout.flush()
            prev_displayed = displayed_chars
        
        if elapsed >= duration:
            break
        
        frame_elapsed = time.time() - frame_start
        sleep_time = max(0.0001, frame_interval - frame_elapsed)
        time.sleep(sleep_time)
    
    clear_line()
    sys.stdout.write(center_text(text, term_width))
    sys.stdout.flush()


def start_audio(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


class LyricPlayer:
    def __init__(self, audio_path, lyric_path, use_audio=True,
                 speed_cps=25.0, gap_ratio=0.85):
        self.audio_path = audio_path
        self.use_audio = use_audio and PYGAME_AVAILABLE and audio_path is not None
        self.speed_cps = speed_cps
        self.gap_ratio = gap_ratio
        self.lyrics = parse_lrc(lyric_path)
        self.stop_event = threading.Event()

        if not self.lyrics:
            raise ValueError("No timed lyric lines found in the LRC file.")

    def elapsed(self, start_time):
        if self.use_audio:
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms is not None and pos_ms >= 0:
                return pos_ms / 1000.0
        return time.time() - start_time

    def run(self):
        term_width = shutil.get_terminal_size().columns

        if self.use_audio:
            try:
                start_audio(self.audio_path)
            except Exception as e:
                print(f"[warning] audio playback failed ({e}); continuing without audio.")
                self.use_audio = False

        start_time = time.time()
        sys.stdout.write('\n\n')

        try:
            for idx, (timestamp, text) in enumerate(self.lyrics):
                while True:
                    remaining = timestamp - self.elapsed(start_time)
                    if remaining <= 0:
                        break
                    time.sleep(min(remaining, 0.005))

                if idx + 1 < len(self.lyrics):
                    gap = max(self.lyrics[idx + 1][0] - timestamp, 0.1)
                else:
                    gap = max(len(text) / self.speed_cps, 1.0) / self.gap_ratio

                target = len(text) / self.speed_cps if self.speed_cps > 0 else 0
                duration = min(target, gap * self.gap_ratio)

                typewriter_reveal(text, duration, term_width, self.stop_event)
                sys.stdout.write('\n')

            sys.stdout.write('\n\n')

        except KeyboardInterrupt:
            self.stop_event.set()
            if self.use_audio:
                pygame.mixer.music.stop()
            sys.stdout.write('\n\nStopped.\n')
            sys.exit(0)


def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


MEDIA_DIR_NAME = "media"
AUDIO_EXTENSIONS = ('.mp3', '.wav', '.ogg', '.flac')
AVG_WORD_LENGTH = 5.0


def wpm_to_cps(wpm):
    return (wpm * AVG_WORD_LENGTH) / 60.0


def analyze_audio(audio_path):
    if not LIBROSA_AVAILABLE:
        return None
    
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.tempo(onset_env=onset_env, sr=sr)
        
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_db = librosa.power_to_db(S, ref=1e-6)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = spec_centroid.mean()
        
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = rms.mean()
        
        suggested_cps = (tempo / 60.0) * AVG_WORD_LENGTH
        
        return {
            'tempo_bpm': float(tempo),
            'spectral_centroid': float(avg_centroid),
            'rms': float(avg_rms),
            'suggested_cps': float(suggested_cps),
            'suggested_wpm': float((suggested_cps * 60.0) / AVG_WORD_LENGTH)
        }
    except Exception as e:
        print(f"[warning] audio analysis failed ({e})")
        return None


def get_media_dir():
    media_dir = os.path.join(get_base_dir(), MEDIA_DIR_NAME)
    is_new = not os.path.isdir(media_dir)
    os.makedirs(media_dir, exist_ok=True)
    if is_new:
        readme_path = os.path.join(media_dir, "PUT_YOUR_FILES_HERE.txt")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(
                "Paste your audio file and your .lrc lyric file in this folder,\n"
                "then run the player again.\n\n"
                "Example:\n"
                "  media/song.mp3\n"
                "  media/song.lrc\n\n"
                f"Supported audio formats: {', '.join(AUDIO_EXTENSIONS)}\n"
                "If no audio file is present, the player runs in visual-only mode.\n"
            )
    return media_dir


def find_media_files():
    media_dir = get_media_dir()
    lyric_path, audio_path = None, None
    lyric_matches, audio_matches = [], []

    for name in sorted(os.listdir(media_dir)):
        full = os.path.join(media_dir, name)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext == '.lrc':
            lyric_matches.append(full)
        elif ext in AUDIO_EXTENSIONS:
            audio_matches.append(full)

    warnings = []
    if lyric_matches:
        lyric_path = lyric_matches[0]
        if len(lyric_matches) > 1:
            warnings.append(
                f"found {len(lyric_matches)} .lrc files, using: {os.path.basename(lyric_path)}")
    if audio_matches:
        audio_path = audio_matches[0]
        if len(audio_matches) > 1:
            warnings.append(
                f"found {len(audio_matches)} audio files, using: {os.path.basename(audio_path)}")

    return audio_path, lyric_path, media_dir, warnings


def auto_setup():
    print("=== Lyric Sync Player ===\n")
    audio_path, lyric_path, media_dir, warnings = find_media_files()

    for w in warnings:
        print(f"[note] {w}")

    if lyric_path is None:
        print(f"No .lrc lyric file found in:\n  {media_dir}\n")
        print("Paste your audio file and .lrc lyric file into that folder, then run this again.")
        _pause_on_exe()
        sys.exit(1)

    print(f"Lyrics: {os.path.basename(lyric_path)}")
    if audio_path:
        print(f"Audio:  {os.path.basename(audio_path)}")
    else:
        print("Audio:  none found — running in visual-only mode.")
    print()

    analysis = None
    if audio_path and LIBROSA_AVAILABLE:
        print("Analyzing audio...")
        analysis = analyze_audio(audio_path)
        if analysis:
            print(f"  Tempo: {analysis['tempo_bpm']:.1f} BPM")
            print(f"  Voice tone (centroid): {analysis['spectral_centroid']:.0f} Hz")
            print(f"  Suggested speed: {analysis['suggested_wpm']:.0f} WPM ({analysis['suggested_cps']:.1f} CPS)")
            print()

    while True:
        default_msg = ""
        if analysis:
            default_msg = f"(default {analysis['suggested_wpm']:.0f} WPM)"
        
        speed_input = input(f"Enter speed {default_msg} or press Enter: ").strip()
        if not speed_input:
            if analysis:
                speed = analysis['suggested_cps']
            else:
                speed = 25.0
            break
        try:
            if speed_input.endswith(' wpm'):
                wpm_value = float(speed_input[:-4].strip())
                speed = wpm_to_cps(wpm_value)
                break
            else:
                speed = float(speed_input)
                break
        except ValueError:
            print("Invalid input. Enter a number or '? wpm' (e.g., '60 wpm')")

    use_audio = audio_path is not None
    return audio_path, lyric_path, use_audio, speed


def main():
    if len(sys.argv) == 1:
        audio_path, lyric_path, use_audio, speed = auto_setup()
    else:
        parser = argparse.ArgumentParser(
            description="Play audio with terminal lyric-sync animation, in any language/script."
        )
        parser.add_argument('audio', help="Path to the audio file (mp3, wav, ogg...), "
                                           "or the lyric file if using --no-audio as the only arg")
        parser.add_argument('lyrics', nargs='?', help="Path to the LRC lyric file")
        parser.add_argument('--no-audio', action='store_true',
                             help="Run without playing audio (visual-only, real-time-clock based)")
        parser.add_argument('--speed', type=float, default=None,
                             help="Typewriter speed in characters/second (default: 25)")
        parser.add_argument('--wpm', type=float, default=None,
                             help="Typewriter speed in words per minute (overrides --speed)")
        args = parser.parse_args()

        if args.no_audio and args.lyrics is None:
            audio_path, lyric_path = None, args.audio
        else:
            audio_path, lyric_path = args.audio, args.lyrics

        if lyric_path is None:
            parser.error("the lyric LRC file is required")

        if not os.path.isfile(lyric_path):
            print(f"Lyric file not found: {lyric_path}")
            _pause_on_exe()
            sys.exit(1)

        use_audio = not args.no_audio
        if use_audio and not os.path.isfile(audio_path):
            print(f"Audio file not found: {audio_path}")
            _pause_on_exe()
            sys.exit(1)

        if args.wpm is not None:
            speed = wpm_to_cps(args.wpm)
        elif args.speed is not None:
            speed = args.speed
        else:
            if use_audio and LIBROSA_AVAILABLE and os.path.isfile(audio_path):
                print("Analyzing audio...")
                analysis = analyze_audio(audio_path)
                if analysis:
                    print(f"  Tempo: {analysis['tempo_bpm']:.1f} BPM")
                    print(f"  Suggested speed: {analysis['suggested_wpm']:.0f} WPM")
                    speed = analysis['suggested_cps']
                else:
                    speed = 25.0
            else:
                speed = 25.0

    if use_audio and not PYGAME_AVAILABLE:
        print("[info] pygame not installed — falling back to visual-only mode.")
        print("       Install audio support with: pip install pygame")
        use_audio = False

    player = LyricPlayer(audio_path if use_audio else None, lyric_path,
                          use_audio=use_audio, speed_cps=speed)
    player.run()


def _pause_on_exe():
    if getattr(sys, 'frozen', False):
        input("\nPress Enter to exit...")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f"\n[error] {exc}")
        _pause_on_exe()
        sys.exit(1)
    else:
        _pause_on_exe()
