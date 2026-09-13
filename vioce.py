"""A small, extensible voice assistant.

Run with a microphone: ``python app.py``
Run in a terminal: ``python app.py --text``
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import webbrowser
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
from urllib.parse import quote_plus

# PyAudio's official wheels currently do not target Python 3.14. On Windows,
# PyAudioWPatch supplies a compatible PortAudio backend. SpeechRecognition
# imports it by the conventional ``pyaudio`` name, so register that alias
# before importing SpeechRecognition.
try:
    import pyaudiowpatch as pyaudio
    sys.modules.setdefault("pyaudio", pyaudio)
except ImportError:
    pass

try:
    import speech_recognition as sr
except ImportError:  # Allows text mode to work before microphone packages are installed.
    sr = None  # type: ignore[assignment]

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None  # type: ignore[assignment]

try:
    from nltk.stem import PorterStemmer
except ImportError:
    PorterStemmer = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class Command:
    """The result of turning a natural-language utterance into an action."""

    intent: str
    query: Optional[str] = None


class IntentParser:
    """Lightweight NLU that accepts conversational wording, not fixed commands.

    Stemming makes variants such as ``searching`` and ``searched`` resolve to the
    same intent. It uses NLTK when installed, but needs no downloaded NLTK data.
    """

    GREETINGS = {"hello", "hi", "hey", "greeting", "morning", "afternoon", "evening"}
    TIME_WORDS = {"time", "clock", "hour"}
    DATE_WORDS = {"date", "day", "today", "calendar"}
    EXIT_WORDS = {"exit", "quit", "goodbye", "bye", "stop"}
    SEARCH_PATTERNS = (
        r"\b(?:search(?:ing)?(?:\s+(?:the\s+)?web)?\s+(?:for\s+)?)\s*(?P<query>.+)",
        r"\b(?:google|lookup|look\s+up|find\s+(?:out\s+)?about)\s+(?P<query>.+)",
        r"\b(?:show\s+me|tell\s+me\s+about)\s+(?P<query>.+)",
    )

    def __init__(self) -> None:
        self.stemmer = PorterStemmer() if PorterStemmer else None

    def parse(self, utterance: str) -> Command:
        clean = " ".join(utterance.lower().strip().split())
        if not clean:
            return Command("unknown")

        for pattern in self.SEARCH_PATTERNS:
            match = re.search(pattern, clean)
            if match:
                query = self._trim_query(match.group("query"))
                return Command("search", query) if query else Command("search_missing_query")

        tokens = set(self._tokens(clean))
        stems = {self._stem(token) for token in tokens}
        if tokens & self.EXIT_WORDS:
            return Command("exit")
        if tokens & self.TIME_WORDS or stems & {"time", "clock", "hour"}:
            return Command("time")
        if tokens & self.DATE_WORDS or stems & {"date", "day", "today", "calendar"}:
            return Command("date")
        if tokens & self.GREETINGS:
            return Command("greeting")
        return Command("unknown")

    @staticmethod
    def _tokens(text: str) -> Iterable[str]:
        return re.findall(r"[a-zA-Z]+", text)

    def _stem(self, token: str) -> str:
        return self.stemmer.stem(token) if self.stemmer else token.rstrip("s")

    @staticmethod
    def _trim_query(query: str) -> str:
        query = re.sub(r"\b(?:please|for me)\b", "", query).strip(" .?!")
        return query


class VoiceAssistant:
    """Coordinates speech I/O and actions while keeping responses spoken."""

    def __init__(
        self,
        *,
        speak_enabled: bool = True,
        browser_opener: Callable[[str], bool] = webbrowser.open,
        now: Callable[[], dt.datetime] = dt.datetime.now,
    ) -> None:
        self.parser = IntentParser()
        self.browser_opener = browser_opener
        self.now = now
        self.engine = None
        if speak_enabled and pyttsx3:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", 175)
            except Exception as error:
                print(f"Text-to-speech is unavailable: {error}")

    def speak(self, message: str) -> None:
        """Print and, when available, say every assistant response."""
        print(f"Assistant: {message}")
        if self.engine:
            self.engine.say(message)
            self.engine.runAndWait()

    def listen(self) -> Optional[str]:
        """Capture one microphone phrase and convert it to text."""
        if sr is None:
            self.speak("Speech recognition is not installed. Run pip install -r requirements.txt.")
            return None

        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                self.speak("Listening.")
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=6, phrase_time_limit=12)
            heard = recognizer.recognize_google(audio)
            print(f"You: {heard}")
            return heard
        except sr.WaitTimeoutError:
            self.speak("I did not hear anything. Please try again.")
        except sr.UnknownValueError:
            self.speak("I did not understand that. Please repeat your request.")
        except sr.RequestError:
            self.speak("The speech recognition service is unavailable. Please check your connection and try again.")
        except AttributeError as error:
            # SpeechRecognition raises AttributeError when its PyAudio backend
            # is absent. Catch it so beginners receive an actionable message.
            if "PyAudio" in str(error):
                self.speak("Microphone support is missing. Install PyAudioWPatch, then run the program again.")
            else:
                self.speak(f"The microphone could not start: {error}")
        except OSError as error:
            self.speak(f"I cannot access the microphone: {error}")
        return None

    def handle(self, utterance: str) -> bool:
        """Act on one utterance. Returns False only when the session should end."""
        command = self.parser.parse(utterance)
        current = self.now()
        if command.intent == "greeting":
            self.speak("Hello! What would you like help with?")
        elif command.intent == "time":
            self.speak(f"It is {current.strftime('%I:%M %p').lstrip('0')}.")
        elif command.intent == "date":
            self.speak(f"Today is {current.strftime('%A, %B %d, %Y')}.")
        elif command.intent == "search":
            assert command.query is not None
            url = f"https://www.google.com/search?q={quote_plus(command.query)}"
            self.browser_opener(url)
            self.speak(f"Searching the web for {command.query}.")
        elif command.intent == "search_missing_query":
            self.speak("What would you like me to search for?")
        elif command.intent == "exit":
            self.speak("Goodbye!")
            return False
        else:
            self.speak("I did not understand that. Please repeat, or ask for the time, date, or a web search.")
        return True

    def run_voice(self) -> None:
        self.speak("Voice assistant ready. Say hello, ask for the time or date, or ask me to search the web.")
        while True:
            utterance = self.listen()
            if utterance and not self.handle(utterance):
                break

    def run_text(self) -> None:
        self.speak("Text mode ready. Type a request, or type quit to exit.")
        while True:
            try:
                utterance = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.speak("Goodbye!")
                break
            if utterance and not self.handle(utterance):
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="A beginner-friendly Python voice assistant")
    parser.add_argument("--text", action="store_true", help="Use typed input instead of a microphone")
    parser.add_argument("--mute", action="store_true", help="Print responses without text-to-speech")
    args = parser.parse_args()
    assistant = VoiceAssistant(speak_enabled=not args.mute)
    if args.text:
        assistant.run_text()
    else:
        assistant.run_voice()


if __name__ == "__main__":
    main()
